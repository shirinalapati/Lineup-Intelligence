"""Offensive hitter archetypes via unsupervised clustering.

Pipeline
--------
1. Load season-level hitter features from undervalued comprehensive_stats
   (2025 preferred for fitting; 2026 for assignment), falling back to rates
   derived from plate_appearances (+ optional DiamondIQ batter baselines).
2. Standardize features (z-scores).
3. Fit KMeans and GaussianMixture for k in {4..8}; select k by mean silhouette
   on a held-out subsample (or full sample when N is modest).
4. Prefer the method (KMeans vs GMM) with the better silhouette at the chosen k.
5. Assign human-readable labels ONLY after inspecting standardized cluster
   centers — never before fitting.
6. Persist ``data/models/archetypes.json`` and
   ``data/models/archetype_assignments.parquet``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..config import settings


# Canonical feature names used inside this module (fractions in [0, 1] where rates).
FEATURE_COLS = [
    "bb_pct",
    "k_pct",
    "iso",
    "xwoba",
    "hardhit_pct",
    "barrel_pct",
    "gb_pct",
    "fb_pct",
    "ld_pct",
    "pull_pct",
    "oppo_pct",
]

# Minimum plate appearances to include a hitter in clustering / assignment.
MIN_PA_TRAIN = 150
MIN_PA_ASSIGN = 40

K_RANGE = range(4, 9)
RANDOM_STATE = 42

# Approximate linear weights for deriving an ISO / xwOBA proxy from PA outcomes
# when Statcast season tables are unavailable.
WOBA_WEIGHTS = {
    "K": 0.0,
    "BB_HBP": 0.690,
    "1B": 0.880,
    "2B": 1.250,
    "3B": 1.600,
    "HR": 2.000,
    "OUT_IP": 0.0,
}


def _as_fraction(series: pd.Series) -> pd.Series:
    """Normalize rates that may be stored as 0–1 or 0–100."""
    s = pd.to_numeric(series, errors="coerce")
    # If the bulk of non-null values exceed 1, treat as percent points.
    nonzero = s.dropna()
    if len(nonzero) and nonzero.median() > 1.0:
        s = s / 100.0
    return s


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_undervalued_features(path: Path, season: int) -> pd.DataFrame:
    """Map heterogeneous undervalued CSV schemas onto FEATURE_COLS."""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, low_memory=False)
    if "player_id" not in raw.columns:
        raise ValueError(f"{path} missing player_id")

    pa_col = _pick_col(raw, ["PA", "pa", "PA_fg_new"])
    mapping = {
        "bb_pct": ["BB%", "bb_percent", "bb_pct"],
        "k_pct": ["K%", "k_percent", "k_pct"],
        "iso": ["ISO", "iso", "xiso"],
        "xwoba": ["xwOBA", "xwoba", "est_woba", "xwOBA_fg_new"],
        "hardhit_pct": ["HardHit%", "hard_hit_percent", "Hard%", "HardHit%_fg_new"],
        "barrel_pct": ["Barrel%", "barrel_batted_rate", "Barrel%_fg_new"],
        "gb_pct": ["GB%", "gb_percent", "GB%_fg_new"],
        "fb_pct": ["FB%", "fb_percent", "FB%_fg_new"],
        "ld_pct": ["LD%", "ld_percent", "LD%_fg_new"],
        "pull_pct": ["Pull%", "pull_percent", "Pull%_fg_new"],
        "oppo_pct": ["Oppo%", "oppo_percent", "Oppo%_fg_new"],
    }

    out = pd.DataFrame({"player_id": pd.to_numeric(raw["player_id"], errors="coerce")})
    out["season"] = int(season)
    if pa_col:
        out["n_pa"] = pd.to_numeric(raw[pa_col], errors="coerce")
    else:
        out["n_pa"] = np.nan

    for feat, cands in mapping.items():
        col = _pick_col(raw, cands)
        if col is None:
            out[feat] = np.nan
        else:
            out[feat] = _as_fraction(raw[col])

    out = out.dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(int)
    # Prefer one row per player (keep highest PA if duplicates)
    out = out.sort_values("n_pa", ascending=False).drop_duplicates("player_id", keep="first")
    return out.reset_index(drop=True)


def _derive_features_from_pa(pa_path: Path, season: int) -> pd.DataFrame:
    """Fallback feature matrix from plate-appearance outcome rates."""
    if not pa_path.exists():
        raise FileNotFoundError(
            f"Neither undervalued stats nor plate appearances available "
            f"(missing {pa_path})"
        )
    pa = pd.read_parquet(pa_path)
    pa = pa[pa["season"] == season].copy()
    if pa.empty:
        raise ValueError(f"No plate appearances for season {season} in {pa_path}")

    rows = []
    for pid, g in pa.groupby("batter_id"):
        n = len(g)
        vc = g["outcome_class"].value_counts()
        k = float(vc.get("K", 0)) / n
        bb = float(vc.get("BB_HBP", 0)) / n
        s = float(vc.get("1B", 0)) / n
        d = float(vc.get("2B", 0)) / n
        t = float(vc.get("3B", 0)) / n
        hr = float(vc.get("HR", 0)) / n
        # ISO proxy from extra bases per PA (not AB); diagnostic, not FanGraphs ISO.
        iso = d + 2.0 * t + 3.0 * hr
        xwoba = sum(WOBA_WEIGHTS.get(c, 0.0) * (float(vc.get(c, 0)) / n) for c in WOBA_WEIGHTS)
        rows.append(
            {
                "player_id": int(pid),
                "season": int(season),
                "n_pa": int(n),
                "bb_pct": bb,
                "k_pct": k,
                "iso": iso,
                "xwoba": xwoba,
                # Spray / contact quality unavailable from outcome-only PA export.
                "hardhit_pct": np.nan,
                "barrel_pct": np.nan,
                "gb_pct": np.nan,
                "fb_pct": np.nan,
                "ld_pct": np.nan,
                "pull_pct": np.nan,
                "oppo_pct": np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_hitter_features(
    season: int,
    *,
    prefer_undervalued: bool = True,
    min_pa: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return (features_df, source_description).

    Tries undervalued comprehensive_stats first for the requested season, then
    derives rates from ``plate_appearances.parquet``.
    """
    min_pa = MIN_PA_TRAIN if min_pa is None else min_pa
    uv_path = {
        2025: settings.undervalued_stats_2025,
        2026: settings.undervalued_stats_2026,
    }.get(season)

    source = ""
    feats: pd.DataFrame | None = None
    if prefer_undervalued and uv_path is not None and Path(uv_path).exists():
        feats = _load_undervalued_features(Path(uv_path), season)
        source = f"undervalued:{uv_path}"
    else:
        pa_path = settings.processed_dir / "plate_appearances.parquet"
        feats = _derive_features_from_pa(pa_path, season)
        source = f"plate_appearances:{pa_path}"

    # Drop columns that are entirely missing so we do not invent values.
    usable = [c for c in FEATURE_COLS if c in feats.columns and feats[c].notna().any()]
    if len(usable) < 3:
        # Hard fail rather than cluster on almost-empty feature space.
        raise RuntimeError(
            f"Insufficient non-null hitter features for season {season} "
            f"(source={source}); need ≥3 of {FEATURE_COLS}"
        )

    keep = ["player_id", "season", "n_pa"] + usable
    out = feats[keep].copy()
    out = out[out["n_pa"].fillna(0) >= min_pa].reset_index(drop=True)
    if len(out) < 30:
        raise RuntimeError(
            f"Only {len(out)} hitters with ≥{min_pa} PA for season {season} "
            f"(source={source}); need more data to cluster"
        )
    return out, source


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    # Drop all-null columns; median-impute remaining (never invent new features).
    keep: list[str] = []
    X = pd.DataFrame(index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        med = s.median()
        if pd.isna(med):
            continue
        X[c] = s.fillna(med)
        keep.append(c)
    if len(keep) < 3:
        raise RuntimeError(f"Need ≥3 usable feature columns after cleaning; got {keep}")
    return X[keep].to_numpy(dtype=np.float64), keep, X.index


def _silhouette_for_labels(X: np.ndarray, labels: np.ndarray) -> float:
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("-inf")
    # Silhouette is O(n^2); subsample for speed on large N.
    if len(labels) > 2500:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(labels), size=2500, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def select_clustering(
    X_std: np.ndarray,
    k_range: range = K_RANGE,
) -> dict[str, Any]:
    """Choose method + k by silhouette; also record GMM BIC for transparency."""
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        km_labels = km.fit_predict(X_std)
        km_sil = _silhouette_for_labels(X_std, km_labels)
        results.append(
            {
                "method": "kmeans",
                "k": k,
                "silhouette": km_sil,
                "inertia": float(km.inertia_),
            }
        )
        cand = {
            "method": "kmeans",
            "k": k,
            "silhouette": km_sil,
            "model": km,
            "labels": km_labels,
            "centers": km.cluster_centers_,
        }
        if best is None or km_sil > best["silhouette"]:
            best = cand

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_STATE,
                n_init=5,
                max_iter=300,
            )
            gmm_labels = gmm.fit_predict(X_std)
        gmm_sil = _silhouette_for_labels(X_std, gmm_labels)
        results.append(
            {
                "method": "gmm",
                "k": k,
                "silhouette": gmm_sil,
                "bic": float(gmm.bic(X_std)),
                "aic": float(gmm.aic(X_std)),
            }
        )
        cand = {
            "method": "gmm",
            "k": k,
            "silhouette": gmm_sil,
            "model": gmm,
            "labels": gmm_labels,
            "centers": gmm.means_,
        }
        if best is None or gmm_sil > best["silhouette"]:
            best = cand

    assert best is not None
    return {"best": best, "grid": results}


def label_clusters_from_centers(
    centers_std: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """Map cluster ids → human labels from dominant standardized traits.

    Labels are assigned after fitting. Ties break by archetype_id order; each
    label is used at most once (fallback suffixes if needed).
    """
    feat = {name: i for i, name in enumerate(feature_names)}

    def z(i: int, name: str, default: float = 0.0) -> float:
        j = feat.get(name)
        return float(centers_std[i, j]) if j is not None else default

    # Score candidates for every cluster, then greedily assign unique labels
    # by descending confidence so distinctive profiles claim names first.
    candidates: list[tuple[float, int, str]] = []
    for i in range(centers_std.shape[0]):
        bb, k = z(i, "bb_pct"), z(i, "k_pct")
        iso = z(i, "iso")
        xw = z(i, "xwoba")
        hard = z(i, "hardhit_pct")
        barrel = z(i, "barrel_pct")
        gb, fb, ld = z(i, "gb_pct"), z(i, "fb_pct"), z(i, "ld_pct")
        pull, oppo = z(i, "pull_pct"), z(i, "oppo_pct")
        power = max(iso, barrel, hard, xw)

        ranked: list[tuple[float, str]] = []
        if bb > 0.35 and power > 0.35 and k > 0.25:
            ranked.append((bb + power + k, "Three True Outcomes"))
        if bb > 0.35 and power > 0.25:
            ranked.append((bb + power, "Patient Power"))
        if bb < -0.2 and power > 0.35:
            ranked.append((power - bb, "Aggressive Power"))
        if k < -0.35 and bb > 0.15 and power < 0.25:
            ranked.append((-k + bb, "Patient Contact"))
        if k < -0.4 and power < 0.15:
            ranked.append((-k, "Contact"))
        if gb > 0.45 and power < 0.3:
            ranked.append((gb, "Ground-Ball Contact"))
        if fb > 0.4 and power > 0.2:
            ranked.append((fb + power, "Fly-Ball Power"))
        if oppo > 0.35 and pull < 0.1:
            ranked.append((oppo - pull, "Spray Contact"))
        if ld > 0.4:
            ranked.append((ld, "Line-Drive Contact"))
        if pull > 0.45 and power > 0.2:
            ranked.append((pull + power, "Pull Power"))
        if abs(bb) < 0.3 and abs(k) < 0.3 and abs(power) < 0.3:
            ranked.append((0.1, "Balanced"))
        if power > 0.45:
            ranked.append((power, "Power"))
        if bb > 0.45:
            ranked.append((bb, "Patient"))
        if not ranked:
            # Fall back to the single most extreme standardized feature.
            j = int(np.argmax(np.abs(centers_std[i])))
            fname = feature_names[j]
            sign = "High" if centers_std[i, j] > 0 else "Low"
            pretty = {
                "bb_pct": "Walk Rate",
                "k_pct": "Strikeout Rate",
                "iso": "ISO",
                "xwoba": "xwOBA",
                "hardhit_pct": "Hard Hit",
                "barrel_pct": "Barrel",
                "gb_pct": "Ground Ball",
                "fb_pct": "Fly Ball",
                "ld_pct": "Line Drive",
                "pull_pct": "Pull",
                "oppo_pct": "Oppo",
            }.get(fname, fname)
            ranked.append((abs(float(centers_std[i, j])), f"{sign} {pretty}"))

        ranked.sort(key=lambda t: t[0], reverse=True)
        conf, name = ranked[0]
        candidates.append((conf, i, name))

    candidates.sort(key=lambda t: t[0], reverse=True)
    used: set[str] = set()
    labels: dict[int, str] = {}

    # Recompute ranked name lists so losers can fall through alternatives.
    def ranked_names(i: int) -> list[str]:
        bb, k = z(i, "bb_pct"), z(i, "k_pct")
        iso = z(i, "iso")
        xw = z(i, "xwoba")
        hard = z(i, "hardhit_pct")
        barrel = z(i, "barrel_pct")
        gb, fb, ld = z(i, "gb_pct"), z(i, "fb_pct"), z(i, "ld_pct")
        pull, oppo = z(i, "pull_pct"), z(i, "oppo_pct")
        power = max(iso, barrel, hard, xw)
        scored: list[tuple[float, str]] = []
        if bb > 0.35 and power > 0.35 and k > 0.25:
            scored.append((bb + power + k, "Three True Outcomes"))
        if bb > 0.35 and power > 0.25:
            scored.append((bb + power, "Patient Power"))
        if bb < -0.2 and power > 0.35:
            scored.append((power - bb, "Aggressive Power"))
        if k < -0.35 and bb > 0.15 and power < 0.25:
            scored.append((-k + bb, "Patient Contact"))
        if k < -0.4 and power < 0.15:
            scored.append((-k, "Contact"))
        if gb > 0.45 and power < 0.3:
            scored.append((gb, "Ground-Ball Contact"))
        if fb > 0.4 and power > 0.2:
            scored.append((fb + power, "Fly-Ball Power"))
        if oppo > 0.35 and pull < 0.1:
            scored.append((oppo - pull, "Spray Contact"))
        if ld > 0.4:
            scored.append((ld, "Line-Drive Contact"))
        if pull > 0.45 and power > 0.2:
            scored.append((pull + power, "Pull Power"))
        if power > 0.45:
            scored.append((power, "Power"))
        if bb > 0.45:
            scored.append((bb, "Patient"))
        if k > 0.45 and power < 0.2:
            scored.append((k, "Strikeout-Prone"))
        if abs(bb) < 0.3 and abs(k) < 0.3 and abs(power) < 0.3:
            scored.append((0.05, "Balanced"))
        # Always offer a unique extreme-feature fallback.
        j = int(np.argmax(np.abs(centers_std[i])))
        fname = feature_names[j]
        sign = "High" if centers_std[i, j] > 0 else "Low"
        pretty = {
            "bb_pct": "Walk Rate",
            "k_pct": "Strikeout Rate",
            "iso": "ISO",
            "xwoba": "xwOBA",
            "hardhit_pct": "Hard Hit",
            "barrel_pct": "Barrel",
            "gb_pct": "Ground Ball",
            "fb_pct": "Fly Ball",
            "ld_pct": "Line Drive",
            "pull_pct": "Pull",
            "oppo_pct": "Oppo",
        }.get(fname, fname)
        scored.append((abs(float(centers_std[i, j])) * 0.01, f"{sign} {pretty}"))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Deduplicate while preserving order
        out: list[str] = []
        for _, name in scored:
            if name not in out:
                out.append(name)
        return out

    for conf, i, _preferred in candidates:
        chosen = None
        for name in ranked_names(i):
            if name not in used:
                chosen = name
                break
        if chosen is None:
            chosen = f"Archetype {i}"
        used.add(chosen)
        labels[i] = chosen
    return labels


def fit_archetypes(
    train_season: int = 2025,
    assign_seasons: list[int] | None = None,
) -> dict[str, Any]:
    """Fit archetypes on ``train_season`` and assign players for ``assign_seasons``."""
    assign_seasons = assign_seasons or [train_season, settings.target_season]

    train_df, train_source = load_hitter_features(train_season, min_pa=MIN_PA_TRAIN)
    X_raw, feat_names, _ = _feature_matrix(train_df)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    selection = select_clustering(X_std)
    best = selection["best"]
    labels = np.asarray(best["labels"])
    centers = np.asarray(best["centers"])
    label_map = label_clusters_from_centers(centers, feat_names)

    # Center means in original units for interpretability.
    centers_raw = scaler.inverse_transform(centers)
    cluster_payload = []
    for cid in range(centers.shape[0]):
        cluster_payload.append(
            {
                "archetype_id": int(cid),
                "archetype_label": label_map[cid],
                "n_train_players": int((labels == cid).sum()),
                "center_std": {f: float(centers[cid, j]) for j, f in enumerate(feat_names)},
                "center_raw": {f: float(centers_raw[cid, j]) for j, f in enumerate(feat_names)},
            }
        )

    meta = {
        "train_season": train_season,
        "train_source": train_source,
        "assign_seasons": assign_seasons,
        "method": best["method"],
        "k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "exploratory": float(best["silhouette"]) < 0.25,
        "display_name": (
            "Exploratory offensive profile groups"
            if float(best["silhouette"]) < 0.25
            else "Offensive profile groups"
        ),
        "features": feat_names,
        "feature_means": {f: float(m) for f, m in zip(feat_names, scaler.mean_)},
        "feature_scales": {f: float(s) for f, s in zip(feat_names, scaler.scale_)},
        "min_pa_train": MIN_PA_TRAIN,
        "min_pa_assign": MIN_PA_ASSIGN,
        "selection_grid": selection["grid"],
        "clusters": cluster_payload,
        "notes": (
            "Labels were assigned after inspecting standardized cluster centers. "
            "Handedness is intentionally excluded from the feature set so clusters "
            "reflect offensive profile rather than platoon side. "
            + (
                "Silhouette is weak (<0.25); treat groups as exploratory, not strong natural archetypes."
                if float(best["silhouette"]) < 0.25
                else ""
            )
        ),
    }

    # Assign train players + additional seasons.
    assignment_frames: list[pd.DataFrame] = []
    train_assign = train_df[["player_id", "season", "n_pa"]].copy()
    train_assign["archetype_id"] = labels.astype(int)
    train_assign["archetype_label"] = train_assign["archetype_id"].map(label_map)
    assignment_frames.append(train_assign)

    for season in assign_seasons:
        if season == train_season:
            continue
        try:
            sdf, src = load_hitter_features(season, min_pa=MIN_PA_ASSIGN)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            meta.setdefault("assign_warnings", []).append(
                {"season": season, "error": str(exc)}
            )
            continue
        # Align columns to training features; median-fill using train medians.
        for f in feat_names:
            if f not in sdf.columns:
                sdf[f] = np.nan
        X_s = sdf[feat_names].copy()
        train_medians = pd.Series(scaler.mean_, index=feat_names)
        for f in feat_names:
            X_s[f] = X_s[f].fillna(train_medians[f])
        Xs = scaler.transform(X_s.to_numpy(dtype=np.float64))
        if best["method"] == "kmeans":
            pred = best["model"].predict(Xs)
        else:
            pred = best["model"].predict(Xs)
        part = sdf[["player_id", "season", "n_pa"]].copy()
        part["archetype_id"] = pred.astype(int)
        part["archetype_label"] = part["archetype_id"].map(label_map)
        part["assign_source"] = src
        assignment_frames.append(part)
        meta.setdefault("assign_sources", {})[str(season)] = src

    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments = assignments.drop_duplicates(["player_id", "season"], keep="last")

    out_models = settings.models_dir
    out_models.mkdir(parents=True, exist_ok=True)
    archetypes_path = out_models / "archetypes.json"
    assign_path = out_models / "archetype_assignments.parquet"
    archetypes_path.write_text(json.dumps(meta, indent=2))
    assignments.to_parquet(assign_path, index=False)
    meta["artifacts"] = {
        "archetypes_json": str(archetypes_path),
        "assignments_parquet": str(assign_path),
        "n_assignments": int(len(assignments)),
    }
    return meta


def main() -> None:
    meta = fit_archetypes()
    print(
        f"method={meta['method']} k={meta['k']} silhouette={meta['silhouette']:.4f}"
    )
    for c in meta["clusters"]:
        print(
            f"  {c['archetype_id']}: {c['archetype_label']} "
            f"(n_train={c['n_train_players']})"
        )
    print(f"wrote {meta['artifacts']['archetypes_json']}")
    print(f"wrote {meta['artifacts']['assignments_parquet']} "
          f"({meta['artifacts']['n_assignments']} rows)")


if __name__ == "__main__":
    main()
