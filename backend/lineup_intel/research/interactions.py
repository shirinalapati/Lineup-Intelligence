"""Adjacent-hitter residual interaction research.

Estimates *associations* (not causal chemistry) between previous and current
batters after controlling for batter baseline quality, pitcher hand, base/out
state, and season.

Artifacts land under ``data/artifacts/research/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from ..config import settings


WOBA_WEIGHTS = {
    "K": 0.0,
    "BB_HBP": 0.690,
    "1B": 0.880,
    "2B": 1.250,
    "3B": 1.600,
    "HR": 2.000,
    "OUT_IP": 0.0,
}

# Reliability tiers for empirical-Bayes pair effects (by posterior reliability).
# reliability = n / (n + n0); thresholds chosen for interpretability, not magic.
TIER_STRONG = 0.70
TIER_MODERATE = 0.40

OUTCOME_CONTEXT_GROUPS = {
    "strikeout": {"K"},
    "walk": {"BB_HBP"},
    "single": {"1B"},
    "extra_base": {"2B", "3B", "HR"},
    "out_in_play": {"OUT_IP"},
    "reached_base": {"BB_HBP", "1B", "2B", "3B", "HR"},
}


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def _load_run_expectancy() -> dict[tuple[int, str], float]:
    path = settings.vendor_models_dir / "run_expectancy_v1.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    table = (raw.get("body") or raw)["payload"]["table"]
    out: dict[tuple[int, str], float] = {}
    for key, cell in table.items():
        outs_s, bases = key.split("|", 1)
        out[(int(outs_s), bases)] = float(cell["re"])
    return out


def _artifact_dir() -> Path:
    d = settings.artifacts_dir / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_adjacent_pa_frame(
    pa_path: Path | None = None,
    archetype_path: Path | None = None,
) -> pd.DataFrame:
    """Attach previous-batter context within each half-inning.

    Drops the first PA of every half-inning (no previous batter). Target columns:
    ``woba_value`` (linear weights) and optional ``re_start`` for diagnostics.
    """
    pa_path = _require(pa_path or (settings.processed_dir / "plate_appearances.parquet"), "PA table")
    pa = pd.read_parquet(pa_path)
    needed = {
        "game_pk", "season", "inning", "half", "at_bat_index", "batter_id",
        "pitcher_hand", "outs_start", "bases_start", "outcome_class", "runs_scored",
    }
    missing = needed - set(pa.columns)
    if missing:
        raise ValueError(f"plate_appearances missing columns: {sorted(missing)}")

    pa = pa[pa["outcome_class"].isin(WOBA_WEIGHTS)].copy()
    pa = pa.sort_values(["game_pk", "inning", "half", "at_bat_index"]).reset_index(drop=True)

    gcols = ["game_pk", "inning", "half"]
    pa["prev_batter_id"] = pa.groupby(gcols, sort=False)["batter_id"].shift(1)
    pa["prev_outcome"] = pa.groupby(gcols, sort=False)["outcome_class"].shift(1)
    pa["prev_runs_scored"] = pa.groupby(gcols, sort=False)["runs_scored"].shift(1)
    adj = pa.dropna(subset=["prev_batter_id"]).copy()
    adj["prev_batter_id"] = adj["prev_batter_id"].astype(int)
    adj["batter_id"] = adj["batter_id"].astype(int)
    adj["woba_value"] = adj["outcome_class"].map(WOBA_WEIGHTS).astype(float)
    adj["reached"] = (~adj["outcome_class"].isin(["K", "OUT_IP"])).astype(int)

    re_table = _load_run_expectancy()
    if re_table:
        adj["re_start"] = [
            re_table.get((int(o), str(b)), np.nan)
            for o, b in zip(adj["outs_start"], adj["bases_start"])
        ]
    else:
        adj["re_start"] = np.nan

    # Approximate batting slot: order of first appearance within game+half.
    first = (
        adj.sort_values("at_bat_index")
        .groupby(["game_pk", "half", "batter_id"], sort=False)["at_bat_index"]
        .first()
        .reset_index(name="first_ab")
    )
    first["slot"] = (
        first.groupby(["game_pk", "half"])["first_ab"]
        .rank(method="dense")
        .clip(upper=9)
        .astype(int)
    )
    adj = adj.merge(first[["game_pk", "half", "batter_id", "slot"]], on=["game_pk", "half", "batter_id"], how="left")

    # Batter season baselines from the same PA table (no leakage across seasons).
    bas = (
        adj.groupby(["season", "batter_id"])
        .agg(
            batter_n=("woba_value", "size"),
            batter_woba=("woba_value", "mean"),
            batter_k=("outcome_class", lambda s: float((s == "K").mean())),
            batter_bb=("outcome_class", lambda s: float((s == "BB_HBP").mean())),
            batter_iso=(
                "outcome_class",
                lambda s: float(
                    ((s == "2B").sum() + 2 * (s == "3B").sum() + 3 * (s == "HR").sum()) / max(len(s), 1)
                ),
            ),
        )
        .reset_index()
    )
    adj = adj.merge(bas, on=["season", "batter_id"], how="left")
    prev_bas = bas.rename(
        columns={
            "batter_id": "prev_batter_id",
            "batter_n": "prev_n",
            "batter_woba": "prev_woba",
            "batter_k": "prev_k",
            "batter_bb": "prev_bb",
            "batter_iso": "prev_iso",
        }
    )
    adj = adj.merge(prev_bas, on=["season", "prev_batter_id"], how="left")

    # Archetypes if available (season-specific assignment).
    archetype_path = archetype_path or (settings.models_dir / "archetype_assignments.parquet")
    if archetype_path.exists():
        arch = pd.read_parquet(archetype_path)
        arch = arch[["player_id", "season", "archetype_id", "archetype_label"]].drop_duplicates(
            ["player_id", "season"]
        )
        adj = adj.merge(
            arch.rename(
                columns={
                    "player_id": "batter_id",
                    "archetype_id": "batter_arch",
                    "archetype_label": "batter_arch_label",
                }
            ),
            on=["batter_id", "season"],
            how="left",
        )
        adj = adj.merge(
            arch.rename(
                columns={
                    "player_id": "prev_batter_id",
                    "archetype_id": "prev_arch",
                    "archetype_label": "prev_arch_label",
                }
            ),
            on=["prev_batter_id", "season"],
            how="left",
        )
    else:
        adj["batter_arch"] = np.nan
        adj["batter_arch_label"] = None
        adj["prev_arch"] = np.nan
        adj["prev_arch_label"] = None

    return adj


def _design_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    *,
    encoder: OneHotEncoder | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, OneHotEncoder | None]:
    X_num = df[feature_cols].astype(float).fillna(0.0).to_numpy(dtype=np.float64)
    if not categorical_cols:
        return X_num, encoder
    cats = df[categorical_cols].astype(str).fillna("NA")
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(cats)
    else:
        if encoder is None:
            raise ValueError("encoder required when fit=False")
        X_cat = encoder.transform(cats)
    return np.hstack([X_num, X_cat]), encoder


def fit_residual_model(adj: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ridge residualization of wOBA value on talent + state controls.

    Controls: batter baseline wOBA/K/BB/ISO, pitcher hand, outs, bases, season.
    Residuals are the primary input to pair / archetype EB shrinkage.
    """
    work = adj.dropna(subset=["woba_value", "batter_woba"]).copy()
    num = ["batter_woba", "batter_k", "batter_bb", "batter_iso"]
    cat = ["pitcher_hand", "outs_start", "bases_start", "season"]
    X, enc = _design_matrix(work, num, cat, fit=True)
    y = work["woba_value"].to_numpy(dtype=np.float64)
    # Mild L2; alpha chosen for stability with collinear one-hots, not CV beauty.
    model = Ridge(alpha=5.0, random_state=42)
    model.fit(X, y)
    work["yhat"] = model.predict(X)
    work["residual"] = work["woba_value"] - work["yhat"]
    meta = {
        "target": "woba_value",
        "model": "Ridge(alpha=5)",
        "n": int(len(work)),
        "rmse": float(np.sqrt(mean_squared_error(y, work["yhat"]))),
        "numeric_features": num,
        "categorical_features": cat,
        "note": (
            "Residuals are wOBA-linear-weight minus expected value given batter "
            "baseline rates, pitcher hand, base/out state, and season."
        ),
    }
    return work, meta


def _eb_shrink_effects(
    group_df: pd.DataFrame,
    group_cols: list[str],
    residual_col: str = "residual",
    min_n: int = 5,
    prior_n0: float = 50.0,
) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of group mean residuals toward 0.

    Uses a fixed prior strength ``prior_n0`` (pseudo-counts) for stable,
    interpretable reliability tiers — analogous to PA-rate shrinkage. Also
    records a method-of-moments τ² estimate for diagnostics.

    Posterior mean μ̂ = (n / (n + n0)) * ȳ.
    """
    g = (
        group_df.groupby(group_cols, dropna=False)[residual_col]
        .agg(n="size", effect_raw="mean", sd="std")
        .reset_index()
    )
    g = g[g["n"] >= min_n].copy()
    if g.empty:
        for col in ("effect", "se", "reliability", "reliability_tier", "shrink_factor",
                    "prior_n0", "sigma2", "tau2"):
            g[col] = pd.Series(dtype=float if col != "reliability_tier" else object)
        return g

    sigma2 = float(group_df[residual_col].var(ddof=1))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.mean(g["sd"].fillna(0.0) ** 2)) or 1e-6

    weights = g["n"].to_numpy(dtype=np.float64)
    ybar = g["effect_raw"].to_numpy(dtype=np.float64)
    mean_all = np.average(ybar, weights=weights)
    between = np.average((ybar - mean_all) ** 2, weights=weights)
    avg_sigma_over_n = float(np.average(sigma2 / weights, weights=weights))
    tau2 = max(between - avg_sigma_over_n, 0.0)

    n0 = float(prior_n0)
    g["shrink_factor"] = g["n"] / (g["n"] + n0)
    g["effect"] = g["shrink_factor"] * g["effect_raw"]
    g["se"] = np.sqrt(sigma2 / (g["n"] + n0))
    g["reliability"] = g["shrink_factor"]
    g["reliability_tier"] = np.where(
        g["reliability"] >= TIER_STRONG,
        "strong",
        np.where(g["reliability"] >= TIER_MODERATE, "moderate", "limited"),
    )
    g["prior_n0"] = n0
    g["sigma2"] = float(sigma2)
    g["tau2_mom"] = float(tau2)
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def _mode_non_null(s: pd.Series):
    """Most common non-null label in a group; None if every value is missing."""
    s = s.dropna()
    if s.empty:
        return None
    s = s[s.astype(str).str.strip().ne("")]
    s = s[~s.astype(str).str.lower().isin({"nan", "none", "null"})]
    if s.empty:
        return None
    return s.mode().iloc[0]


def player_pair_effects(resid_df: pd.DataFrame) -> pd.DataFrame:
    """EB-shrunk adjacency effects for (prev_batter_id → batter_id)."""
    pairs = _eb_shrink_effects(resid_df, ["prev_batter_id", "batter_id"], min_n=8)
    # Attach archetype labels when present (mode of non-null labels, not first PA).
    if "prev_arch_label" in resid_df.columns:
        lab = (
            resid_df.groupby(["prev_batter_id", "batter_id"], dropna=False)
            .agg(
                prev_arch_label=("prev_arch_label", _mode_non_null),
                batter_arch_label=("batter_arch_label", _mode_non_null),
                n_pa=("residual", "size"),
            )
            .reset_index()
        )
        pairs = pairs.merge(lab.drop(columns=["n_pa"]), on=["prev_batter_id", "batter_id"], how="left")
    return pairs


def archetype_pair_matrix(resid_df: pd.DataFrame) -> pd.DataFrame:
    """EB-shrunk previous-archetype × next-archetype residual matrix."""
    work = resid_df.dropna(subset=["prev_arch", "batter_arch"]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "prev_arch", "batter_arch", "prev_arch_label", "batter_arch_label",
                "n", "effect_raw", "effect", "se", "reliability", "reliability_tier",
            ]
        )
    work["prev_arch"] = work["prev_arch"].astype(int)
    work["batter_arch"] = work["batter_arch"].astype(int)
    mat = _eb_shrink_effects(work, ["prev_arch", "batter_arch"], min_n=25)
    labels = (
        work.groupby(["prev_arch", "batter_arch"])
        .agg(
            prev_arch_label=("prev_arch_label", "first"),
            batter_arch_label=("batter_arch_label", "first"),
        )
        .reset_index()
    )
    return mat.merge(labels, on=["prev_arch", "batter_arch"], how="left")


def previous_outcome_context(resid_df: pd.DataFrame) -> dict[str, Any]:
    """Residual performance after previous-batter outcome groups.

    Because residuals already control for current outs/bases (and batter
    baseline / pitcher hand / season), remaining differences are closer to
    "identity of the prior event" than to "runners on base."
    """
    work = resid_df.dropna(subset=["prev_outcome", "residual"]).copy()
    rows = []
    for name, outcomes in OUTCOME_CONTEXT_GROUPS.items():
        mask = work["prev_outcome"].isin(outcomes)
        sub = work.loc[mask, "residual"]
        if len(sub) < 50:
            continue
        rows.append(
            {
                "prev_outcome_group": name,
                "n": int(len(sub)),
                "mean_residual": float(sub.mean()),
                "se": float(sub.std(ddof=1) / np.sqrt(len(sub))),
                "mean_woba": float(work.loc[mask, "woba_value"].mean()),
            }
        )
    # Also raw outcome class breakdown
    by_class = (
        work.groupby("prev_outcome")
        .agg(
            n=("residual", "size"),
            mean_residual=("residual", "mean"),
            se=("residual", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return {
        "groups": rows,
        "by_outcome_class": by_class.to_dict(orient="records"),
        "note": (
            "Means are residuals after base/out controls, so they are not the "
            "trivial 'runner on base boost.'"
        ),
    }


def _model_feature_sets() -> dict[str, dict[str, list[str]]]:
    """Incremental feature bundles for Models 1–5."""
    return {
        "m1_talent": {
            "num": ["batter_woba", "batter_k", "batter_bb", "batter_iso"],
            "cat": ["pitcher_hand", "season"],
        },
        "m2_slot": {
            "num": ["batter_woba", "batter_k", "batter_bb", "batter_iso"],
            "cat": ["pitcher_hand", "season", "slot"],
        },
        "m3_state": {
            "num": ["batter_woba", "batter_k", "batter_bb", "batter_iso"],
            "cat": ["pitcher_hand", "season", "slot", "outs_start", "bases_start"],
        },
        "m4_prev_feats": {
            "num": [
                "batter_woba", "batter_k", "batter_bb", "batter_iso",
                "prev_woba", "prev_k", "prev_bb", "prev_iso",
            ],
            "cat": ["pitcher_hand", "season", "slot", "outs_start", "bases_start", "prev_outcome"],
        },
        "m5_arch_interact": {
            "num": [
                "batter_woba", "batter_k", "batter_bb", "batter_iso",
                "prev_woba", "prev_k", "prev_bb", "prev_iso",
            ],
            "cat": [
                "pitcher_hand", "season", "slot", "outs_start", "bases_start",
                "prev_outcome", "arch_pair",
            ],
        },
    }


def incremental_predictive_value(
    adj: pd.DataFrame,
    train_season: int = 2024,
    valid_season: int = 2025,
    apply_season: int | None = 2026,
) -> dict[str, Any]:
    """Temporal train→validate lift for Models 1–5.

    Primary metric: RMSE on wOBA linear weights.
    Secondary: log-loss on binary reached-base indicator (calibration-friendly).
    Reports honest deltas vs Model 1; negative lift means the richer model did
    not help out of sample.

    To limit leakage, validation/application rows use *prior-season* batter and
    previous-batter talent rates (e.g. 2024 rates when scoring 2025) rather than
    same-season pooled rates that include the PA being predicted.
    """
    work = adj.copy()
    if "prev_arch" in work.columns and "batter_arch" in work.columns:
        work["arch_pair"] = (
            work["prev_arch"].fillna(-1).astype(int).astype(str)
            + "->"
            + work["batter_arch"].fillna(-1).astype(int).astype(str)
        )
    else:
        work["arch_pair"] = "NA"

    talent_cols = ["batter_woba", "batter_k", "batter_bb", "batter_iso",
                   "prev_woba", "prev_k", "prev_bb", "prev_iso"]
    # Build prior-season talent lookup from batter_id rates.
    batter_rates = (
        work.groupby(["season", "batter_id"], as_index=False)
        .agg(
            batter_woba=("woba_value", "mean"),
            batter_k=("outcome_class", lambda s: float((s == "K").mean())),
            batter_bb=("outcome_class", lambda s: float((s == "BB_HBP").mean())),
            batter_iso=(
                "outcome_class",
                lambda s: float(
                    ((s == "2B").sum() + 2 * (s == "3B").sum() + 3 * (s == "HR").sum())
                    / max(len(s), 1)
                ),
            ),
        )
    )

    def attach_prior_talent(df: pd.DataFrame, feature_season: int) -> pd.DataFrame:
        rates = batter_rates[batter_rates["season"] == feature_season].drop(columns=["season"])
        out = df.drop(columns=[c for c in talent_cols if c in df.columns], errors="ignore")
        out = out.merge(rates, on="batter_id", how="left")
        prev_rates = rates.rename(
            columns={
                "batter_id": "prev_batter_id",
                "batter_woba": "prev_woba",
                "batter_k": "prev_k",
                "batter_bb": "prev_bb",
                "batter_iso": "prev_iso",
            }
        )
        out = out.merge(prev_rates, on="prev_batter_id", how="left")
        return out

    train = attach_prior_talent(
        work[work["season"] == train_season], train_season
    ).dropna(subset=["woba_value", "batter_woba"])
    # Validation uses train_season talent only (no 2025 same-season rates).
    valid = attach_prior_talent(
        work[work["season"] == valid_season], train_season
    ).dropna(subset=["woba_value", "batter_woba"])
    if len(train) < 1000 or len(valid) < 1000:
        raise RuntimeError(
            f"Insufficient PAs for temporal split "
            f"(train {train_season}={len(train)}, valid {valid_season}={len(valid)})"
        )

    bundles = _model_feature_sets()
    # Skip m5 if archetypes missing
    if work["arch_pair"].eq("NA").all() or work["prev_arch"].isna().all():
        bundles.pop("m5_arch_interact", None)

    results = []
    baseline_rmse = None
    baseline_ll = None

    for name, feats in bundles.items():
        need = [c for c in feats["num"] if c in train.columns]
        tr = train.dropna(subset=need)
        va = valid.dropna(subset=need)
        if len(tr) < 500 or len(va) < 500:
            results.append(
                {
                    "model": name,
                    "skipped": True,
                    "reason": f"insufficient rows after dropna (train={len(tr)}, valid={len(va)})",
                }
            )
            continue
        Xtr, enc = _design_matrix(tr, need, feats["cat"], fit=True)
        Xva, _ = _design_matrix(va, need, feats["cat"], encoder=enc, fit=False)
        ytr = tr["woba_value"].to_numpy(dtype=np.float64)
        yva = va["woba_value"].to_numpy(dtype=np.float64)

        reg = Ridge(alpha=5.0, random_state=42)
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xva)
        rmse = float(np.sqrt(mean_squared_error(yva, pred)))

        clf = LogisticRegression(max_iter=400, C=0.5, solver="lbfgs")
        ytr_b = tr["reached"].to_numpy(dtype=np.int64)
        yva_b = va["reached"].to_numpy(dtype=np.int64)
        try:
            clf.fit(Xtr, ytr_b)
            proba = np.clip(clf.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
            ll = float(log_loss(yva_b, proba))
            ll_error = None
        except Exception as exc:  # noqa: BLE001
            ll = float("nan")
            ll_error = str(exc)

        if baseline_rmse is None:
            baseline_rmse = rmse
            baseline_ll = ll

        row = {
            "model": name,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "train_season": train_season,
            "valid_season": valid_season,
            "talent_feature_season": train_season,
            "rmse": rmse,
            "rmse_lift_vs_m1": float(baseline_rmse - rmse),
            "logloss_reached": ll,
            "logloss_lift_vs_m1": (
                float(baseline_ll - ll) if np.isfinite(ll) and np.isfinite(baseline_ll) else None
            ),
            "numeric_features": need,
            "categorical_features": feats["cat"],
        }
        if ll_error:
            row["logloss_error"] = ll_error
        results.append(row)

    apply_payload = None
    if apply_season is not None and (work["season"] == apply_season).any():
        apply_payload = {
            "season": apply_season,
            "n_pa": int((work["season"] == apply_season).sum()),
            "note": (
                "Target-season rows are available for optional scoring; metrics above are "
                "strictly train→validate temporal lifts and do not peek at the apply season."
            ),
        }

    return {
        "train_season": train_season,
        "valid_season": valid_season,
        "models": results,
        "apply": apply_payload,
        "interpretation": (
            "Positive rmse_lift_vs_m1 / logloss_lift_vs_m1 means the richer model "
            "improved out-of-sample prediction relative to individual talent. "
            "Zero/negative lift is an informative negative result about interaction value. "
            "Validation talent features are taken from the train season to avoid "
            "same-season leakage."
        ),
    }


def multi_fold_incremental_predictive_value(
    adj: pd.DataFrame,
    folds: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Rolling temporal folds for interaction lift; compare Model 5 vs M1 and M4."""
    seasons = sorted(int(s) for s in adj["season"].dropna().unique())
    if folds is None:
        folds = []
        for a, b in ((2023, 2024), (2024, 2025), (2025, 2026)):
            if a in seasons and b in seasons:
                n_a = int((adj["season"] == a).sum())
                n_b = int((adj["season"] == b).sum())
                if n_a >= 1000 and n_b >= 1000:
                    folds.append((a, b))
        if not folds and len(seasons) >= 2:
            folds = [(seasons[-2], seasons[-1])]

    fold_results = []
    for tr, va in folds:
        try:
            one = incremental_predictive_value(adj, train_season=tr, valid_season=va, apply_season=None)
        except Exception as exc:  # noqa: BLE001
            fold_results.append({
                "train_season": tr,
                "valid_season": va,
                "error": str(exc),
            })
            continue
        models = {m["model"]: m for m in one.get("models") or [] if not m.get("skipped")}
        m1 = models.get("m1_talent") or models.get("m1") or next(
            (m for k, m in models.items() if k.startswith("m1")), None
        )
        m4 = next((m for k, m in models.items() if k.startswith("m4")), None)
        m5 = next((m for k, m in models.items() if k.startswith("m5")), None)
        fold_results.append({
            "train_season": tr,
            "valid_season": va,
            "models": one.get("models"),
            "m5_vs_m1_rmse_lift": (
                None
                if not (m1 and m5)
                else float(m1["rmse"] - m5["rmse"])
            ),
            "m5_vs_m4_rmse_lift": (
                None
                if not (m4 and m5)
                else float(m4["rmse"] - m5["rmse"])
            ),
            "m5_vs_m1_logloss_lift": (
                None
                if not (m1 and m5 and m1.get("logloss_reached") is not None and m5.get("logloss_reached") is not None)
                else float(m1["logloss_reached"] - m5["logloss_reached"])
            ),
            "m5_vs_m4_logloss_lift": (
                None
                if not (m4 and m5 and m4.get("logloss_reached") is not None and m5.get("logloss_reached") is not None)
                else float(m4["logloss_reached"] - m5["logloss_reached"])
            ),
        })

    def _agg(key: str) -> dict[str, Any]:
        vals = [f[key] for f in fold_results if isinstance(f.get(key), (int, float)) and np.isfinite(f[key])]
        if not vals:
            return {"n": 0}
        arr = np.asarray(vals, dtype=float)
        # Simple bootstrap CI for mean lift
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(1000):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boots.append(float(sample.mean()))
        return {
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "sign_stability": float(np.mean(arr > 0)),
            "folds": vals,
        }

    m5_diag = diagnose_model5_logloss(adj, folds)

    return {
        "folds": fold_results,
        "aggregate": {
            "m5_vs_m1_rmse_lift": _agg("m5_vs_m1_rmse_lift"),
            "m5_vs_m4_rmse_lift": _agg("m5_vs_m4_rmse_lift"),
            "m5_vs_m1_logloss_lift": _agg("m5_vs_m1_logloss_lift"),
            "m5_vs_m4_logloss_lift": _agg("m5_vs_m4_logloss_lift"),
        },
        "model5_diagnosis": m5_diag,
        "interpretation": (
            "Multi-fold temporal validation. Positive lift favors Model 5. "
            "Sign stability is the fraction of folds with positive lift."
        ),
    }


def diagnose_model5_logloss(
    adj: pd.DataFrame,
    folds: list[tuple[int, int]],
) -> dict[str, Any]:
    """Investigate why Model 5 log-loss may degrade vs Models 1/4."""
    notes: list[str] = []
    evidence: dict[str, Any] = {}
    if not folds:
        return {"status": "skipped", "reason": "no folds"}

    # Use the most recent fold
    tr_s, va_s = folds[-1]
    try:
        one = incremental_predictive_value(adj, train_season=tr_s, valid_season=va_s, apply_season=None)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}

    models = {m["model"]: m for m in one.get("models") or [] if not m.get("skipped")}
    m5_key = next((k for k in models if k.startswith("m5")), None)
    m4_key = next((k for k in models if k.startswith("m4")), None)
    m1_key = next((k for k in models if k.startswith("m1")), None)
    if not m5_key:
        return {"status": "skipped", "reason": "m5 not present (archetypes missing?)"}

    m5 = models[m5_key]
    m4 = models.get(m4_key) if m4_key else None
    m1 = models.get(m1_key) if m1_key else None

    # Category cardinality / sparsity check for arch_pair
    work = adj.copy()
    if "prev_arch" in work.columns and "batter_arch" in work.columns:
        work["arch_pair"] = (
            work["prev_arch"].fillna(-1).astype(int).astype(str)
            + "->"
            + work["batter_arch"].fillna(-1).astype(int).astype(str)
        )
        tr_pairs = set(work.loc[work["season"] == tr_s, "arch_pair"].unique())
        va_pairs = set(work.loc[work["season"] == va_s, "arch_pair"].unique())
        evidence["n_arch_pairs_train"] = len(tr_pairs)
        evidence["n_arch_pairs_valid"] = len(va_pairs)
        evidence["n_arch_pairs_valid_unseen"] = len(va_pairs - tr_pairs)
        evidence["frac_valid_rows_unseen_pair"] = float(
            work.loc[work["season"] == va_s, "arch_pair"].isin(va_pairs - tr_pairs).mean()
        )
        if evidence["frac_valid_rows_unseen_pair"] > 0.05:
            notes.append(
                "Train/validation category mismatch: material share of validation rows use "
                "arch_pair levels unseen in training — classic OHE generalization issue."
            )

    if m5.get("logloss_reached") is not None and m4 and m4.get("logloss_reached") is not None:
        delta = float(m5["logloss_reached"] - m4["logloss_reached"])
        evidence["m5_minus_m4_logloss"] = delta
        if delta > 0.01:
            notes.append(
                "Model 5 log-loss degrades vs Model 4 by a material amount on this fold."
            )
            notes.append(
                "Likely causes to weigh: overfitting via high-cardinality arch_pair encodings, "
                "regularization too weak for the expanded design matrix, or genuine lack of "
                "residual interaction signal (noise dominates)."
            )
        elif delta > 0:
            notes.append("Mild log-loss degradation vs Model 4; treat as weak/no signal.")
        else:
            notes.append("Model 5 does not degrade log-loss vs Model 4 on this fold.")

    if m1 and m5.get("rmse") is not None and m1.get("rmse") is not None:
        evidence["m5_minus_m1_rmse"] = float(m5["rmse"] - m1["rmse"])

    # Regularization probe: re-fit m5-like with stronger C if we can
    evidence["regularization_note"] = (
        "Baseline uses Ridge(alpha=5) and LogisticRegression(C=0.5). "
        "High-cardinality interaction features often need stronger regularization."
    )

    status = "degrades" if evidence.get("m5_minus_m4_logloss", 0) > 0.01 else (
        "mild_or_none" if evidence.get("m5_minus_m4_logloss", 0) > 0 else "improves_or_flat"
    )
    return {
        "status": status,
        "fold": {"train": tr_s, "valid": va_s},
        "evidence": evidence,
        "notes": notes,
        "models_snapshot": {
            k: {
                "rmse": models[k].get("rmse"),
                "logloss_reached": models[k].get("logloss_reached"),
                "n_train": models[k].get("n_train"),
                "n_valid": models[k].get("n_valid"),
            }
            for k in (m1_key, m4_key, m5_key)
            if k
        },
    }


def run_interaction_research(
    *,
    fit_archetypes_if_missing: bool = True,
) -> dict[str, Any]:
    """End-to-end research run; writes artifacts under data/artifacts/research/."""
    arch_path = settings.models_dir / "archetype_assignments.parquet"
    if fit_archetypes_if_missing and not arch_path.exists():
        from .archetypes import fit_archetypes

        fit_archetypes()

    adj = build_adjacent_pa_frame()
    resid_df, resid_meta = fit_residual_model(adj)
    pairs = player_pair_effects(resid_df)
    arch_mat = archetype_pair_matrix(resid_df)
    prev_ctx = previous_outcome_context(resid_df)
    incremental = incremental_predictive_value(adj)
    multi_fold = multi_fold_incremental_predictive_value(adj)

    out = _artifact_dir()
    resid_path = out / "adjacency_residuals.parquet"
    pairs_path = out / "player_pair_effects.parquet"
    arch_path_out = out / "archetype_pair_effects.parquet"
    prev_path = out / "previous_outcome_context.json"
    incr_path = out / "incremental_predictive_value.json"
    multi_path = out / "interaction_multifold.json"
    summary_path = out / "interaction_summary.json"

    # Persist a lean residual frame (not every PA column).
    lean_cols = [
        c for c in [
            "game_pk", "season", "game_date", "batter_id", "prev_batter_id",
            "outcome_class", "prev_outcome", "woba_value", "yhat", "residual",
            "outs_start", "bases_start", "pitcher_hand", "slot",
            "batter_arch", "prev_arch", "batter_arch_label", "prev_arch_label",
        ]
        if c in resid_df.columns
    ]
    resid_df[lean_cols].to_parquet(resid_path, index=False)
    pairs.to_parquet(pairs_path, index=False)
    arch_mat.to_parquet(arch_path_out, index=False)
    prev_path.write_text(json.dumps(prev_ctx, indent=2))
    incr_path.write_text(json.dumps(incremental, indent=2))
    multi_path.write_text(json.dumps(multi_fold, indent=2, default=str))

    # Summary counts for findings / UI — numbers only, no causal language.
    n_pairs = int(len(pairs))
    n_strong = int((pairs["reliability_tier"] == "strong").sum()) if n_pairs else 0
    n_mod = int((pairs["reliability_tier"] == "moderate").sum()) if n_pairs else 0
    summary = {
        "status": "ok",
        "residual_model": resid_meta,
        "n_adjacent_pa": int(len(resid_df)),
        "n_player_pairs": n_pairs,
        "n_player_pairs_strong": n_strong,
        "n_player_pairs_moderate": n_mod,
        "n_player_pairs_limited": int((pairs["reliability_tier"] == "limited").sum()) if n_pairs else 0,
        "pair_effect_mean_abs_strong": (
            float(pairs.loc[pairs["reliability_tier"] == "strong", "effect"].abs().mean())
            if n_strong
            else None
        ),
        "n_archetype_pairs": int(len(arch_mat)),
        "incremental_predictive_value": incremental,
        "multi_fold_validation": multi_fold,
        "previous_outcome_context": prev_ctx,
        "artifacts": {
            "adjacency_residuals": str(resid_path),
            "player_pair_effects": str(pairs_path),
            "archetype_pair_effects": str(arch_path_out),
            "previous_outcome_context": str(prev_path),
            "incremental_predictive_value": str(incr_path),
            "interaction_multifold": str(multi_path),
        },
        "language": (
            "Reported quantities are estimated associations after controls, "
            "not causal claims about lineup chemistry. "
            "Mechanical sequencing value (Markov ordering) is distinct from residual interaction."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["artifacts"]["interaction_summary"] = str(summary_path)
    return summary


def main() -> None:
    summary = run_interaction_research()
    print(f"adjacent PAs: {summary['n_adjacent_pa']}")
    print(
        f"player pairs: {summary['n_player_pairs']} "
        f"(strong={summary['n_player_pairs_strong']}, "
        f"moderate={summary['n_player_pairs_moderate']}, "
        f"limited={summary['n_player_pairs_limited']})"
    )
    for m in summary["incremental_predictive_value"]["models"]:
        print(
            f"  {m['model']}: rmse={m['rmse']:.5f} "
            f"lift={m['rmse_lift_vs_m1']:+.5f} "
            f"logloss_lift={m.get('logloss_lift_vs_m1')}"
        )
    print(f"wrote {summary['artifacts']['interaction_summary']}")


if __name__ == "__main__":
    main()
