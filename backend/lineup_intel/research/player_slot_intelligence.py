"""Player lineup intelligence: season ranks, modeled slot fit, observed splits.

Modeled slot fit holds the other eight hitters' *relative order* fixed and
inserts the focal player into each batting slot 1–9, then evaluates team
expected runs with the Markov engine. Aggregates are weighted by games in
each personnel/context group.

Observed slot splits join starting-lineup slots to plate appearances in those
games and are labeled descriptive — not causal placement effects.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..engine.markov import LineupEngine
from ..engine.pa_probs import PAProbabilityStore
from ..identity import personnel_id
from ..teams import normalize_abbrev
from .ranking import (
    MIN_PA_OVERALL,
    MIN_PA_PLATOON,
    MIN_PA_SLOT,
    RankedMetric,
    attach_team_rank,
    ordinal,
    rank_frame,
    rank_metric,
)

SLOT_COLS = [f"slot{i}" for i in range(1, 10)]
RISP_BASES = {"-2-", "--3", "12-", "1-3", "-23", "123"}
WOBA_W = {
    "K": 0.0,
    "BB_HBP": 0.690,
    "1B": 0.880,
    "2B": 1.250,
    "3B": 1.600,
    "HR": 2.000,
    "OUT_IP": 0.0,
}

# Operational equivalence for slot placement (runs/game).
SLOT_EQUIV_EPS = 0.01
SLOT_EQUIV_EPS_WIDE = 0.02

# Season-stat columns from Undervalued CSV (performance only — no salary/UV index).
UV_METRIC_MAP = {
    "woba": "woba",
    "xwoba": "xwoba",
    "obp": "obp",
    "slg": "slg",
    "iso": "iso",
    "wrc_plus": "wrc_plus",
    "xslg": "xslg",
    "k_pct": "k_percent",
    "bb_pct": "bb_percent",
    "chase_pct": "o_swing_percent",
    "contact_pct": "contact_percent",
    "z_contact_pct": "z_contact_percent",
    "avg_exit_velocity": "avg_exit_velocity",
    "hardhit_pct": "hard_hit_percent",
    "barrel_pct": "barrel_batted_rate",
    "gb_pct": "gb_percent",
    "ld_pct": "ld_percent",
    "fb_pct": "fb_percent",
    "pull_pct": "pull_percent",
    "oppo_pct": "oppo_percent",
}


def _as_frac(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if abs(x) > 1.5 and abs(x) <= 100:  # percent points
        return x / 100.0
    return x


def data_cutoff(lineups: pd.DataFrame) -> str:
    return str(lineups["game_date"].max())[:10] if len(lineups) else datetime.now().date().isoformat()


def move_player_to_slot(order: list[int], player_id: int, slot: int) -> list[int]:
    """Insert ``player_id`` at batting slot (1–9), preserving relative order of others."""
    if slot < 1 or slot > 9:
        raise ValueError("slot must be 1–9")
    if player_id not in order:
        raise ValueError("player not in order")
    others = [p for p in order if p != player_id]
    out = list(others)
    out.insert(slot - 1, player_id)
    if len(out) != 9 or len(set(out)) != 9:
        raise ValueError("invalid order after insertion")
    return out


def _context_from_hand(hand: Any) -> str:
    h = str(hand or "").upper()
    if h.startswith("L"):
        return "vs_L"
    if h.startswith("R"):
        return "vs_R"
    return "neutral"


def _flow_metrics(flow: list[dict] | None, slot: int) -> dict[str, float]:
    if not flow:
        return {}
    cell = next((f for f in flow if int(f.get("slot") or 0) == slot), None)
    if not cell:
        return {}
    dist = cell.get("base_state_distribution") or {}
    risp = 0.0
    two_out = 0.0
    empty = 0.0
    avg_runners = 0.0
    for key, mass in dist.items():
        try:
            outs_s, bases = str(key).split("|", 1)
            outs = int(outs_s)
            m = float(mass)
        except Exception:
            continue
        if bases == "---":
            empty += m
        if bases in RISP_BASES:
            risp += m
        if outs == 2:
            two_out += m
        runners = sum(1 for ch in bases if ch != "-")
        avg_runners += m * runners
    pa = float(cell.get("expected_pa") or 0.0)
    return {
        "expected_pa": pa,
        "expected_pa_162": pa * 162.0,
        "prob_runners_on": float(cell.get("prob_runners_on") or 0.0),
        "prob_risp": float(risp),
        "prob_bases_empty": float(empty),
        "prob_two_out": float(two_out),
        "avg_runners_on": float(avg_runners),
        "prob_lead_inning": None,  # filled from slot_start_share when available
    }


def load_season_stats(season: int | None = None) -> pd.DataFrame:
    """Load Undervalued comprehensive stats for performance metrics (no salary fields)."""
    season = season or settings.target_season
    path = (
        settings.undervalued_stats_2026
        if season >= 2026
        else settings.undervalued_stats_2025
    )
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "player_id" not in df.columns:
        return pd.DataFrame()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)
    # Normalize rate columns to fractions
    out = pd.DataFrame({"player_id": df["player_id"]})
    if "name" in df.columns:
        out["name"] = df["name"]
    elif "last_name" in df.columns:
        out["name"] = df.get("first_name", "").astype(str) + " " + df["last_name"].astype(str)
    out["pa"] = pd.to_numeric(df.get("pa"), errors="coerce")
    for metric, col in UV_METRIC_MAP.items():
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        if metric.endswith("_pct") or metric in {
            "obp", "slg", "iso", "woba", "xwoba", "xslg", "chase_pct",
            "contact_pct", "z_contact_pct", "hardhit_pct", "barrel_pct",
            "gb_pct", "ld_pct", "fb_pct", "pull_pct", "oppo_pct",
        }:
            out[metric] = raw.map(_as_frac)
        else:
            out[metric] = raw
    if "pull_pct" in out.columns and "oppo_pct" in out.columns:
        out["center_pct"] = (1.0 - out["pull_pct"].fillna(0) - out["oppo_pct"].fillna(0)).clip(0, 1)
    return out


def build_season_metric_ranks(
    season_stats: pd.DataFrame,
    lineups: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    """MLB (+ team when possible) ranks for season metrics."""
    if season_stats.empty:
        return {}

    # Primary team from most starts
    team_by_player: dict[int, str] = {}
    for _, row in lineups.iterrows():
        team = normalize_abbrev(row["team"]) or str(row["team"])
        for c in SLOT_COLS:
            pid = int(row[c])
            team_by_player.setdefault(pid, team)
    # Prefer majority team
    counts: dict[int, Counter] = defaultdict(Counter)
    for _, row in lineups.iterrows():
        team = normalize_abbrev(row["team"]) or str(row["team"])
        for c in SLOT_COLS:
            counts[int(row[c])][team] += 1
    for pid, ctr in counts.items():
        team_by_player[pid] = ctr.most_common(1)[0][0]

    stats = season_stats.copy()
    stats["team"] = stats["player_id"].map(team_by_player)

    metrics = [
        "woba", "xwoba", "obp", "slg", "iso", "wrc_plus", "xslg",
        "k_pct", "bb_pct", "chase_pct", "contact_pct", "z_contact_pct",
        "avg_exit_velocity", "hardhit_pct", "barrel_pct",
        "gb_pct", "ld_pct", "fb_pct", "pull_pct", "center_pct", "oppo_pct",
    ]
    # Precompute peer sets
    league_ranks: dict[str, dict[int, RankedMetric]] = {}
    for m in metrics:
        if m not in stats.columns:
            continue
        league_ranks[m] = rank_frame(
            stats,
            value_col=m,
            metric=m,
            sample_col="pa",
            min_sample=MIN_PA_OVERALL,
            id_col="player_id",
        )

    by_player: dict[int, dict[str, Any]] = {}
    for _, row in stats.iterrows():
        pid = int(row["player_id"])
        team = row.get("team")
        pa = float(row["pa"]) if pd.notna(row.get("pa")) else None
        payload: dict[str, Any] = {
            "player_id": pid,
            "pa": pa,
            "team": team,
            "metrics": {},
        }
        for m, ranks in league_ranks.items():
            league = ranks.get(pid)
            if league is None:
                continue
            team_peers = []
            if team and pd.notna(team):
                tdf = stats[(stats["team"] == team) & (stats["pa"] >= MIN_PA_OVERALL)]
                team_peers = [
                    float(x) for x in tdf[m].dropna().tolist()
                ] if m in tdf.columns else []
            payload["metrics"][m] = attach_team_rank(
                league,
                team_value=league.value,
                team_peer_values=team_peers,
                metric=m,
                sample_size=pa,
                min_sample=MIN_PA_OVERALL,
            )
        by_player[pid] = payload
    return by_player


def _primary_team_by_player(lineups: pd.DataFrame) -> dict[int, str]:
    parts = []
    for c in SLOT_COLS:
        if c not in lineups.columns:
            continue
        part = lineups[["team", c]].rename(columns={c: "player_id"})
        parts.append(part)
    if not parts:
        return {}
    long = pd.concat(parts, ignore_index=True).dropna(subset=["player_id"])
    long["team"] = long["team"].map(lambda t: normalize_abbrev(t) or str(t))
    long["player_id"] = long["player_id"].astype(int)
    vc = (
        long.groupby(["player_id", "team"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("player_id")
    )
    return dict(zip(vc["player_id"].tolist(), vc["team"].tolist()))


def _modeled_woba_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "n_pa", "woba"])
    df = pd.read_parquet(path)
    if df.empty or "player_id" not in df.columns:
        return pd.DataFrame(columns=["player_id", "n_pa", "woba"])
    woba = 0.0
    for col, wt in WOBA_W.items():
        if col in df.columns:
            woba = woba + wt * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    out = pd.DataFrame({
        "player_id": pd.to_numeric(df["player_id"], errors="coerce"),
        "n_pa": pd.to_numeric(df["n_pa"], errors="coerce") if "n_pa" in df.columns else np.nan,
        "woba": woba,
    }).dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(int)
    return out[out["player_id"] >= 0].copy()


def build_platoon_model_woba_ranks(
    lineups: pd.DataFrame | None = None,
    team_by_player: dict[int, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """League + team ranks for modeled wOBA vs RHP / vs LHP (PA-prob tables)."""
    if team_by_player is None:
        team_by_player = (
            _primary_team_by_player(lineups)
            if lineups is not None and len(lineups)
            else {}
        )
    by_player: dict[int, dict[str, Any]] = {}
    for metric, fname in (("woba_vs_R", "pa_probs_vs_R.parquet"), ("woba_vs_L", "pa_probs_vs_L.parquet")):
        stats = _modeled_woba_table(settings.models_dir / fname)
        if stats.empty:
            continue
        stats["team"] = stats["player_id"].map(team_by_player)
        league_ranks = rank_frame(
            stats,
            value_col="woba",
            metric=metric,
            sample_col="n_pa",
            min_sample=MIN_PA_PLATOON,
            id_col="player_id",
        )
        team_peers_by_team: dict[str, list[float]] = {}
        qualified = stats[stats["n_pa"] >= MIN_PA_PLATOON]
        for team, grp in qualified.groupby("team"):
            if not team or (isinstance(team, float) and pd.isna(team)):
                continue
            team_peers_by_team[str(team)] = [float(x) for x in grp["woba"].dropna().tolist()]
        for rec in stats[["player_id", "team", "n_pa"]].to_dict(orient="records"):
            pid = int(rec["player_id"])
            league = league_ranks.get(pid)
            if league is None:
                continue
            team = rec.get("team")
            n_pa = float(rec["n_pa"]) if rec.get("n_pa") is not None and pd.notna(rec.get("n_pa")) else None
            team_key = str(team) if team is not None and pd.notna(team) else None
            team_peers = team_peers_by_team.get(team_key, []) if team_key else []
            payload = by_player.setdefault(pid, {"player_id": pid, "metrics": {}})
            payload["metrics"][metric] = attach_team_rank(
                league,
                team_value=league.value,
                team_peer_values=team_peers,
                metric=metric,
                sample_size=n_pa,
                min_sample=MIN_PA_PLATOON,
            )
    return by_player


_PLATOON_WOBA_RANKS: dict[int, dict[str, Any]] | None = None


def _teams_from_profiles(profiles: Any) -> dict[int, str]:
    rows: list = []
    if isinstance(profiles, list):
        rows = profiles
    elif isinstance(profiles, dict) and isinstance(profiles.get("players"), list):
        rows = profiles["players"]
    elif isinstance(profiles, dict):
        rows = [v for v in profiles.values() if isinstance(v, dict) and "player_id" in v]
    out: dict[int, str] = {}
    for p in rows:
        try:
            pid = int(p["player_id"])
        except (TypeError, ValueError, KeyError):
            continue
        teams = p.get("teams") or []
        if isinstance(teams, list) and teams:
            out[pid] = str(teams[0])
        elif p.get("team"):
            out[pid] = str(p["team"])
    return out


def platoon_model_woba_ranks_cached() -> dict[int, dict[str, Any]]:
    global _PLATOON_WOBA_RANKS
    if _PLATOON_WOBA_RANKS is not None:
        return _PLATOON_WOBA_RANKS
    from ..db.store import get_store

    store = get_store()
    team_by_player = _teams_from_profiles(store.load_player_profiles())
    _PLATOON_WOBA_RANKS = build_platoon_model_woba_ranks(team_by_player=team_by_player)
    return _PLATOON_WOBA_RANKS


def merge_platoon_woba_metrics(season_metrics: dict | None, pid: int) -> dict:
    sm = dict(season_metrics or {})
    extra = platoon_model_woba_ranks_cached().get(int(pid))
    if not extra:
        return sm
    mets = dict(sm.get("metrics") or {})
    mets.update(extra.get("metrics") or {})
    sm["metrics"] = mets
    return sm


def build_observed_slot_splits(
    lineups: pd.DataFrame,
    pa: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Descriptive batting-slot splits for starts in each slot."""
    # Map (game_pk, batter_id) → starting slot
    starter_slots: dict[tuple[int, int], int] = {}
    for _, row in lineups.iterrows():
        gpk = int(row["game_pk"])
        for slot, col in enumerate(SLOT_COLS, start=1):
            starter_slots[(gpk, int(row[col]))] = slot

    work = pa[pa["season"] == season].copy()
    if work.empty:
        return pd.DataFrame()
    work["slot"] = [
        starter_slots.get((int(g), int(b)))
        for g, b in zip(work["game_pk"], work["batter_id"])
    ]
    work = work[work["slot"].notna()].copy()
    work["slot"] = work["slot"].astype(int)
    work["woba_value"] = work["outcome_class"].map(lambda c: WOBA_W.get(str(c), 0.0))
    work["is_k"] = (work["outcome_class"] == "K").astype(float)
    work["is_bb"] = (work["outcome_class"] == "BB_HBP").astype(float)
    work["tb"] = work["outcome_class"].map(
        {"1B": 1.0, "2B": 2.0, "3B": 3.0, "HR": 4.0}
    ).fillna(0.0)
    work["reached"] = work["outcome_class"].isin(
        ["BB_HBP", "1B", "2B", "3B", "HR"]
    ).astype(float)
    work["extra_bases"] = work["outcome_class"].map(
        {"2B": 1.0, "3B": 2.0, "HR": 3.0}
    ).fillna(0.0)

    g = (
        work.groupby(["batter_id", "slot"], as_index=False)
        .agg(
            pa=("outcome_class", "size"),
            starts=("game_pk", "nunique"),
            woba=("woba_value", "mean"),
            k_pct=("is_k", "mean"),
            bb_pct=("is_bb", "mean"),
            obp=("reached", "mean"),
            slg=("tb", "mean"),
            iso=("extra_bases", "mean"),
        )
        .rename(columns={"batter_id": "player_id"})
    )
    return g


def _slot_split_ranks(splits: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if splits.empty:
        return out
    for slot in range(1, 10):
        sdf = splits[splits["slot"] == slot]
        for metric in ("woba", "obp", "slg", "iso", "k_pct", "bb_pct"):
            if metric not in sdf.columns:
                continue
            ranks = rank_frame(
                sdf,
                value_col=metric,
                metric=metric,
                sample_col="pa",
                min_sample=MIN_PA_SLOT,
                id_col="player_id",
            )
            for pid, rm in ranks.items():
                key = (pid, slot)
                out.setdefault(key, {"player_id": pid, "slot": slot})
                row = sdf[(sdf["player_id"] == pid)].iloc[0]
                out[key]["pa"] = int(row["pa"])
                out[key]["starts"] = int(row["starts"])
                out[key].setdefault("metrics", {})
                out[key]["metrics"][metric] = {"mlb": rm.to_dict()}
                out[key]["raw"] = {
                    m: float(row[m]) if pd.notna(row.get(m)) else None
                    for m in ("woba", "obp", "slg", "iso", "k_pct", "bb_pct")
                    if m in row.index
                }
    return out


def compute_player_slot_fit(
    player_id: int,
    lineups: pd.DataFrame,
    *,
    pa_store: PAProbabilityStore,
    engine: LineupEngine,
    contexts: tuple[str, ...] = ("neutral", "vs_R", "vs_L"),
) -> dict[str, Any]:
    """Modeled placement value for one player across slots, by context."""
    # Games where player started
    mask = False
    for c in SLOT_COLS:
        mask = mask | (lineups[c] == player_id)
    mine = lineups.loc[mask].copy()
    if mine.empty:
        return {"available": False, "reason": "no starting appearances"}

    # Actual primary slot
    slot_counts = Counter()
    for _, row in mine.iterrows():
        for slot, col in enumerate(SLOT_COLS, start=1):
            if int(row[col]) == player_id:
                slot_counts[slot] += 1
                break
    primary_slot = slot_counts.most_common(1)[0][0]

    by_context: dict[str, Any] = {}
    usage_rows: list[dict] = []

    for ctx in contexts:
        # Filter games by opp hand when context-specific
        if ctx == "vs_R":
            sub = mine[mine["opp_sp_hand"].astype(str).str.upper().str.startswith("R")]
        elif ctx == "vs_L":
            sub = mine[mine["opp_sp_hand"].astype(str).str.upper().str.startswith("L")]
        else:
            sub = mine
        if len(sub) < 3 and ctx != "neutral":
            # Fall back: still compute neutral-quality estimate on all games labeled ctx
            # using ctx probs but all appearances — only skip if empty
            if sub.empty:
                continue

        # Group by personnel
        groups: dict[str, list[dict]] = defaultdict(list)
        for _, row in sub.iterrows():
            order = [int(row[c]) for c in SLOT_COLS]
            pid_key = str(row.get("personnel_id") or personnel_id(order))
            groups[pid_key].append({
                "order": order,
                "actual_slot": next(i for i, p in enumerate(order, 1) if p == player_id),
                "game_pk": int(row["game_pk"]),
                "team": normalize_abbrev(row["team"]) or str(row["team"]),
            })

        # Weight by games; evaluate unique personnel
        slot_er = {s: 0.0 for s in range(1, 10)}
        slot_w = {s: 0.0 for s in range(1, 10)}
        slot_flow_acc: dict[int, list[dict]] = defaultdict(list)
        actual_gaps = []
        actual_ranks = []
        top1 = top3 = within01 = within02 = 0
        n_starts = 0

        for _pid, refs in groups.items():
            w = float(len(refs))
            # Representative order: most common
            order_counts = Counter(tuple(r["order"]) for r in refs)
            rep = list(order_counts.most_common(1)[0][0])
            # Evaluate each placement
            results = []
            for slot in range(1, 10):
                placed = move_player_to_slot(rep, player_id, slot)
                probs = pa_store.probs_matrix(placed, ctx)
                ev = engine.evaluate(probs, with_flow=True)
                flow_m = _flow_metrics(ev.lineup_flow, slot)
                if ev.slot_start_share is not None:
                    flow_m["prob_lead_inning"] = float(ev.slot_start_share[slot - 1])
                results.append({
                    "slot": slot,
                    "expected_runs": float(ev.expected_runs_9),
                    "flow": flow_m,
                    "order": placed,
                })
                slot_er[slot] += w * float(ev.expected_runs_9)
                slot_w[slot] += w
                slot_flow_acc[slot].append((w, flow_m))

            ranked = sorted(results, key=lambda r: -r["expected_runs"])
            best = ranked[0]["expected_runs"]
            avg_er = float(np.mean([r["expected_runs"] for r in results]))

            for ref in refs:
                n_starts += 1
                actual_slot = ref["actual_slot"]
                actual_er = next(r["expected_runs"] for r in results if r["slot"] == actual_slot)
                # Rank of actual slot
                order_by_er = sorted(results, key=lambda r: -r["expected_runs"])
                rank = next(i for i, r in enumerate(order_by_er, 1) if r["slot"] == actual_slot)
                gap = best - actual_er
                actual_gaps.append(gap)
                actual_ranks.append(rank)
                if rank == 1:
                    top1 += 1
                if rank <= 3:
                    top3 += 1
                if gap <= SLOT_EQUIV_EPS:
                    within01 += 1
                if gap <= SLOT_EQUIV_EPS_WIDE:
                    within02 += 1
                usage_rows.append({
                    "context": ctx,
                    "game_pk": ref["game_pk"],
                    "team": ref["team"],
                    "actual_slot": actual_slot,
                    "actual_er": actual_er,
                    "best_slot": order_by_er[0]["slot"],
                    "best_er": best,
                    "gap": gap,
                    "slot_rank": rank,
                })

            # accumulate weighted — already done via slot_er

        if not any(slot_w.values()):
            continue

        slots_out = []
        for slot in range(1, 10):
            er = slot_er[slot] / max(slot_w[slot], 1e-9)
            # Weighted average flow
            flows = slot_flow_acc[slot]
            tw = sum(w for w, _ in flows) or 1.0
            flow_avg: dict[str, float] = {}
            keys = set().union(*(f.keys() for _, f in flows)) if flows else set()
            for k in keys:
                vals = [(w, f.get(k)) for w, f in flows if f.get(k) is not None]
                if vals:
                    flow_avg[k] = float(sum(w * float(v) for w, v in vals) / sum(w for w, _ in vals))
            slots_out.append({
                "slot": slot,
                "expected_runs": float(er),
                "weight_games": float(slot_w[slot]),
                **flow_avg,
            })

        ers = [s["expected_runs"] for s in slots_out]
        avg_placement = float(np.mean(ers))
        for s in slots_out:
            s["delta_vs_avg"] = float(s["expected_runs"] - avg_placement)
            primary_er = next(x["expected_runs"] for x in slots_out if x["slot"] == primary_slot)
            s["delta_vs_primary"] = float(s["expected_runs"] - primary_er)

        ordered = sorted(slots_out, key=lambda s: -s["expected_runs"])
        for i, s in enumerate(ordered, 1):
            for cell in slots_out:
                if cell["slot"] == s["slot"]:
                    cell["fit_rank"] = i

        best_slot = ordered[0]["slot"]
        best_er = ordered[0]["expected_runs"]
        near = [
            s["slot"] for s in ordered
            if best_er - s["expected_runs"] <= SLOT_EQUIV_EPS + 1e-12
        ]
        near_wide = [
            s["slot"] for s in ordered
            if best_er - s["expected_runs"] <= SLOT_EQUIV_EPS_WIDE + 1e-12
        ]
        primary_er = next(s["expected_runs"] for s in slots_out if s["slot"] == primary_slot)
        primary_rank = next(s["fit_rank"] for s in slots_out if s["slot"] == primary_slot)
        opportunity = float(best_er - primary_er)

        by_context[ctx] = {
            "slots": slots_out,
            "best_slot": best_slot,
            "best_expected_runs": best_er,
            "near_equivalent_slots": near,
            "near_equivalent_slots_02": near_wide,
            "primary_slot": primary_slot,
            "primary_expected_runs": primary_er,
            "primary_fit_rank": primary_rank,
            "placement_opportunity": opportunity,
            "avg_placement_runs": avg_placement,
            "equivalence_eps": SLOT_EQUIV_EPS,
            "n_starts": n_starts,
            "n_personnel_groups": len(groups),
            "actual_usage_fit": {
                "avg_slot_rank": float(np.mean(actual_ranks)) if actual_ranks else None,
                "avg_opportunity_gap": float(np.mean(actual_gaps)) if actual_gaps else None,
                "pct_top1": float(top1 / n_starts) if n_starts else None,
                "pct_top3": float(top3 / n_starts) if n_starts else None,
                "pct_within_01": float(within01 / n_starts) if n_starts else None,
                "pct_within_02": float(within02 / n_starts) if n_starts else None,
                "n_starts": n_starts,
            },
        }

    return {
        "available": True,
        "player_id": player_id,
        "primary_actual_slot": primary_slot,
        "slot_usage": {str(k): int(v) for k, v in sorted(slot_counts.items())},
        "by_context": by_context,
        "usage_game_rows": usage_rows,
        "method": (
            "same_nine_relative_order_insertion; "
            f"equivalence_eps={SLOT_EQUIV_EPS}; "
            "Markov LineupEngine with_flow"
        ),
    }


def explain_slot_fit(
    player_name: str,
    profile_label: str | None,
    fit: dict[str, Any],
    context: str = "neutral",
) -> list[dict[str, str]]:
    """Structured explanations from computed slot-fit numbers only."""
    ctx = (fit.get("by_context") or {}).get(context) or {}
    if not ctx:
        return []
    slots = {s["slot"]: s for s in ctx.get("slots") or []}
    best = ctx.get("best_slot")
    primary = ctx.get("primary_slot")
    near = ctx.get("near_equivalent_slots") or []
    opp = ctx.get("placement_opportunity")
    bullets: list[dict[str, str]] = []

    if best is not None:
        bullets.append({
            "id": "best",
            "text": (
                f"Best modeled placement for {player_name} is slot #{best} "
                f"({ctx.get('best_expected_runs'):.3f} expected team runs/game)."
            ),
        })
    if near and len(near) > 1:
        others = [s for s in near if s != best]
        bullets.append({
            "id": "equivalence",
            "text": (
                f"Slots {', '.join('#'+str(s) for s in near)} are within "
                f"{SLOT_EQUIV_EPS:.2f} runs/game of each other — essentially equivalent "
                f"under the model’s operational band."
            ),
        })
        if others:
            b = slots.get(best) or {}
            o = slots.get(others[0]) or {}
            if b and o:
                bullets.append({
                    "id": "tiny_gap",
                    "text": (
                        f"Difference between #{best} and #{others[0]}: "
                        f"{abs(float(b['expected_runs']) - float(o['expected_runs'])):.3f} R/G."
                    ),
                })
    if primary is not None and best is not None and primary != best and opp is not None:
        bullets.append({
            "id": "opportunity",
            "text": (
                f"Primary actual slot #{primary} ranks {ctx.get('primary_fit_rank')} of 9; "
                f"estimated placement opportunity vs best is {float(opp):+.3f} R/G."
            ),
        })
    # PA mechanism
    if best is not None and primary is not None and best in slots and primary in slots:
        bpa = float(slots[best].get("expected_pa") or 0)
        ppa = float(slots[primary].get("expected_pa") or 0)
        if abs(bpa - ppa) > 0.02:
            bullets.append({
                "id": "pa_allocation",
                "text": (
                    f"Slot #{best} yields {bpa:.2f} expected PA/game vs "
                    f"{ppa:.2f} at primary slot #{primary}. "
                    "Batting order changes both game states and plate appearances."
                ),
            })
        bron = float(slots[best].get("prob_runners_on") or 0)
        pron = float(slots[primary].get("prob_runners_on") or 0)
        if abs(bron - pron) > 0.01:
            bullets.append({
                "id": "runners_on",
                "text": (
                    f"Runner-on exposure: {bron:.0%} at #{best} vs {pron:.0%} at #{primary}."
                ),
            })
    if profile_label:
        bullets.append({
            "id": "profile",
            "text": (
                f"Offensive profile group: {profile_label} "
                "(exploratory unsupervised cluster — not a definitive archetype)."
            ),
        })
    # Power vs top of order heuristic from deltas
    cleanup = slots.get(4)
    if cleanup and best and best >= 6:
        bullets.append({
            "id": "avoid_cleanup",
            "text": (
                f"Cleanup (#{4}) models {float(cleanup.get('delta_vs_avg') or 0):+.3f} R/G "
                f"vs his average placement — typically less value than mid/lower-order spots "
                f"when power/run-production profiles are modest relative to the nine."
            ),
        })
    return bullets


def build_neighbors(
    player_id: int,
    lineups: pd.DataFrame,
    pair_effects: pd.DataFrame | None,
    names: dict[int, str],
    arch: dict[int, str],
) -> dict[str, Any]:
    before: Counter = Counter()
    after: Counter = Counter()
    for _, row in lineups.iterrows():
        order = [int(row[c]) for c in SLOT_COLS]
        if player_id not in order:
            continue
        i = order.index(player_id)
        if i > 0:
            before[order[i - 1]] += 1
        if i < 8:
            after[order[i + 1]] += 1

    def enrich(ctr: Counter, side: str) -> list[dict]:
        rows = []
        for pid, n in ctr.most_common(8):
            item = {
                "player_id": int(pid),
                "name": names.get(int(pid), str(pid)),
                "n_adjacent_starts": int(n),
                "offensive_profile": arch.get(int(pid)),
                "association": None,
            }
            if pair_effects is not None and len(pair_effects):
                if side == "before":
                    m = pair_effects[
                        (pair_effects["prev_batter_id"] == int(pid))
                        & (pair_effects["batter_id"] == int(player_id))
                    ]
                else:
                    m = pair_effects[
                        (pair_effects["prev_batter_id"] == int(player_id))
                        & (pair_effects["batter_id"] == int(pid))
                    ]
                if len(m):
                    r = m.iloc[0]
                    item["association"] = {
                        "effect": float(r["effect"]) if pd.notna(r.get("effect")) else None,
                        "n": int(r["n"]) if pd.notna(r.get("n")) else None,
                        "reliability_tier": r.get("reliability_tier"),
                        "label": "Exploratory residual association",
                    }
            rows.append(item)
        return rows

    return {
        "before": enrich(before, "before"),
        "after": enrich(after, "after"),
        "note": "Neighbor counts are starting-lineup adjacency; pair effects are exploratory.",
    }


def precompute_player_lineup_intelligence(
    season: int | None = None,
    *,
    limit_players: int | None = None,
) -> dict[str, Any]:
    season = season or settings.target_season
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    out_dir = artifacts / "player_lineup_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    lu_path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    pl_path = settings.processed_dir / f"players_{season}.parquet"
    pa_path = settings.processed_dir / "plate_appearances.parquet"
    if not lu_path.exists():
        raise FileNotFoundError(lu_path)

    lineups = pd.read_parquet(lu_path)
    lineups["team"] = lineups["team"].map(lambda x: normalize_abbrev(x) or x)
    cutoff = data_cutoff(lineups)
    players = pd.read_parquet(pl_path) if pl_path.exists() else pd.DataFrame()
    names = {
        int(r["player_id"]): str(r.get("name") or r["player_id"])
        for r in players.to_dict(orient="records")
    } if len(players) else {}

    # Archetype labels
    arch_path = settings.models_dir / "archetype_assignments.parquet"
    arch: dict[int, str] = {}
    if arch_path.exists():
        adf = pd.read_parquet(arch_path)
        adf = adf[adf["season"] == season] if "season" in adf.columns else adf
        for r in adf.to_dict(orient="records"):
            arch[int(r["player_id"])] = str(r.get("archetype_label") or r.get("archetype_id"))

    season_stats = load_season_stats(season)
    season_ranks = build_season_metric_ranks(season_stats, lineups)
    platoon_woba = build_platoon_model_woba_ranks(lineups)
    global _PLATOON_WOBA_RANKS
    _PLATOON_WOBA_RANKS = platoon_woba

    pa = pd.read_parquet(pa_path) if pa_path.exists() else pd.DataFrame()
    splits = build_observed_slot_splits(lineups, pa, season) if len(pa) else pd.DataFrame()
    split_ranks = _slot_split_ranks(splits)

    pairs_path = artifacts / "research" / "player_pair_effects.parquet"
    pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else None

    pa_store = PAProbabilityStore(settings.models_dir)
    engine = LineupEngine()

    # Players with starts
    starter_ids: set[int] = set()
    for c in SLOT_COLS:
        starter_ids.update(int(x) for x in lineups[c].tolist())
    starter_list = sorted(starter_ids)
    if limit_players is not None:
        starter_list = starter_list[: int(limit_players)]

    placement_opps: list[dict] = []
    index_rows = []

    print(f"[player_lineup_intel] {len(starter_list)} players, cutoff={cutoff}")
    for i, pid in enumerate(starter_list, 1):
        fit = compute_player_slot_fit(pid, lineups, pa_store=pa_store, engine=engine)
        neu = (fit.get("by_context") or {}).get("neutral") or {}
        neighbors = build_neighbors(pid, lineups, pairs, names, arch)
        explanations = explain_slot_fit(
            names.get(pid, str(pid)),
            arch.get(pid),
            fit,
            context="neutral",
        )

        # Observed splits for this player
        obs = []
        for slot in range(1, 10):
            key = (pid, slot)
            if key in split_ranks:
                cell = split_ranks[key]
                cell["label"] = "Observed performance — descriptive, not estimated causal slot effect."
                obs.append(cell)

        # Platoon best slots
        platoon = {}
        for ctx in ("neutral", "vs_R", "vs_L"):
            c = (fit.get("by_context") or {}).get(ctx)
            if c:
                platoon[ctx] = {
                    "best_slot": c.get("best_slot"),
                    "near_equivalent_slots": c.get("near_equivalent_slots"),
                    "placement_opportunity": c.get("placement_opportunity"),
                    "n_starts": c.get("n_starts"),
                }

        usage = neu.get("actual_usage_fit") or {}
        opp = neu.get("placement_opportunity")
        if opp is not None and (neu.get("n_starts") or 0) >= 10:
            placement_opps.append({
                "player_id": pid,
                "placement_opportunity": float(opp),
                "n_starts": int(neu.get("n_starts") or 0),
                "primary_slot": fit.get("primary_actual_slot"),
                "best_slot": neu.get("best_slot"),
            })

        season_metrics = dict(season_ranks.get(pid, {}))
        extra_platoon = platoon_woba.get(pid)
        if extra_platoon:
            mets = dict(season_metrics.get("metrics") or {})
            mets.update(extra_platoon.get("metrics") or {})
            season_metrics["metrics"] = mets

        profile = {
            "available": True,
            "player_id": pid,
            "name": names.get(pid, str(pid)),
            "data_cutoff": cutoff,
            "season": season,
            "offensive_profile": {
                "label": arch.get(pid),
                "display_name": "Offensive profile",
                "tooltip": (
                    "Exploratory profile group derived from unsupervised clustering "
                    "of hitter characteristics."
                ),
            },
            "summary": {
                "primary_actual_slot": fit.get("primary_actual_slot"),
                "best_modeled_slot": neu.get("best_slot"),
                "placement_opportunity": opp,
                "actual_slot_fit_rank": neu.get("primary_fit_rank"),
                "near_equivalent_slots": neu.get("near_equivalent_slots"),
                "best_context": _best_context_label(fit),
                "offensive_profile": arch.get(pid),
                "actual_usage_fit": usage,
            },
            "season_metrics": season_metrics,
            "slot_usage": fit.get("slot_usage"),
            "modeled_slot_fit": fit.get("by_context"),
            "run_opportunity_profile": {
                "context": "neutral",
                "slots": neu.get("slots"),
                "note": (
                    "Batting order changes both the game states a hitter encounters "
                    "and the number of plate appearances he receives."
                ),
            },
            "why_this_slot": explanations,
            "observed_slot_splits": obs,
            "platoon_slot_fit": platoon,
            "lineup_neighbors": neighbors,
            "method": fit.get("method"),
            "qualification": {
                "overall_min_pa": MIN_PA_OVERALL,
                "platoon_min_pa": MIN_PA_PLATOON,
                "slot_min_pa": MIN_PA_SLOT,
                "slot_equivalence_eps": SLOT_EQUIV_EPS,
            },
        }

        path = out_dir / f"{pid}.json"
        path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        index_rows.append({
            "player_id": pid,
            "name": names.get(pid, str(pid)),
            "primary_slot": fit.get("primary_actual_slot"),
            "best_slot": neu.get("best_slot"),
            "placement_opportunity": opp,
            "n_starts": neu.get("n_starts"),
            "path": str(path),
        })
        if i % 25 == 0 or i == len(starter_list):
            print(f"[player_lineup_intel] {i}/{len(starter_list)}")

    # League placement opportunity ranks
    opp_df = pd.DataFrame(placement_opps)
    opp_ranks: dict[int, RankedMetric] = {}
    if len(opp_df):
        opp_ranks = rank_frame(
            opp_df,
            value_col="placement_opportunity",
            metric="placement_opportunity",
            sample_col="n_starts",
            min_sample=10,
            id_col="player_id",
        )
        for row in index_rows:
            pid = int(row["player_id"])
            if pid in opp_ranks:
                row["placement_opportunity_rank"] = opp_ranks[pid].to_dict()
                # Patch individual profile files
                ppath = out_dir / f"{pid}.json"
                if ppath.exists():
                    payload = json.loads(ppath.read_text())
                    payload["summary"]["placement_opportunity_rank"] = opp_ranks[pid].to_dict()
                    ppath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    index = {
        "available": True,
        "season": season,
        "data_cutoff": cutoff,
        "n_players": len(index_rows),
        "players": index_rows,
        "qualification": {
            "overall_min_pa": MIN_PA_OVERALL,
            "platoon_min_pa": MIN_PA_PLATOON,
            "slot_min_pa": MIN_PA_SLOT,
            "slot_equivalence_eps": SLOT_EQUIV_EPS,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (artifacts / "player_lineup_intelligence_index.json").write_text(
        json.dumps(index, indent=2, default=str), encoding="utf-8"
    )

    # League research summary
    league = _league_slot_research(index_rows, out_dir, cutoff, season)
    (artifacts / "research" / "player_slot_intelligence.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (artifacts / "research" / "player_slot_intelligence.json").write_text(
        json.dumps(league, indent=2, default=str), encoding="utf-8"
    )

    return {"index": index, "league": league, "n_players": len(index_rows)}


def _best_context_label(fit: dict) -> str | None:
    by = fit.get("by_context") or {}
    if not by:
        return None
    # Context with largest placement opportunity (most room to improve) is not ideal;
    # report context where best slot differs from primary most clearly.
    best_ctx = None
    best_opp = None
    for ctx, payload in by.items():
        opp = payload.get("placement_opportunity")
        if opp is None:
            continue
        if best_opp is None or abs(float(opp)) > abs(float(best_opp)):
            best_opp = opp
            best_ctx = ctx
    labels = {"neutral": "Neutral", "vs_R": "vs RHP", "vs_L": "vs LHP"}
    return labels.get(best_ctx or "neutral", best_ctx)


def _league_slot_research(
    index_rows: list[dict],
    out_dir: Path,
    cutoff: str,
    season: int,
) -> dict[str, Any]:
    gaps = []
    top1 = top3 = w01 = w02 = 0
    n = 0
    best_slot_by_profile: dict[str, Counter] = defaultdict(Counter)
    for row in index_rows:
        path = Path(row["path"])
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        neu = (p.get("modeled_slot_fit") or {}).get("neutral") or {}
        usage = neu.get("actual_usage_fit") or {}
        ns = int(usage.get("n_starts") or 0)
        if ns < 5:
            continue
        n += ns
        gaps.append(float(usage.get("avg_opportunity_gap") or 0) * ns)
        top1 += float(usage.get("pct_top1") or 0) * ns
        top3 += float(usage.get("pct_top3") or 0) * ns
        w01 += float(usage.get("pct_within_01") or 0) * ns
        w02 += float(usage.get("pct_within_02") or 0) * ns
        prof = (p.get("offensive_profile") or {}).get("label") or "Unknown"
        if neu.get("best_slot") is not None:
            best_slot_by_profile[str(prof)][int(neu["best_slot"])] += 1

    largest = sorted(
        [r for r in index_rows if r.get("placement_opportunity") is not None],
        key=lambda r: -float(r["placement_opportunity"]),
    )[:25]

    return {
        "available": True,
        "season": season,
        "data_cutoff": cutoff,
        "n_player_starts_weighted": n,
        "mean_opportunity_gap": float(sum(gaps) / n) if n else None,
        "pct_starts_top1_slot": float(top1 / n) if n else None,
        "pct_starts_top3_slot": float(top3 / n) if n else None,
        "pct_starts_within_01": float(w01 / n) if n else None,
        "pct_starts_within_02": float(w02 / n) if n else None,
        "best_slot_distribution_by_profile": {
            k: dict(v) for k, v in best_slot_by_profile.items()
        },
        "largest_placement_opportunities": largest,
        "notes": [
            "Placement opportunity is modeled same-nine slot value, not a managerial grade.",
            "Observed slot batting stats are descriptive only.",
            "Injuries, platoons, and unmodeled constraints can explain gaps.",
        ],
    }


def load_player_lineup_profile(player_id: int) -> dict[str, Any] | None:
    path = settings.artifacts_dir / "player_lineup_profiles" / f"{int(player_id)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--limit-players", type=int, default=None)
    args = p.parse_args(argv)
    result = precompute_player_lineup_intelligence(
        season=args.season,
        limit_players=args.limit_players,
    )
    print(f"wrote profiles for {result['n_players']} players")


if __name__ == "__main__":
    main()
