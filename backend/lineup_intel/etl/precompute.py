"""Precompute lineup evaluations and summary artifacts.

Caches optimizer work by (personnel_id, context): exhaustive fast ranking once
per nine-man group, then O(1) rank lookup + one full Markov eval per observed order.

CLI:
  PYTHONPATH=backend python -m lineup_intel.etl.precompute
  PYTHONPATH=backend python -m lineup_intel.etl.precompute --limit 50 --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..engine.markov import LineupEngine
from ..engine.optimizer import (
    HAS_NUMBA,
    _build_kernels_for_players,
    _fast_features,
    _index_of_perm,
    _local_search_full,
    _perm_from_index,
    _score_all_perms_fast,
)
from ..engine.pa_probs import PAProbabilityStore
from ..engine.transitions import load_transitions
from ..identity import order_id, personnel_id
from ..research.ranking import attach_team_metric_ranks
from ..teams import CANONICAL_ABBREVS, TEAMS, normalize_abbrev

SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def _context_from_hand(hand: Any) -> str:
    if hand == "R":
        return "vs_R"
    if hand == "L":
        return "vs_L"
    return "neutral"


def _row_slots(row: pd.Series) -> list[int]:
    return [int(row[c]) for c in SLOT_COLS]


def _eval_personnel_orders(
    player_ids: list[int],
    orders: list[list[int]],
    context: str,
    pa_store: PAProbabilityStore,
    engine: LineupEngine,
    trans,
    equivalence_eps: float,
) -> list[dict]:
    """Evaluate many batting orders that share the same nine players + context."""
    # Canonical personnel order = sorted ids for stable indexing
    canon = sorted(player_ids)
    id_to_idx = {pid: i for i, pid in enumerate(canon)}
    probs = pa_store.probs_matrix(canon, context)
    R, P, E = _build_kernels_for_players(probs, trans)
    slot_value, seq_bonus, lead_extra = _fast_features(R, P, E)

    if not HAS_NUMBA:
        raise RuntimeError("Numba is required for precompute exhaustive ranking")

    scores = _score_all_perms_fast(slot_value, seq_bonus, lead_extra)
    best_pi = int(scores.argmax())
    worst_pi = int(scores.argmin())
    best_arr = np.empty(9, dtype=np.int64)
    worst_arr = np.empty(9, dtype=np.int64)
    _perm_from_index(best_pi, best_arr)
    _perm_from_index(worst_pi, worst_arr)
    best_perm = tuple(int(x) for x in best_arr)
    worst_perm = tuple(int(x) for x in worst_arr)

    # Refine best among top fast candidates + local search
    top_idx = np.argpartition(-scores, 20)[:20]
    cand = {best_perm, worst_perm}
    tmp = np.empty(9, dtype=np.int64)
    for pi in top_idx:
        _perm_from_index(int(pi), tmp)
        cand.add(tuple(int(x) for x in tmp))

    def full(perm: tuple[int, ...]) -> float:
        return float(engine.expected_runs(probs[list(perm)]))

    refined = {p: full(p) for p in cand}
    loc_best, loc_sc = _local_search_full(engine, probs, list(max(refined, key=refined.get)), restarts=1)
    refined[tuple(loc_best)] = loc_sc
    best_perm = max(refined, key=refined.get)
    worst_perm = min(refined, key=refined.get)
    best_runs = refined[best_perm]
    worst_runs = refined[worst_perm]
    best_fast = float(scores[best_pi])
    scale = best_runs / best_fast if abs(best_fast) > 1e-12 else 1.0
    mean_runs = float(scores.mean() * scale)
    median_runs = float(np.median(scores) * scale)
    lo, hi = min(worst_runs, best_runs), max(worst_runs, best_runs)
    mean_runs = float(min(max(mean_runs, lo), hi))
    median_runs = float(min(max(median_runs, lo), hi))
    fast_eps = equivalence_eps / max(scale, 1e-6)
    n_near = int((scores >= scores.max() - fast_eps).sum())
    n_near_01 = int((scores >= scores.max() - (0.01 / max(scale, 1e-6))).sum())
    n_near_02 = int((scores >= scores.max() - (0.02 / max(scale, 1e-6))).sum())

    best_order_ids = [canon[i] for i in best_perm]
    worst_order_ids = [canon[i] for i in worst_perm]

    out = []
    for order in orders:
        order_idx = tuple(id_to_idx[pid] for pid in order)
        pi = _index_of_perm(order_idx)
        actual_fast = float(scores[pi])
        rank = int((scores > actual_fast).sum()) + 1
        actual_runs = float(engine.expected_runs(probs[list(order_idx)]))
        gap = best_runs - actual_runs
        out.append({
            "actual_runs": actual_runs,
            "best_runs": best_runs,
            "worst_runs": worst_runs,
            "mean_runs": mean_runs,
            "median_runs": median_runs,
            "rank": rank,
            "n_perms": 362880,
            "percentile": 100.0 * (1.0 - (rank - 1) / 362880),
            "gap": gap,
            "ordering_value": actual_runs - mean_runs,
            "value_vs_median": actual_runs - median_runs,
            "best_worst_spread": best_runs - worst_runs,
            "n_near_optimal": n_near,
            "n_near_optimal_01": n_near_01,
            "n_near_optimal_02": n_near_02,
            "pct_near_optimal_01": 100.0 * n_near_01 / 362880,
            "pct_near_optimal_02": 100.0 * n_near_02 / 362880,
            "operationally_equivalent": gap <= equivalence_eps,
            "method": "personnel_cache_fast_exhaustive",
            "best_order_ids": best_order_ids,
            "worst_order_ids": worst_order_ids,
            "actual_order_ids": list(order),
        })
    return out


def precompute(
    season: int | None = None,
    limit: int | None = None,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    season = season or settings.target_season
    processed = settings.processed_dir
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    lineups_path = processed / f"starting_lineups_{season}.parquet"
    players_path = processed / f"players_{season}.parquet"
    evals_path = artifacts / "lineup_evaluations.parquet"

    if not lineups_path.exists():
        raise FileNotFoundError(f"Missing {lineups_path}; run extract_lineups first")

    print(f"[precompute] loading lineups from {lineups_path}")
    all_lineups = pd.read_parquet(lineups_path)
    all_lineups["team"] = all_lineups["team"].map(lambda x: normalize_abbrev(x) or x)
    if "opponent" in all_lineups.columns:
        all_lineups["opponent"] = all_lineups["opponent"].map(lambda x: normalize_abbrev(x) or x)

    lineups = all_lineups.head(int(limit)).copy() if limit is not None else all_lineups
    if limit is not None:
        print(f"[precompute] --limit={limit} → {len(lineups)} rows")

    existing = pd.DataFrame()
    done_keys: set[tuple[int, str]] = set()
    if not force and evals_path.exists():
        existing = pd.read_parquet(evals_path)
        existing["team"] = existing["team"].map(lambda x: normalize_abbrev(x) or x)
        want_ctx: dict[tuple[int, str], str] = {}
        for rec in lineups.to_dict(orient="records"):
            want_ctx[(int(rec["game_pk"]), str(rec["team"]))] = _context_from_hand(
                rec.get("opp_sp_hand")
            )
        if not existing.empty and "context" in existing.columns:
            want = [
                want_ctx.get((int(r.game_pk), str(r.team)))
                for r in existing.itertuples(index=False)
            ]
            stale = pd.Series(want, index=existing.index).notna() & (
                existing["context"].astype(str) != pd.Series(want, index=existing.index).astype(str)
            )
            n_stale = int(stale.sum())
            if n_stale:
                print(
                    f"[precompute] recompute {n_stale} rows whose pitcher-hand context changed"
                )
                existing = existing.loc[~stale].copy()
        done_keys = set(zip(existing["game_pk"].astype(int), existing["team"].astype(str)))
        print(f"[precompute] skipping {len(done_keys)} already-computed rows")

    # Group remaining rows by personnel_id + context
    groups: dict[tuple[str, str], list[dict]] = {}
    for _, row in lineups.iterrows():
        team = str(row["team"])
        gpk = int(row["game_pk"])
        if (gpk, team) in done_keys:
            continue
        slots = _row_slots(row)
        pid = str(row.get("personnel_id") or personnel_id(slots))
        oid = str(row.get("order_id") or order_id(slots))
        ctx = _context_from_hand(row.get("opp_sp_hand"))
        groups.setdefault((pid, ctx), []).append({
            "game_pk": gpk,
            "team": team,
            "season": int(row.get("season") or season),
            "game_date": row.get("game_date"),
            "opponent": row.get("opponent"),
            "is_home": bool(row.get("is_home")),
            "opp_sp_hand": row.get("opp_sp_hand"),
            "order_id": oid,
            "personnel_id": pid,
            "context": ctx,
            "player_ids": slots,
            "runs_scored": row.get("runs_scored"),
            "result": row.get("result"),
        })

    print(f"[precompute] {sum(len(v) for v in groups.values())} lineups in {len(groups)} personnel/context groups")

    pa_store = PAProbabilityStore()
    trans = load_transitions()
    engine = LineupEngine(trans)
    eps = settings.equivalence_eps

    new_rows: list[dict] = []
    t0 = time.time()
    for gi, ((pid, ctx), refs) in enumerate(groups.items(), start=1):
        # unique orders within group
        order_key_to_refs: dict[str, list[dict]] = {}
        for ref in refs:
            order_key_to_refs.setdefault(ref["order_id"], []).append(ref)
        unique_orders = [r[0]["player_ids"] for r in order_key_to_refs.values()]
        # player set from first order
        player_ids = sorted(unique_orders[0])
        # sanity: all orders same set
        for o in unique_orders:
            if sorted(o) != player_ids:
                # fallback: evaluate orders individually via group of one
                pass
        try:
            results = _eval_personnel_orders(
                player_ids=player_ids,
                orders=unique_orders,
                context=ctx,
                pa_store=pa_store,
                engine=engine,
                trans=trans,
                equivalence_eps=eps,
            )
        except Exception as e:
            print(f"[precompute] ERROR group {pid}/{ctx}: {e}")
            continue

        by_order = {order_id(o): res for o, res in zip(unique_orders, results)}
        for oid, refs_o in order_key_to_refs.items():
            opt = by_order.get(oid) or by_order.get(order_id(refs_o[0]["player_ids"]))
            if not opt:
                continue
            for ref in refs_o:
                new_rows.append({
                    **{k: ref[k] for k in (
                        "game_pk", "team", "season", "game_date", "opponent", "is_home",
                        "opp_sp_hand", "order_id", "personnel_id", "context",
                        "runs_scored", "result",
                    )},
                    "cache_key": f"{pid}|{oid}|{ctx}",
                    "actual_runs": opt["actual_runs"],
                    "best_runs": opt["best_runs"],
                    "worst_runs": opt["worst_runs"],
                    "mean_runs": opt["mean_runs"],
                    "median_runs": opt["median_runs"],
                    "rank": opt["rank"],
                    "n_perms": opt["n_perms"],
                    "percentile": opt["percentile"],
                    "gap": opt["gap"],
                    "ordering_value": opt["ordering_value"],
                    "value_vs_median": opt.get("value_vs_median"),
                    "best_worst_spread": opt.get("best_worst_spread"),
                    "n_near_optimal": opt["n_near_optimal"],
                    "n_near_optimal_01": opt.get("n_near_optimal_01"),
                    "n_near_optimal_02": opt.get("n_near_optimal_02"),
                    "pct_near_optimal_01": opt.get("pct_near_optimal_01"),
                    "pct_near_optimal_02": opt.get("pct_near_optimal_02"),
                    "operationally_equivalent": opt["operationally_equivalent"],
                    "method": opt["method"],
                    "best_order_ids": json.dumps(opt["best_order_ids"]),
                    "worst_order_ids": json.dumps(opt["worst_order_ids"]),
                    "actual_order_ids": json.dumps(ref["player_ids"]),
                })

        if gi % 10 == 0 or gi == len(groups):
            elapsed = time.time() - t0
            rate = gi / max(elapsed, 1e-6)
            eta = (len(groups) - gi) / max(rate, 1e-6)
            print(f"[precompute] {gi}/{len(groups)} groups ({elapsed:.1f}s, ETA {eta/60:.1f}m)")

    if force or existing.empty:
        evals_df = pd.DataFrame(new_rows)
    else:
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            existing_f = existing.copy()
            merged = pd.concat([
                existing_f[~existing_f.set_index(["game_pk", "team"]).index.isin(
                    new_df.set_index(["game_pk", "team"]).index
                )],
                new_df,
            ], ignore_index=True)
            evals_df = merged
        else:
            evals_df = existing

    evals_df.to_parquet(evals_path, index=False)
    print(f"[precompute] wrote {evals_path} ({len(evals_df)} rows)")

    # Team summaries
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()
    name_map = {int(r.player_id): str(r.name) for r in players.itertuples(index=False)} if len(players) else {}

    team_summaries: dict[str, Any] = {"season": season, "teams": {}}
    for abbr in CANONICAL_ABBREVS:
        tdf = all_lineups[all_lineups["team"] == abbr]
        edf = evals_df[evals_df["team"] == abbr] if len(evals_df) else pd.DataFrame()
        summary: dict[str, Any] = {
            "abbr": abbr,
            "name": TEAMS[abbr]["name"],
            "division": TEAMS[abbr]["division"],
            "games": int(len(tdf)),
            "unique_orders": int(tdf["order_id"].nunique()) if len(tdf) else 0,
            "unique_personnel": int(tdf["personnel_id"].nunique()) if len(tdf) else 0,
        }
        if len(edf):
            summary.update({
                "avg_actual_runs": float(edf["actual_runs"].mean()),
                "avg_best_runs": float(edf["best_runs"].mean()),
                "avg_gap": float(edf["gap"].mean()),
                "median_gap": float(edf["gap"].median()),
                "avg_percentile": float(edf["percentile"].mean()),
                "avg_value_vs_median": float(edf["value_vs_median"].mean())
                if "value_vs_median" in edf.columns
                else None,
                "avg_best_worst_spread": float(edf["best_worst_spread"].mean())
                if "best_worst_spread" in edf.columns
                else (
                    float((edf["best_runs"] - edf["worst_runs"]).mean())
                    if "worst_runs" in edf.columns
                    else None
                ),
                "pct_within_01": float((edf["gap"] <= 0.01).mean()),
                "pct_within_02": float((edf["gap"] <= 0.02).mean()),
                "avg_pct_near_01": float(edf["pct_near_optimal_01"].mean())
                if "pct_near_optimal_01" in edf.columns
                else None,
                "avg_pct_near_02": float(edf["pct_near_optimal_02"].mean())
                if "pct_near_optimal_02" in edf.columns
                else None,
                "pct_top10": float((edf["percentile"] >= 90).mean()),
                "pct_operationally_equivalent": float(edf["operationally_equivalent"].mean()),
                "avg_ordering_value": float(edf["ordering_value"].mean()),
                "evaluated_games": int(len(edf)),
                "metrics_available": True,
            })
            if len(tdf):
                top = tdf["order_id"].value_counts().head(1)
                if len(top):
                    oid = top.index[0]
                    sample = tdf[tdf["order_id"] == oid].iloc[0]
                    order = _row_slots(sample)
                    summary["most_common_order"] = {
                        "order_id": oid,
                        "n": int(top.iloc[0]),
                        "batting_order": order,
                        "batter_names": [name_map.get(p, str(p)) for p in order],
                    }
                # stability
                from ..research.stability import lineup_stability
                summary["stability"] = lineup_stability(tdf)
        else:
            summary["metrics_available"] = False
            summary["reason"] = "No evaluations for this team yet"
        team_summaries["teams"][abbr] = summary

    attach_team_metric_ranks(team_summaries["teams"])

    (artifacts / "team_summaries.json").write_text(
        json.dumps(team_summaries, indent=2, default=str), encoding="utf-8"
    )

    team_list = list(team_summaries["teams"].values())
    evaluated = [t for t in team_list if t.get("evaluated_games")]
    overview: dict[str, Any] = {
        "season": season,
        "n_teams": len(CANONICAL_ABBREVS),
        "n_lineups": int(len(all_lineups)),
        "n_evaluated": int(len(evals_df)),
        "teams": team_list,
    }
    if evaluated:
        overview["league_avg_gap"] = float(np.mean([t["avg_gap"] for t in evaluated]))
        overview["league_median_gap"] = float(np.median([t["median_gap"] for t in evaluated]))
        overview["league_avg_percentile"] = float(np.mean([t["avg_percentile"] for t in evaluated]))
        overview["league_avg_actual_runs"] = float(np.mean([t["avg_actual_runs"] for t in evaluated]))
        overview["league_avg_value_vs_median"] = float(
            np.nanmean([t.get("avg_value_vs_median") for t in evaluated])
        )
        overview["league_avg_best_worst_spread"] = float(
            np.nanmean([t.get("avg_best_worst_spread") for t in evaluated])
        )
        overview["pct_lineups_within_01"] = float(np.mean([t["pct_within_01"] for t in evaluated]))
        overview["pct_lineups_within_02"] = float(np.mean([t["pct_within_02"] for t in evaluated]))
        overview["ordering_opportunity_162"] = float(overview["league_avg_gap"] * 162)
        overview["pct_operationally_equivalent"] = float(
            np.mean([t["pct_operationally_equivalent"] for t in evaluated])
        )
        ranked = sorted(evaluated, key=lambda t: t.get("avg_gap", 0))
        overview["most_efficient_ordering"] = [
            {
                "abbr": t["abbr"],
                "avg_gap": t["avg_gap"],
                "avg_value_vs_median": t.get("avg_value_vs_median"),
                "pct_within_02": t.get("pct_within_02"),
                "avg_percentile": t["avg_percentile"],
            }
            for t in ranked[:10]
        ]
        ranked_gap = sorted(evaluated, key=lambda t: t.get("avg_gap", 0), reverse=True)
        overview["largest_avg_gaps"] = [
            {
                "abbr": t["abbr"],
                "avg_gap": t["avg_gap"],
                "avg_best_worst_spread": t.get("avg_best_worst_spread"),
                "avg_percentile": t["avg_percentile"],
            }
            for t in ranked_gap[:10]
        ]
    else:
        overview["metrics_available"] = False
        overview["reason"] = "No lineup evaluations available yet"

    (artifacts / "league_overview.json").write_text(
        json.dumps(overview, indent=2, default=str), encoding="utf-8"
    )

    # Player profiles
    slot_apps: dict[int, dict[str, Any]] = {}
    for _, row in all_lineups.iterrows():
        for slot, col in enumerate(SLOT_COLS, start=1):
            pid = int(row[col])
            if pid not in slot_apps:
                slot_apps[pid] = {
                    "player_id": pid,
                    "name": name_map.get(pid, str(pid)),
                    "games": 0,
                    "teams": set(),
                    "slot_counts": {str(i): 0 for i in range(1, 10)},
                    "gaps": [],
                    "percentiles": [],
                }
            slot_apps[pid]["games"] += 1
            slot_apps[pid]["teams"].add(row["team"])
            slot_apps[pid]["slot_counts"][str(slot)] += 1

    if len(evals_df):
        for _, erow in evals_df.iterrows():
            try:
                order = json.loads(erow["actual_order_ids"]) if isinstance(erow.get("actual_order_ids"), str) else erow.get("actual_order_ids")
            except (TypeError, json.JSONDecodeError):
                order = None
            if not order:
                continue
            for pid in order:
                pid = int(pid)
                if pid in slot_apps:
                    if pd.notna(erow.get("gap")):
                        slot_apps[pid]["gaps"].append(float(erow["gap"]))
                    if pd.notna(erow.get("percentile")):
                        slot_apps[pid]["percentiles"].append(float(erow["percentile"]))

    # Attach archetypes if available
    arch_path = settings.models_dir / "archetype_assignments.parquet"
    arch_map = {}
    if arch_path.exists():
        adf = pd.read_parquet(arch_path)
        if "season" in adf.columns:
            # prefer 2026 then 2025
            for season_pref in (season, season - 1):
                sub = adf[adf["season"] == season_pref]
                for r in sub.itertuples(index=False):
                    arch_map[int(r.player_id)] = {
                        "archetype_id": int(getattr(r, "archetype_id", -1)),
                        "archetype_label": str(getattr(r, "archetype_label", "Unknown")),
                    }

    profiles = []
    for pid, info in sorted(slot_apps.items(), key=lambda kv: -kv[1]["games"]):
        prof = {
            "player_id": pid,
            "name": info["name"],
            "games": info["games"],
            "teams": sorted(info["teams"]),
            "slot_counts": info["slot_counts"],
            "primary_slot": max(info["slot_counts"].items(), key=lambda kv: kv[1])[0],
        }
        if info["gaps"]:
            prof["avg_lineup_gap"] = float(np.mean(info["gaps"]))
        if info["percentiles"]:
            prof["avg_lineup_percentile"] = float(np.mean(info["percentiles"]))
        if pid in arch_map:
            prof.update(arch_map[pid])
        profiles.append(prof)

    (artifacts / "player_profiles.json").write_text(
        json.dumps({"season": season, "players": profiles}, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(profiles).to_parquet(artifacts / "player_profiles.parquet", index=False)

    elapsed = time.time() - t0
    print(f"[precompute] done in {elapsed:.1f}s — {len(evals_df)} evaluations")
    return {"evaluations": len(evals_df), "new_rows": len(new_rows), "groups": len(groups), "elapsed_sec": elapsed}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Precompute lineup evaluations")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    precompute(season=args.season, limit=args.limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
