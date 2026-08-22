#!/usr/bin/env python3
"""Daily season refresh for deployed / local Lineup Intelligence.

Pulls newly completed MLB games into starting_lineups and evaluates new
lineups. Safe to run every day through the end of the 2026 regular season /
postseason.

Usage (repo root):
  PYTHONPATH=backend python scripts/daily_refresh.py
  PYTHONPATH=backend python scripts/daily_refresh.py --skip-precompute
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily MLB lineup refresh")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--through", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, str(BACKEND))
    from lineup_intel.etl.refresh_lineups import refresh_lineups

    through = args.through or date.today().isoformat()
    t0 = time.time()
    print(f"[daily] refresh_lineups through {through}", flush=True)
    summary = refresh_lineups(season=args.season, through=through)
    print(
        f"[daily] +{summary.get('n_new_lineups', 0)} lineups "
        f"(total {summary.get('n_total')}, max {summary.get('max_game_date')})",
        flush=True,
    )

    # Roster rebuild hits MLB Stats API for many monthly transaction chunks.
    # Morning scheduled runs often see transient 503s; do not abort lineup
    # evaluation / artifact commit when roster refresh fails.
    print("[daily] update rosters / transactions", flush=True)
    try:
        from lineup_intel.etl.update_rosters import build_roster_history

        roster_report = build_roster_history(season=args.season, through=through)
        lv = roster_report.get("lineup_validation") or {}
        print(
            f"[daily] roster tenures={roster_report.get('n_tenure_intervals')} "
            f"lineup_ok={lv.get('validated')}/{lv.get('total_starting_player_observations')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[daily] WARNING: roster refresh failed (continuing with lineups/"
            f"precompute): {exc!r}",
            flush=True,
        )

    if not args.skip_precompute:
        print("[daily] precompute new evaluations", flush=True)
        from lineup_intel.etl.precompute import main as precompute_main

        pre_argv = ["--workers", str(args.workers)]
        if args.season is not None:
            pre_argv += ["--season", str(args.season)]
        precompute_main(pre_argv)

    # Invalidate API in-process caches if imported under uvicorn workers
    try:
        from lineup_intel.db.store import get_store

        get_store().clear_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"[daily] cache clear skipped: {exc}", flush=True)

    print(f"[daily] done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
