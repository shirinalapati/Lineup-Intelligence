#!/usr/bin/env python3
"""Run the MLB Lineup Intelligence offline pipeline end-to-end.

Order: extract → pa_probs → archetypes → interactions → precompute → findings

Usage (from repo root):
  PYTHONPATH=backend python scripts/run_pipeline.py
  PYTHONPATH=backend python scripts/run_pipeline.py --limit 50
  PYTHONPATH=backend python scripts/run_pipeline.py --steps extract,pa_probs,precompute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

STEPS = [
    ("extract", [sys.executable, "-m", "lineup_intel.etl.extract_lineups"]),
    ("rosters", [sys.executable, "-m", "lineup_intel.etl.update_rosters"]),
    ("pa_probs", [sys.executable, "-m", "lineup_intel.engine.pa_probs"]),
    ("archetypes", [sys.executable, "-m", "lineup_intel.research.archetypes"]),
    ("interactions", [sys.executable, "-m", "lineup_intel.research.interactions"]),
    ("precompute", [sys.executable, "-m", "lineup_intel.etl.precompute"]),
    ("markov_validation", [sys.executable, "-m", "lineup_intel.research.markov_validation"]),
    ("model_cards", [sys.executable, "-m", "lineup_intel.research.model_cards"]),
    ("findings", [sys.executable, "-m", "lineup_intel.research.findings"]),
    ("data_quality", [sys.executable, "-m", "lineup_intel.etl.data_quality"]),
    ("player_slot_intel", [sys.executable, "-m", "lineup_intel.research.player_slot_intelligence"]),
]


def run_step(name: str, cmd: list[str], extra: list[str], env: dict) -> int:
    full = cmd + extra
    print(f"\n=== [{name}] {' '.join(full)} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(full, cwd=str(ROOT), env=env)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"=== [{name}] {status} in {dt:.1f}s ===", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="MLB Lineup Intelligence pipeline")
    parser.add_argument(
        "--steps",
        type=str,
        default=",".join(s for s, _ in STEPS),
        help="Comma-separated step names",
    )
    parser.add_argument("--limit", type=int, default=None, help="Forwarded to extract/precompute")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Precompute workers")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-precompute", action="store_true")
    args = parser.parse_args()

    wanted = {s.strip() for s in args.steps.split(",") if s.strip()}
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(BACKEND) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    failures = []
    for name, cmd in STEPS:
        if name not in wanted:
            continue
        extra: list[str] = []
        if name == "extract":
            if args.season is not None:
                extra += ["--season", str(args.season)]
            if args.limit is not None:
                extra += ["--limit", str(args.limit)]
            # Also export PA table when running full extract
            extra += ["--export-pa"]
        elif name == "rosters":
            if args.season is not None:
                extra += ["--season", str(args.season)]
        elif name == "precompute":
            if args.season is not None:
                extra += ["--season", str(args.season)]
            if args.limit is not None:
                extra += ["--limit", str(args.limit)]
            if args.workers:
                extra += ["--workers", str(args.workers)]
            if args.force_precompute:
                extra += ["--force"]
        elif name == "player_slot_intel":
            if args.season is not None:
                extra += ["--season", str(args.season)]
            if args.limit is not None:
                extra += ["--limit-players", str(args.limit)]
        rc = run_step(name, cmd, extra, env)
        if rc != 0:
            failures.append(name)
            if not args.continue_on_error:
                print(f"Pipeline stopped at step '{name}'", flush=True)
                return rc

    if failures:
        print(f"Pipeline finished with failures: {failures}", flush=True)
        return 1
    print("\nPipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
