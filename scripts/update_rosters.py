#!/usr/bin/env python3
"""Refresh MLB roster/transaction history without retraining models.

Usage (repo root):
  PYTHONPATH=backend python scripts/update_rosters.py
  PYTHONPATH=backend python scripts/update_rosters.py --through 2026-08-18
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from lineup_intel.etl.update_rosters import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
