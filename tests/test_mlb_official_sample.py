"""Stratified sample: extracted starters vs MLB live-feed starter codes.

Requires network access to statsapi.mlb.com. Skips cleanly offline.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest

from lineup_intel.etl.extract_lineups import starting_batting_order_ids

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


@pytest.mark.network
def test_stratified_sample_matches_mlb_starter_codes():
    path = PROCESSED / "starting_lineups_2026.parquet"
    if not path.exists():
        pytest.skip("lineups missing")
    lu = pd.read_parquet(path)
    # Stratify by month
    lu = lu.copy()
    lu["month"] = lu["game_date"].astype(str).str.slice(5, 7)
    samples = []
    for month, g in lu.groupby("month"):
        n = min(4, len(g))
        if n:
            samples.append(g.sample(n, random_state=int(month or "1")))
    if not samples:
        pytest.skip("no rows")
    sample = pd.concat(samples, ignore_index=True)

    mismatches = []
    checked = 0
    with httpx.Client(timeout=30) as client:
        for _, row in sample.iterrows():
            gpk = int(row["game_pk"])
            team = str(row["team"])
            try:
                r = client.get(f"https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live")
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"network unavailable: {exc}")
            if r.status_code != 200:
                continue
            d = r.json()
            teams = (d.get("gameData") or {}).get("teams") or {}
            box = ((d.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
            side = None
            for s in ("home", "away"):
                abbr = (teams.get(s) or {}).get("abbreviation")
                if abbr == team or (team == "ATH" and abbr in ("ATH", "OAK")):
                    side = s
                    break
            if side is None:
                continue
            official = starting_batting_order_ids(box.get(side) or {})
            if not official:
                continue
            ours = [int(row[c]) for c in SLOT_COLS]
            checked += 1
            if ours != official:
                mismatches.append({
                    "game_pk": gpk,
                    "team": team,
                    "ours": ours,
                    "official": official,
                })

    if checked < 5:
        pytest.skip(f"too few comparable games ({checked})")
    assert not mismatches, f"{len(mismatches)}/{checked} mismatches e.g. {mismatches[:3]}"
