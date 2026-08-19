"""Invariants and regressions for STARTING batting-order extraction.

GUMBO/live ``boxscore.teams.*.battingOrder`` is the *current/final* slot
occupant. True starters use battingOrder codes 100, 200, …, 900.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lineup_intel.etl.extract_lineups import (
    _open_gumbo,
    _starter_pitcher,
    starting_batting_order_ids,
)
from lineup_intel.config import settings

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHAPMAN_ID = 547973
SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def _box_for_game(game_pk: int, side: str = "home") -> dict | None:
    path = Path(settings.gumbo_cache) / f"{game_pk}.json.gz"
    if not path.exists():
        return None
    d = _open_gumbo(path)
    box = ((d.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    return box.get(side)


def test_starter_codes_exclude_substitute_pitcher_chapman():
    """Regression: Aroldis Chapman entered via sub codes, not as a starter."""
    # Known BOS games where Chapman appears in the *live* battingOrder array.
    cases = [824782, 824283]  # 2026-04-04, 2026-05-04
    found = 0
    for gpk in cases:
        path = Path(settings.gumbo_cache) / f"{gpk}.json.gz"
        if not path.exists():
            continue
        d = _open_gumbo(path)
        teams = (d.get("gameData") or {}).get("teams") or {}
        box = ((d.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("home", "away"):
            abbr = (teams.get(side) or {}).get("abbreviation")
            if abbr != "BOS":
                continue
            bt = box.get(side) or {}
            arr = [int(x) for x in (bt.get("battingOrder") or [])]
            starters = starting_batting_order_ids(bt)
            assert starters is not None, f"missing starters for {gpk}"
            assert CHAPMAN_ID in arr, f"fixture drifted: Chapman not in live array {gpk}"
            assert CHAPMAN_ID not in starters, f"Chapman incorrectly treated as starter {gpk}"
            assert len(starters) == 9
            assert len(set(starters)) == 9
            found += 1
    if found == 0:
        pytest.skip("Chapman fixture GUMBO files not present")


def test_starting_batting_order_unit_codes():
    box = {
        "players": {
            f"ID{1000 + i}": {
                "person": {"id": 1000 + i, "fullName": f"P{i}"},
                "battingOrder": str(i * 100),
                "gameStatus": {"isSubstitute": False},
                "position": {"abbreviation": "OF"},
            }
            for i in range(1, 10)
        }
    }
    # Inject a substitute pitcher in slot 3 (code 302) who would pollute the array.
    box["players"]["ID9999"] = {
        "person": {"id": 9999, "fullName": "Reliever"},
        "battingOrder": "302",
        "gameStatus": {"isSubstitute": True},
        "position": {"abbreviation": "P"},
    }
    box["battingOrder"] = [1001, 1002, 9999, 1004, 1005, 1006, 1007, 1008, 1009]
    ids = starting_batting_order_ids(box)
    assert ids == [1000 + i for i in range(1, 10)]
    assert 9999 not in ids


def test_incomplete_starters_return_none():
    box = {
        "players": {
            "ID1": {
                "person": {"id": 1},
                "battingOrder": "100",
                "gameStatus": {"isSubstitute": False},
            }
        },
        "battingOrder": [1] * 9,
    }
    assert starting_batting_order_ids(box) is None


@pytest.fixture(scope="module")
def lineups():
    path = PROCESSED / "starting_lineups_2026.parquet"
    if not path.exists():
        pytest.skip("lineups not extracted")
    return pd.read_parquet(path)


def test_exactly_nine_slots_and_unique_ids(lineups):
    for _, row in lineups.iterrows():
        ids = [int(row[c]) for c in SLOT_COLS]
        assert len(ids) == 9
        assert len(set(ids)) == 9


def test_no_chapman_as_starting_hitter(lineups):
    mask = False
    for c in SLOT_COLS:
        mask = mask | (lineups[c] == CHAPMAN_ID)
    assert int(mask.sum()) == 0, (
        f"Chapman still appears as starter in {int(mask.sum())} rows — re-extract"
    )


def test_pitchers_in_starting_order_investigated(lineups):
    """After correct extraction, known relief pitchers must not be starters.

    Note: ``batter_positions`` can mislabel utility players as ``P`` when they
    also pitched; use identity checks for known pitchers instead of filtering
    on position alone.
    """
    known_pitchers = {CHAPMAN_ID}
    bad = []
    for _, row in lineups.iterrows():
        ids = [int(row[c]) for c in SLOT_COLS]
        for slot, pid in enumerate(ids, start=1):
            if pid in known_pitchers:
                bad.append((int(row["game_pk"]), row["team"], slot, pid))
    assert not bad, f"known pitchers in starting order: {bad}"


def test_thirty_teams_and_slots(lineups):
    assert lineups["team"].nunique() == 30
    for c in SLOT_COLS:
        assert c in lineups.columns


def test_starter_pitch_hand_from_gamedata_not_boxscore():
    """Cached GUMBO boxscore players omit pitchHand; gameData.players has it."""
    pid = 547179
    box_team = {
        "pitchers": [pid],
        "players": {
            f"ID{pid}": {
                "person": {"id": pid, "fullName": "Michael Lorenzen", "link": "/api/v1/people/x"},
            }
        },
    }
    game_players = {
        f"ID{pid}": {
            "id": pid,
            "fullName": "Michael Lorenzen",
            "pitchHand": {"code": "R", "description": "Right"},
        }
    }
    got_id, hand, name = _starter_pitcher(box_team, {}, "away")
    assert got_id == pid
    assert name == "Michael Lorenzen"
    assert hand is None

    got_id, hand, name = _starter_pitcher(box_team, {}, "away", game_players)
    assert got_id == pid
    assert hand == "R"
    assert name == "Michael Lorenzen"


def test_fill_opp_sp_hands_copies_same_pitcher_without_api():
    from lineup_intel.etl.extract_lineups import fill_opp_sp_hands

    df = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "opp_sp_id": [592288, 592288],
            "opp_sp_hand": ["L", None],
        }
    )
    stats = fill_opp_sp_hands(df, use_people_api=False)
    assert stats["known"] == 2
    assert stats["filled"] == 1
    assert df.loc[1, "opp_sp_hand"] == "L"


def test_starter_pitch_hand_from_probable_when_pitchers_empty():
    pid = 592288
    box_team = {"pitchers": [], "players": {}}
    game_data = {
        "probablePitchers": {
            "away": {
                "id": pid,
                "fullName": "Kent Emanuel",
                "pitchHand": {"code": "L"},
            }
        }
    }
    got_id, hand, name = _starter_pitcher(box_team, {}, "away", {}, game_data)
    assert got_id == pid
    assert hand == "L"
    assert name == "Kent Emanuel"
