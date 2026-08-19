"""Roster membership vs availability — no hard-coded player exceptions."""

from __future__ import annotations

from datetime import date

from lineup_intel.roster_history import (
    Tenure,
    badge_with_counterparty,
    build_status_intervals,
    build_tenures,
    classify_transaction,
    mlb_lineup_available,
    overlapping_tenure_violations,
    snapshot_status_to_canonical,
    status_at,
    tenure_at,
)


def _tx(**kwargs):
    person = {"id": kwargs.pop("player_id"), "fullName": kwargs.pop("name", "X")}
    raw = {
        "id": kwargs.pop("id", 1),
        "typeCode": kwargs.pop("typeCode"),
        "typeDesc": kwargs.pop("typeDesc", ""),
        "date": kwargs.get("effectiveDate"),
        "effectiveDate": kwargs.pop("effectiveDate"),
        "description": kwargs.pop("description", ""),
        "person": person,
        "fromTeam": kwargs.pop("fromTeam", None),
        "toTeam": kwargs.pop("toTeam", None),
    }
    raw.update(kwargs)
    return classify_transaction(raw)


def _team(tid, name):
    return {"id": tid, "name": name}


def test_trade_closes_old_and_opens_new_tenure():
    opening = {
        110: [{"player_id": 1, "status_code": "A", "status": "Active"}],  # BAL
    }
    events = [
        _tx(
            id=10,
            player_id=1,
            typeCode="TR",
            typeDesc="Trade",
            effectiveDate="2026-08-03",
            description="Baltimore Orioles traded OF Pat to Boston Red Sox.",
            fromTeam=_team(110, "Baltimore Orioles"),
            toTeam=_team(111, "Boston Red Sox"),
        )
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    assert overlapping_tenure_violations(tenures) == []
    bal = tenure_at(tenures, 1, date(2026, 7, 15))
    bos_early = tenure_at(tenures, 1, date(2026, 7, 15))
    assert bal is not None and bal.team == "BAL"
    assert bos_early is not None and bos_early.team == "BAL"
    assert tenure_at(tenures, 1, date(2026, 8, 2)).team == "BAL"
    after = tenure_at(tenures, 1, date(2026, 8, 3))
    assert after is not None and after.team == "BOS"
    assert tenure_at(tenures, 1, date(2026, 8, 18)).team == "BOS"
    # Future trade must not change July 15
    assert tenure_at(tenures, 1, date(2026, 7, 15)).team == "BAL"


def test_release_ends_tenure_sign_starts_new():
    opening = {110: [{"player_id": 2, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=11,
            player_id=2,
            typeCode="REL",
            typeDesc="Released",
            effectiveDate="2026-07-12",
            description="Baltimore Orioles released INF Pat.",
            fromTeam=_team(110, "Baltimore Orioles"),
        ),
        _tx(
            id=12,
            player_id=2,
            typeCode="SFA",
            typeDesc="Signed as Free Agent",
            effectiveDate="2026-07-20",
            description="Boston Red Sox signed free agent INF Pat.",
            toTeam=_team(111, "Boston Red Sox"),
        ),
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    assert tenure_at(tenures, 2, date(2026, 7, 11)).team == "BAL"
    assert tenure_at(tenures, 2, date(2026, 7, 12)) is None
    assert tenure_at(tenures, 2, date(2026, 7, 19)) is None
    assert tenure_at(tenures, 2, date(2026, 7, 20)).team == "BOS"


def test_il_does_not_end_tenure():
    opening = {110: [{"player_id": 3, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=20,
            player_id=3,
            typeCode="SC",
            typeDesc="Status Change",
            effectiveDate="2026-07-06",
            description="Baltimore Orioles placed OF Pat on the 10-day injured list.",
            toTeam=_team(110, "Baltimore Orioles"),
        ),
        _tx(
            id=21,
            player_id=3,
            typeCode="SC",
            typeDesc="Status Change",
            effectiveDate="2026-08-02",
            description="Baltimore Orioles activated OF Pat from the 10-day injured list.",
            toTeam=_team(110, "Baltimore Orioles"),
        ),
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    assert tenure_at(tenures, 3, date(2026, 7, 20)).team == "BAL"
    assert tenure_at(tenures, 3, date(2026, 8, 10)).team == "BAL"
    intervals = build_status_intervals(
        opening, events, tenures, opening_date=date(2026, 3, 26)
    )
    il = status_at(intervals, 3, 110, date(2026, 7, 20))
    assert il is not None
    assert il["roster_status"] == "IL_10"
    assert il["mlb_lineup_available"] is False
    act = status_at(intervals, 3, 110, date(2026, 8, 2))
    assert act["roster_status"] == "ACTIVE"
    assert act["mlb_lineup_available"] is True


def test_option_does_not_end_org_tenure_recall_restores_availability():
    opening = {110: [{"player_id": 4, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=30,
            player_id=4,
            typeCode="OPT",
            typeDesc="Optioned",
            effectiveDate="2026-06-20",
            description="Baltimore Orioles optioned INF Pat to Norfolk Tides.",
            fromTeam=_team(110, "Baltimore Orioles"),
            toTeam=_team(568, "Norfolk Tides"),
        ),
        _tx(
            id=31,
            player_id=4,
            typeCode="CU",
            typeDesc="Recalled",
            effectiveDate="2026-07-05",
            description="Baltimore Orioles recalled INF Pat from Norfolk Tides.",
            fromTeam=_team(568, "Norfolk Tides"),
            toTeam=_team(110, "Baltimore Orioles"),
        ),
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    assert tenure_at(tenures, 4, date(2026, 6, 25)).team == "BAL"
    intervals = build_status_intervals(
        opening, events, tenures, opening_date=date(2026, 3, 26)
    )
    opt = status_at(intervals, 4, 110, date(2026, 6, 25))
    assert opt["roster_status"] == "OPTIONED"
    assert not mlb_lineup_available(opt["roster_status"])
    rec = status_at(intervals, 4, 110, date(2026, 7, 5))
    assert rec["roster_status"] == "ACTIVE"
    assert mlb_lineup_available(rec["roster_status"])


def test_as_of_deterministic_and_current_excludes_traded():
    opening = {110: [{"player_id": 5, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=40,
            player_id=5,
            typeCode="TR",
            typeDesc="Trade",
            effectiveDate="2026-08-03",
            description="Baltimore Orioles traded OF Pat to Boston Red Sox.",
            fromTeam=_team(110, "Baltimore Orioles"),
            toTeam=_team(111, "Boston Red Sox"),
        )
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    a = tenure_at(tenures, 5, date(2026, 7, 15))
    b = tenure_at(tenures, 5, date(2026, 7, 15))
    assert a.team == b.team == "BAL"
    # season pool: both teams in history
    teams = {t.team for t in tenures if t.player_id == 5}
    assert teams == {"BAL", "BOS"}
    # current (Aug 18): BOS only
    now = tenure_at(tenures, 5, date(2026, 8, 18))
    assert now.team == "BOS"
    badge = badge_with_counterparty(tenures, 5, "BAL", as_of=date(2026, 8, 18))
    assert badge and "BOS" in badge and "Traded" in badge
    badge_bos = badge_with_counterparty(tenures, 5, "BOS", as_of=date(2026, 8, 18))
    assert badge_bos and "BAL" in badge_bos and "Acquired" in badge_bos


def test_waiver_claim_and_dfa_availability():
    opening = {110: [{"player_id": 6, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=50,
            player_id=6,
            typeCode="DES",
            typeDesc="Designated for Assignment",
            effectiveDate="2026-05-01",
            description="Baltimore Orioles designated INF Pat for assignment.",
            toTeam=_team(110, "Baltimore Orioles"),
        ),
        _tx(
            id=51,
            player_id=6,
            typeCode="CLW",
            typeDesc="Claimed Off Waivers",
            effectiveDate="2026-05-06",
            description="Boston Red Sox claimed INF Pat off waivers from Baltimore Orioles.",
            fromTeam=_team(110, "Baltimore Orioles"),
            toTeam=_team(111, "Boston Red Sox"),
        ),
    ]
    tenures = build_tenures(opening, events, opening_date=date(2026, 3, 26))
    intervals = build_status_intervals(
        opening, events, tenures, opening_date=date(2026, 3, 26)
    )
    assert tenure_at(tenures, 6, date(2026, 5, 3)).team == "BAL"
    dfa = status_at(intervals, 6, 110, date(2026, 5, 3))
    assert dfa["roster_status"] == "DFA"
    assert dfa["mlb_lineup_available"] is False
    assert tenure_at(tenures, 6, date(2026, 5, 6)).team == "BOS"


def test_il60_and_snapshot_status_map():
    assert snapshot_status_to_canonical("D60", "Injured 60-Day") == "IL_60"
    assert snapshot_status_to_canonical("RM", "Reassigned to Minors") == "OPTIONED"
    assert mlb_lineup_available("IL_60") is False
    assert mlb_lineup_available("ACTIVE") is True


def test_through_cutoff_ignores_future_events():
    opening = {110: [{"player_id": 7, "status_code": "A", "status": "Active"}]}
    events = [
        _tx(
            id=60,
            player_id=7,
            typeCode="TR",
            typeDesc="Trade",
            effectiveDate="2026-08-03",
            description="Baltimore Orioles traded OF Pat to Boston Red Sox.",
            fromTeam=_team(110, "Baltimore Orioles"),
            toTeam=_team(111, "Boston Red Sox"),
        )
    ]
    tenures = build_tenures(
        opening, events, opening_date=date(2026, 3, 26), through=date(2026, 7, 15)
    )
    assert tenure_at(tenures, 7, date(2026, 7, 15)).team == "BAL"
    # Trade after cutoff is not applied, so Aug 18 still BAL in this reconstruction
    assert tenure_at(tenures, 7, date(2026, 8, 18)).team == "BAL"


# ── Artifact / Explorer pool golden checks (data-driven, no name exceptions) ──

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TENURE_PATH = ROOT / "data" / "processed" / "player_team_tenure.parquet"
INTERVAL_PATH = ROOT / "data" / "processed" / "player_roster_intervals.parquet"
LINEUPS_PATH = ROOT / "data" / "processed" / "starting_lineups_2026.parquet"


def _as_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    return pd.Timestamp(v).date()


def _ids(payload: dict) -> set[int]:
    return {int(p["player_id"]) for p in payload.get("players") or []}


@pytest.fixture(scope="module")
def tenures_df():
    if not TENURE_PATH.exists():
        pytest.skip("player_team_tenure.parquet missing")
    return pd.read_parquet(TENURE_PATH)


@pytest.fixture(scope="module")
def traded_stints(tenures_df):
    """Closed TRADE tenures that have a same-day TRADE start on another club.

    Prefer hitters who actually started for the old club so Explorer season-pool
    assertions are meaningful. No player-specific exceptions.
    """
    if not LINEUPS_PATH.exists():
        pytest.skip("lineups missing")
    df = tenures_df.copy()
    df["start_d"] = df["start_at"].map(_as_date)
    df["end_d"] = df["end_at"].map(_as_date)
    closed = df[(df["end_reason"] == "TRADE") & df["end_d"].notna()]
    lineups = pd.read_parquet(LINEUPS_PATH)
    slots = [c for c in lineups.columns if c.startswith("slot") and c[4:].isdigit()]
    rows = []
    for rec in closed.to_dict(orient="records"):
        pid = int(rec["player_id"])
        old_team = rec["team"]
        trade_d = rec["end_d"]
        nxt = df[
            (df["player_id"] == pid)
            & (df["start_reason"] == "TRADE")
            & (df["start_d"] == trade_d)
            & (df["team"] != old_team)
        ]
        if nxt.empty:
            continue
        started_old = not lineups[
            (lineups["team"] == old_team) & lineups[slots].eq(pid).any(axis=1)
        ].empty
        if not started_old:
            continue
        new = nxt.iloc[0]
        rows.append(
            {
                "player_id": pid,
                "old_team": old_team,
                "new_team": new["team"],
                "trade_date": trade_d,
            }
        )
        if len(rows) >= 5:
            break
    if not rows:
        pytest.skip("no TRADE stints with old-club starts in tenure artifacts")
    return rows


def test_artifact_tenures_do_not_overlap(tenures_df):
    from lineup_intel.roster_history import (
        index_tenures,
        overlapping_tenure_violations,
    )

    rows = [t for ts in index_tenures(tenures_df).values() for t in ts]
    assert overlapping_tenure_violations(rows) == []


def test_real_trades_move_membership_and_explorer_pools(traded_stints):
    from datetime import timedelta

    from lineup_intel.db.store import ArtifactStore

    if not LINEUPS_PATH.exists():
        pytest.skip("lineups missing")
    store = ArtifactStore()
    lineups = pd.read_parquet(LINEUPS_PATH)
    checked = 0
    for stint in traded_stints:
        pid = stint["player_id"]
        old_t = stint["old_team"]
        new_t = stint["new_team"]
        trade_d = stint["trade_date"]
        before = trade_d - timedelta(days=1)
        after = trade_d + timedelta(days=1)

        slots = [c for c in lineups.columns if c.startswith("slot") and c[4:].isdigit()]
        started_old = not lineups[
            (lineups["team"] == old_t) & lineups[slots].eq(pid).any(axis=1)
        ].empty
        if not started_old:
            continue

        season_old = store.team_roster(old_t, mode="season")
        assert pid in _ids(season_old)

        current_old = store.team_roster(old_t, mode="current", include_unavailable=True)
        assert pid not in _ids(current_old)

        as_before_old = store.team_roster(
            old_t, mode="as_of", as_of=before.isoformat(), include_unavailable=True
        )
        as_before_new = store.team_roster(
            new_t, mode="as_of", as_of=before.isoformat(), include_unavailable=True
        )
        as_after_old = store.team_roster(
            old_t, mode="as_of", as_of=after.isoformat(), include_unavailable=True
        )
        as_after_new = store.team_roster(
            new_t, mode="as_of", as_of=after.isoformat(), include_unavailable=True
        )
        as_trade_old = store.team_roster(
            old_t, mode="as_of", as_of=trade_d.isoformat(), include_unavailable=True
        )
        as_trade_new = store.team_roster(
            new_t, mode="as_of", as_of=trade_d.isoformat(), include_unavailable=True
        )

        assert pid in _ids(as_before_old)
        assert pid not in _ids(as_before_new)
        assert pid not in _ids(as_after_old)
        assert pid in _ids(as_after_new)
        # Half-open [start, end): trade date belongs to the new club.
        assert pid not in _ids(as_trade_old)
        assert pid in _ids(as_trade_new)
        # Future trade must not leak into the earlier as-of query.
        assert pid in _ids(as_before_old)
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1


def test_unavailable_toggle_does_not_resurrect_traded_players(traded_stints):
    from lineup_intel.db.store import ArtifactStore

    store = ArtifactStore()
    stint = traded_stints[0]
    off = store.team_roster(stint["old_team"], mode="current", include_unavailable=False)
    on = store.team_roster(stint["old_team"], mode="current", include_unavailable=True)
    assert stint["player_id"] not in _ids(off)
    assert stint["player_id"] not in _ids(on)


def test_il_status_does_not_end_tenure_in_artifacts():
    if not INTERVAL_PATH.exists() or not TENURE_PATH.exists():
        pytest.skip("roster interval artifacts missing")
    intervals = pd.read_parquet(INTERVAL_PATH)
    tenures = pd.read_parquet(TENURE_PATH)
    il = intervals[intervals["roster_status"].astype(str).str.startswith("IL")]
    if il.empty:
        pytest.skip("no IL intervals")
    rec = il.iloc[0]
    pid = int(rec["player_id"])
    team_id = int(rec["team_id"])
    mid = _as_date(rec["start_at"])
    hit = tenures[
        (tenures["player_id"] == pid)
        & (tenures["team_id"] == team_id)
        & (tenures["start_at"].map(_as_date) <= mid)
        & (
            tenures["end_at"].map(_as_date).isna()
            | (tenures["end_at"].map(_as_date) > mid)
        )
    ]
    assert not hit.empty, "IL interval without covering org tenure"
