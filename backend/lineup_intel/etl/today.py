"""Optional CLI: build a one-day lineup snapshot from the MLB Stats API.

Not used by the product UI. Daily season ingest is ``refresh_lineups``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..config import settings
from ..identity import order_id, personnel_id, validate_lineup
from ..teams import normalize_abbrev
from .extract_lineups import _bat_side_code, fetch_pitch_hands, starting_batting_order_ids


def fetch_schedule(day: str | None = None) -> list[dict]:
    day = day or date.today().isoformat()
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": day,
        "hydrate": "probablePitcher,team,lineups,linescore",
    }
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    games: list[dict] = []
    for d in data.get("dates") or []:
        for g in d.get("games") or []:
            games.append(g)
    return games


def _probable(side_team: dict) -> tuple[int | None, str | None, str | None]:
    pp = side_team.get("probablePitcher") or {}
    pid = pp.get("id")
    name = pp.get("fullName")
    hand = ((pp.get("pitchHand") or {}).get("code")) if isinstance(pp.get("pitchHand"), dict) else None
    return (int(pid) if pid else None), hand, name


def _fill_missing_opp_hands(lineups: list[dict]) -> None:
    need: list[int] = []
    for lu in lineups:
        if lu.get("opp_sp_hand"):
            continue
        pid = lu.get("opp_sp_id")
        if pid:
            try:
                need.append(int(pid))
            except (TypeError, ValueError):
                continue
    if not need:
        return
    hands = fetch_pitch_hands(need)
    for lu in lineups:
        if lu.get("opp_sp_hand"):
            continue
        pid = lu.get("opp_sp_id")
        try:
            pid_i = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid_i = None
        if pid_i is not None and pid_i in hands:
            lu["opp_sp_hand"] = hands[pid_i]


def _enrich_opp_sp_from_schedule(lineups: list[dict], schedule: list[dict]) -> None:
    """Attach opposing starter id/hand/name from schedule probablePitcher.

    Extracted parquet rows often have a name but no id/hand — without this,
    the UI falls back to \"Hand TBD\" even when the starter is known.
    """
    by_pk: dict[int, dict] = {}
    for g in schedule:
        try:
            gpk = int(g["gamePk"])
        except (KeyError, TypeError, ValueError):
            continue
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        by_pk[gpk] = {
            "home": _probable(home),
            "away": _probable(away),
            "home_abbr": normalize_abbrev(
                ((home.get("team") or {}).get("abbreviation"))
            ),
            "away_abbr": normalize_abbrev(
                ((away.get("team") or {}).get("abbreviation"))
            ),
        }

    for lu in lineups:
        try:
            gpk = int(lu.get("game_pk"))
        except (TypeError, ValueError):
            continue
        meta = by_pk.get(gpk)
        if not meta:
            continue
        team = normalize_abbrev(str(lu.get("team") or "")) or str(lu.get("team") or "")
        # Opposing starter = the other club's probable/actual SP
        if team == meta.get("home_abbr"):
            pid, hand, name = meta["away"]
        elif team == meta.get("away_abbr"):
            pid, hand, name = meta["home"]
        else:
            # Fall back using is_home flag
            if lu.get("is_home") is True:
                pid, hand, name = meta["away"]
            elif lu.get("is_home") is False:
                pid, hand, name = meta["home"]
            else:
                continue
        if pid and not lu.get("opp_sp_id"):
            lu["opp_sp_id"] = pid
        if name and not lu.get("opp_sp_name"):
            lu["opp_sp_name"] = name
        if hand and not lu.get("opp_sp_hand"):
            lu["opp_sp_hand"] = hand


def _box_meta(box_team: dict, game_players: dict | None = None) -> dict[int, dict]:
    game_players = game_players or {}
    out: dict[int, dict] = {}
    for key, pdata in (box_team.get("players") or {}).items():
        person = pdata.get("person") or {}
        pid = person.get("id")
        if pid is None:
            continue
        gp = (
            game_players.get(f"ID{int(pid)}")
            or game_players.get(str(pid))
            or game_players.get(key)
            or {}
        )
        bat_side = _bat_side_code(pdata, person, gp)
        pos = (
            (pdata.get("position") or {}).get("abbreviation")
            or ((pdata.get("allPositions") or [{}])[0] or {}).get("abbreviation")
        )
        out[int(pid)] = {
            "player_id": int(pid),
            "name": person.get("fullName") or person.get("boxscoreName") or gp.get("fullName") or str(pid),
            "bat_side": bat_side,
            "position": pos,
        }
    return out


def _starter_from_box(box_team: dict) -> tuple[int | None, str | None, str | None]:
    pitchers = box_team.get("pitchers") or []
    players = box_team.get("players") or {}
    if not pitchers:
        return None, None, None
    pid = int(pitchers[0])
    pdata = players.get(f"ID{pid}") or players.get(str(pid)) or {}
    person = pdata.get("person") or {}
    hand = (
        (pdata.get("pitchHand") or {}).get("code")
        or (person.get("pitchHand") or {}).get("code")
    )
    name = person.get("fullName")
    return pid, hand, name


def fetch_live_lineups(game_pk: int) -> dict[str, Any] | None:
    """Pull batting orders + opp SP from live feed when posted."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        r = httpx.get(url, timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:
        return None

    gd = d.get("gameData") or {}
    live = d.get("liveData") or {}
    box = (live.get("boxscore") or {}).get("teams") or {}
    teams = gd.get("teams") or {}
    status = (gd.get("status") or {}).get("detailedState")
    out_sides: dict[str, dict] = {}

    for side, opp_side, is_home in (("home", "away", True), ("away", "home", False)):
        bt = box.get(side) or {}
        ot = box.get(opp_side) or {}
        # Prefer starter codes (100..900). Fall back to live array only when
        # the game has not started and no substitutes exist yet.
        ids = starting_batting_order_ids(bt)
        if not ids:
            order = bt.get("battingOrder") or []
            if len(order) != 9:
                continue
            # Reject if any listed player is already a substitute (post-start pollution).
            players = bt.get("players") or {}
            polluted = False
            for pid in order:
                pdata = players.get(f"ID{pid}") or players.get(str(pid)) or {}
                if (pdata.get("gameStatus") or {}).get("isSubstitute"):
                    polluted = True
                    break
                bo = pdata.get("battingOrder")
                try:
                    bo_i = int(bo) if bo is not None else None
                except (TypeError, ValueError):
                    bo_i = None
                if bo_i is not None and bo_i % 100 != 0:
                    polluted = True
                    break
            if polluted:
                continue
            try:
                ids = validate_lineup(order)
            except ValueError:
                continue
        team_info = teams.get(side) or {}
        opp_info = teams.get(opp_side) or {}
        abbr = normalize_abbrev(team_info.get("abbreviation"))
        opp_abbr = normalize_abbrev(opp_info.get("abbreviation"))
        meta = _box_meta(bt, gd.get("players") or {})
        batter_names = [meta.get(pid, {}).get("name") or str(pid) for pid in ids]
        opp_sp_id, opp_sp_hand, opp_sp_name = _starter_from_box(ot)
        out_sides[side] = {
            "game_pk": int(game_pk),
            "team": abbr,
            "opponent": opp_abbr,
            "is_home": is_home,
            "batting_order": ids,
            "batter_names": batter_names,
            "order_id": order_id(ids),
            "personnel_id": personnel_id(ids),
            "opp_sp_id": opp_sp_id,
            "opp_sp_hand": opp_sp_hand,
            "opp_sp_name": opp_sp_name,
            "status": status,
            "source": "mlb_live_feed",
            "extraction": "battingOrder_codes_100_900",
        }
    return out_sides or None


def _lineups_from_schedule_hydrate(g: dict, day: str) -> list[dict]:
    """Fallback when live feed has no battingOrder yet: schedule hydrate lineups."""
    teams = g.get("teams") or {}
    home_t = (teams.get("home") or {}).get("team") or {}
    away_t = (teams.get("away") or {}).get("team") or {}
    home_abbr = normalize_abbrev(home_t.get("abbreviation"))
    away_abbr = normalize_abbrev(away_t.get("abbreviation"))
    home_pp = _probable(teams.get("home") or {})
    away_pp = _probable(teams.get("away") or {})
    status = (g.get("status") or {}).get("detailedState")
    game_pk = int(g["gamePk"])
    lu = g.get("lineups") or {}
    rows: list[dict] = []

    for side, players_key, team, opp, is_home, opp_pp in (
        ("home", "homePlayers", home_abbr, away_abbr, True, away_pp),
        ("away", "awayPlayers", away_abbr, home_abbr, False, home_pp),
    ):
        players = lu.get(players_key) or []
        if len(players) < 9:
            continue
        # Prefer explicit starter battingOrder codes (100/200/…/900 or 1–9).
        # Do not take the first 9 players from a mixed list that may include subs.
        if any(p.get("battingOrder") for p in players):
            starters = []
            for p in players:
                bo = p.get("battingOrder")
                if bo is None:
                    continue
                try:
                    bo_i = int(bo)
                except (TypeError, ValueError):
                    continue
                # Schedule hydrate may use 1–9 or 100–900.
                if 1 <= bo_i <= 9:
                    slot = bo_i
                elif bo_i % 100 == 0 and 100 <= bo_i <= 900:
                    slot = bo_i // 100
                else:
                    continue
                starters.append((slot, p))
            if len({s for s, _ in starters}) != 9:
                continue
            ordered = [p for _, p in sorted(starters, key=lambda t: t[0])]
        else:
            ordered = list(players)[:9]
        ids_raw = [int(p["id"]) for p in ordered if p.get("id")]
        if len(ids_raw) != 9:
            continue
        try:
            ids = validate_lineup(ids_raw)
        except ValueError:
            continue
        names = [
            (p.get("fullName") or p.get("boxscoreName") or str(p.get("id")))
            for p in ordered
        ]
        opp_sp_id, opp_sp_hand, opp_sp_name = opp_pp
        rows.append({
            "game_pk": game_pk,
            "game_date": day,
            "team": team,
            "opponent": opp,
            "is_home": is_home,
            "batting_order": ids,
            "batter_names": names,
            "order_id": order_id(ids),
            "personnel_id": personnel_id(ids),
            "opp_sp_id": opp_sp_id,
            "opp_sp_hand": opp_sp_hand,
            "opp_sp_name": opp_sp_name,
            "status": status,
            "source": "mlb_schedule_lineups",
        })
    return rows


def build_today(day: str | None = None, *, force_live: bool = True) -> dict:
    day = day or date.today().isoformat()
    schedule = fetch_schedule(day)
    lineups_out: list[dict] = []
    seen: set[tuple[int, str]] = set()

    # 1) Prefer already-extracted season lineups for this date (completed games).
    lu_path = settings.processed_dir / "starting_lineups_2026.parquet"
    if lu_path.exists():
        df = pd.read_parquet(lu_path)
        day_df = df[df["game_date"].astype(str) == day]
        for _, row in day_df.iterrows():
            slots = [int(row[f"slot{i}"]) for i in range(1, 10)]
            team = normalize_abbrev(row["team"]) or str(row["team"])
            key = (int(row["game_pk"]), team)
            if key in seen:
                continue
            seen.add(key)
            lineups_out.append({
                "game_pk": int(row["game_pk"]),
                "game_date": day,
                "team": team,
                "opponent": normalize_abbrev(row["opponent"]) or row["opponent"],
                "is_home": bool(row["is_home"]),
                "batting_order": slots,
                "order_id": row["order_id"],
                "personnel_id": row["personnel_id"],
                "opp_sp_id": (
                    int(row["opp_sp_id"])
                    if "opp_sp_id" in row and pd.notna(row.get("opp_sp_id"))
                    else None
                ),
                "opp_sp_hand": (
                    None
                    if ("opp_sp_hand" not in row or pd.isna(row.get("opp_sp_hand")))
                    else row.get("opp_sp_hand")
                ),
                "opp_sp_name": (
                    None
                    if ("opp_sp_name" not in row or pd.isna(row.get("opp_sp_name")))
                    else row.get("opp_sp_name")
                ),
                "status": row.get("status"),
                "source": "extracted_lineups",
            })

    # 2) Live MLB feed / schedule hydrate for games not yet in the season parquet.
    if force_live:
        for g in schedule:
            gpk = int(g["gamePk"])
            teams = g.get("teams") or {}
            home_abbr = normalize_abbrev(
                ((teams.get("home") or {}).get("team") or {}).get("abbreviation")
            )
            away_abbr = normalize_abbrev(
                ((teams.get("away") or {}).get("team") or {}).get("abbreviation")
            )
            need_live = (gpk, home_abbr) not in seen or (gpk, away_abbr) not in seen
            if need_live:
                live = fetch_live_lineups(gpk)
                if live:
                    for side_row in live.values():
                        team = side_row["team"]
                        key = (gpk, team)
                        if key in seen:
                            continue
                        seen.add(key)
                        side_row = dict(side_row)
                        side_row["game_date"] = day
                        # Fill probable pitcher if box lacked starters
                        if not side_row.get("opp_sp_name"):
                            opp_side = "away" if side_row["is_home"] else "home"
                            pid, hand, name = _probable(teams.get(opp_side) or {})
                            side_row["opp_sp_id"] = side_row.get("opp_sp_id") or pid
                            side_row["opp_sp_hand"] = side_row.get("opp_sp_hand") or hand
                            side_row["opp_sp_name"] = name
                        lineups_out.append(side_row)

            # Schedule hydrate fallback
            if (gpk, home_abbr) not in seen or (gpk, away_abbr) not in seen:
                for row in _lineups_from_schedule_hydrate(g, day):
                    key = (row["game_pk"], row["team"])
                    if key in seen:
                        continue
                    seen.add(key)
                    lineups_out.append(row)

    schedule_slim = []
    status_by_pk: dict[int, dict] = {}
    for g in schedule:
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        try:
            gpk = int(g.get("gamePk"))
        except (TypeError, ValueError):
            continue
        ls = (g.get("linescore") or {}).get("teams") or {}
        home_runs = (ls.get("home") or {}).get("runs")
        away_runs = (ls.get("away") or {}).get("runs")
        if home_runs is None:
            home_runs = home.get("score")
        if away_runs is None:
            away_runs = away.get("score")
        live_status = (g.get("status") or {}).get("detailedState")
        status_by_pk[gpk] = {
            "status": live_status,
            "abstract": (g.get("status") or {}).get("abstractGameState"),
            "home": normalize_abbrev(home_team.get("abbreviation")),
            "away": normalize_abbrev(away_team.get("abbreviation")),
            "home_runs": home_runs,
            "away_runs": away_runs,
        }
        schedule_slim.append({
            "game_pk": g.get("gamePk"),
            "status": live_status,
            "home": normalize_abbrev(home_team.get("abbreviation")),
            "away": normalize_abbrev(away_team.get("abbreviation")),
            "game_date": day,
            "home_sp": (_probable(home)[2]),
            "away_sp": (_probable(away)[2]),
        })

    # Extracted parquet freezes status/score at pull time — overlay live MLB.
    for lu in lineups_out:
        try:
            gpk = int(lu.get("game_pk"))
        except (TypeError, ValueError):
            continue
        live = status_by_pk.get(gpk) or {}
        if live.get("status"):
            lu["status"] = live["status"]
        team = normalize_abbrev(str(lu.get("team") or "")) or str(lu.get("team") or "")
        if team == live.get("home"):
            if live.get("home_runs") is not None:
                lu["runs_scored"] = live["home_runs"]
            if live.get("away_runs") is not None:
                lu["runs_allowed"] = live["away_runs"]
        elif team == live.get("away"):
            if live.get("away_runs") is not None:
                lu["runs_scored"] = live["away_runs"]
            if live.get("home_runs") is not None:
                lu["runs_allowed"] = live["home_runs"]
        elif lu.get("is_home") is True and live.get("home_runs") is not None:
            lu["runs_scored"] = live["home_runs"]
            lu["runs_allowed"] = live.get("away_runs")
        elif lu.get("is_home") is False and live.get("away_runs") is not None:
            lu["runs_scored"] = live["away_runs"]
            lu["runs_allowed"] = live.get("home_runs")
        rs, ra = lu.get("runs_scored"), lu.get("runs_allowed")
        abstract = str(live.get("abstract") or "")
        if (
            rs is not None
            and ra is not None
            and abstract == "Final"
        ):
            try:
                rs_i, ra_i = int(rs), int(ra)
            except (TypeError, ValueError):
                rs_i = ra_i = None
            if rs_i is not None:
                lu["result"] = "W" if rs_i > ra_i else ("L" if rs_i < ra_i else "T")

    # Attach precomputed evaluations when present
    evals_path = settings.artifacts_dir / "lineup_evaluations.parquet"
    if evals_path.exists() and lineups_out:
        edf = pd.read_parquet(evals_path)
        eval_map = {}
        for _, e in edf.iterrows():
            team = normalize_abbrev(str(e["team"])) or str(e["team"])
            eval_map[(int(e["game_pk"]), team)] = {
                "actual_runs": float(e["actual_runs"]),
                "best_runs": float(e["best_runs"]),
                "gap": float(e["gap"]),
                "percentile": float(e["percentile"]),
                "rank": int(e["rank"]),
                "n_near_optimal": int(e.get("n_near_optimal") or 0),
                "operationally_equivalent": bool(e.get("operationally_equivalent")),
            }
        for lu in lineups_out:
            key = (lu["game_pk"], lu["team"])
            if key in eval_map:
                lu["evaluation"] = eval_map[key]

    # Extracted rows often lack opp SP id/hand — fill from schedule, then people API.
    _enrich_opp_sp_from_schedule(lineups_out, schedule)
    _fill_missing_opp_hands(lineups_out)

    calendar_today = date.today().isoformat()
    note = (
        "Official starting lineups from MLB when posted; season evaluations "
        "attach when precomputed for that game."
    )
    if day != calendar_today:
        note = f"Showing {day} (requested). Calendar today is {calendar_today}. " + note

    payload = {
        "date": day,
        "calendar_today": calendar_today,
        "n_schedule_games": len(schedule_slim),
        "n_lineups": len(lineups_out),
        "schedule": schedule_slim,
        "lineups": lineups_out,
        "available": True,
        "note": note,
    }
    try:
        out = settings.artifacts_dir / "today_lineups.json"
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (settings.artifacts_dir / "today.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        # Read-only deploy volumes still get a live response.
        payload["note"] = f"{note} (artifact write skipped: {exc})"
    return payload


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    args = p.parse_args()
    result = build_today(args.date)
    print(
        f"today {result['date']}: {result['n_schedule_games']} games, "
        f"{result['n_lineups']} lineups"
    )
