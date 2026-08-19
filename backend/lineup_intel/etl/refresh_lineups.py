"""Incrementally refresh starting lineups from MLB Stats API.

Pulls completed games after the latest date in starting_lineups_{season}.parquet
(or an explicit --since date), merges them into the processed parquet, and
updates the players table.

Does not write to the DiamondIQ GUMBO cache (read-only).

CLI:
  PYTHONPATH=backend python -m lineup_intel.etl.refresh_lineups
  PYTHONPATH=backend python -m lineup_intel.etl.refresh_lineups --since 2026-08-12
  PYTHONPATH=backend python -m lineup_intel.etl.refresh_lineups --through 2026-08-18
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import httpx
import pandas as pd

from ..config import settings
from ..identity import order_id, personnel_id
from ..teams import normalize_abbrev
from .extract_lineups import _player_meta, _starter_pitcher, fill_opp_sp_hands, starting_batting_order_ids


SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def _daterange(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _schedule_for_day(day: str, client: httpx.Client) -> list[dict]:
    r = client.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": day, "hydrate": "team"},
        timeout=30,
    )
    r.raise_for_status()
    games: list[dict] = []
    for d in r.json().get("dates") or []:
        games.extend(d.get("games") or [])
    return games


def _lineups_from_live(game_pk: int, client: httpx.Client) -> list[dict]:
    """Extract starting lineups from the MLB live feed (same shape as GUMBO extract)."""
    r = client.get(
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        timeout=45,
    )
    if r.status_code != 200:
        return []
    d = r.json()
    gd = d.get("gameData") or {}
    live = d.get("liveData") or {}
    box = (live.get("boxscore") or {}).get("teams") or {}
    dt = gd.get("datetime") or {}
    status = gd.get("status") or {}
    teams = gd.get("teams") or {}
    venue = (gd.get("venue") or {}).get("name")
    game_date = dt.get("officialDate")
    game_season = int((game_date or "0")[:4]) if game_date else None
    abstract = status.get("abstractGameState")
    detailed = status.get("detailedState")

    # Prefer completed / in-progress games that already have batting orders
    if abstract not in ("Final", "Live") and detailed not in (
        "Final",
        "Game Over",
        "Completed Early",
        "In Progress",
    ):
        return []

    linescore = live.get("linescore") or {}
    teams_ls = linescore.get("teams") or {}
    home_runs = (teams_ls.get("home") or {}).get("runs")
    away_runs = (teams_ls.get("away") or {}).get("runs")

    out: list[dict] = []
    for side, opp_side, is_home in (("home", "away", True), ("away", "home", False)):
        bt = box.get(side) or {}
        ot = box.get(opp_side) or {}
        ids = starting_batting_order_ids(bt)
        if not ids:
            continue
        team_info = teams.get(side) or {}
        opp_info = teams.get(opp_side) or {}
        abbr = normalize_abbrev(team_info.get("abbreviation"))
        opp_abbr = normalize_abbrev(opp_info.get("abbreviation"))
        if not abbr or not opp_abbr:
            continue
        meta = _player_meta(bt, gd.get("players") or {})
        batters = []
        for slot, pid in enumerate(ids, start=1):
            m = meta.get(
                pid,
                {"player_id": pid, "name": str(pid), "bat_side": None, "position": None},
            )
            batters.append({
                "slot": slot,
                "player_id": pid,
                "name": m.get("name") or str(pid),
                "bat_side": m.get("bat_side"),
                "position": m.get("position"),
            })
        opp_sp_id, opp_sp_hand, opp_sp_name = _starter_pitcher(
            ot, live, opp_side, gd.get("players") or {}, gd
        )
        runs_scored = home_runs if is_home else away_runs
        runs_allowed = away_runs if is_home else home_runs
        result = None
        if home_runs is not None and away_runs is not None and abstract == "Final":
            if runs_scored > runs_allowed:
                result = "W"
            elif runs_scored < runs_allowed:
                result = "L"
            else:
                result = "T"
        out.append({
            "game_pk": int(game_pk),
            "season": game_season,
            "game_date": game_date,
            "game_datetime": dt.get("dateTime"),
            "status": detailed,
            "abstract_state": abstract,
            "venue": venue,
            "team": abbr,
            "team_id": team_info.get("id"),
            "opponent": opp_abbr,
            "opponent_id": opp_info.get("id"),
            "is_home": is_home,
            "batting_order": ids,
            "order_id": order_id(ids),
            "personnel_id": personnel_id(ids),
            "batters": batters,
            "opp_sp_id": opp_sp_id,
            "opp_sp_hand": opp_sp_hand,
            "opp_sp_name": opp_sp_name,
            "runs_scored": runs_scored,
            "runs_allowed": runs_allowed,
            "result": result,
            "source": "mlb_live_refresh",
            "extraction": "battingOrder_codes_100_900",
        })
    return out


def _row_from_lu(lu: dict) -> dict:
    return {
        "game_pk": lu["game_pk"],
        "season": lu["season"],
        "game_date": lu["game_date"],
        "game_datetime": lu["game_datetime"],
        "status": lu["status"],
        "abstract_state": lu["abstract_state"],
        "venue": lu["venue"],
        "team": lu["team"],
        "team_id": lu["team_id"],
        "opponent": lu["opponent"],
        "opponent_id": lu["opponent_id"],
        "is_home": lu["is_home"],
        "slot1": lu["batting_order"][0],
        "slot2": lu["batting_order"][1],
        "slot3": lu["batting_order"][2],
        "slot4": lu["batting_order"][3],
        "slot5": lu["batting_order"][4],
        "slot6": lu["batting_order"][5],
        "slot7": lu["batting_order"][6],
        "slot8": lu["batting_order"][7],
        "slot9": lu["batting_order"][8],
        "order_id": lu["order_id"],
        "personnel_id": lu["personnel_id"],
        "opp_sp_id": lu["opp_sp_id"],
        "opp_sp_hand": lu["opp_sp_hand"],
        "opp_sp_name": lu["opp_sp_name"],
        "runs_scored": lu["runs_scored"],
        "runs_allowed": lu["runs_allowed"],
        "result": lu["result"],
        "batter_names": "|".join(b["name"] for b in lu["batters"]),
        "batter_sides": "|".join((b.get("bat_side") or "?") for b in lu["batters"]),
        "batter_positions": "|".join((b.get("position") or "?") for b in lu["batters"]),
    }


def refresh_lineups(
    season: int | None = None,
    since: str | None = None,
    through: str | None = None,
) -> dict[str, Any]:
    season = season or settings.target_season
    lu_path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    players_path = settings.processed_dir / f"players_{season}.parquet"

    if lu_path.exists():
        existing = pd.read_parquet(lu_path)
    else:
        existing = pd.DataFrame()

    if since:
        start = date.fromisoformat(since)
    elif len(existing) and "game_date" in existing.columns:
        max_d = str(existing["game_date"].max())[:10]
        # Re-check the latest date so delayed games pick up final scores.
        start = date.fromisoformat(max_d)
    else:
        start = date(season, 3, 20)

    end = date.fromisoformat(through) if through else date.today()
    if start > end:
        return {
            "available": True,
            "season": season,
            "since": start.isoformat(),
            "through": end.isoformat(),
            "n_new_lineups": 0,
            "n_total": int(len(existing)),
            "note": "already up to date",
        }

    known: set[tuple[int, str]] = set()
    if len(existing):
        for rec in existing.to_dict(orient="records"):
            known.add((int(rec["game_pk"]), str(rec["team"])))

    new_rows: list[dict] = []
    score_updates: list[dict] = []
    players: dict[int, dict] = {}
    if players_path.exists():
        for rec in pd.read_parquet(players_path).to_dict(orient="records"):
            players[int(rec["player_id"])] = {
                "player_id": int(rec["player_id"]),
                "name": rec.get("name"),
                "bat_side": rec.get("bat_side"),
                "position": rec.get("position"),
            }

    fetched_games = 0
    with httpx.Client() as client:
        for day in _daterange(start, end):
            day_s = day.isoformat()
            try:
                games = _schedule_for_day(day_s, client)
            except Exception as exc:  # noqa: BLE001
                print(f"[refresh] schedule {day_s} failed: {exc}")
                continue
            for g in games:
                status = (g.get("status") or {})
                abstract = status.get("abstractGameState")
                detailed = status.get("detailedState")
                if abstract not in ("Final", "Live") and detailed not in (
                    "Final",
                    "Game Over",
                    "Completed Early",
                ):
                    continue
                gpk = int(g["gamePk"])
                fetched_games += 1
                try:
                    lus = _lineups_from_live(gpk, client)
                except Exception as exc:  # noqa: BLE001
                    print(f"[refresh] live {gpk} failed: {exc}")
                    continue
                for lu in lus:
                    key = (lu["game_pk"], lu["team"])
                    row = _row_from_lu(lu)
                    if key in known:
                        score_updates.append({
                            "game_pk": row["game_pk"],
                            "team": row["team"],
                            "status": row["status"],
                            "abstract_state": row["abstract_state"],
                            "runs_scored": row["runs_scored"],
                            "runs_allowed": row["runs_allowed"],
                            "result": row["result"],
                        })
                        continue
                    known.add(key)
                    new_rows.append(row)
                    for b in lu["batters"]:
                        pid = int(b["player_id"])
                        prev = players.get(pid, {})
                        players[pid] = {
                            "player_id": pid,
                            "name": b.get("name") or prev.get("name") or str(pid),
                            "bat_side": b.get("bat_side") or prev.get("bat_side"),
                            "position": b.get("position") or prev.get("position"),
                        }
            print(
                f"[refresh] {day_s}: schedule games pulled, "
                f"new lineups so far {len(new_rows)}"
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if len(existing):
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["game_pk", "team"], keep="last")
        else:
            merged = new_df
    else:
        merged = existing

    n_score_updates = 0
    if score_updates and len(merged):
        upd = pd.DataFrame(score_updates)
        merged = merged.merge(
            upd,
            on=["game_pk", "team"],
            how="left",
            suffixes=("", "_live"),
        )
        for col in ("status", "abstract_state", "runs_scored", "runs_allowed", "result"):
            live_col = f"{col}_live"
            if live_col not in merged.columns:
                continue
            changed = merged[live_col].notna() & (
                merged[live_col].astype(str) != merged[col].astype(str)
            )
            n_score_updates += int(changed.sum())
            merged[col] = merged[live_col].where(merged[live_col].notna(), merged[col])
            merged = merged.drop(columns=[live_col])

    hand_stats = fill_opp_sp_hands(merged, use_people_api=True) if len(merged) else {"filled": 0}
    if len(merged) and (new_rows or score_updates or (hand_stats.get("filled") or 0) > 0):
        merged = merged.sort_values(["game_date", "game_pk", "team"]).reset_index(drop=True)
        merged.to_parquet(lu_path, index=False)

    players_df = pd.DataFrame(list(players.values()))
    if len(players_df):
        players_df = players_df.sort_values("name").reset_index(drop=True)
        players_df.to_parquet(players_path, index=False)

    summary = {
        "available": True,
        "season": season,
        "since": start.isoformat(),
        "through": end.isoformat(),
        "n_schedule_games_considered": fetched_games,
        "n_new_lineups": len(new_rows),
        "n_score_updates": n_score_updates,
        "n_total": int(len(merged)),
        "max_game_date": str(merged["game_date"].max()) if len(merged) else None,
        "cle_games": int((merged["team"] == "CLE").sum()) if len(merged) else 0,
        "written": str(lu_path),
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = settings.artifacts_dir / "lineup_refresh.json"
    meta_path.write_text(
        pd.Series(summary).to_json(),
        encoding="utf-8",
    )
    # prettier json
    import json

    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Refresh starting lineups from MLB API")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--since", type=str, default=None, help="YYYY-MM-DD inclusive start")
    p.add_argument("--through", type=str, default=None, help="YYYY-MM-DD inclusive end")
    args = p.parse_args(argv)
    result = refresh_lineups(season=args.season, since=args.since, through=args.through)
    print(
        f"refresh {result['since']}→{result['through']}: "
        f"+{result['n_new_lineups']} lineups, total {result['n_total']}, "
        f"max date {result['max_game_date']}, CLE={result['cle_games']}"
    )


if __name__ == "__main__":
    main()
