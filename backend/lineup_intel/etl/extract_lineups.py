"""Extract starting lineups from DiamondIQ GUMBO cache (read-only)."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..config import settings
from ..identity import order_id, personnel_id, validate_lineup
from ..teams import normalize_abbrev


OUTCOME_CLASSES = ["K", "BB_HBP", "1B", "2B", "3B", "HR", "OUT_IP"]


def _open_gumbo(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _player_meta(box_team: dict, game_players: dict | None = None) -> dict[int, dict]:
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
        # Prefer a fielding/hitting position over Pitcher when both appear.
        pos = None
        for candidate in (
            (pdata.get("position") or {}).get("abbreviation"),
            *((ap or {}).get("abbreviation") for ap in (pdata.get("allPositions") or [])),
        ):
            if not candidate:
                continue
            if str(candidate).upper() == "P" and pos is None:
                pos = "P"
                continue
            if str(candidate).upper() != "P":
                pos = candidate
                break
        if pos is None:
            pos = (pdata.get("position") or {}).get("abbreviation")
        out[int(pid)] = {
            "player_id": int(pid),
            "name": person.get("fullName") or person.get("boxscoreName") or gp.get("fullName") or str(pid),
            "bat_side": bat_side,
            "position": pos,
            "batting_order_code": pdata.get("battingOrder"),
            "is_substitute": bool((pdata.get("gameStatus") or {}).get("isSubstitute")),
        }
    return out


def _pitch_hand_code(*objs: dict | None) -> str | None:
    """R/L from a GUMBO/MLB player object (boxscore row, person, or gameData player)."""
    for obj in objs:
        if not obj:
            continue
        for key in ("pitchHand", "throwHand"):
            raw = obj.get(key)
            if isinstance(raw, dict):
                code = raw.get("code")
            else:
                code = raw
            if not code:
                continue
            h = str(code).strip().upper()[:1]
            if h in {"R", "L"}:
                return h
    return None


def _bat_side_code(*objs: dict | None) -> str | None:
    """R/L/S from a GUMBO/MLB player object. Boxscore rows usually omit batSide;
    gameData.players has it."""
    for obj in objs:
        if not obj:
            continue
        raw = obj.get("batSide")
        if isinstance(raw, dict):
            code = raw.get("code")
        else:
            code = raw
        if not code:
            continue
        h = str(code).strip().upper()[:1]
        if h in {"R", "L", "S"}:
            return h
    return None


def _starter_pitcher(
    box_team: dict,
    live: dict,
    side: str,
    game_players: dict | None = None,
    game_data: dict | None = None,
) -> tuple[int | None, str | None, str | None]:
    """Return (pitcher_id, hand, name) for the starting pitcher.

    Boxscore ``players`` entries usually have name/id only. Throw hand is on
    ``gameData.players``. If that is also missing, the caller can fill from the
    MLB people API.
    """
    pitchers = box_team.get("pitchers") or []
    players = box_team.get("players") or {}
    game_players = game_players or {}
    pid: int | None = None
    if pitchers:
        try:
            pid = int(pitchers[0])
        except (TypeError, ValueError):
            pid = None
    if pid is None:
        pp = ((game_data or {}).get("probablePitchers") or {}).get(side) or {}
        if isinstance(pp, dict) and pp.get("id") is not None:
            try:
                pid = int(pp["id"])
            except (TypeError, ValueError):
                pid = None
    if pid is None:
        return None, None, None
    pdata = players.get(f"ID{pid}") or players.get(str(pid)) or {}
    person = pdata.get("person") or {}
    gp = game_players.get(f"ID{pid}") or game_players.get(str(pid)) or {}
    pp = ((game_data or {}).get("probablePitchers") or {}).get(side) or {}
    if not isinstance(pp, dict):
        pp = {}
    hand = _pitch_hand_code(pdata, person, gp, pp)
    name = person.get("fullName") or gp.get("fullName") or pp.get("fullName")
    return pid, hand, name


def fetch_pitch_hands(player_ids: list[int]) -> dict[int, str]:
    """Look up throw-hand codes from the MLB Stats API people endpoint."""
    ids = sorted({int(i) for i in player_ids if i})
    if not ids:
        return {}
    import httpx

    out: dict[int, str] = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i : i + 40]
        try:
            r = httpx.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={"personIds": ",".join(str(x) for x in chunk)},
                timeout=30,
            )
            if r.status_code != 200:
                continue
            for person in r.json().get("people") or []:
                pid = person.get("id")
                code = _pitch_hand_code(person)
                if pid is not None and code:
                    out[int(pid)] = code
        except Exception:  # noqa: BLE001
            continue
    return out


def fill_opp_sp_hands(df: pd.DataFrame, *, use_people_api: bool = True) -> dict[str, int]:
    """Fill missing opp_sp_hand from same-pitcher rows, GUMBO, then people API.

    Mutates ``df`` in place. Returns coverage stats.
    """
    if df is None or df.empty or "opp_sp_hand" not in df.columns:
        return {"known": 0, "total": 0, "filled": 0}

    def _missing_mask() -> pd.Series:
        raw = df["opp_sp_hand"]
        return raw.isna() | raw.astype(str).str.strip().isin(["", "None", "nan", "NaN"])

    before = int((~_missing_mask()).sum())

    # 1) Copy a known hand from another game with the same opposing starter.
    known_by_id: dict[int, str] = {}
    for rec in df.loc[~_missing_mask(), ["opp_sp_id", "opp_sp_hand"]].to_dict(orient="records"):
        code = str(rec.get("opp_sp_hand") or "").strip().upper()[:1]
        if code not in {"R", "L"}:
            continue
        try:
            known_by_id[int(rec["opp_sp_id"])] = code
        except (TypeError, ValueError):
            continue
    miss = _missing_mask()
    if miss.any() and known_by_id:
        mapped = df.loc[miss, "opp_sp_id"].map(
            lambda pid: known_by_id.get(int(pid)) if pd.notna(pid) else None
        )
        df.loc[miss, "opp_sp_hand"] = mapped

    # 2) GUMBO gameData.players.pitchHand (boxscore rows usually omit it).
    miss = _missing_mask()
    if miss.any() and "game_pk" in df.columns:
        cache = Path(settings.gumbo_cache)
        hands: dict[int, str] = {}
        for gpk in df.loc[miss, "game_pk"].dropna().unique():
            gpath = cache / f"{int(gpk)}.json.gz"
            if not gpath.exists():
                continue
            try:
                d = _open_gumbo(gpath)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            for gp in ((d.get("gameData") or {}).get("players") or {}).values():
                if not isinstance(gp, dict):
                    continue
                pid = gp.get("id")
                code = _pitch_hand_code(gp)
                if pid is not None and code:
                    hands[int(pid)] = code
        if hands:
            mapped = df.loc[miss, "opp_sp_id"].map(
                lambda pid: hands.get(int(pid)) if pd.notna(pid) else None
            )
            df.loc[miss, "opp_sp_hand"] = mapped

    # 3) MLB people API for remaining ids (covers live-refresh games not in GUMBO).
    miss = _missing_mask()
    if use_people_api and miss.any() and "opp_sp_id" in df.columns:
        need: list[int] = []
        for pid in df.loc[miss, "opp_sp_id"].dropna().unique():
            try:
                need.append(int(pid))
            except (TypeError, ValueError):
                continue
        people = fetch_pitch_hands(need)
        if people:
            mapped = df.loc[miss, "opp_sp_id"].map(
                lambda pid: people.get(int(pid)) if pd.notna(pid) else None
            )
            df.loc[miss, "opp_sp_hand"] = mapped

    known = int((~_missing_mask()).sum())
    return {
        "known": known,
        "total": int(len(df)),
        "filled": known - before,
    }


def starting_batting_order_ids(box_team: dict) -> list[int] | None:
    """Extract the official STARTING 1–9 order from GUMBO/live boxscore.

    MLB's ``boxscore.teams.*.battingOrder`` array is the *current/final*
    occupant of each batting-order slot (substitutes overwrite starters).
    Starters are players whose ``battingOrder`` code is exactly 100, 200, …, 900.

    Substitutes use codes like 301, 302 (slot 3, replacement #1/#2). Using the
    live array incorrectly inserts pitchers (e.g. Aroldis Chapman) into
    "starting" lineups after they enter as defensive/pitching substitutes.
    """
    starters: dict[int, int] = {}
    for pdata in (box_team.get("players") or {}).values():
        bo = pdata.get("battingOrder")
        if bo is None:
            continue
        try:
            bo_i = int(bo)
        except (TypeError, ValueError):
            continue
        # Starter codes are exact hundreds; anything else is a replacement.
        if bo_i % 100 != 0 or not (100 <= bo_i <= 900):
            continue
        gs = pdata.get("gameStatus") or {}
        if gs.get("isSubstitute"):
            continue
        person = pdata.get("person") or {}
        pid = person.get("id")
        if pid is None:
            continue
        slot = bo_i // 100
        starters[slot] = int(pid)

    if set(starters.keys()) != set(range(1, 10)):
        return None
    try:
        return validate_lineup([starters[i] for i in range(1, 10)])
    except ValueError:
        return None


def iter_game_lineups(path: Path, season: int | None = None) -> Iterator[dict]:
    d = _open_gumbo(path)
    gd = d.get("gameData") or {}
    live = d.get("liveData") or {}
    box = (live.get("boxscore") or {}).get("teams") or {}
    dt = gd.get("datetime") or {}
    status = gd.get("status") or {}
    teams = gd.get("teams") or {}
    venue = (gd.get("venue") or {}).get("name")
    game_pk = (gd.get("game") or {}).get("pk") or int(path.name.split(".")[0])
    game_date = dt.get("officialDate")
    game_season = int((game_date or "0")[:4]) if game_date else None
    game_players = gd.get("players") or {}
    if season is not None and game_season != season:
        return
    abstract = status.get("abstractGameState")
    detailed = status.get("detailedState")
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
        meta = _player_meta(bt, game_players)
        batters = []
        for slot, pid in enumerate(ids, start=1):
            m = meta.get(pid, {"player_id": pid, "name": str(pid), "bat_side": None, "position": None})
            batters.append({
                "slot": slot,
                "player_id": pid,
                "name": m.get("name") or str(pid),
                "bat_side": m.get("bat_side"),
                "position": m.get("position"),
            })
        opp_sp_id, opp_sp_hand, opp_sp_name = _starter_pitcher(
            ot, live, opp_side, game_players, gd
        )
        linescore = live.get("linescore") or {}
        teams_ls = linescore.get("teams") or {}
        home_runs = (teams_ls.get("home") or {}).get("runs")
        away_runs = (teams_ls.get("away") or {}).get("runs")
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
        yield {
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
            "extraction": "battingOrder_codes_100_900",
        }


def extract_lineups(season: int = 2026, limit: int | None = None) -> pd.DataFrame:
    cache = Path(settings.gumbo_cache)
    rows: list[dict] = []
    players: dict[int, dict] = {}
    files = sorted(cache.glob("*.json.gz"))
    for i, path in enumerate(files):
        if limit is not None and i >= limit:
            break
        for lu in iter_game_lineups(path, season=season):
            rows.append({
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
            })
            for b in lu["batters"]:
                pid = int(b["player_id"])
                prev = players.get(pid, {})
                players[pid] = {
                    "player_id": pid,
                    "name": b.get("name") or prev.get("name") or str(pid),
                    "bat_side": b.get("bat_side") or prev.get("bat_side"),
                    "position": b.get("position") or prev.get("position"),
                }
    df = pd.DataFrame(rows)
    players_df = pd.DataFrame(list(players.values()))
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"starting_lineups_{season}.parquet", index=False)
    players_df.to_parquet(out / f"players_{season}.parquet", index=False)
    return df


def backfill_opp_sp_hands(season: int | None = None) -> dict[str, int]:
    """Fill missing opp_sp_hand from GUMBO, same-pitcher rows, and MLB people API."""
    season = season or settings.target_season
    path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    df = pd.read_parquet(path)
    stats = fill_opp_sp_hands(df, use_people_api=True)
    df.to_parquet(path, index=False)
    return stats


def _normalize_bat_side(v: object) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    h = str(v).strip().upper()[:1]
    return h if h in {"R", "L", "S"} else None


def backfill_bat_sides(season: int | None = None) -> dict[str, int]:
    """Fill players_{season}.parquet bat_side from GUMBO gameData, PA table, and players_all."""
    season = season or settings.target_season
    path = settings.processed_dir / f"players_{season}.parquet"
    df = pd.read_parquet(path)
    sides: dict[int, str] = {}

    all_path = settings.processed_dir / f"players_{season}_all.parquet"
    if all_path.exists():
        all_df = pd.read_parquet(all_path)
        if "bat_side" in all_df.columns:
            for rec in all_df.to_dict(orient="records"):
                code = _normalize_bat_side(rec.get("bat_side"))
                if code is None:
                    continue
                try:
                    sides[int(rec["player_id"])] = code
                except (TypeError, ValueError):
                    continue

    pa_path = settings.processed_dir / "plate_appearances.parquet"
    if pa_path.exists():
        pa = pd.read_parquet(pa_path, columns=["batter_id", "batter_side"])
        pa = pa.dropna(subset=["batter_id", "batter_side"])
        if not pa.empty:
            mode = (
                pa.groupby("batter_id")["batter_side"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
            )
            for pid, side in mode.items():
                code = _normalize_bat_side(side)
                if code is None:
                    continue
                try:
                    pid_i = int(pid)
                except (TypeError, ValueError):
                    continue
                # Prefer explicit S from roster files over PA mode (switch hitters).
                if pid_i not in sides or sides[pid_i] != "S":
                    if pid_i not in sides:
                        sides[pid_i] = code

    cache = Path(settings.gumbo_cache)
    gumbo_mapped = 0
    for gpath in cache.glob("*.json.gz"):
        try:
            d = _open_gumbo(gpath)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        for gp in ((d.get("gameData") or {}).get("players") or {}).values():
            if not isinstance(gp, dict):
                continue
            pid = gp.get("id")
            code = _bat_side_code(gp)
            if pid is None or not code:
                continue
            sides[int(pid)] = code
            gumbo_mapped += 1

    def mapped(pid: object) -> str | None:
        try:
            return sides.get(int(pid))
        except (TypeError, ValueError):
            return None

    filled = df["player_id"].map(mapped)
    missing = df["bat_side"].map(_normalize_bat_side).isna()
    df.loc[missing, "bat_side"] = filled[missing]
    df.to_parquet(path, index=False)
    known = int(df["bat_side"].map(_normalize_bat_side).notna().sum())
    return {
        "known": known,
        "total": int(len(df)),
        "mapped": len(sides),
        "gumbo_player_rows": gumbo_mapped,
    }


def export_pa_table(seasons: list[int] | None = None) -> pd.DataFrame:
    seasons = seasons or list(settings.train_seasons) + [settings.target_season]
    db = settings.diamondiq_db
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    placeholders = ",".join("?" for _ in seasons)
    q = f"""
    SELECT pa.game_pk, g.season, g.game_date, g.home_abbrev, g.away_abbrev, g.venue,
           pa.at_bat_index, pa.batter_id, pa.pitcher_id, pa.batter_side, pa.pitcher_hand,
           pa.inning, pa.half, pa.outs_start, pa.bases_start,
           pa.outcome_class, pa.runs_scored, pa.event_type
    FROM plate_appearances pa
    JOIN games g ON g.game_pk = pa.game_pk
    WHERE g.season IN ({placeholders}) AND g.backfilled = 1
      AND pa.outcome_class IS NOT NULL
      AND pa.outcome_class != 'OTHER'
    """
    df = pd.read_sql_query(q, con, params=seasons)
    con.close()
    # Map ATH/OAK consistency later at join time
    path = settings.processed_dir / "plate_appearances.parquet"
    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--export-pa", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--backfill-hands",
        action="store_true",
        help="Fill opp_sp_hand from GUMBO gameData without a full re-extract",
    )
    p.add_argument(
        "--backfill-bat-sides",
        action="store_true",
        help="Fill players bat_side from GUMBO gameData without a full re-extract",
    )
    args = p.parse_args()
    if args.export_pa:
        pa = export_pa_table()
        print(f"exported {len(pa)} PAs")
    if args.backfill_hands:
        stats = backfill_opp_sp_hands(season=args.season)
        print(
            f"opp_sp_hand {stats['known']}/{stats['total']} "
            f"(filled {stats.get('filled', 0)})"
        )
    if args.backfill_bat_sides:
        stats = backfill_bat_sides(season=args.season)
        print(
            f"bat_side {stats['known']}/{stats['total']} "
            f"({stats['mapped']} players mapped)"
        )
    if not args.backfill_hands and not args.backfill_bat_sides:
        lu = extract_lineups(season=args.season, limit=args.limit)
        print(f"extracted {len(lu)} starting lineups for {args.season}")
        if len(lu):
            print(lu.groupby("team").size().sort_values(ascending=False).head(10))
            print("teams", lu["team"].nunique())
