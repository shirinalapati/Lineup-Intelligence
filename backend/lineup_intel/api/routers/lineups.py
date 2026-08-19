"""Lineup list and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...db.store import SLOT_COLS, get_store, unavailable
from ...teams import normalize_abbrev

router = APIRouter(tags=["lineups"])


@router.get("/lineups")
def list_lineups(
    team: str | None = None,
    opponent: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    game_pk: int | None = None,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    store = get_store()
    df = store.load_lineups()
    if df is None:
        return unavailable(f"starting_lineups_{store.season}.parquet not found")
    if team:
        t = normalize_abbrev(team) or team.upper()
        df = df[df["team"] == t]
    if opponent:
        o = normalize_abbrev(opponent) or opponent.upper()
        df = df[df["opponent"] == o]
    if game_pk is not None:
        df = df[df["game_pk"] == int(game_pk)]
    if date_from:
        df = df[df["game_date"].astype(str) >= date_from]
    if date_to:
        df = df[df["game_date"].astype(str) <= date_to]
    df = df.sort_values(["game_date", "game_pk", "team"], ascending=[False, False, True])
    total = int(len(df))
    page = df.iloc[offset : offset + limit]
    rows = []
    for rec in page.to_dict(orient="records"):
        rows.append({
            "game_pk": int(rec["game_pk"]),
            "game_date": rec.get("game_date"),
            "team": rec.get("team"),
            "opponent": rec.get("opponent"),
            "is_home": bool(rec.get("is_home")),
            "order_id": rec.get("order_id"),
            "personnel_id": rec.get("personnel_id"),
            "batting_order": [int(rec[c]) for c in SLOT_COLS],
            "runs_scored": rec.get("runs_scored"),
            "result": rec.get("result"),
        })
    return {"available": True, "total": total, "offset": offset, "limit": limit, "lineups": rows}


@router.get("/lineups/{game_pk}/{team}")
def lineup_detail(game_pk: int, team: str):
    store = get_store()
    row = store.lineup_row(game_pk, team)
    if row is None:
        raise HTTPException(status_code=404, detail="Lineup not found")
    names = store.player_name_map()
    order = row.get("batting_order") or []

    # Prefer stored batter meta when present
    raw_names = row.get("batter_names")
    raw_sides = row.get("batter_sides")
    raw_pos = row.get("batter_positions")
    name_list = (
        raw_names.split("|") if isinstance(raw_names, str)
        else (list(raw_names) if isinstance(raw_names, list) else [])
    )
    side_list = (
        raw_sides.split("|") if isinstance(raw_sides, str)
        else (list(raw_sides) if isinstance(raw_sides, list) else [])
    )
    pos_list = (
        raw_pos.split("|") if isinstance(raw_pos, str)
        else (list(raw_pos) if isinstance(raw_pos, list) else [])
    )

    batters = []
    for i, pid in enumerate(order):
        pid = int(pid)
        nm = name_list[i] if i < len(name_list) and name_list[i] not in ("", "?") else None
        side = side_list[i] if i < len(side_list) and side_list[i] not in ("", "?") else None
        pos = pos_list[i] if i < len(pos_list) and pos_list[i] not in ("", "?") else None
        batters.append({
            "slot": i + 1,
            "player_id": pid,
            "name": nm or names.get(pid, str(pid)),
            "bat_side": side,
            "position": pos,
        })
    row["batters"] = batters
    row["batter_names"] = [b["name"] for b in batters]

    # Enrich evaluation with parsed orders + display names
    ev = row.get("evaluation")
    if isinstance(ev, dict) and ev.get("available") is not False:
        import json

        def _parse_ids(val):
            if val is None:
                return None
            if isinstance(val, list):
                return [int(x) for x in val]
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                except (TypeError, json.JSONDecodeError, ValueError):
                    return None
            return None

        for key in ("best_order_ids", "worst_order_ids", "actual_order_ids"):
            parsed = _parse_ids(ev.get(key))
            if parsed:
                ev[key] = parsed
                ev[key.replace("_ids", "_names")] = [names.get(i, str(i)) for i in parsed]
        ev["player_names"] = {str(pid): names.get(int(pid), str(pid)) for pid in order}
        if ev.get("best_order_ids"):
            for pid in ev["best_order_ids"]:
                ev["player_names"][str(pid)] = names.get(int(pid), str(pid))
        row["evaluation"] = ev
    elif "evaluation" not in row or row["evaluation"] is None:
        row["evaluation"] = unavailable(
            "lineup_evaluations.parquet missing or this game/team not precomputed"
        )
    return {"available": True, "lineup": row}
