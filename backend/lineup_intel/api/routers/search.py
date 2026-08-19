"""Global search across teams, players, and games."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ...db.store import get_store
from ...teams import CANONICAL_ABBREVS, TEAMS

router = APIRouter(tags=["search"])


@router.get("/search")
def global_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    store = get_store()
    query = q.strip()
    ql = query.lower()
    results: list[dict] = []

    # Teams
    for abbr in CANONICAL_ABBREVS:
        meta = TEAMS[abbr]
        if ql in abbr.lower() or ql in meta["name"].lower() or ql in meta["division"].lower():
            results.append({
                "type": "team",
                "abbr": abbr,
                "name": meta["name"],
                "division": meta["division"],
            })

    # Players (hitters only — exclude pitchers from directory search)
    players = store.load_players()
    if players is not None:
        for rec in players.itertuples(index=False):
            pos = str(getattr(rec, "position", "") or "").upper()
            if pos == "P":
                continue
            name = str(getattr(rec, "name", "") or "")
            if ql in name.lower() or ql == str(rec.player_id):
                results.append({
                    "type": "player",
                    "player_id": int(rec.player_id),
                    "name": name,
                    "bat_side": getattr(rec, "bat_side", None),
                    "position": getattr(rec, "position", None),
                })
                if len([r for r in results if r["type"] == "player"]) >= limit:
                    break
    else:
        # still allow team-only search
        pass

    # Games / lineups by date or game_pk
    lineups = store.load_lineups()
    if lineups is not None:
        # numeric game_pk
        if query.isdigit():
            gpk = int(query)
            hits = lineups[lineups["game_pk"] == gpk]
            for rec in hits.head(limit).to_dict(orient="records"):
                results.append({
                    "type": "game",
                    "game_pk": int(rec["game_pk"]),
                    "game_date": rec.get("game_date"),
                    "team": rec.get("team"),
                    "opponent": rec.get("opponent"),
                })
        # date-like
        date_hits = lineups[lineups["game_date"].astype(str).str.contains(query, na=False)]
        seen = set()
        for rec in date_hits.head(limit * 2).to_dict(orient="records"):
            key = (rec["game_pk"], rec["team"])
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "type": "lineup",
                "game_pk": int(rec["game_pk"]),
                "game_date": rec.get("game_date"),
                "team": rec.get("team"),
                "opponent": rec.get("opponent"),
                "order_id": rec.get("order_id"),
            })
            if len([r for r in results if r["type"] in ("game", "lineup")]) >= limit:
                break

        # batter name search via batter_names column
        if "batter_names" in lineups.columns and not query.isdigit():
            name_hits = lineups[
                lineups["batter_names"].astype(str).str.lower().str.contains(ql, na=False)
            ]
            for rec in name_hits.head(10).to_dict(orient="records"):
                results.append({
                    "type": "lineup",
                    "game_pk": int(rec["game_pk"]),
                    "game_date": rec.get("game_date"),
                    "team": rec.get("team"),
                    "opponent": rec.get("opponent"),
                    "order_id": rec.get("order_id"),
                    "match": "batter_name",
                })

    # Deduplicate while preserving order
    deduped = []
    seen_keys = set()
    for r in results:
        key = tuple(sorted((k, str(v)) for k, v in r.items()))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)
        if len(deduped) >= limit:
            break

    return {
        "available": True,
        "query": query,
        "results": deduped,
        "lineups_index": lineups is not None,
        "players_index": players is not None,
    }
