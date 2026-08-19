"""Team endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ...db.store import get_store, unavailable
from ...research.ranking import rank_team_metrics
from ...teams import CANONICAL_ABBREVS, TEAMS, normalize_abbrev

router = APIRouter(tags=["teams"])


def _teams_from_summaries(summaries: dict | None) -> dict[str, dict]:
    if not summaries or not isinstance(summaries, dict):
        return {}
    teams = summaries.get("teams") or summaries
    if isinstance(teams, dict):
        out: dict[str, dict] = {}
        for key, val in teams.items():
            if not isinstance(val, dict):
                continue
            row = dict(val)
            abbr = str(row.get("abbr") or key)
            row.setdefault("abbr", abbr)
            out[abbr] = row
        return out
    if isinstance(teams, list):
        return {
            str(t["abbr"]): dict(t)
            for t in teams
            if isinstance(t, dict) and t.get("abbr")
        }
    return {}


def _overlay_live_counts(teams: dict[str, dict], lineups: Any) -> None:
    if lineups is None:
        return
    for abbr in CANONICAL_ABBREVS:
        tdf = lineups[lineups["team"] == abbr]
        row = teams.setdefault(abbr, {"abbr": abbr})
        row["games"] = int(len(tdf))
        row["unique_orders"] = int(tdf["order_id"].nunique()) if len(tdf) else 0
        row["unique_personnel"] = int(tdf["personnel_id"].nunique()) if len(tdf) else 0


@router.get("/teams")
def list_teams():
    store = get_store()
    summaries = store.load_team_summaries()
    lineups = store.load_lineups()
    out = []
    for abbr in CANONICAL_ABBREVS:
        meta = TEAMS[abbr]
        row = {
            "abbr": abbr,
            "name": meta["name"],
            "division": meta["division"],
            "id": meta["id"],
        }
        if summaries and isinstance(summaries, dict):
            teams = summaries.get("teams") or summaries
            if isinstance(teams, dict) and abbr in teams:
                row["summary"] = teams[abbr]
            elif isinstance(teams, list):
                match = next((t for t in teams if t.get("abbr") == abbr), None)
                if match:
                    row["summary"] = match
        if lineups is not None:
            row["games_games"] = int((lineups["team"] == abbr).sum())
        out.append(row)
    return {"available": True, "teams": out}


@router.get("/teams/{abbr}")
def team_detail(abbr: str):
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    store = get_store()
    meta = TEAMS[abbr_n]
    payload: dict = {
        "available": True,
        "abbr": abbr_n,
        "name": meta["name"],
        "division": meta["division"],
        "id": meta["id"],
    }
    summaries = store.load_team_summaries()
    if summaries and isinstance(summaries, dict):
        teams = summaries.get("teams") or summaries
        if isinstance(teams, dict) and abbr_n in teams:
            payload["summary"] = teams[abbr_n]
        elif isinstance(teams, list):
            match = next((t for t in teams if t.get("abbr") == abbr_n), None)
            if match:
                payload["summary"] = match
        else:
            payload["summary"] = unavailable("team summary entry not found in team_summaries.json")
    else:
        payload["summary"] = unavailable("team_summaries.json not found — run precompute")

    df = store.team_lineups(abbr_n)
    if df is not None:
        payload["games"] = int(len(df))
        payload["unique_orders"] = int(df["order_id"].nunique()) if len(df) else 0
        payload["unique_personnel"] = int(df["personnel_id"].nunique()) if len(df) else 0

    team_rows = _teams_from_summaries(summaries if isinstance(summaries, dict) else None)
    _overlay_live_counts(team_rows, store.load_lineups())
    payload["ranks"] = rank_team_metrics(team_rows, abbr_n)
    return payload


@router.get("/teams/{abbr}/lineups")
def team_lineups(
    abbr: str,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    store = get_store()
    df = store.team_lineups(abbr_n)
    if df is None:
        return unavailable(f"starting_lineups_{store.season}.parquet not found")
    df = df.sort_values("game_date", ascending=False)
    total = int(len(df))
    page = df.iloc[offset : offset + limit]
    evals = store.load_lineup_evaluations()
    rows = []
    for rec in page.to_dict(orient="records"):
        item = {
            "game_pk": int(rec["game_pk"]),
            "game_date": rec.get("game_date"),
            "opponent": normalize_abbrev(rec.get("opponent")) or rec.get("opponent"),
            "is_home": bool(rec.get("is_home")),
            "venue": rec.get("venue"),
            "order_id": rec.get("order_id"),
            "personnel_id": rec.get("personnel_id"),
            "opp_sp_hand": rec.get("opp_sp_hand"),
            "opp_sp_name": rec.get("opp_sp_name"),
            "runs_scored": rec.get("runs_scored"),
            "runs_allowed": rec.get("runs_allowed"),
            "result": rec.get("result"),
            "batting_order": [int(rec[f"slot{i}"]) for i in range(1, 10)],
            "batter_names": (rec.get("batter_names") or "").split("|") if rec.get("batter_names") else None,
        }
        if evals is not None and not evals.empty:
            em = evals[(evals["game_pk"] == rec["game_pk"]) & (evals["team"] == abbr_n)]
            if len(em) == 0:
                em = evals[
                    (evals["game_pk"] == rec["game_pk"])
                    & (evals["team"].map(lambda x: normalize_abbrev(x) or x) == abbr_n)
                ]
            if len(em):
                ev = em.iloc[0].to_dict()
                batting_ids = item["batting_order"]
                name_by_id = {}
                names_list = item.get("batter_names") or []
                for pid, name in zip(batting_ids, names_list):
                    name_by_id[int(pid)] = name

                def _parse_ids(val):
                    if val is None:
                        return None
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except (TypeError, json.JSONDecodeError, ValueError):
                            return None
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    if isinstance(val, (list, tuple)):
                        return [int(x) for x in val]
                    return None

                for key in ("best_order_ids", "worst_order_ids", "actual_order_ids"):
                    parsed = _parse_ids(ev.get(key))
                    if parsed:
                        ev[key] = parsed
                        ev[key.replace("_ids", "_names")] = [
                            name_by_id.get(i, str(i)) for i in parsed
                        ]
                item["evaluation"] = ev
        rows.append(item)
    return {"available": True, "team": abbr_n, "total": total, "offset": offset, "limit": limit, "lineups": rows}


@router.get("/teams/{abbr}/heatmap")
def team_heatmap(abbr: str):
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    return get_store().slot_heatmap(abbr_n)


@router.get("/teams/{abbr}/roster")
def team_roster(
    abbr: str,
    mode: str = Query("season", pattern="^(season|current|as_of)$"),
    as_of: str | None = Query(None, description="YYYY-MM-DD for as_of mode"),
    include_unavailable: bool = Query(
        False,
        description=(
            "If true, include players who belong to the club but are not "
            "MLB-lineup available (IL, optioned, etc.). Does not resurrect "
            "traded/released players. Ignored for season pool (always includes leavers)."
        ),
    ),
):
    """Roster pool for the explorer / team tools."""
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    return get_store().team_roster(
        abbr_n,
        mode=mode,
        as_of=as_of,
        include_unavailable=include_unavailable,
    )


@router.get("/teams/{abbr}/timeline")
def team_timeline(abbr: str):
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    return get_store().lineup_timeline(abbr_n)


@router.get("/teams/{abbr}/most-used")
def team_most_used(
    abbr: str,
    top_n: int = Query(
        0,
        ge=0,
        le=5000,
        description="Max unique orders to return. 0 = all unique orders.",
    ),
    rank_by: str = Query("effectiveness", pattern="^(effectiveness|usage)$"),
):
    abbr_n = normalize_abbrev(abbr)
    if not abbr_n or abbr_n not in CANONICAL_ABBREVS:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbr}")
    return get_store().most_used_lineups(abbr_n, top_n=top_n, rank_by=rank_by)
