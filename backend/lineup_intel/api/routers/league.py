"""League overview endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...db.store import get_store, unavailable
from ...teams import CANONICAL_ABBREVS, TEAMS

router = APIRouter(tags=["league"])


@router.get("/league/overview")
def league_overview():
    store = get_store()
    overview = store.load_league_overview()
    if overview is not None:
        if isinstance(overview, dict) and "available" in overview:
            return overview
        return {"available": True, **overview}

    # Soft fallback: team list without invented metrics
    summaries = store.load_team_summaries()
    if summaries is not None:
        teams = summaries.get("teams") if isinstance(summaries, dict) else None
        if teams is not None:
            return {
                "available": True,
                "source": "team_summaries.json",
                "n_teams": len(teams),
                "teams": teams,
            }

    lineups = store.load_lineups()
    if lineups is None:
        return unavailable("league_overview.json and starting lineups are missing")

    team_rows = []
    for abbr in CANONICAL_ABBREVS:
        meta = TEAMS.get(abbr, {})
        n = int((lineups["team"] == abbr).sum())
        team_rows.append({
            "abbr": abbr,
            "name": meta.get("name"),
            "division": meta.get("division"),
            "games": n,
            "metrics_available": False,
        })
    return {
        "available": True,
        "source": "starting_lineups_only",
        "reason": "Precomputed league_overview.json not found; returning game counts only.",
        "n_teams": len(team_rows),
        "teams": team_rows,
        "metrics": unavailable("league_overview.json not found — run precompute to populate summary metrics"),
    }
