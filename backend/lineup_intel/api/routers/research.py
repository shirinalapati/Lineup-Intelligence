"""Research methodology, findings, and model cards."""

from __future__ import annotations

from fastapi import APIRouter

from ...db.store import get_store, unavailable
from ...config import settings

router = APIRouter(tags=["research"])


# Static methodology payload describing implemented modules (not invented results).
_METHODOLOGY = {
    "title": "MLB Lineup Intelligence — Methodology",
    "season": settings.target_season,
    "sections": [
        {
            "id": "research_question",
            "title": "Research Question",
            "body": (
                "How much does batting order matter among the same nine hitters, "
                "and do adjacent-hitter interactions improve predictive models beyond "
                "talent, slot, and base-out state?"
            ),
        },
        {
            "id": "dataset",
            "title": "Lineup Dataset",
            "body": (
                "Starting batting orders are extracted from MLB GUMBO boxscores "
                "(read-only). Each lineup stores ordered and unordered personnel identities, "
                "opponent starter context when available, and game results."
            ),
        },
        {
            "id": "pa_outcomes",
            "title": "Player Outcome Modeling",
            "body": (
                "Plate-appearance outcome probabilities use empirical-Bayes shrinkage "
                "toward league rates within neutral / vs-RHP / vs-LHP contexts. "
                "Outcome classes: K, BB_HBP, 1B, 2B, 3B, HR, OUT_IP."
            ),
        },
        {
            "id": "markov",
            "title": "Run Expectancy / Markov Model",
            "body": (
                "A deterministic Markov model combines each hitter's outcome probabilities "
                "with historical base-out transitions to calculate the expected number of "
                "runs a batting order would score over nine innings. The same process "
                "estimates how often each lineup slot comes to the plate and the base/out "
                "situations it is likely to encounter. Product pages report this as "
                "projected runs per game (R/G)."
            ),
        },
        {
            "id": "optimizer",
            "title": "Same-Nine Optimization",
            "body": (
                "For a fixed set of nine batters, the optimizer searches the 9! batting-order "
                "space (exhaustive Numba path when available) and reports actual vs best vs "
                "worst expected runs, percentile, gap, and near-optimal orders within an "
                "operational equivalence band."
            ),
        },
        {
            "id": "simulation",
            "title": "Monte Carlo Simulation",
            "body": (
                "Stochastic game simulations sample PA outcomes and transitions to produce "
                "run distributions for a given order."
            ),
        },
        {
            "id": "archetypes",
            "title": "Offensive Archetypes",
            "body": (
                "Hitters are clustered into offensive archetypes from season features. "
                "Assignments and labels are written to model artifacts when research modules run."
            ),
        },
        {
            "id": "interactions",
            "title": "Pair Interaction Modeling",
            "body": (
                "Adjacent-hitter residual models estimate player-pair and archetype-pair "
                "effects with empirical-Bayes shrinkage and reliability tiers. "
                "Incremental predictive value is assessed with temporal train/validation splits. "
                "Validated conclusion: adjacent hitter interaction effects were generally "
                "small and did not provide reliable incremental out-of-sample predictive "
                "value beyond hitter talent and game-state controls, so they are exploratory "
                "and are not used by the optimizer. Player-pair residual associations remain "
                "available on Research as an exploratory table. Model-comparison metrics and "
                "profile-group residual matrices stay in methodology for transparency only."
            ),
        },
        {
            "id": "uncertainty",
            "title": "Uncertainty & Limitations",
            "body": (
                "Small samples shrink toward zero. Interaction findings are only "
                "returned when research artifacts exist — the API does not invent metrics."
            ),
        },
        {
            "id": "player_slot_intelligence",
            "title": "Player Slot Intelligence",
            "body": (
                "Observed batting-slot splits are descriptive only — confounded by "
                "personnel, pitchers, platoon usage, and sample size. Modeled slot fit "
                "holds the other eight hitters’ relative order fixed, inserts the focal "
                "player into each slot 1–9, and evaluates team expected runs with the "
                "Markov engine. Batting order affects both PA allocation and base-state "
                "exposure. Slot differences within 0.01 runs/game are reported as "
                "operationally equivalent. Every league rank includes a qualifying "
                "population denominator (overall ≥100 PA; slot splits ≥30 PA)."
            ),
        },
        {
            "id": "roster_history",
            "title": "Roster History",
            "body": (
                "Explorer roster pools separate organizational membership from "
                "MLB-lineup availability. Membership comes from opening-day 40-man "
                "rosters plus MLB Stats API transactions (trades, releases, claims, "
                "signings). IL, options, and recalls change availability only. "
                "Season pool is every 2026 starter for the club (leavers remain). "
                "Current and as-of pools reconstruct who belonged on that date. "
                "The unavailable toggle never resurrects traded/released players. "
                "As-of analysis uses historical roster reconstruction; hitter "
                "probabilities remain 2024–2025 trained rates, not 2026 stats "
                "through the selected date."
            ),
        },
        {
            "id": "reproducibility",
            "title": "Reproducibility",
            "body": (
                "Pipeline: extract lineups → refresh roster history → build PA "
                "probabilities → archetypes → interactions → precompute lineup "
                "evaluations → player slot intelligence → findings. Roster refresh "
                "does not retrain lineup models."
            ),
        },
    ],
}


@router.get("/research/methodology")
def methodology():
    store = get_store()
    art = store.load_methodology()
    if art is not None:
        if isinstance(art, dict) and "available" in art:
            return art
        return {"available": True, "source": "artifact", "methodology": art}
    return {"available": True, "source": "builtin", "methodology": _METHODOLOGY}


@router.get("/research/findings")
def findings():
    store = get_store()
    data = store.load_findings()
    if data is None:
        return unavailable(
            "findings artifact not found under data/artifacts/ — run research findings step"
        )
    if isinstance(data, dict) and "available" in data:
        return data
    return {"available": True, "findings": data}


@router.get("/research/model-cards")
def model_cards():
    store = get_store()
    data = store.load_model_cards()
    if data is None:
        # Generate artifact on demand so the UI never shows a missing-artifact fallback.
        try:
            from ...research.model_cards import build_model_cards

            data = build_model_cards()
        except Exception as exc:  # noqa: BLE001
            return unavailable(f"model_cards.json artifact not found and could not generate: {exc}")
    if isinstance(data, dict) and "available" in data:
        return data
    return {"available": True, "source": "artifact", "cards": data}


@router.get("/research/markov-validation")
def markov_validation():
    store = get_store()
    path = settings.artifacts_dir / "research" / "markov_validation.json"
    if path.exists():
        import json

        data = json.loads(path.read_text())
        if isinstance(data, dict) and "available" in data:
            return data
        return {"available": True, "validation": data}
    return unavailable("markov_validation.json not found — run markov_validation step")


@router.get("/research/data-quality")
def data_quality():
    path = settings.artifacts_dir / "data_quality_report.json"
    if path.exists():
        import json

        data = json.loads(path.read_text())
        if isinstance(data, dict) and "available" in data:
            return data
        return {"available": True, "report": data}
    return unavailable("data_quality_report.json not found — run data_quality step")


@router.get("/research/player-slot-intelligence")
def player_slot_intelligence():
    path = settings.artifacts_dir / "research" / "player_slot_intelligence.json"
    if path.exists():
        import json

        data = json.loads(path.read_text())
        if isinstance(data, dict) and "available" in data:
            return data
        return {"available": True, "findings": data}
    return unavailable(
        "player_slot_intelligence.json not found — run player_slot_intelligence step"
    )
