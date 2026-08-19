"""Live optimize / evaluate / simulate endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from ...config import settings
from ...db.store import get_store, unavailable
from ...engine.explain import explain_order_delta
from ...engine.markov import LineupEngine, lineup_breakdown_narrative
from ...engine.optimizer import optimize_lineup
from ...engine.pa_probs import PAProbabilityStore
from ...engine.simulate import simulate_lineup
from ...identity import validate_lineup
from ..schemas import EvaluateRequest, OptimizeRequest, SimulateRequest

router = APIRouter(tags=["optimizer"])

_PA: PAProbabilityStore | None = None
_ENGINE: LineupEngine | None = None


def _pa_store() -> PAProbabilityStore | dict:
    global _PA
    models = settings.models_dir
    needed = [models / f"pa_probs_{c}.parquet" for c in ("neutral", "vs_R", "vs_L")]
    missing = [str(p.name) for p in needed if not p.exists()]
    if missing:
        return unavailable(f"PA probability tables missing: {', '.join(missing)}")
    if _PA is None:
        _PA = PAProbabilityStore(models)
    return _PA


def _engine() -> LineupEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LineupEngine()
    return _ENGINE


def _resolve_context(context: str) -> str:
    c = (context or "neutral").strip()
    if c in ("neutral", "vs_R", "vs_L"):
        return c
    if c.upper() in ("R", "RHP"):
        return "vs_R"
    if c.upper() in ("L", "LHP"):
        return "vs_L"
    return "neutral"


@router.post("/optimize")
def optimize(req: OptimizeRequest):
    pa = _pa_store()
    if isinstance(pa, dict):
        return pa
    try:
        player_ids = validate_lineup(req.player_ids)
    except ValueError as e:
        return unavailable(str(e))
    order = req.order
    if order is not None:
        try:
            order = validate_lineup(order)
        except ValueError as e:
            return unavailable(str(e))
        if sorted(order) != sorted(player_ids):
            return unavailable("order must be a permutation of player_ids")
    else:
        order = list(player_ids)

    ctx = _resolve_context(req.context)
    # optimize_lineup expects probs aligned to player_ids list order
    probs = pa.probs_matrix(player_ids, ctx)
    result = optimize_lineup(
        player_ids=player_ids,
        probs=probs,
        actual_order_ids=order,
        equivalence_eps=settings.equivalence_eps,
    )
    names = get_store().player_name_map()
    probs_by_id = {pid: pa.probs_for(pid, ctx) for pid in player_ids}
    actual_order_ids = [player_ids[i] for i in result.actual_order]
    best_order_ids = [player_ids[i] for i in result.best_order]
    explanations = explain_order_delta(
        player_ids=player_ids,
        player_names=names,
        actual_order=actual_order_ids,
        best_order=best_order_ids,
        probs_by_id=probs_by_id,
        actual_runs=result.actual_runs,
        best_runs=result.best_runs,
    )
    payload = asdict(result)
    payload["actual_order_ids"] = actual_order_ids
    payload["best_order_ids"] = best_order_ids
    payload["worst_order_ids"] = [player_ids[i] for i in result.worst_order]
    payload["context"] = ctx
    payload["explanations"] = explanations
    payload["player_names"] = {str(pid): names.get(pid, str(pid)) for pid in player_ids}
    return {"available": True, "result": payload}


@router.post("/evaluate")
def evaluate(req: EvaluateRequest):
    pa = _pa_store()
    if isinstance(pa, dict):
        return pa
    try:
        player_ids = validate_lineup(req.player_ids)
    except ValueError as e:
        return unavailable(str(e))
    ctx = _resolve_context(req.context)
    probs = pa.probs_matrix(player_ids, ctx)
    ev = _engine().evaluate(probs, with_flow=True)
    summary, observations = lineup_breakdown_narrative(ev)
    return {
        "available": True,
        "context": ctx,
        "player_ids": player_ids,
        "expected_runs_9": float(ev.expected_runs_9),
        "expected_runs_per_inning": float(ev.expected_runs_per_inning),
        "expected_pa_by_slot": [float(x) for x in ev.expected_pa_by_slot],
        "slot_start_share": [float(x) for x in ev.slot_start_share],
        "lineup_flow": ev.lineup_flow or [],
        "summary_sentence": summary,
        "observations": observations,
    }


@router.post("/simulate")
def simulate(req: SimulateRequest):
    pa = _pa_store()
    if isinstance(pa, dict):
        return pa
    try:
        player_ids = validate_lineup(req.player_ids)
    except ValueError as e:
        return unavailable(str(e))
    ctx = _resolve_context(req.context)
    probs = pa.probs_matrix(player_ids, ctx)
    det = float(_engine().expected_runs(probs))
    sim = simulate_lineup(
        probs,
        n_games=req.n_games,
        seed=req.seed,
        deterministic_expected=det,
    )
    return {
        "available": True,
        "context": ctx,
        "player_ids": player_ids,
        "result": asdict(sim),
    }
