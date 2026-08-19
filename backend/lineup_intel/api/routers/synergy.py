"""Synergy / interaction research endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

import pandas as pd

from ...db.store import _jsonable, get_store, unavailable

router = APIRouter(tags=["synergy"])


def _blank_label(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    text = str(v).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _player_archetype_labels(store) -> dict[int, str]:
    """Latest-season offensive profile label per player."""
    arch = store.load_research_artifact("archetype_assignments.parquet")
    if arch is None or not hasattr(arch, "columns"):
        return {}
    if "player_id" not in arch.columns or "archetype_label" not in arch.columns:
        return {}
    work = arch.copy()
    sort_cols = [c for c in ("season", "n_pa") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
    latest = work.groupby("player_id", dropna=False).tail(1)
    out: dict[int, str] = {}
    for pid, label in zip(latest["player_id"], latest["archetype_label"]):
        if _blank_label(label):
            continue
        try:
            out[int(pid)] = str(label).strip()
        except (TypeError, ValueError):
            continue
    return out


def _load_or_missing(store, *names: str):
    for name in names:
        data = store.load_research_artifact(name)
        if data is not None:
            if hasattr(data, "to_dict"):
                return {
                    "available": True,
                    "source": name,
                    "data": _jsonable(data.to_dict(orient="records")),
                }
            if isinstance(data, dict):
                if "available" in data:
                    return _jsonable(data)
                return {"available": True, "source": name, "data": _jsonable(data)}
            return {"available": True, "source": name, "data": _jsonable(data)}
    return unavailable(
        "Research synergy artifacts not found under data/artifacts/research/ "
        f"(tried: {', '.join(names)})"
    )


def _pair_player_ids(row: dict) -> tuple[int | None, int | None]:
    def _as_int(v) -> int | None:
        try:
            if v is None or v == "":
                return None
            return int(v)
        except (TypeError, ValueError):
            return None

    prev_id = _as_int(row.get("player_a", row.get("prev_batter_id")))
    bat_id = _as_int(row.get("player_b", row.get("batter_id")))
    return prev_id, bat_id


_TIER_RANK = {"unknown": 0, "limited": 0, "moderate": 1, "strong": 2}


def _reliability_tier(row: dict) -> str:
    raw = str(row.get("reliability_tier") or row.get("reliability") or "").lower()
    if "strong" in raw:
        return "strong"
    if "moderate" in raw:
        return "moderate"
    if "limited" in raw:
        return "limited"
    return "unknown"


@router.get("/synergy/pairs")
def pair_explorer(
    player_a: int | None = None,
    player_b: int | None = None,
    q: str | None = Query(
        None,
        description="Case-insensitive search on either player's name or MLBAM id",
    ),
    min_n: int = Query(0, ge=0),
    min_tier: str = Query(
        "strong",
        description="Minimum reliability: strong | moderate | all",
    ),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    store = get_store()
    payload = _load_or_missing(
        store,
        "player_pair_effects.parquet",
        "player_pair_effects",
        "pair_effects",
        "player_pairs",
        "adjacency_player_pairs",
    )
    if not payload.get("available"):
        return payload
    data = payload.get("data")
    rows = data if isinstance(data, list) else (data.get("pairs") if isinstance(data, dict) else None)
    if rows is None:
        return payload
    filtered = rows
    if player_a is not None:
        filtered = [
            r for r in filtered
            if int(r.get("player_a", r.get("prev_batter_id", -1)) or -1) == player_a
            or int(r.get("player_b", r.get("batter_id", -1)) or -1) == player_a
        ]
    if player_b is not None:
        filtered = [
            r for r in filtered
            if int(r.get("player_a", r.get("prev_batter_id", -1)) or -1) == player_b
            or int(r.get("player_b", r.get("batter_id", -1)) or -1) == player_b
        ]
    if min_n:
        filtered = [r for r in filtered if int(r.get("n", r.get("n_pa", 0)) or 0) >= min_n]
    tier_req = (min_tier or "strong").strip().lower()
    if tier_req not in {"strong", "moderate", "all"}:
        tier_req = "strong"
    if tier_req != "all":
        min_rank = _TIER_RANK["strong"] if tier_req == "strong" else _TIER_RANK["moderate"]
        filtered = [
            r for r in filtered
            if _TIER_RANK.get(_reliability_tier(r), 0) >= min_rank
        ]

    names = store.player_name_map()
    needle = (q or "").strip().lower()
    if needle:
        matching_ids: set[int] = set()
        for pid, name in names.items():
            try:
                pid_i = int(pid)
            except (TypeError, ValueError):
                continue
            if needle in str(name).lower() or needle == str(pid_i):
                matching_ids.add(pid_i)
        filtered = [
            r
            for r in filtered
            if any(pid is not None and pid in matching_ids for pid in _pair_player_ids(r))
        ]

    # Rank by |effect| * reliability so strongest associations surface first
    def sort_key(r: dict) -> tuple:
        try:
            effect = abs(float(r.get("effect") or r.get("shrunk_effect") or 0))
        except (TypeError, ValueError):
            effect = 0.0
        try:
            n = int(r.get("n") or r.get("n_pa") or 0)
        except (TypeError, ValueError):
            n = 0
        return (effect, n)

    filtered = sorted(filtered, key=sort_key, reverse=True)
    page = filtered[offset : offset + limit]
    arch_labels = _player_archetype_labels(store)
    enriched = []
    for r in page:
        row = dict(r)
        prev_id = row.get("prev_batter_id", row.get("player_a"))
        bat_id = row.get("batter_id", row.get("player_b"))
        try:
            prev_id_i = int(prev_id) if prev_id is not None else None
        except (TypeError, ValueError):
            prev_id_i = None
        try:
            bat_id_i = int(bat_id) if bat_id is not None else None
        except (TypeError, ValueError):
            bat_id_i = None
        if prev_id_i is not None:
            row["player_a"] = prev_id_i
            row["player_a_name"] = names.get(prev_id_i, str(prev_id_i))
            row["prev_name"] = row["player_a_name"]
            if _blank_label(row.get("prev_arch_label")):
                row["prev_arch_label"] = arch_labels.get(prev_id_i)
        if bat_id_i is not None:
            row["player_b"] = bat_id_i
            row["player_b_name"] = names.get(bat_id_i, str(bat_id_i))
            row["name"] = row["player_b_name"]
            if _blank_label(row.get("batter_arch_label")):
                row["batter_arch_label"] = arch_labels.get(bat_id_i)
        row["reliability"] = row.get("reliability_tier") or row.get("reliability")
        enriched.append(row)
    return _jsonable({
        "available": True,
        "source": payload.get("source"),
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "q": needle or None,
        "min_n": min_n,
        "min_tier": tier_req,
        "pairs": enriched,
    })


@router.get("/synergy/archetypes")
def archetype_matrix():
    store = get_store()
    payload = _load_or_missing(
        store,
        "archetype_pair_effects.parquet",
        "archetype_pair_effects",
        "archetype_pair_matrix",
        "archetype_pairs",
        "archetype_interactions",
    )
    archetypes = store.load_research_artifact("archetypes")
    if not payload.get("available"):
        if archetypes is not None:
            return _jsonable({
                "available": True,
                "matrix": unavailable("archetype pair matrix artifact missing"),
                "archetypes": archetypes if not hasattr(archetypes, "to_dict") else archetypes.to_dict(orient="records"),
            })
        return payload
    out = dict(payload)
    # Alias for frontend matrix consumers
    if isinstance(out.get("data"), list):
        out["matrix"] = out["data"]
        out["pairs"] = out["data"]
    if archetypes is not None:
        out["archetypes"] = archetypes if not hasattr(archetypes, "to_dict") else archetypes.to_dict(orient="records")
    return _jsonable(out)


@router.get("/synergy/incremental")
def incremental_value():
    store = get_store()
    return _load_or_missing(
        store,
        "incremental_predictive_value.json",
        "incremental_predictive_value",
        "incremental_value",
        "model_comparison",
        "does_synergy_matter",
        "interaction_summary.json",
    )
