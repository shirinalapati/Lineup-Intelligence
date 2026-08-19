"""Player list, search, and profile endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ...config import settings
from ...db.store import SLOT_COLS, _jsonable, get_store, unavailable
from ...research.archetypes import WOBA_WEIGHTS
from ...teams import normalize_abbrev
from ...research.player_slot_intelligence import (
    load_player_lineup_profile,
    merge_platoon_woba_metrics,
)

router = APIRouter(tags=["players"])

_OUTCOME_COLS = ["K", "BB_HBP", "1B", "2B", "3B", "HR", "OUT_IP"]
_SLOT_INTEL_BY_ID: dict[int, dict] | None = None


def _slot_intel_by_id() -> dict[int, dict]:
    """Cached primary actual / best modeled slots from the lineup-intel index."""
    global _SLOT_INTEL_BY_ID
    if _SLOT_INTEL_BY_ID is not None:
        return _SLOT_INTEL_BY_ID
    path = settings.artifacts_dir / "player_lineup_intelligence_index.json"
    out: dict[int, dict] = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("players") if isinstance(raw, dict) else raw
        for rec in rows or []:
            if not isinstance(rec, dict):
                continue
            try:
                pid = int(rec["player_id"])
            except (TypeError, ValueError, KeyError):
                continue
            item: dict = {}
            primary = rec.get("primary_slot")
            best = rec.get("best_slot")
            if primary is not None:
                try:
                    item["primary_actual_slot"] = int(primary)
                except (TypeError, ValueError):
                    pass
            if best is not None:
                try:
                    item["best_modeled_slot"] = int(best)
                except (TypeError, ValueError):
                    pass
            if item:
                out[pid] = item
    _SLOT_INTEL_BY_ID = out
    return out


def _talent_rates(player_id: int) -> dict | None:
    """Derive modeled rates from empirical-Bayes PA outcome probabilities."""
    path = Path(settings.models_dir) / "pa_probs_neutral.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    m = df[df["player_id"] == int(player_id)]
    if m.empty:
        return None
    row = m.iloc[0]
    rates = {c: float(row[c]) for c in _OUTCOME_COLS if c in row.index}
    if not rates:
        return None
    woba = sum(WOBA_WEIGHTS.get(c, 0.0) * rates.get(c, 0.0) for c in WOBA_WEIGHTS)
    # Per-PA ISO proxy (extra bases beyond singles), matching archetype feature construction.
    iso = (
        rates.get("2B", 0.0)
        + 2.0 * rates.get("3B", 0.0)
        + 3.0 * rates.get("HR", 0.0)
    )
    out = {
        "woba": float(woba),
        "est_woba": float(woba),
        "k_rate": float(rates.get("K", 0.0)),
        "bb_rate": float(rates.get("BB_HBP", 0.0)),
        "iso": float(iso),
        "n_pa": int(row["n_pa"]) if "n_pa" in row.index and pd.notna(row["n_pa"]) else None,
        "pa_outcome_rates": rates,
        "talent_source": "pa_probs_neutral",
    }
    # Platoon splits when available
    for ctx, fname in (("vs_R", "pa_probs_vs_R.parquet"), ("vs_L", "pa_probs_vs_L.parquet")):
        p = Path(settings.models_dir) / fname
        if not p.exists():
            continue
        sdf = pd.read_parquet(p)
        sm = sdf[sdf["player_id"] == int(player_id)]
        if sm.empty:
            continue
        srow = sm.iloc[0]
        srates = {c: float(srow[c]) for c in _OUTCOME_COLS if c in srow.index}
        swoba = sum(WOBA_WEIGHTS.get(c, 0.0) * srates.get(c, 0.0) for c in WOBA_WEIGHTS)
        siso = (
            srates.get("2B", 0.0)
            + 2.0 * srates.get("3B", 0.0)
            + 3.0 * srates.get("HR", 0.0)
        )
        out[ctx] = {
            "woba": float(swoba),
            "k_rate": float(srates.get("K", 0.0)),
            "bb_rate": float(srates.get("BB_HBP", 0.0)),
            "iso": float(siso),
            "n_pa": int(srow["n_pa"]) if "n_pa" in srow.index and pd.notna(srow["n_pa"]) else None,
        }
    return out


def _merge_profile(base: dict | None, talent: dict | None) -> dict | None:
    if base is None and talent is None:
        return None
    out = dict(base or {})
    if talent:
        out.update(talent)
    return out


@router.get("/players")
def list_players(
    q: str | None = None,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    include_pitchers: bool = Query(False),
):
    store = get_store()
    players = store.load_players()
    if players is None:
        return unavailable(f"players_{store.season}.parquet not found")
    df = players
    if not include_pitchers and "position" in df.columns:
        pos = df["position"].fillna("").astype(str).str.upper()
        df = df[pos != "P"]
    if q:
        ql = q.strip().lower()
        df = df[df["name"].astype(str).str.lower().str.contains(ql, na=False)]

    profiles = store.load_player_profiles()
    profile_by_id: dict[int, dict] = {}
    if isinstance(profiles, list):
        profile_by_id = {int(p["player_id"]): p for p in profiles if "player_id" in p}
    elif isinstance(profiles, dict) and "players" in profiles:
        profile_by_id = {int(p["player_id"]): p for p in profiles["players"] if "player_id" in p}
    elif isinstance(profiles, dict):
        for k, v in profiles.items():
            if k in ("available", "reason"):
                continue
            try:
                profile_by_id[int(k)] = v if isinstance(v, dict) else {"player_id": int(k)}
            except (TypeError, ValueError):
                continue

    slot_intel = _slot_intel_by_id()
    rows = []
    for rec in df.to_dict(orient="records"):
        pid = int(rec["player_id"])
        item: dict = {
            "player_id": pid,
            "name": rec.get("name"),
            "bat_side": rec.get("bat_side"),
            "position": rec.get("position"),
        }
        prof = profile_by_id.get(pid)
        if prof:
            item["profile"] = prof
            teams = prof.get("teams") or []
            if isinstance(teams, list) and teams:
                item["team"] = teams[0]
                item["teams"] = teams
            if prof.get("games") is not None:
                item["games"] = prof.get("games")
            if prof.get("archetype_label") is not None:
                item["archetype"] = prof.get("archetype_label")
            elif prof.get("archetype") is not None:
                item["archetype"] = prof.get("archetype")
        intel = slot_intel.get(pid)
        if intel:
            item.update(intel)
        elif prof and prof.get("primary_slot") is not None:
            try:
                item["primary_actual_slot"] = int(prof["primary_slot"])
            except (TypeError, ValueError):
                pass
        rows.append(item)

    if q:
        rows.sort(key=lambda r: str(r.get("name") or "").lower())
    else:
        rows.sort(
            key=lambda r: (
                -(int(r.get("games") or 0)),
                str(r.get("name") or "").lower(),
            )
        )

    total = len(rows)
    page = rows[offset : offset + limit]
    return _jsonable({
        "available": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "players": page,
    })


@router.get("/players/{player_id}")
def player_profile(player_id: int):
    store = get_store()
    players = store.load_players()
    if players is None:
        return unavailable(f"players_{store.season}.parquet not found")
    m = players[players["player_id"] == int(player_id)]
    if m.empty:
        raise HTTPException(status_code=404, detail="Player not found")
    base = m.iloc[0].to_dict()
    payload: dict = {
        "available": True,
        "player_id": int(player_id),
        "name": base.get("name"),
        "bat_side": base.get("bat_side"),
        "position": base.get("position"),
    }

    profiles = store.load_player_profiles()
    profile = None
    if isinstance(profiles, list):
        profile = next((p for p in profiles if int(p.get("player_id", -1)) == int(player_id)), None)
    elif isinstance(profiles, dict):
        if "players" in profiles:
            profile = next(
                (p for p in profiles["players"] if int(p.get("player_id", -1)) == int(player_id)),
                None,
            )
        elif str(player_id) in profiles:
            profile = profiles[str(player_id)]
        elif player_id in profiles:
            profile = profiles[player_id]
    if profile is not None:
        payload["profile"] = profile
    else:
        payload["profile"] = unavailable("player_profiles artifact not found — run precompute")

    talent = _talent_rates(int(player_id))
    if isinstance(payload.get("profile"), dict):
        payload["profile"] = _merge_profile(payload["profile"], talent)
    elif talent:
        payload["profile"] = talent

    # Appearances
    lineups = store.load_lineups()
    if lineups is not None:
        mask = False
        for c in SLOT_COLS:
            mask = mask | (lineups[c] == int(player_id))
        mine = lineups.loc[mask]
        teams = sorted({normalize_abbrev(t) or str(t) for t in mine["team"].tolist()}) if len(mine) else []
        slot_counts = {str(i): 0 for i in range(1, 10)}
        for _, row in mine.iterrows():
            for slot, col in enumerate(SLOT_COLS, start=1):
                if int(row[col]) == int(player_id):
                    slot_counts[str(slot)] += 1
                    break
        payload["appearances"] = {
            "games": int(len(mine)),
            "teams": teams,
            "slot_counts": slot_counts,
        }
        if isinstance(payload.get("profile"), dict):
            payload["profile"]["slot_counts"] = slot_counts
            payload["profile"]["games"] = int(len(mine))
            payload["profile"]["teams"] = teams

    # Attach precomputed lineup intelligence when present
    li = load_player_lineup_profile(int(player_id))
    if li:
        li = dict(li)
        li["season_metrics"] = merge_platoon_woba_metrics(
            li.get("season_metrics") if isinstance(li.get("season_metrics"), dict) else {},
            int(player_id),
        )
        payload["lineup_intelligence"] = li

    payload["team_history"] = store.player_team_history(int(player_id))

    return _jsonable(payload)


@router.get("/players/{player_id}/lineup-profile")
def player_lineup_profile(player_id: int):
    """Full lineup-intelligence profile (ranks, slot fit, splits, neighbors)."""
    store = get_store()
    players = store.load_players()
    if players is None:
        return unavailable(f"players_{store.season}.parquet not found")
    m = players[players["player_id"] == int(player_id)]
    if m.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    li = load_player_lineup_profile(int(player_id))
    if li is None:
        return unavailable(
            "player lineup-profile artifact missing — run "
            "`python -m lineup_intel.research.player_slot_intelligence`"
        )
    base = m.iloc[0].to_dict()
    li = dict(li)
    li["bat_side"] = base.get("bat_side")
    li["position"] = base.get("position")
    li["name"] = li.get("name") or base.get("name")
    li["season_metrics"] = merge_platoon_woba_metrics(
        li.get("season_metrics") if isinstance(li.get("season_metrics"), dict) else {},
        int(player_id),
    )
    return _jsonable(li)
