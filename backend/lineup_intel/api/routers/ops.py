"""Ops endpoints for scheduled daily refresh after deploy."""

from __future__ import annotations

import os
import threading
from datetime import date
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from ...db.store import get_store

router = APIRouter(tags=["ops"])

_lock = threading.Lock()
_last: dict[str, Any] | None = None


def _token_ok(provided: str | None) -> bool:
    expected = (os.environ.get("LI_REFRESH_TOKEN") or "").strip()
    if not expected:
        return False
    return bool(provided) and provided.strip() == expected


def run_daily_refresh(*, skip_precompute: bool = False) -> dict[str, Any]:
    """Refresh season lineups and optionally precompute evaluations."""
    global _last
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Daily refresh already running")
    try:
        from ...etl.refresh_lineups import refresh_lineups

        through = date.today().isoformat()
        summary = refresh_lineups(through=through)
        precompute_summary = None
        if not skip_precompute:
            from ...etl.precompute import precompute

            precompute_summary = precompute(workers=1)
        get_store().clear_cache()
        payload = {
            "available": True,
            "ran_on": through,
            "refresh": summary,
            "precompute": precompute_summary,
        }
        _last = payload
        return payload
    finally:
        _lock.release()


@router.get("/ops/daily-refresh")
def daily_refresh_status():
    return {
        "available": True,
        "last": _last,
        "token_configured": bool((os.environ.get("LI_REFRESH_TOKEN") or "").strip()),
        "scheduler_enabled": (os.environ.get("LI_DAILY_REFRESH") or "").strip().lower()
        in {"1", "true", "yes"},
    }


@router.post("/ops/daily-refresh")
def daily_refresh(
    x_refresh_token: str | None = Header(default=None, alias="X-Refresh-Token"),
    skip_precompute: bool = False,
):
    """Pull new completed games into season tables and precompute gaps.

    Requires env ``LI_REFRESH_TOKEN`` and matching ``X-Refresh-Token`` header.
    """
    if not _token_ok(x_refresh_token):
        raise HTTPException(status_code=401, detail="Invalid or missing refresh token")
    try:
        return run_daily_refresh(skip_precompute=skip_precompute)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
