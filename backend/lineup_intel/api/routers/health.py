"""Health and data-validation endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...db.store import get_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "MLB Lineup Intelligence"}


@router.get("/health/data")
def data_health():
    return get_store().data_health_report()
