"""Shared request/response helpers for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..db.store import unavailable


class OptimizeRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=9, max_length=9)
    order: list[int] | None = None
    context: str = "neutral"


class EvaluateRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=9, max_length=9)
    context: str = "neutral"


class SimulateRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=9, max_length=9)
    context: str = "neutral"
    n_games: int = Field(1000, ge=100, le=20000)
    seed: int = 42


def missing_artifact(path_desc: str) -> dict[str, Any]:
    return unavailable(f"Artifact not available: {path_desc}")
