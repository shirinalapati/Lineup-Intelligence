"""Lineup identity hashing and helpers."""

from __future__ import annotations

import hashlib
from typing import Sequence


def order_id(player_ids: Sequence[int]) -> str:
    """Deterministic identity for an exact 1–9 batting order."""
    if len(player_ids) != 9:
        raise ValueError(f"expected 9 players, got {len(player_ids)}")
    raw = ",".join(str(int(p)) for p in player_ids)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def personnel_id(player_ids: Sequence[int]) -> str:
    """Deterministic identity for unordered nine starters."""
    if len(player_ids) != 9:
        raise ValueError(f"expected 9 players, got {len(player_ids)}")
    raw = ",".join(str(int(p)) for p in sorted(player_ids))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def validate_lineup(player_ids: Sequence[int]) -> list[int]:
    ids = [int(p) for p in player_ids]
    if len(ids) != 9:
        raise ValueError("lineup must have exactly 9 batters")
    if len(set(ids)) != 9:
        raise ValueError("lineup must have 9 unique batters")
    return ids
