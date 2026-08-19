"""Transparent lineup stability metrics (no proprietary composite score)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"lineups dataframe missing columns: {missing}")


def unique_orders(df: pd.DataFrame) -> int:
    """Number of distinct ordered lineups (order_id)."""
    _require_cols(df, ["order_id"])
    return int(df["order_id"].nunique(dropna=True))


def unique_personnel(df: pd.DataFrame) -> int:
    """Number of distinct unordered nines (personnel_id)."""
    _require_cols(df, ["personnel_id"])
    return int(df["personnel_id"].nunique(dropna=True))


def lineup_order_entropy(df: pd.DataFrame, base: float = 2.0) -> float:
    """Shannon entropy of order_id frequencies (bits when base=2).

    Higher entropy ⇒ more dispersed order usage. Zero iff a single order
    accounts for every game in ``df``.
    """
    _require_cols(df, ["order_id"])
    if len(df) == 0:
        return float("nan")
    counts = df["order_id"].value_counts(dropna=True).to_numpy(dtype=np.float64)
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    # 0 * log(0) convention: skip zeros (value_counts already drops them)
    return float(-(p * np.log(p) / np.log(base)).sum())


def most_common_order_pct(df: pd.DataFrame) -> float:
    """Share of games using the single most-common batting order."""
    _require_cols(df, ["order_id"])
    if len(df) == 0:
        return float("nan")
    counts = df["order_id"].value_counts(dropna=True)
    if counts.empty:
        return float("nan")
    return float(counts.iloc[0] / len(df))


def avg_position_changes(df: pd.DataFrame, date_col: str = "game_date") -> float:
    """Mean number of slot-position changes between consecutive games.

    Games are ordered by ``date_col`` (then ``game_pk`` if present). For each
    adjacent pair of games, count how many of the nine slots have a different
    player_id. Result is in [0, 9].
    """
    _require_cols(df, SLOT_COLS)
    if len(df) < 2:
        return float("nan")

    sort_cols = [c for c in (date_col, "game_pk") if c in df.columns]
    ordered = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    slots = ordered[SLOT_COLS].to_numpy()
    # Absolute differences as inequality counts across consecutive rows
    changes = (slots[1:] != slots[:-1]).sum(axis=1).astype(np.float64)
    return float(changes.mean()) if len(changes) else float("nan")


def same_nine_recurrence(df: pd.DataFrame) -> float:
    """Share of consecutive game pairs that reuse the same personnel_id."""
    _require_cols(df, ["personnel_id"])
    if "game_date" not in df.columns and "game_pk" not in df.columns:
        # Still computable in row order, but prefer chronological when available
        ordered = df.reset_index(drop=True)
    else:
        sort_cols = [c for c in ("game_date", "game_pk") if c in df.columns]
        ordered = df.sort_values(sort_cols).reset_index(drop=True)
    if len(ordered) < 2:
        return float("nan")
    pid = ordered["personnel_id"].to_numpy()
    return float(np.mean(pid[1:] == pid[:-1]))


def stability_summary(df: pd.DataFrame, team: str | None = None) -> dict[str, Any]:
    """Compute the full interpretable stability bundle for a lineups frame.

    If ``team`` is provided and a ``team`` column exists, filter first.
    """
    work = df
    if team is not None:
        if "team" not in df.columns:
            raise ValueError("team filter requested but 'team' column missing")
        work = df[df["team"] == team].copy()
    n_games = int(len(work))
    return {
        "team": team,
        "n_games": n_games,
        "unique_orders": unique_orders(work) if n_games else 0,
        "unique_personnel": unique_personnel(work) if n_games else 0,
        "order_entropy_bits": lineup_order_entropy(work) if n_games else float("nan"),
        "most_common_order_pct": most_common_order_pct(work) if n_games else float("nan"),
        "avg_position_changes": avg_position_changes(work) if n_games else float("nan"),
        "same_nine_recurrence": same_nine_recurrence(work) if n_games else float("nan"),
    }


def stability_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """One stability summary row per team."""
    _require_cols(df, ["team"])
    rows = [stability_summary(df, team=t) for t in sorted(df["team"].dropna().unique())]
    return pd.DataFrame(rows)


def lineup_stability(df: pd.DataFrame) -> dict[str, Any]:
    """Bundle of interpretable stability metrics for a team lineup history."""
    return {
        "unique_orders": unique_orders(df) if len(df) else 0,
        "unique_personnel": unique_personnel(df) if len(df) else 0,
        "order_entropy": lineup_order_entropy(df) if len(df) else None,
        "most_common_order_pct": most_common_order_pct(df) if len(df) else None,
        "avg_position_changes": avg_position_changes(df) if len(df) else None,
        "same_nine_recurrence": same_nine_recurrence(df) if len(df) else None,
        "n_games": int(len(df)),
    }
