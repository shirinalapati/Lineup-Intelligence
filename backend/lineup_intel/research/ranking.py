"""League ranking framework for player metrics.

Every rank object includes value, rank, population_n, percentile, qualifying
threshold, and direction. Players who fail the sample gate still retain their
raw value with ``qualified=False`` and no fabricated rank.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

Direction = Literal["higher_better", "lower_better", "neutral"]


# Transparent minimum-PA gates (documented for Research methodology).
MIN_PA_OVERALL = 100
MIN_PA_PLATOON = 50
MIN_PA_SLOT = 30

METRIC_DIRECTION: dict[str, Direction] = {
    # Core offense
    "woba": "higher_better",
    "xwoba": "higher_better",
    "woba_vs_R": "higher_better",
    "woba_vs_L": "higher_better",
    "obp": "higher_better",
    "slg": "higher_better",
    "iso": "higher_better",
    "wrc_plus": "higher_better",
    "xslg": "higher_better",
    # Discipline
    "k_pct": "lower_better",
    "bb_pct": "higher_better",
    "chase_pct": "lower_better",
    "contact_pct": "higher_better",
    "z_contact_pct": "higher_better",
    # Contact quality
    "avg_exit_velocity": "higher_better",
    "hardhit_pct": "higher_better",
    "barrel_pct": "higher_better",
    # Batted ball — descriptive / neutral unless noted
    "gb_pct": "neutral",
    "ld_pct": "neutral",
    "fb_pct": "neutral",
    "pull_pct": "neutral",
    "center_pct": "neutral",
    "oppo_pct": "neutral",
    # Lineup / placement
    "placement_opportunity": "higher_better",
    "actual_usage_fit_pct": "higher_better",
    "marginal_runs_vs_avg": "higher_better",
    "expected_runs": "higher_better",
    # Team season aggregates (30-team ranks)
    "avg_gap": "lower_better",
    "avg_percentile": "higher_better",
    "avg_actual_runs": "higher_better",
    "unique_orders": "higher_better",
    "unique_personnel": "higher_better",
}

# Value key + direction for 30-team ranks on team pages.
TEAM_METRIC_SPECS: tuple[tuple[str, Direction], ...] = (
    ("avg_gap", "lower_better"),
    ("avg_percentile", "higher_better"),
    ("avg_actual_runs", "higher_better"),
    ("unique_orders", "higher_better"),
    ("unique_personnel", "higher_better"),
)


@dataclass
class RankedMetric:
    value: float | None
    rank: int | None
    population_n: int | None
    percentile: int | None
    qualifying_threshold: int | float | None
    direction: Direction
    qualified: bool
    metric: str
    display: str | None = None
    note: str | None = None
    sample_size: int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.value is not None and isinstance(self.value, float):
            d["value"] = float(self.value)
        return d


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _competition_rank(values: np.ndarray, *, higher_better: bool) -> np.ndarray:
    """Competition ranking (1 = best). Ties share the minimum rank."""
    n = len(values)
    order = np.argsort(-values if higher_better else values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isclose(sorted_vals[j], sorted_vals[i], equal_nan=False):
            j += 1
        # competition rank: 1-based position of first in tied group
        r = i + 1
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def favorable_percentile(rank: int, population_n: int) -> int:
    """100 = most favorable extreme given competition rank."""
    if population_n <= 1:
        return 100
    # rank 1 → ~100, rank N → ~0
    pct = 100.0 * (1.0 - (rank - 1) / population_n)
    return int(round(pct))


def rank_metric(
    value: float | None,
    peer_values: Iterable[float],
    *,
    metric: str,
    direction: Direction | None = None,
    qualifying_threshold: int | float | None = None,
    sample_size: int | float | None = None,
    qualified: bool | None = None,
) -> RankedMetric:
    direction = direction or METRIC_DIRECTION.get(metric, "neutral")
    peers = [float(v) for v in peer_values if v is not None and np.isfinite(float(v))]
    if qualified is None:
        if qualifying_threshold is None or sample_size is None:
            qualified = value is not None and np.isfinite(float(value)) and len(peers) > 0
        else:
            qualified = (
                sample_size is not None
                and float(sample_size) >= float(qualifying_threshold)
                and value is not None
                and np.isfinite(float(value))
            )

    if value is None or not np.isfinite(float(value)):
        return RankedMetric(
            value=None,
            rank=None,
            population_n=len(peers) if peers else None,
            percentile=None,
            qualifying_threshold=qualifying_threshold,
            direction=direction,
            qualified=False,
            metric=metric,
            sample_size=sample_size,
            note="Statistic unavailable",
            display=None,
        )

    val = float(value)
    if not qualified:
        return RankedMetric(
            value=val,
            rank=None,
            population_n=None,
            percentile=None,
            qualifying_threshold=qualifying_threshold,
            direction=direction,
            qualified=False,
            metric=metric,
            sample_size=sample_size,
            note="Limited sample — rank unavailable",
            display=_fmt_value(metric, val),
        )

    # Ensure the focal value is in the peer set for ranking consistency
    arr = np.asarray(peers + [val], dtype=float)
    # Deduplicate? No — population is peers who qualified; focal may already be in peers.
    # Prefer: peers already include everyone qualified including focal.
    if peers:
        arr = np.asarray(peers, dtype=float)
        # Find closest match to val for ranking (focal should be in peers)
        if not any(np.isclose(arr, val)):
            arr = np.append(arr, val)
            idx = len(arr) - 1
        else:
            idx = int(np.argmin(np.abs(arr - val)))
    else:
        arr = np.asarray([val], dtype=float)
        idx = 0

    if direction == "neutral":
        # Descriptive: percentile of raw distribution (higher value = higher %ile)
        higher_better = True
        ranks = _competition_rank(arr, higher_better=True)
        rank = int(ranks[idx])
        # For neutral, report raw percentile of value (not "favorable")
        pct = int(round(100.0 * (arr <= val).mean())) if len(arr) else None
        note = "Descriptive distribution percentile (not a quality judgment)"
    elif direction == "higher_better":
        ranks = _competition_rank(arr, higher_better=True)
        rank = int(ranks[idx])
        pct = favorable_percentile(rank, len(arr))
        note = None
    else:  # lower_better
        ranks = _competition_rank(arr, higher_better=False)
        rank = int(ranks[idx])
        pct = favorable_percentile(rank, len(arr))
        note = None

    return RankedMetric(
        value=val,
        rank=rank,
        population_n=int(len(arr)),
        percentile=pct,
        qualifying_threshold=qualifying_threshold,
        direction=direction,
        qualified=True,
        metric=metric,
        sample_size=sample_size,
        note=note,
        display=_fmt_value(metric, val),
    )


def _fmt_value(metric: str, val: float) -> str:
    rate_like = metric.endswith("_pct") or metric in {
        "k_rate",
        "bb_rate",
        "obp",
        "slg",
        "ba",
    }
    if metric in {"woba", "xwoba", "iso", "xslg", "woba_vs_R", "woba_vs_L"} or rate_like and val <= 1.5:
        if metric.endswith("_pct") or metric in {"k_rate", "bb_rate"}:
            # store as fraction → display percent
            return f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
        return f"{val:.3f}"
    if metric in {"avg_exit_velocity"}:
        return f"{val:.1f}"
    if metric in {"wrc_plus"}:
        return f"{val:.0f}"
    if abs(val) < 1:
        return f"{val:.3f}"
    return f"{val:.2f}"


def rank_frame(
    df: pd.DataFrame,
    *,
    value_col: str,
    metric: str,
    sample_col: str | None = None,
    min_sample: int | float | None = None,
    direction: Direction | None = None,
    id_col: str = "player_id",
) -> dict[int, RankedMetric]:
    """Rank all rows in ``df`` for one metric; returns map player_id → RankedMetric."""
    direction = direction or METRIC_DIRECTION.get(metric, "neutral")
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if sample_col and min_sample is not None:
        work["_sample"] = pd.to_numeric(work[sample_col], errors="coerce").fillna(0)
        qualified = work[work["_sample"] >= float(min_sample)].copy()
    else:
        work["_sample"] = np.nan
        qualified = work[work[value_col].notna()].copy()

    peer_vals = qualified[value_col].dropna().astype(float).tolist()
    out: dict[int, RankedMetric] = {}
    for _, row in work.iterrows():
        pid = int(row[id_col])
        samp = float(row["_sample"]) if pd.notna(row["_sample"]) else None
        is_q = (
            samp is not None and min_sample is not None and samp >= float(min_sample)
            and pd.notna(row[value_col])
        ) if min_sample is not None else pd.notna(row[value_col])
        out[pid] = rank_metric(
            float(row[value_col]) if pd.notna(row[value_col]) else None,
            peer_vals if is_q else peer_vals,
            metric=metric,
            direction=direction,
            qualifying_threshold=min_sample,
            sample_size=samp,
            qualified=bool(is_q),
        )
    return out


def attach_team_rank(
    league: RankedMetric,
    *,
    team_value: float | None,
    team_peer_values: list[float],
    metric: str,
    sample_size: int | float | None,
    min_sample: int | float | None,
    direction: Direction | None = None,
) -> dict[str, Any]:
    """Bundle MLB + team ranks for API payloads."""
    team = rank_metric(
        team_value if team_value is not None else league.value,
        team_peer_values,
        metric=metric,
        direction=direction or league.direction,
        qualifying_threshold=min_sample,
        sample_size=sample_size,
        qualified=league.qualified and len(team_peer_values) >= 2,
    )
    return {
        "mlb": league.to_dict(),
        "team": team.to_dict() if league.qualified else {
            **team.to_dict(),
            "note": league.note or "Limited sample — rank unavailable",
            "qualified": False,
            "rank": None,
            "percentile": None,
            "population_n": None,
        },
    }


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def rank_team_metrics(
    teams: dict[str, dict[str, Any]],
    abbr: str,
) -> dict[str, dict[str, Any]]:
    """30-team ranks for season aggregates. Rank 1 is the favorable extreme
    (smallest gap; highest percentile, talent, unique orders, unique personnel).
    """
    out: dict[str, dict[str, Any]] = {}
    focal = teams.get(abbr) or {}
    for metric, direction in TEAM_METRIC_SPECS:
        peers: list[float] = []
        for row in teams.values():
            v = _finite_float(row.get(metric))
            if v is not None:
                peers.append(v)
        fval = _finite_float(focal.get(metric))
        out[metric] = rank_metric(
            fval,
            peers,
            metric=metric,
            direction=direction,
            qualified=fval is not None and len(peers) >= 2,
        ).to_dict()
    return out


def attach_team_metric_ranks(teams: dict[str, dict[str, Any]]) -> None:
    """Mutate each team dict with a ``ranks`` payload."""
    for abbr, row in teams.items():
        row["ranks"] = rank_team_metrics(teams, abbr)
