"""Research modules: archetypes, adjacency interactions, findings, lineup stability."""

from .stability import (
    avg_position_changes,
    lineup_order_entropy,
    most_common_order_pct,
    stability_summary,
    unique_orders,
    unique_personnel,
)

__all__ = [
    "avg_position_changes",
    "lineup_order_entropy",
    "most_common_order_pct",
    "stability_summary",
    "unique_orders",
    "unique_personnel",
]
