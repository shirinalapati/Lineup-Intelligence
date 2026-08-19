"""Generate structured explanations for why one order beats another."""

from __future__ import annotations

import numpy as np

from .transitions import OUTCOME_CLASSES


def explain_order_delta(
    player_ids: list[int],
    player_names: dict[int, str],
    actual_order: list[int],
    best_order: list[int],
    probs_by_id: dict[int, np.ndarray],
    actual_runs: float,
    best_runs: float,
    actual_pa: np.ndarray | None = None,
    best_pa: np.ndarray | None = None,
) -> list[dict]:
    """Return structured explanation bullets backed by computed deltas."""
    explanations: list[dict] = []
    gap = best_runs - actual_runs
    if abs(gap) < 1e-9:
        explanations.append({
            "code": "identical",
            "text": (
                "Your current batting order is already the model’s best order "
                "for these nine hitters."
            ),
            "support": {"gap": 0.0},
        })
        return explanations

    if gap <= 0.02:
        explanations.append({
            "code": "operationally_equivalent",
            "text": (
                f"Compared with the order on your card, the model’s best batting "
                f"order of these same nine hitters projects only {gap:.3f} more "
                f"runs per game. That is a tiny difference — many arrangements "
                f"of this nine are effectively tied."
            ),
            "support": {"gap": gap},
        })

    def obp(pid: int) -> float:
        p = probs_by_id[pid]
        # OBP proxy: 1 - K - OUT_IP (rough; BB_HBP+hits)
        idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
        return float(1.0 - p[idx["K"]] - p[idx["OUT_IP"]])

    def iso_proxy(pid: int) -> float:
        p = probs_by_id[pid]
        idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
        return float(p[idx["2B"]] + 2 * p[idx["3B"]] + 3 * p[idx["HR"]])

    def hr_rate(pid: int) -> float:
        return float(probs_by_id[pid][OUTCOME_CLASSES.index("HR")])

    # Top-of-order OBP
    act_top_obp = np.mean([obp(p) for p in actual_order[:3]])
    best_top_obp = np.mean([obp(p) for p in best_order[:3]])
    if best_top_obp - act_top_obp > 0.005:
        explanations.append({
            "code": "obp_up_top",
            "text": (
                f"The best order places higher on-base profiles in slots 1–3 "
                f"(on-base {best_top_obp:.3f} vs {act_top_obp:.3f} in your current order)."
            ),
            "support": {"best_top_obp": best_top_obp, "actual_top_obp": act_top_obp},
        })

    # Power behind OBP
    def power_after_obp(order: list[int]) -> float:
        score = 0.0
        for i in range(8):
            score += obp(order[i]) * iso_proxy(order[i + 1])
        return score

    act_seq = power_after_obp(actual_order)
    best_seq = power_after_obp(best_order)
    if best_seq - act_seq > 0.0005:
        explanations.append({
            "code": "obp_before_power",
            "text": (
                "The best order puts higher on-base hitters ahead of extra-base "
                "hitters more often, so power bats see more runners on base."
            ),
            "support": {"actual_seq": act_seq, "best_seq": best_seq},
        })

    # PA distribution to best hitters
    if actual_pa is not None and best_pa is not None:
        # quality = obp + iso
        q = {pid: obp(pid) + iso_proxy(pid) for pid in player_ids}
        act_map = {pid: actual_pa[i] for i, pid in enumerate(actual_order)}
        best_map = {pid: best_pa[i] for i, pid in enumerate(best_order)}
        # expected PA-weighted quality
        act_wq = sum(act_map[pid] * q[pid] for pid in player_ids)
        best_wq = sum(best_map[pid] * q[pid] for pid in player_ids)
        if best_wq > act_wq + 1e-6:
            # who gained PA
            gains = sorted(
                ((best_map[pid] - act_map[pid], pid) for pid in player_ids),
                reverse=True,
            )
            top = gains[0]
            if top[0] > 0.02:
                explanations.append({
                    "code": "more_pa_best_hitters",
                    "text": (
                        f"In the best order, stronger hitters get more plate appearances; "
                        f"{player_names.get(top[1], top[1])} gains "
                        f"{top[0]:.2f} expected PA/game versus your current order."
                    ),
                    "support": {"pa_gain": top[0], "player_id": top[1]},
                })

    # Cluster of low-OBP
    def low_obp_cluster(order: list[int]) -> int:
        flags = [obp(p) < 0.300 for p in order]
        # count adjacent low-OBP pairs
        return sum(1 for i in range(8) if flags[i] and flags[i + 1])

    if low_obp_cluster(best_order) < low_obp_cluster(actual_order):
        explanations.append({
            "code": "reduce_low_obp_cluster",
            "text": (
                "The best order clusters fewer low on-base hitters in a row "
                "than your current order."
            ),
            "support": {
                "actual_clusters": low_obp_cluster(actual_order),
                "best_clusters": low_obp_cluster(best_order),
            },
        })

    if not explanations or (len(explanations) == 1 and explanations[0]["code"] == "operationally_equivalent"):
        explanations.append({
            "code": "small_mechanical",
            "text": (
                f"The {gap:.3f} run-per-game difference between your current order "
                f"and the model’s best order comes from the whole sequence "
                f"(who bats with runners on, and who gets extra plate appearances), "
                f"not from one player swap."
            ),
            "support": {"gap": gap},
        })

    return explanations
