"""Deterministic Markov lineup expected-run engine.

State within an inning: (base/out state, next batter index).
Solves expected runs for 9 innings given player-specific PA outcome probabilities
and empirical base-out transition distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transitions import (
    N_STATE,
    TransitionModel,
    build_player_kernels,
    load_transitions,
    state_index,
)


@dataclass
class LineupEvaluation:
    expected_runs_9: float
    expected_runs_per_inning: float
    expected_pa_by_slot: np.ndarray  # shape (9,)
    slot_start_share: np.ndarray  # P(each batter starts an inning), approx
    value_empty_0outs: np.ndarray  # E[runs in inning | start batter i, 0 outs empty]
    lineup_flow: list[dict] | None = None


class LineupEngine:
    def __init__(self, trans: TransitionModel | None = None):
        self.trans = trans or load_transitions()

    def evaluate(
        self,
        probs: np.ndarray,
        n_innings: int = 9,
        tol: float = 1e-10,
        *,
        with_flow: bool = False,
    ) -> LineupEvaluation:
        """probs: shape (9, 7) outcome probabilities for batting order slots 0..8."""
        if probs.shape != (9, 7):
            raise ValueError(f"probs must be (9,7), got {probs.shape}")
        # Normalize rows
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)

        Rs, Ps, Es = [], [], []
        for i in range(9):
            R, P, E = build_player_kernels(probs[i], self.trans)
            Rs.append(R)
            Ps.append(P)
            Es.append(E)
        R_arr = np.stack(Rs)  # (9, 24)
        P_arr = np.stack(Ps)  # (9, 24, 24)
        E_arr = np.stack(Es)  # (9, 24)

        # Solve V[batter, state] = E[runs from here until 3 outs]
        V = np.zeros((9, N_STATE), dtype=np.float64)
        for _ in range(500):
            V_new = np.empty_like(V)
            for b in range(9):
                nb = (b + 1) % 9
                V_new[b] = R_arr[b] + P_arr[b] @ V[nb]
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        start_state = state_index(0, "---")
        v_start = V[:, start_state]

        start_dist = np.zeros(9, dtype=np.float64)
        start_dist[0] = 1.0
        total_runs = 0.0
        pa_by_slot = np.zeros(9, dtype=np.float64)
        start_share = np.zeros(9, dtype=np.float64)
        # Full-game expected visits by (slot, base-out state)
        game_visits = np.zeros((9, N_STATE), dtype=np.float64)

        end_batter = np.zeros((9, 9), dtype=np.float64)
        pa_from_start = np.zeros((9, 9), dtype=np.float64)
        visits_from_start = np.zeros((9, 9, N_STATE), dtype=np.float64)

        for b0 in range(9):
            visits = np.zeros((9, N_STATE), dtype=np.float64)
            visits[b0, start_state] = 1.0
            flow = visits.copy()
            for _ in range(60):
                new_flow = np.zeros_like(flow)
                for b in range(9):
                    nb = (b + 1) % 9
                    to_states = flow[b] @ P_arr[b]
                    new_flow[nb] += to_states
                visits += new_flow
                if new_flow.sum() < 1e-12:
                    break
                flow = new_flow
            pa_from_start[b0] = visits.sum(axis=1)
            visits_from_start[b0] = visits
            absorb_to = np.zeros(9, dtype=np.float64)
            for b in range(9):
                mass = float((visits[b] * E_arr[b]).sum())
                absorb_to[(b + 1) % 9] += mass
            s = absorb_to.sum()
            if s > 0:
                absorb_to /= s
            else:
                absorb_to[(b0) % 9] = 1.0
            end_batter[b0] = absorb_to

        for inn in range(n_innings):
            start_share += start_dist
            total_runs += float(start_dist @ v_start)
            pa_by_slot += start_dist @ pa_from_start
            for b0 in range(9):
                game_visits += start_dist[b0] * visits_from_start[b0]
            start_dist = start_dist @ end_batter

        flow = None
        if with_flow:
            from .transitions import decode_state

            risp_bases = {"-2-", "--3", "12-", "1-3", "-23", "123"}
            flow = []
            for slot in range(9):
                visits_s = game_visits[slot]
                pa = float(visits_s.sum())
                runners_on = 0.0
                risp = 0.0
                two_out = 0.0
                leadoff = 0.0
                runner_mass = 0.0
                base_dist: dict[str, float] = {}
                for si in range(N_STATE):
                    mass = float(visits_s[si])
                    if mass <= 0:
                        continue
                    outs, bases = decode_state(si)
                    key = f"{outs}|{bases}"
                    base_dist[key] = base_dist.get(key, 0.0) + mass
                    n_run = sum(1 for ch in bases if ch != "-")
                    runner_mass += mass * n_run
                    if bases != "---":
                        runners_on += mass
                    if bases in risp_bases:
                        risp += mass
                    if outs == 2:
                        two_out += mass
                    if outs == 0 and bases == "---":
                        leadoff += mass
                if pa > 0:
                    base_dist = {
                        k: v / pa
                        for k, v in sorted(base_dist.items(), key=lambda x: -x[1])
                    }
                    top = dict(list(base_dist.items())[:8])
                else:
                    top = {}
                flow.append({
                    "slot": slot + 1,
                    "expected_pa": float(pa_by_slot[slot]),
                    "prob_runners_on": float(runners_on / pa) if pa > 0 else 0.0,
                    "runners_on_pct": float(runners_on / pa) if pa > 0 else 0.0,
                    "risp_pct": float(risp / pa) if pa > 0 else 0.0,
                    "avg_runners_on": float(runner_mass / pa) if pa > 0 else 0.0,
                    "leadoff_pct": float(leadoff / pa) if pa > 0 else 0.0,
                    "two_out_pct": float(two_out / pa) if pa > 0 else 0.0,
                    "expected_runner_on_pa": float(runners_on),
                    "base_state_distribution": top,
                    "value_empty_0outs": float(v_start[slot]),
                })

        return LineupEvaluation(
            expected_runs_9=float(total_runs),
            expected_runs_per_inning=float(total_runs / n_innings),
            expected_pa_by_slot=pa_by_slot,
            slot_start_share=start_share / n_innings,
            value_empty_0outs=v_start,
            lineup_flow=flow,
        )

    def expected_runs(self, probs: np.ndarray, n_innings: int = 9) -> float:
        return self.evaluate(probs, n_innings=n_innings).expected_runs_9


def lineup_breakdown_narrative(ev: LineupEvaluation) -> tuple[str, list[str]]:
    """Plain-English summary + 2–4 observations from computed flow only."""
    runs = float(ev.expected_runs_9)
    flow = ev.lineup_flow or []
    if len(flow) != 9:
        return (f"This order projects {runs:.2f} runs per game.", [])

    pa = [float(f.get("expected_pa") or 0.0) for f in flow]
    on = [float(f.get("runners_on_pct") or f.get("prob_runners_on") or 0.0) for f in flow]
    risp = [float(f.get("risp_pct") or 0.0) for f in flow]

    bits = [f"This order projects {runs:.2f} runs per game."]
    if pa and pa[0] >= max(pa) - 1e-9:
        bits.append("Higher lineup spots receive more plate appearances")
    mid = sum(on[2:6]) / 4.0
    top = sum(on[:2]) / 2.0
    bot = sum(on[6:9]) / 3.0
    if mid > top and mid > bot:
        bits.append(
            "while the middle of the order is projected to bat with runners aboard more frequently"
        )
    if len(bits) == 1:
        summary = bits[0]
    elif len(bits) == 2:
        summary = bits[0] + " " + bits[1] + "."
    else:
        summary = bits[0] + " " + bits[1] + ", " + bits[2] + "."

    notes: list[str] = []
    i_pa = int(max(range(9), key=lambda i: pa[i]))
    notes.append(
        f"Slot {i_pa + 1} receives the most plate appearances at {pa[i_pa]:.2f} per game."
    )
    i_on = int(max(range(9), key=lambda i: on[i]))
    notes.append(
        f"Slot {i_on + 1} has the highest runners-on probability at {100.0 * on[i_on]:.0f}%."
    )
    if mid >= top and mid >= bot:
        notes.append("Slots 3–6 receive the most frequent run-producing situations.")
    i_risp = int(max(range(9), key=lambda i: risp[i]))
    if i_risp != i_on:
        notes.append(
            f"Slot {i_risp + 1} has the highest RISP rate at {100.0 * risp[i_risp]:.0f}%."
        )
    return summary, notes[:4]
