"""Monte Carlo lineup simulation for run distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transitions import N_STATE, TransitionModel, load_transitions, decode_state


@dataclass
class SimResult:
    n_games: int
    mean: float
    median: float
    std: float
    p0: float
    p3_plus: float
    p5_plus: float
    p8_plus: float
    histogram: dict[int, int]
    deterministic_expected: float | None = None
    p05: float = 0.0
    p95: float = 0.0
    p0_2: float = 0.0
    p3_5: float = 0.0
    p6_plus: float = 0.0
    mean_runs: float = 0.0
    median_runs: float = 0.0
    std_runs: float = 0.0


def _sample_transition(trans: TransitionModel, outcome_i: int, state_i: int, rng: np.random.Generator):
    k = int(trans.n_trans[outcome_i, state_i])
    probs = trans.next_prob[outcome_i, state_i, :k].astype(np.float64)
    probs = probs / probs.sum()
    j = int(rng.choice(k, p=probs))
    return int(trans.next_idx[outcome_i, state_i, j]), float(trans.next_runs[outcome_i, state_i, j])


def simulate_lineup(
    probs: np.ndarray,
    n_games: int = 1000,
    n_innings: int = 9,
    seed: int = 42,
    trans: TransitionModel | None = None,
    deterministic_expected: float | None = None,
) -> SimResult:
    """Simulate n_games of n_innings using player PA outcome probs (9,7)."""
    trans = trans or load_transitions()
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    runs_arr = np.zeros(n_games, dtype=np.float64)

    for g in range(n_games):
        batter = 0
        total = 0.0
        for _inn in range(n_innings):
            state = 0  # 0 outs, ---
            while True:
                # sample outcome
                oi = int(rng.choice(7, p=probs[batter]))
                nxt, r = _sample_transition(trans, oi, state, rng)
                total += r
                batter = (batter + 1) % 9
                if nxt < 0:
                    break
                state = nxt
        runs_arr[g] = total

    hist = {}
    for v in runs_arr.astype(int):
        hist[int(v)] = hist.get(int(v), 0) + 1

    mean = float(runs_arr.mean())
    median = float(np.median(runs_arr))
    std = float(runs_arr.std())
    return SimResult(
        n_games=n_games,
        mean=mean,
        median=median,
        std=std,
        p0=float((runs_arr == 0).mean()),
        p3_plus=float((runs_arr >= 3).mean()),
        p5_plus=float((runs_arr >= 5).mean()),
        p8_plus=float((runs_arr >= 8).mean()),
        histogram=hist,
        deterministic_expected=deterministic_expected,
        p05=float(np.percentile(runs_arr, 5)),
        p95=float(np.percentile(runs_arr, 95)),
        p0_2=float(((runs_arr >= 0) & (runs_arr <= 2)).mean()),
        p3_5=float(((runs_arr >= 3) & (runs_arr <= 5)).mean()),
        p6_plus=float((runs_arr >= 6).mean()),
        mean_runs=mean,
        median_runs=median,
        std_runs=std,
    )
