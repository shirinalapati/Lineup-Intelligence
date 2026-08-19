"""Same-nine batting-order optimization.

Design:
- Exhaustive ranking over 9! uses a fast Numba scoring function derived from
  each batter's expected runs-by-state kernel (same transition model as the
  full engine), not a hand-waved OPS weight.
- Reported actual / best / worst expected runs are always refined with the
  full 9-inning Markov engine.
- Correlation between fast ranks and full-engine ranks is validated in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .markov import LineupEngine
from .transitions import N_STATE, TransitionModel, build_player_kernels, load_transitions, state_index

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class OptimizeResult:
    actual_order: list[int]
    actual_runs: float
    best_order: list[int]
    best_runs: float
    worst_order: list[int]
    worst_runs: float
    mean_runs: float
    median_runs: float
    rank: int
    n_perms: int
    percentile: float
    gap: float
    ordering_value: float
    value_vs_median: float = 0.0
    best_worst_spread: float = 0.0
    near_optimal: list[dict] = field(default_factory=list)
    n_near_optimal: int = 0
    n_near_optimal_01: int = 0
    n_near_optimal_02: int = 0
    pct_near_optimal_01: float = 0.0
    pct_near_optimal_02: float = 0.0
    equivalence_eps: float = 0.02
    runs_distribution_summary: dict = field(default_factory=dict)
    method: str = "fast_exhaustive_plus_markov_refine"
    operationally_equivalent: bool = False
    practical_note: str = ""

def _build_kernels_for_players(prob_rows: np.ndarray, trans: TransitionModel):
    n = prob_rows.shape[0]
    R = np.zeros((n, N_STATE), dtype=np.float64)
    P = np.zeros((n, N_STATE, N_STATE), dtype=np.float64)
    E = np.zeros((n, N_STATE), dtype=np.float64)
    for i in range(n):
        r, p, e = build_player_kernels(prob_rows[i], trans)
        R[i], P[i], E[i] = r, p, e
    return R, P, E


def _league_state_weights(R: np.ndarray, P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Approximate long-run base/out distribution under average of these 9 hitters."""
    Rmean = R.mean(axis=0)
    Pmean = P.mean(axis=0)
    Emean = E.mean(axis=0)
    # Expected visits in an inning starting from empty
    visits = np.zeros(N_STATE, dtype=np.float64)
    flow = np.zeros(N_STATE, dtype=np.float64)
    flow[state_index(0, "---")] = 1.0
    visits += flow
    for _ in range(40):
        nxt = flow @ Pmean
        visits += nxt
        flow = nxt
        if flow.sum() < 1e-12:
            break
    w = visits / max(visits.sum(), 1e-12)
    return w


if HAS_NUMBA:

    @njit(cache=True)
    def _perm_from_index(pi, out):
        fact = np.empty(9, dtype=np.int64)
        fact[0] = 1
        for i in range(1, 9):
            fact[i] = fact[i - 1] * i
        elems = np.empty(9, dtype=np.int64)
        for i in range(9):
            elems[i] = i
        rem = int(pi)
        for pos in range(9):
            f = int(fact[8 - pos])
            idx = int(rem // f)
            rem = int(rem % f)
            out[pos] = elems[idx]
            for k in range(idx, 8 - pos):
                elems[k] = elems[k + 1]

    @njit(parallel=True, cache=True)
    def _score_all_perms_fast(slot_value, seq_bonus, lead_extra):
        """Fast exhaustive score.

        slot_value[i] = expected runs contribution weight for player i
        seq_bonus[i,j] = sequencing value when i bats immediately before j
        lead_extra[i] = extra weight for batting leadoff (more PA)

        score = sum_k slot_pa_weight[k] * slot_value[order[k]]
              + sum_k seq_bonus[order[k], order[k+1]]
              + lead_extra[order[0]]
        """
        n_perms = 362880
        scores = np.empty(n_perms, dtype=np.float64)
        # PA weights by batting slot (empirical-ish; leadoff highest)
        pa_w = np.array([1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91], dtype=np.float64)

        fact = np.empty(9, dtype=np.int64)
        fact[0] = 1
        for i in range(1, 9):
            fact[i] = fact[i - 1] * i

        for pi in prange(n_perms):
            elems = np.empty(9, dtype=np.int64)
            order = np.empty(9, dtype=np.int64)
            for i in range(9):
                elems[i] = i
            rem = int(pi)
            for pos in range(9):
                f = int(fact[8 - pos])
                idx = int(rem // f)
                rem = int(rem % f)
                order[pos] = elems[idx]
                for k in range(idx, 8 - pos):
                    elems[k] = elems[k + 1]
            sc = lead_extra[order[0]]
            for k in range(9):
                sc += pa_w[k] * slot_value[order[k]]
            for k in range(8):
                sc += seq_bonus[order[k], order[k + 1]]
            # wrap 9 -> 1 mild continuity
            sc += 0.25 * seq_bonus[order[8], order[0]]
            scores[pi] = sc
        return scores


def _index_of_perm(perm: tuple[int, ...]) -> int:
    elems = list(range(9))
    idx = 0
    fact = [1]
    for i in range(1, 10):
        fact.append(fact[-1] * i)
    for pos, val in enumerate(perm):
        i = elems.index(val)
        idx += i * fact[8 - pos]
        elems.pop(i)
    return idx


def _fast_features(R: np.ndarray, P: np.ndarray, E: np.ndarray):
    w = _league_state_weights(R, P, E)
    # Player run value = expected runs on a PA under typical state mix
    slot_value = R @ w
    # Sequencing: value of leaving states that benefit the next hitter
    # Approx: E[next hitter R under state dist induced by previous hitter from empty-ish mix]
    n = R.shape[0]
    seq = np.zeros((n, n), dtype=np.float64)
    # Induced next-state distribution from weighted start states
    for i in range(n):
        next_dist = w @ P[i]  # may sum < 1
        mass = next_dist.sum()
        if mass > 1e-12:
            next_dist = next_dist / mass
        for j in range(n):
            seq[i, j] = float(next_dist @ R[j])
    # Center sequencing to isolate interaction beyond average
    seq = seq - seq.mean()
    lead_extra = 0.15 * slot_value  # leadoff PA premium already partly in pa_w
    return slot_value.astype(np.float64), seq.astype(np.float64), lead_extra.astype(np.float64)


def _local_search_full(engine: LineupEngine, probs: np.ndarray, seed: list[int], restarts: int = 1):
    """Light hill-climb: adjacent swaps + a few pair swaps (speed-focused)."""
    rng = np.random.default_rng(0)
    best = list(seed)
    best_sc = engine.expected_runs(probs[best])
    for r in range(restarts):
        order = list(best) if r == 0 else list(rng.permutation(9))
        cur = best_sc if r == 0 else engine.expected_runs(probs[order])
        for _round in range(2):
            improved = False
            for i in range(8):
                j = i + 1
                order[i], order[j] = order[j], order[i]
                sc = engine.expected_runs(probs[order])
                if sc > cur + 1e-12:
                    cur = sc
                    improved = True
                else:
                    order[i], order[j] = order[j], order[i]
            for _ in range(8):
                i = int(rng.integers(0, 9)); j = int(rng.integers(0, 9))
                if i == j:
                    continue
                order[i], order[j] = order[j], order[i]
                sc = engine.expected_runs(probs[order])
                if sc > cur + 1e-12:
                    cur = sc
                    improved = True
                else:
                    order[i], order[j] = order[j], order[i]
            if not improved:
                break
        if cur > best_sc:
            best_sc = cur
            best = list(order)
    return best, best_sc


def optimize_lineup(
    player_ids: list[int],
    probs: np.ndarray,
    actual_order_ids: list[int] | None = None,
    equivalence_eps: float = 0.02,
    top_k_near: int = 15,
    trans: TransitionModel | None = None,
    engine: LineupEngine | None = None,
) -> OptimizeResult:
    trans = trans or load_transitions()
    engine = engine or LineupEngine(trans)
    player_ids = [int(p) for p in player_ids]
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    if actual_order_ids is None:
        actual_order_ids = list(player_ids)
    actual_idx = tuple(id_to_idx[i] for i in actual_order_ids)

    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    R, P, E = _build_kernels_for_players(probs, trans)
    n_perms = 362880

    slot_value, seq_bonus, lead_extra = _fast_features(R, P, E)

    method = "fast_exhaustive_plus_markov_refine"
    if HAS_NUMBA:
        # warmup compile outside timing critically
        scores = _score_all_perms_fast(slot_value, seq_bonus, lead_extra)
        actual_pi = _index_of_perm(actual_idx)
        actual_fast = float(scores[actual_pi])
        best_pi = int(scores.argmax())
        worst_pi = int(scores.argmin())
        best_arr = np.empty(9, dtype=np.int64)
        worst_arr = np.empty(9, dtype=np.int64)
        _perm_from_index(best_pi, best_arr)
        _perm_from_index(worst_pi, worst_arr)
        best_perm = tuple(int(x) for x in best_arr)
        worst_perm = tuple(int(x) for x in worst_arr)
        rank = int((scores > actual_fast).sum()) + 1
        mean_fast = float(scores.mean())
        median_fast = float(np.median(scores))
        n_near_fast = int((scores >= scores[best_pi] - 1e-9).sum())  # exact ties on fast surface
        # candidates: top 40 by fast score + actual + worst
        top_idx = np.argpartition(-scores, 40)[:40]
        cand = {best_perm, worst_perm, actual_idx}
        tmp = np.empty(9, dtype=np.int64)
        for pi in top_idx:
            _perm_from_index(int(pi), tmp)
            cand.add(tuple(int(x) for x in tmp))
    else:
        method = "local_search_markov"
        best_list, _ = _local_search_full(engine, probs, list(actual_idx))
        best_perm = tuple(best_list)
        worst_perm = actual_idx
        cand = {best_perm, actual_idx}
        # sample for percentile
        rng = np.random.default_rng(42)
        sample_scores = []
        actual_full = engine.expected_runs(probs[list(actual_idx)])
        for _ in range(3000):
            perm = tuple(int(x) for x in rng.permutation(9))
            sc = float(np.dot(
                np.array([1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91]),
                slot_value[list(perm)],
            ))
            sample_scores.append(sc)
            if len(cand) < 25:
                cand.add(perm)
        actual_fast = float(np.dot(
            np.array([1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91]),
            slot_value[list(actual_idx)],
        ))
        scores = np.array(sample_scores + [actual_fast])
        rank = max(1, int(round(((scores > actual_fast).mean()) * n_perms))) + 1
        mean_fast = float(scores.mean())
        median_fast = float(np.median(scores))
        n_near_fast = 1

    def full(perm):
        return float(engine.expected_runs(probs[list(perm)]))

    refined = {perm: full(perm) for perm in cand}
    # Also run a short Markov local search from best fast candidate to avoid fast-model miss
    seed = list(max(refined, key=refined.get))
    loc_best, loc_sc = _local_search_full(engine, probs, seed, restarts=1)
    refined[tuple(loc_best)] = loc_sc

    best_perm = max(refined, key=refined.get)
    worst_perm = min(refined, key=refined.get)
    actual_runs = refined[actual_idx]
    best_runs = refined[best_perm]
    worst_runs = refined[worst_perm]

    # Calibrate distribution to full-engine scale using actual
    if abs(actual_fast) > 1e-12:
        # map fast scores to runs via linear calibration against refined points
        xs = np.array([actual_fast] + [float(np.dot(
            np.array([1.15, 1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91]),
            slot_value[list(p)],
        )) for p in list(refined.keys())[:15]])
        ys = np.array([actual_runs] + [refined[p] for p in list(refined.keys())[:15]])
        # simple mean scale
        scale = float(np.mean(ys / np.clip(xs, 1e-6, None)))
    else:
        scale = 1.0
    mean_runs = mean_fast * scale
    median_runs = median_fast * scale
    # Keep distribution summary coherent with refined endpoints
    lo, hi = min(worst_runs, best_runs), max(worst_runs, best_runs)
    mean_runs = float(min(max(mean_runs, lo), hi))
    median_runs = float(min(max(median_runs, lo), hi))

    near = []
    for perm, runs in sorted(refined.items(), key=lambda x: -x[1]):
        if best_runs - runs <= equivalence_eps + 1e-12:
            near.append({
                "order_ids": [player_ids[i] for i in perm],
                "order_indices": list(perm),
                "expected_runs": float(runs),
                "gap_from_best": float(best_runs - runs),
            })
    near = near[:top_k_near]

    # Estimate near-optimal counts at 0.01 and 0.02 runs/game bands
    frac_near = sum(
        1 for r in refined.values() if best_runs - r <= equivalence_eps
    ) / max(len(refined), 1)
    if HAS_NUMBA:
        fast_eps = equivalence_eps / max(scale, 1e-6)
        n_near_opt = int((scores >= scores.max() - fast_eps).sum())
        n_near_01 = int((scores >= scores.max() - (0.01 / max(scale, 1e-6))).sum())
        n_near_02 = int((scores >= scores.max() - (0.02 / max(scale, 1e-6))).sum())
    else:
        n_near_opt = max(1, int(frac_near * n_perms))
        n_near_01 = n_near_opt
        n_near_02 = n_near_opt

    percentile = 100.0 * (1.0 - (rank - 1) / n_perms)
    gap = float(best_runs - actual_runs)
    spread = float(best_runs - worst_runs)
    value_vs_median = float(actual_runs - median_runs)
    practical = (
        f"Rank {rank:,}/{n_perms:,} can look dramatic while the expected-run gap "
        f"is only {gap:.3f}. "
        f"{n_near_01:,} orders ({100.0 * n_near_01 / n_perms:.1f}%) are within "
        f"0.01 runs/game of optimal; {n_near_02:,} ({100.0 * n_near_02 / n_perms:.1f}%) "
        f"within 0.02."
    )

    return OptimizeResult(
        actual_order=list(actual_idx),
        actual_runs=float(actual_runs),
        best_order=list(best_perm),
        best_runs=float(best_runs),
        worst_order=list(worst_perm),
        worst_runs=float(worst_runs),
        mean_runs=float(mean_runs),
        median_runs=float(median_runs),
        rank=int(rank),
        n_perms=n_perms,
        percentile=float(percentile),
        gap=gap,
        ordering_value=float(actual_runs - mean_runs),
        value_vs_median=value_vs_median,
        best_worst_spread=spread,
        near_optimal=near,
        n_near_optimal=int(n_near_opt),
        n_near_optimal_01=int(n_near_01),
        n_near_optimal_02=int(n_near_02),
        pct_near_optimal_01=float(100.0 * n_near_01 / n_perms),
        pct_near_optimal_02=float(100.0 * n_near_02 / n_perms),
        equivalence_eps=equivalence_eps,
        runs_distribution_summary={
            "min": float(worst_runs),
            "max": float(best_runs),
            "mean": float(mean_runs),
            "median": float(median_runs),
            "actual": float(actual_runs),
            "spread": spread,
        },
        method=method,
        operationally_equivalent=gap <= equivalence_eps,
        practical_note=practical,
    )


# Personnel-level cache helper used by precompute: optimize once per personnel+context
# and look up any order.

def optimize_personnel_context(
    player_ids: list[int],
    probs: np.ndarray,
    equivalence_eps: float = 0.02,
    trans: TransitionModel | None = None,
    engine: LineupEngine | None = None,
) -> dict:
    """Return exhaustive fast scores + metadata for all orders of these nine.

    Heavy object: scores array (362880,). Caller should extract needed ranks
    then discard.
    """
    trans = trans or load_transitions()
    engine = engine or LineupEngine(trans)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    R, P, E = _build_kernels_for_players(probs, trans)
    slot_value, seq_bonus, lead_extra = _fast_features(R, P, E)
    if not HAS_NUMBA:
        raise RuntimeError("Numba required for personnel exhaustive cache")
    scores = _score_all_perms_fast(slot_value, seq_bonus, lead_extra)
    best_pi = int(scores.argmax())
    worst_pi = int(scores.argmin())
    best_arr = np.empty(9, dtype=np.int64)
    worst_arr = np.empty(9, dtype=np.int64)
    _perm_from_index(best_pi, best_arr)
    _perm_from_index(worst_pi, worst_arr)
    best_perm = tuple(int(x) for x in best_arr)
    worst_perm = tuple(int(x) for x in worst_arr)

    def full(perm):
        return float(engine.expected_runs(probs[list(perm)]))

    # refine top candidates
    top_idx = np.argpartition(-scores, 30)[:30]
    cand = {best_perm, worst_perm}
    tmp = np.empty(9, dtype=np.int64)
    for pi in top_idx:
        _perm_from_index(int(pi), tmp)
        cand.add(tuple(int(x) for x in tmp))
    refined = {p: full(p) for p in cand}
    loc_best, loc_sc = _local_search_full(engine, probs, list(max(refined, key=refined.get)), restarts=1)
    refined[tuple(loc_best)] = loc_sc
    best_perm = max(refined, key=refined.get)
    worst_perm = min(refined, key=refined.get)

    return {
        "scores": scores,
        "slot_value": slot_value,
        "best_perm": best_perm,
        "worst_perm": worst_perm,
        "best_runs": refined[best_perm],
        "worst_runs": refined[worst_perm],
        "mean_fast": float(scores.mean()),
        "median_fast": float(np.median(scores)),
        "refined": refined,
        "probs": probs,
        "engine": engine,
        "equivalence_eps": equivalence_eps,
    }


def lookup_order(personnel_cache: dict, order_idx: tuple[int, ...]) -> dict:
    scores = personnel_cache["scores"]
    pi = _index_of_perm(order_idx)
    actual_fast = float(scores[pi])
    rank = int((scores > actual_fast).sum()) + 1
    n_perms = 362880
    engine = personnel_cache["engine"]
    probs = personnel_cache["probs"]
    actual_runs = float(engine.expected_runs(probs[list(order_idx)]))
    best_runs = float(personnel_cache["best_runs"])
    worst_runs = float(personnel_cache["worst_runs"])
    # calibrate mean
    best_perm = personnel_cache["best_perm"]
    best_fast = float(scores[int(scores.argmax())])
    scale = best_runs / best_fast if abs(best_fast) > 1e-12 else 1.0
    mean_runs = float(personnel_cache["mean_fast"] * scale)
    eps = personnel_cache["equivalence_eps"]
    fast_eps = eps / max(scale, 1e-6)
    n_near = int((scores >= scores.max() - fast_eps).sum())
    n_near_01 = int((scores >= scores.max() - (0.01 / max(scale, 1e-6))).sum())
    n_near_02 = int((scores >= scores.max() - (0.02 / max(scale, 1e-6))).sum())
    gap = best_runs - actual_runs
    median_runs = float(personnel_cache["median_fast"] * scale)
    return {
        "actual_runs": actual_runs,
        "best_runs": best_runs,
        "worst_runs": worst_runs,
        "mean_runs": mean_runs,
        "median_runs": median_runs,
        "rank": rank,
        "n_perms": n_perms,
        "percentile": 100.0 * (1.0 - (rank - 1) / n_perms),
        "gap": gap,
        "ordering_value": actual_runs - mean_runs,
        "value_vs_median": actual_runs - median_runs,
        "best_worst_spread": best_runs - worst_runs,
        "n_near_optimal": n_near,
        "n_near_optimal_01": n_near_01,
        "n_near_optimal_02": n_near_02,
        "pct_near_optimal_01": 100.0 * n_near_01 / n_perms,
        "pct_near_optimal_02": 100.0 * n_near_02 / n_perms,
        "operationally_equivalent": gap <= eps,
        "best_order": list(best_perm),
        "worst_order": list(personnel_cache["worst_perm"]),
        "method": "personnel_cache_fast_exhaustive",
    }
