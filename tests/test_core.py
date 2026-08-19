"""Automated tests for MLB Lineup Intelligence core engine and data integrity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
ARTIFACTS = ROOT / "data" / "artifacts"


@pytest.fixture(scope="module")
def lineups():
    path = PROCESSED / "starting_lineups_2026.parquet"
    if not path.exists():
        pytest.skip("lineups not extracted")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def pa_store():
    from lineup_intel.engine.pa_probs import PAProbabilityStore
    if not (MODELS / "pa_probs_neutral.parquet").exists():
        pytest.skip("pa probs missing")
    return PAProbabilityStore()


@pytest.fixture(scope="module")
def engine():
    from lineup_intel.engine.markov import LineupEngine
    return LineupEngine()


# ── Data integrity ──────────────────────────────────────────────────────────

def test_thirty_teams(lineups):
    assert lineups["team"].nunique() == 30


def test_nine_unique_starters(lineups):
    slots = [f"slot{i}" for i in range(1, 10)]
    for _, row in lineups.sample(min(200, len(lineups)), random_state=0).iterrows():
        ids = [int(row[c]) for c in slots]
        assert len(ids) == 9
        assert len(set(ids)) == 9


def test_no_duplicate_game_team(lineups):
    dup = lineups.duplicated(subset=["game_pk", "team"]).sum()
    assert dup == 0


def test_two_lineups_per_game(lineups):
    counts = lineups.groupby("game_pk").size()
    # Most games should have exactly 2; allow rare exceptions but require majority
    assert (counts == 2).mean() > 0.95


# ── Transitions / Markov ────────────────────────────────────────────────────

def test_transition_probs_sum():
    from lineup_intel.engine.transitions import load_transitions, OUTCOME_CLASSES, N_STATE
    trans = load_transitions()
    for oi in range(len(OUTCOME_CLASSES)):
        for si in range(N_STATE):
            k = int(trans.n_trans[oi, si])
            s = float(trans.next_prob[oi, si, :k].sum())
            assert abs(s - 1.0) < 1e-4


def test_markov_reasonable_runs(engine, pa_store):
    # League-average-ish nine
    league = pa_store.probs_for(-1, "neutral")
    probs = np.vstack([league] * 9)
    er = engine.expected_runs(probs)
    assert 2.5 < er < 7.5


def test_better_hitters_score_more(engine, pa_store):
    league = pa_store.probs_for(-1, "neutral")
    good = league.copy()
    # shift outs to HR/BB
    from lineup_intel.engine.transitions import OUTCOME_CLASSES
    idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
    good[idx["OUT_IP"]] *= 0.7
    good[idx["HR"]] += 0.05
    good[idx["BB_HBP"]] += 0.05
    good = good / good.sum()
    weak = league.copy()
    weak[idx["K"]] += 0.1
    weak[idx["OUT_IP"]] += 0.05
    weak[idx["HR"]] = max(0.01, weak[idx["HR"]] - 0.03)
    weak = weak / weak.sum()
    good_lineup = np.vstack([good] * 9)
    weak_lineup = np.vstack([weak] * 9)
    assert engine.expected_runs(good_lineup) > engine.expected_runs(weak_lineup)


# ── Optimizer ───────────────────────────────────────────────────────────────

def test_permutation_count_and_same_nine(pa_store, engine):
    from lineup_intel.engine.optimizer import optimize_lineup
    # Use 9 real players from PA store
    import pandas as pd
    df = pd.read_parquet(MODELS / "pa_probs_neutral.parquet")
    ids = [int(x) for x in df[df.player_id > 0].nlargest(9, "n_pa")["player_id"].tolist()]
    assert len(ids) == 9
    probs = pa_store.probs_matrix(ids, "neutral")
    res = optimize_lineup(ids, probs, actual_order_ids=ids, engine=engine)
    assert res.n_perms == 362880
    assert set(res.best_order) == set(range(9))
    assert res.best_runs + 1e-9 >= res.actual_runs
    assert 0 <= res.percentile <= 100


def test_optimizer_deterministic(pa_store, engine):
    from lineup_intel.engine.optimizer import optimize_lineup
    import pandas as pd
    df = pd.read_parquet(MODELS / "pa_probs_neutral.parquet")
    ids = [int(x) for x in df[df.player_id > 0].nlargest(9, "n_pa")["player_id"].tolist()]
    probs = pa_store.probs_matrix(ids, "neutral")
    a = optimize_lineup(ids, probs, actual_order_ids=ids, engine=engine)
    b = optimize_lineup(ids, probs, actual_order_ids=ids, engine=engine)
    assert abs(a.actual_runs - b.actual_runs) < 1e-9
    assert a.rank == b.rank


# ── Simulation convergence ──────────────────────────────────────────────────

def test_monte_carlo_converges_toward_markov(engine, pa_store):
    from lineup_intel.engine.simulate import simulate_lineup
    league = pa_store.probs_for(-1, "neutral")
    probs = np.vstack([league] * 9)
    det = engine.expected_runs(probs)
    sim = simulate_lineup(probs, n_games=3000, seed=7, deterministic_expected=det)
    assert abs(sim.mean - det) < 0.35  # sampling noise band


# ── Interactions shrinkage ──────────────────────────────────────────────────

def test_interaction_artifacts_honest():
    path = ARTIFACTS / "research" / "incremental_predictive_value.json"
    if not path.exists():
        pytest.skip("research artifacts missing")
    import json
    data = json.loads(path.read_text())
    assert "models" in data or "results" in data or "available" in data


def test_small_samples_shrink(pa_store):
    # League fallback for unknown player should equal league row
    unknown = pa_store.probs_for(999999999, "neutral")
    league = pa_store.probs_for(-1, "neutral")
    assert np.allclose(unknown, league)


# ── Identity ────────────────────────────────────────────────────────────────

def test_order_vs_personnel_identity():
    from lineup_intel.identity import order_id, personnel_id
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert order_id(a) != order_id(b)
    assert personnel_id(a) == personnel_id(b)


def test_simulate_quantiles_and_agrees_with_markov(engine, pa_store):
    from lineup_intel.engine.simulate import simulate_lineup

    league = pa_store.probs_for(-1, "neutral")
    probs = np.vstack([league] * 9)
    markov = engine.expected_runs(probs)
    sim = simulate_lineup(
        probs, n_games=4000, seed=7, deterministic_expected=markov
    )
    assert sim.p95 >= sim.p05
    assert sim.p05 <= sim.median <= sim.p95
    se = sim.std / np.sqrt(sim.n_games)
    assert abs(sim.mean - markov) < max(0.20, 4.0 * se)


def test_lineup_flow_has_baseball_metrics(engine, pa_store):
    from lineup_intel.engine.markov import lineup_breakdown_narrative

    league = pa_store.probs_for(-1, "neutral")
    probs = np.vstack([league] * 9)
    ev = engine.evaluate(probs, with_flow=True)
    assert ev.lineup_flow is not None and len(ev.lineup_flow) == 9
    row = ev.lineup_flow[0]
    assert 3.5 < row["expected_pa"] < 5.5
    assert 0 <= row["runners_on_pct"] <= 1
    assert 0 <= row["risp_pct"] <= 1
    assert row["avg_runners_on"] >= 0
    summary, notes = lineup_breakdown_narrative(ev)
    assert "projects" in summary
    assert 2 <= len(notes) <= 4
