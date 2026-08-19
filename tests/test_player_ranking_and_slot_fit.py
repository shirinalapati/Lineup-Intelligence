"""Tests for ranking framework and same-nine slot insertion."""

from __future__ import annotations

import numpy as np
import pytest

from lineup_intel.research.ranking import (
    MIN_PA_OVERALL,
    favorable_percentile,
    ordinal,
    rank_frame,
    rank_metric,
    rank_team_metrics,
)
from lineup_intel.research.player_slot_intelligence import move_player_to_slot
import pandas as pd


def test_ordinal():
    assert ordinal(1) == "1st"
    assert ordinal(2) == "2nd"
    assert ordinal(3) == "3rd"
    assert ordinal(11) == "11th"
    assert ordinal(42) == "42nd"


def test_higher_better_rank_and_percentile():
    peers = [0.300, 0.320, 0.350, 0.360, 0.400]
    r = rank_metric(0.360, peers, metric="woba", direction="higher_better", sample_size=200, qualifying_threshold=100)
    assert r.qualified
    assert r.rank == 2
    assert r.population_n == 5
    assert r.percentile == favorable_percentile(2, 5)


def test_lower_better_k_pct():
    peers = [0.30, 0.25, 0.20, 0.15, 0.10]
    r = rank_metric(0.10, peers, metric="k_pct", direction="lower_better", sample_size=200, qualifying_threshold=100)
    assert r.rank == 1
    assert r.percentile == 100


def test_limited_sample_no_rank():
    r = rank_metric(
        0.400,
        [0.3, 0.35, 0.4],
        metric="woba",
        sample_size=20,
        qualifying_threshold=MIN_PA_OVERALL,
    )
    assert r.value == 0.400
    assert not r.qualified
    assert r.rank is None
    assert "Limited sample" in (r.note or "")


def test_tied_ranks_competition():
    peers = [0.350, 0.350, 0.300]
    r = rank_metric(0.350, peers, metric="woba", direction="higher_better", qualified=True)
    assert r.rank == 1
    assert r.population_n == 3


def test_neutral_direction_not_quality():
    peers = [0.2, 0.4, 0.6]
    r = rank_metric(0.4, peers, metric="pull_pct", direction="neutral", qualified=True)
    assert r.note and "Descriptive" in r.note


def test_rank_frame_denominator():
    df = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "woba": [0.4, 0.35, 0.3, 0.45],
        "pa": [200, 200, 50, 200],
    })
    ranks = rank_frame(df, value_col="woba", metric="woba", sample_col="pa", min_sample=100)
    assert ranks[4].rank == 1
    assert ranks[4].population_n == 3  # player 3 excluded
    assert ranks[3].qualified is False
    assert ranks[3].rank is None


def test_move_player_preserves_personnel():
    order = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for slot in range(1, 10):
        placed = move_player_to_slot(order, 50, slot)
        assert set(placed) == set(order)
        assert placed[slot - 1] == 50
        # relative order of others preserved
        others = [p for p in placed if p != 50]
        assert others == [10, 20, 30, 40, 60, 70, 80, 90]


def test_move_player_deterministic():
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = move_player_to_slot(order, 3, 7)
    b = move_player_to_slot(order, 3, 7)
    assert a == b == [1, 2, 4, 5, 6, 7, 3, 8, 9]


def test_expected_pa_decreases_later_slots_smoke():
    """Aggregate Markov PA weights are higher early; smoke via engine if models present."""
    from pathlib import Path
    from lineup_intel.config import settings
    if not (settings.models_dir / "pa_probs_neutral.parquet").exists():
        pytest.skip("pa probs missing")
    from lineup_intel.engine.markov import LineupEngine
    from lineup_intel.engine.pa_probs import PAProbabilityStore

    pa = PAProbabilityStore()
    engine = LineupEngine()
    # League-average nine
    ids = [-1] * 9
    # Use nine distinct placeholders: store supports -1 only; duplicate ok for smoke
    probs = np.vstack([pa.probs_for(-1, "neutral")] * 9)
    ev = engine.evaluate(probs, with_flow=True)
    pas = list(ev.expected_pa_by_slot)
    # Slot 1 should get at least as many PA as slot 9 in aggregate
    assert pas[0] > pas[8]


def test_team_metric_ranks():
    teams = {
        "AAA": {
            "avg_gap": 0.01,
            "avg_percentile": 80,
            "avg_actual_runs": 5.0,
            "unique_orders": 10,
            "unique_personnel": 5,
        },
        "BBB": {
            "avg_gap": 0.05,
            "avg_percentile": 40,
            "avg_actual_runs": 4.0,
            "unique_orders": 50,
            "unique_personnel": 20,
        },
        "CCC": {
            "avg_gap": 0.03,
            "avg_percentile": 60,
            "avg_actual_runs": 4.5,
            "unique_orders": 30,
            "unique_personnel": 10,
        },
    }
    aaa = rank_team_metrics(teams, "AAA")
    bbb = rank_team_metrics(teams, "BBB")
    assert aaa["avg_gap"]["rank"] == 1
    assert aaa["avg_percentile"]["rank"] == 1
    assert aaa["avg_actual_runs"]["rank"] == 1
    assert aaa["unique_orders"]["rank"] == 3
    assert bbb["avg_gap"]["rank"] == 3
    assert bbb["unique_orders"]["rank"] == 1
    assert bbb["unique_personnel"]["rank"] == 1
    assert aaa["avg_gap"]["population_n"] == 3


def test_platoon_woba_metric_direction():
    from lineup_intel.research.ranking import METRIC_DIRECTION

    assert METRIC_DIRECTION["woba_vs_R"] == "higher_better"
    assert METRIC_DIRECTION["woba_vs_L"] == "higher_better"


def test_modeled_woba_table_and_platoon_ranks(tmp_path, monkeypatch):
    from lineup_intel.research import player_slot_intelligence as psi
    from lineup_intel.config import settings

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(settings, "models_dir", models)
    rows = []
    for i, (n, hr) in enumerate([(80, 0.05), (80, 0.02), (20, 0.10), (80, 0.08)]):
        rows.append({
            "player_id": i + 1,
            "n_pa": n,
            "K": 0.2,
            "BB_HBP": 0.08,
            "1B": 0.15,
            "2B": 0.04,
            "3B": 0.0,
            "HR": hr,
            "OUT_IP": 0.53 - hr,
        })
    pd.DataFrame(rows).to_parquet(models / "pa_probs_vs_R.parquet", index=False)
    pd.DataFrame(rows).to_parquet(models / "pa_probs_vs_L.parquet", index=False)
    lineups = pd.DataFrame([
        {"team": "PHI", "slot1": 1, "slot2": 2, "slot3": 3, "slot4": 4,
         "slot5": 1, "slot6": 1, "slot7": 1, "slot8": 1, "slot9": 1},
    ])
    ranks = psi.build_platoon_model_woba_ranks(lineups)
    r = ranks[4]["metrics"]["woba_vs_R"]["mlb"]
    assert r["qualified"]
    assert r["rank"] == 1
    assert r["population_n"] == 3  # player 3 has only 20 PA
    low = ranks[2]["metrics"]["woba_vs_R"]["mlb"]
    assert low["rank"] == 3
    unqualified = ranks[3]["metrics"]["woba_vs_R"]["mlb"]
    assert unqualified["qualified"] is False

