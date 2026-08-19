"""Generate model_cards.json for the Research page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import settings


def build_model_cards() -> dict[str, Any]:
    cards = [
        {
            "id": "markov_expected_runs",
            "name": "Markov expected-run engine",
            "role": "Optimization currency and same-nine ordering evaluation",
            "inputs": [
                "Player PA outcome probabilities (neutral / vs RHP / vs LHP)",
                "Empirical base-out transition distributions",
            ],
            "outputs": [
                "Expected runs per 9 innings",
                "Expected PA by batting slot",
            ],
            "limitations": [
                "Does not model mid-game substitutions or intentional walks as decisions",
                "Single-game observed runs are high-variance; validate on aggregates",
            ],
            "validation": "See markov_validation.json (out-of-time predicted vs actual R/G)",
        },
        {
            "id": "same_nine_optimizer",
            "name": "Same-nine batting-order optimizer",
            "role": "Exhaustive ranking of 9! orders for fixed personnel",
            "inputs": ["Nine player IDs", "PA probability matrix for a platoon context"],
            "outputs": [
                "Best / worst / median expected runs",
                "Rank, percentile, optimization gap",
                "Near-optimal counts within 0.01 and 0.02 runs/game",
            ],
            "limitations": [
                "Percentile can look dramatic when thousands of orders are nearly equivalent",
                "Fast exhaustive scores are refined with full Markov evaluations at endpoints",
            ],
            "validation": "Correlation checks between fast ranks and full-engine ranks in tests",
        },
        {
            "id": "pa_probability_store",
            "name": "PA outcome probability store",
            "role": "Batter talent / platoon skill inputs to the Markov engine",
            "inputs": ["Historical plate appearances with outcome classes"],
            "outputs": ["7-class outcome probabilities per player and context"],
            "limitations": [
                "Small-sample players shrink toward league averages",
                "Does not encode park or pitcher identity beyond hand splits",
            ],
        },
        {
            "id": "interaction_models",
            "name": "Residual interaction / chemistry models",
            "role": "Test whether adjacent-hitter identity improves prediction after talent/state controls",
            "inputs": ["Adjacent PA frame with previous batter, state, pitcher hand"],
            "outputs": [
                "Out-of-sample log-loss / RMSE lift vs baseline models",
                "Empirical-Bayes pair associations (exploratory)",
            ],
            "limitations": [
                "Pair associations are not causal",
                "One temporal split is insufficient; prefer multi-fold rolling validation",
            ],
            "validation": "incremental_predictive_value.json and multi-fold interaction folds",
        },
        {
            "id": "offensive_profile_groups",
            "name": "Exploratory offensive profile groups",
            "role": "Descriptive clustering of hitter skill profiles",
            "inputs": ["Season batting skill features"],
            "outputs": ["Cluster labels and pair matrices"],
            "limitations": [
                "Low silhouette implies weak natural separation",
                "Treat as exploratory unless assignment stability is validated",
            ],
        },
    ]

    payload = {
        "available": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "season": settings.target_season,
        "cards": cards,
        "notes": [
            "Expected runs remain the optimization currency for same-nine comparisons.",
            "Cross-team ranking should prioritize ordering gaps and near-optimal rates over raw expected runs.",
        ],
    }

    research_dir = settings.artifacts_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        settings.artifacts_dir / "model_cards.json",
        research_dir / "model_cards.json",
        settings.models_dir / "model_cards.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["written"] = [
        str(settings.artifacts_dir / "model_cards.json"),
        str(research_dir / "model_cards.json"),
        str(settings.models_dir / "model_cards.json"),
    ]
    return payload


def main() -> None:
    print(json.dumps(build_model_cards(), indent=2))


if __name__ == "__main__":
    main()
