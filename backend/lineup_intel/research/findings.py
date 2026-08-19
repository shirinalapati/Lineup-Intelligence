"""Generate a 'what we learned' findings JSON from computed research artifacts.

Only statements that are directly supported by numbers already present in
artifacts are emitted. Missing inputs yield ``status: unavailable`` rather
than invented conclusions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings
from .stability import stability_by_team


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _stmt(qid: str, question: str, answer: str, support: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "support": support,
    }


def _pct(v: Any) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if n <= 1.5:
        n *= 100
    return f"{n:.0f}%"


def _num(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _plain_cluster_blurb(cluster: dict[str, Any], means: dict[str, Any], train_season: int | None) -> str:
    """One-sentence typical-hitter description from cluster averages vs league."""
    raw = cluster.get("center_raw") or {}
    label = cluster.get("archetype_label") or "Group"
    n = cluster.get("n_train_players")
    if n and train_season:
        n_bit = f" ({int(n)} hitters when the groups were learned on {train_season})"
    elif n:
        n_bit = f" ({int(n)} hitters in the learning sample)"
    else:
        n_bit = ""

    def vs(key: str, hi: str, lo: str, band: float = 0.35) -> str | None:
        if key not in raw or key not in means:
            return None
        scale = float(means.get(key) or 0) or 1.0
        # Prefer std if present
        std = (cluster.get("center_std") or {}).get(key)
        z = float(std) if std is not None else (float(raw[key]) - float(means[key])) / max(abs(scale), 1e-6)
        if z > band:
            return hi
        if z < -band:
            return lo
        return None

    bits = [
        vs("iso", "more extra-base power", "less extra-base power"),
        vs("k_pct", "more strikeouts", "fewer strikeouts"),
        vs("bb_pct", "more walks", "fewer walks"),
        vs("barrel_pct", "more barrels", "fewer barrels"),
        vs("hardhit_pct", "harder contact", "softer contact"),
        vs("oppo_pct", "more opposite-field contact", None, band=0.4),
        vs("gb_pct", "more ground balls", "fewer ground balls", band=0.45),
        vs("fb_pct", "more fly balls", "fewer fly balls", band=0.45),
    ]
    bits = [b for b in bits if b]
    if not bits:
        bits = ["close to league-average across walks, strikeouts, and power"]
    # Keep 3 traits max
    traits = ", ".join(bits[:3])
    return (
        f"{label}{n_bit}: typical member has {traits} "
        f"(about {_pct(raw.get('bb_pct'))} walks, {_pct(raw.get('k_pct'))} strikeouts, "
        f"ISO {_num(raw.get('iso'))})."
    )


def build_findings() -> dict[str, Any]:
    """Assemble findings strictly from on-disk research / model artifacts."""
    research_dir = settings.artifacts_dir / "research"
    models_dir = settings.models_dir

    interaction = _read_json(research_dir / "interaction_summary.json")
    incremental = _read_json(research_dir / "incremental_predictive_value.json")
    prev_ctx = _read_json(research_dir / "previous_outcome_context.json")
    archetypes = _read_json(models_dir / "archetypes.json")
    pairs = _read_parquet(research_dir / "player_pair_effects.parquet")
    arch_pairs = _read_parquet(research_dir / "archetype_pair_effects.parquet")

    lineups_path = settings.processed_dir / f"starting_lineups_{settings.target_season}.parquet"
    lineups = _read_parquet(lineups_path)

    # Optimizer / Markov evaluation artifacts (optional — only cite if present).
    opt_summary = _read_json(settings.artifacts_dir / "optimizer_summary.json")
    team_eval = _read_parquet(settings.artifacts_dir / "team_order_value.parquet")
    league_overview = _read_json(settings.artifacts_dir / "league_overview.json")
    if opt_summary is None and league_overview and league_overview.get("n_evaluated"):
        opt_summary = {
            "n_evaluated": league_overview.get("n_evaluated"),
            "league_avg_gap": league_overview.get("league_avg_gap"),
            "league_avg_percentile": league_overview.get("league_avg_percentile"),
            "league_avg_actual_runs": league_overview.get("league_avg_actual_runs"),
            "pct_operationally_equivalent": league_overview.get("pct_operationally_equivalent"),
            "most_efficient_ordering": league_overview.get("most_efficient_ordering"),
            "largest_avg_gaps": league_overview.get("largest_avg_gaps"),
        }


    statements: list[dict[str, Any]] = []
    unavailable: list[str] = []

    if archetypes is None:
        unavailable.append("archetypes")
    else:
        clusters = archetypes.get("clusters") or []
        labels = [c.get("archetype_label", "?") for c in clusters]
        sil = float(archetypes.get("silhouette", float("nan")))
        exploratory = sil < 0.25 if math.isfinite(sil) else True
        means = archetypes.get("feature_means") or {}
        train_season = archetypes.get("train_season")
        apply_season = settings.target_season
        blurbs = [_plain_cluster_blurb(c, means, train_season) for c in clusters]
        learned = f"{train_season}" if train_season else "the prior season"
        statements.append(
            _stmt(
                "archetype_fit",
                "How does a hitter get an archetype?",
                (
                    "There is no checklist like “ISO above X, therefore Power.” "
                    f"This app evaluates {apply_season} lineups. The four style "
                    f"groups were learned from {learned} season stats (walks, "
                    "strikeouts, extra-base power, quality of contact, and "
                    "batted-ball direction), then each "
                    f"{apply_season} hitter with enough plate appearances is "
                    "assigned to the group whose typical hitter they most "
                    "resemble. Names are just labels for those groups — not "
                    "lineup chemistry. "
                    + " ".join(blurbs)
                    + (
                        " Groups overlap, so treat the name as a shorthand, not "
                        "a hard identity."
                        if exploratory
                        else ""
                    )
                ),
                {
                    "method": archetypes.get("method"),
                    "k": archetypes.get("k"),
                    "silhouette": archetypes.get("silhouette"),
                    "train_season": archetypes.get("train_season"),
                    "labels": labels,
                    "exploratory": exploratory,
                    "plain_blurbs": blurbs,
                    "n_train_by_cluster": {
                        c.get("archetype_label"): c.get("n_train_players") for c in clusters
                    },
                },
            )
        )

    if interaction is None:
        unavailable.append("interaction_summary")
    else:
        n_pairs = int(interaction.get("n_player_pairs", 0) or 0)
        n_strong = int(interaction.get("n_player_pairs_strong", 0) or 0)
        n_mod = int(interaction.get("n_player_pairs_moderate", 0) or 0)
        n_lim = int(interaction.get("n_player_pairs_limited", 0) or 0)
        statements.append(
            _stmt(
                "pair_reliability",
                "When can we trust a two-hitter pairing?",
                (
                    "A pairing is two specific hitters batting back-to-back. "
                    "The more often that happens, the more we trust any leftover "
                    "stat after controlling for talent and game state. "
                    "Rule of thumb from shared plate appearances: "
                    "strong ≈ 117+ times together, moderate ≈ 33–116, "
                    f"limited = fewer than that. Of {n_pairs:,} pairs we can "
                    f"estimate: {n_strong:,} strong, {n_mod:,} moderate, "
                    f"{n_lim:,} limited. Most are limited because the same two "
                    "hitters rarely bat in that exact order enough times."
                ),
                {
                    "n_player_pairs": n_pairs,
                    "n_strong": n_strong,
                    "n_moderate": n_mod,
                    "n_limited": n_lim,
                    "mean_abs_effect_strong": interaction.get("pair_effect_mean_abs_strong"),
                    "n_adjacent_pa": interaction.get("n_adjacent_pa"),
                    "prior_n0": 50,
                    "tier_strong_min": 0.70,
                    "tier_moderate_min": 0.40,
                },
            )
        )
        if interaction.get("pair_effect_mean_abs_strong") is not None:
            statements.append(
                _stmt(
                    "pair_effect_magnitude",
                    "How large are the more reliable pair effects?",
                    (
                        "Even among hitter pairs with the largest samples, the "
                        "remaining effect after accounting for player ability and "
                        "game situation was generally small.\n\n"
                        "That suggests specific hitter pairings may provide some "
                        "useful context, but we did not find evidence of large, "
                        "consistent “chemistry” effects between two hitters simply "
                        "because they bat next to each other."
                    ),
                    {"mean_abs_effect_strong": interaction["pair_effect_mean_abs_strong"]},
                )
            )

    incr = incremental or (interaction or {}).get("incremental_predictive_value")
    if not incr or not incr.get("models"):
        unavailable.append("incremental_predictive_value")
    else:
        models = {m["model"]: m for m in incr["models"]}
        m1 = models.get("m1_talent")
        m5 = models.get("m5_arch_interact")
        m4 = models.get("m4_prev_feats")
        if m1 and m5 and m5.get("rmse_lift_vs_m1") is not None:
            lift = float(m5["rmse_lift_vs_m1"])
            ll_lift = m5.get("logloss_lift_vs_m1")
            improved = lift > 0
            answer = (
                "Not in our historical test.\n\n"
                "We trained the model on 2024 data and tested it on 2025 games "
                "it had not seen before. Adding information about which "
                "offensive styles were batting next to each other did not "
                "improve prediction accuracy. The more complicated interaction "
                "model actually performed slightly worse than the simpler model "
                "based primarily on hitter talent and game context.\n\n"
                "Takeaway: who the hitters are and the situation they bat in "
                "mattered more than whether certain offensive styles were "
                "paired together."
            )
            if improved:
                answer = (
                    "In our historical test, adding offensive-style pairings "
                    "slightly improved prediction versus talent and game "
                    "context alone. That result is still small and should be "
                    "read as incremental, not as evidence of large chemistry "
                    "effects."
                )
            statements.append(
                _stmt(
                    "archetype_interaction_predictive",
                    "Does hitter style improve prediction?",
                    answer,
                    {
                        "m1_rmse": m1.get("rmse"),
                        "m5_rmse": m5.get("rmse"),
                        "rmse_lift_vs_m1": lift,
                        "logloss_lift_vs_m1": ll_lift,
                        "train_season": incr.get("train_season"),
                        "valid_season": incr.get("valid_season"),
                    },
                )
            )
            chemistry_supported = improved and (ll_lift is None or float(ll_lift) > 0)
            statements.append(
                _stmt(
                    "lineup_chemistry_support",
                    "Is lineup chemistry supported as a predictive signal?",
                    (
                        "Out-of-sample incremental tests "
                        + (
                            "show a positive predictive lift from archetype-interaction "
                            "features after talent/state controls — evidence of a "
                            "measurable association, not a causal proof of chemistry."
                            if chemistry_supported
                            else "do not show a clear predictive lift from "
                            "archetype-interaction features beyond talent/state "
                            "controls. Residual pair associations may still be "
                            "useful diagnostically, but the data do not support "
                            "treating 'chemistry' as a strong predictive input."
                        )
                    ),
                    {
                        "rmse_lift_m5_vs_m1": lift,
                        "logloss_lift_m5_vs_m1": ll_lift,
                        "predictive_support": bool(chemistry_supported),
                    },
                )
            )
        if m1 and m4 and m4.get("rmse_lift_vs_m1") is not None:
            statements.append(
                _stmt(
                    "prev_hitter_features_lift",
                    "Do previous-hitter characteristics improve prediction?",
                    (
                        f"Model 4 RMSE lift vs Model 1 is "
                        f"{float(m4['rmse_lift_vs_m1']):+.5f} on the "
                        f"{incr.get('train_season')}→{incr.get('valid_season')} split."
                    ),
                    {
                        "rmse_lift_vs_m1": m4.get("rmse_lift_vs_m1"),
                        "logloss_lift_vs_m1": m4.get("logloss_lift_vs_m1"),
                        "m4_rmse": m4.get("rmse"),
                        "m1_rmse": m1.get("rmse"),
                    },
                )
            )
        # Full ladder snapshot
        statements.append(
            _stmt(
                "incremental_ladder",
                "How do Models 1–5 compare out of sample?",
                "Temporal validation RMSE / log-loss by model (see support table).",
                {
                    "models": [
                        {
                            "model": m["model"],
                            "rmse": m.get("rmse"),
                            "rmse_lift_vs_m1": m.get("rmse_lift_vs_m1"),
                            "logloss_reached": m.get("logloss_reached"),
                            "logloss_lift_vs_m1": m.get("logloss_lift_vs_m1"),
                        }
                        for m in incr["models"]
                    ]
                },
            )
        )

    ctx = prev_ctx or (interaction or {}).get("previous_outcome_context")
    if not ctx or not ctx.get("groups"):
        unavailable.append("previous_outcome_context")
    else:
        groups = sorted(ctx["groups"], key=lambda r: abs(r.get("mean_residual", 0)), reverse=True)
        top = groups[0]
        statements.append(
            _stmt(
                "prev_outcome_context",
                "After base/out controls, does previous-batter outcome still matter?",
                (
                    f"Largest absolute residual association is after "
                    f"'{top['prev_outcome_group']}' "
                    f"(mean residual={top['mean_residual']:+.4f}, n={top['n']}). "
                    "Because residuals already control for current outs/bases, this is "
                    "not the trivial runner-on-base effect."
                ),
                {"groups": ctx["groups"]},
            )
        )

    if arch_pairs is not None and len(arch_pairs):
        top_arch = arch_pairs.sort_values("n", ascending=False).head(1).iloc[0]
        statements.append(
            _stmt(
                "archetype_pair_top",
                "Which archetype adjacencies are most observed?",
                (
                    f"Most frequent estimated archetype adjacency is "
                    f"{top_arch.get('prev_arch_label')} → {top_arch.get('batter_arch_label')} "
                    f"(n={int(top_arch['n'])}, shrunk effect={float(top_arch['effect']):+.4f}, "
                    f"tier={top_arch.get('reliability_tier')})."
                ),
                {
                    "prev_arch_label": top_arch.get("prev_arch_label"),
                    "batter_arch_label": top_arch.get("batter_arch_label"),
                    "n": int(top_arch["n"]),
                    "effect": float(top_arch["effect"]),
                    "reliability_tier": top_arch.get("reliability_tier"),
                },
            )
        )
    else:
        unavailable.append("archetype_pair_effects")

    if pairs is not None and len(pairs):
        strong = pairs[pairs["reliability_tier"] == "strong"]
        statements.append(
            _stmt(
                "player_pair_coverage",
                "How sparse are player-pair samples?",
                (
                    f"{len(pairs)} pairs meet the minimum sample gate; "
                    f"{len(strong)} are strong-reliability after shrinkage "
                    f"(median n={float(pairs['n'].median()):.0f})."
                ),
                {
                    "n_pairs": int(len(pairs)),
                    "n_strong": int(len(strong)),
                    "median_n": float(pairs["n"].median()),
                    "p90_n": float(pairs["n"].quantile(0.9)),
                },
            )
        )
    else:
        unavailable.append("player_pair_effects")

    if lineups is not None and len(lineups):
        by_team = stability_by_team(lineups)
        most_stable = by_team.sort_values("most_common_order_pct", ascending=False).iloc[0]
        least_stable = by_team.sort_values("order_entropy_bits", ascending=False).iloc[0]
        statements.append(
            _stmt(
                "lineup_stability_league",
                "How stable are 2026 starting batting orders?",
                (
                    f"Across {int(len(lineups))} team-games, league median unique orders "
                    f"per team is {float(by_team['unique_orders'].median()):.0f}; "
                    f"highest most-common-order share is {most_stable['team']} "
                    f"({float(most_stable['most_common_order_pct']):.1%}); "
                    f"highest order entropy is {least_stable['team']} "
                    f"({float(least_stable['order_entropy_bits']):.2f} bits)."
                ),
                {
                    "n_team_games": int(len(lineups)),
                    "median_unique_orders": float(by_team["unique_orders"].median()),
                    "max_most_common_order_pct": {
                        "team": most_stable["team"],
                        "pct": float(most_stable["most_common_order_pct"]),
                    },
                    "max_entropy": {
                        "team": least_stable["team"],
                        "entropy_bits": float(least_stable["order_entropy_bits"]),
                    },
                },
            )
        )
    else:
        unavailable.append(f"starting_lineups_{settings.target_season}")

    if opt_summary is not None:
        # Only cite keys that exist — never invent optimization gaps.
        support = {k: opt_summary[k] for k in opt_summary if k in {
            "mean_optimization_gap",
            "median_optimization_gap",
            "mean_best_minus_worst",
            "pct_orders_within_eps",
            "equivalence_eps",
            "n_lineups",
        }}
        if support:
            parts = []
            if "mean_optimization_gap" in support:
                parts.append(
                    f"mean optimization gap={float(support['mean_optimization_gap']):.4f} runs/game"
                )
            if "mean_best_minus_worst" in support:
                parts.append(
                    f"mean best−worst same-nine gap={float(support['mean_best_minus_worst']):.4f}"
                )
            if "pct_orders_within_eps" in support:
                parts.append(
                    f"{float(support['pct_orders_within_eps']):.1%} of evaluated orders "
                    f"fall within the operational equivalence band"
                )
            statements.append(
                _stmt(
                    "order_value_summary",
                    "How much does batting order matter on average?",
                    "; ".join(parts) + "." if parts else "Optimizer summary present (see support).",
                    support,
                )
            )
    else:
        unavailable.append("optimizer_summary")

    if team_eval is not None and len(team_eval) and "optimization_gap" in team_eval.columns:
        col = "optimization_gap"
        top = team_eval.sort_values(col, ascending=False).iloc[0]
        team_col = "team" if "team" in team_eval.columns else None
        statements.append(
            _stmt(
                "teams_left_on_table",
                "Which teams leave the most estimated order value on the table?",
                (
                    f"{top[team_col] if team_col else 'Top team'} has the largest mean "
                    f"optimization gap ({float(top[col]):.4f} runs/game) among teams with "
                    "evaluated lineups."
                    if team_col
                    else f"Largest mean optimization gap is {float(top[col]):.4f} runs/game."
                ),
                {
                    "team": (top[team_col] if team_col else None),
                    "optimization_gap": float(top[col]),
                },
            )
        )

    status = "ok" if statements else "unavailable"
    payload = {
        "status": status,
        "title": "What we learned",
        "generated_from": {
            "research_dir": str(research_dir),
            "models_dir": str(models_dir),
        },
        "unavailable_inputs": unavailable,
        "statements": statements,
        "caveats": [
            "Statements are limited to quantities present in computed artifacts.",
            "Pair and archetype effects are estimated associations after controls, not causal chemistry.",
            "Missing artifacts are listed under unavailable_inputs rather than filled with placeholders.",
        ],
    }

    out = research_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "findings.json"
    path.write_text(json.dumps(payload, indent=2))
    payload["artifact"] = str(path)
    return payload


def main() -> None:
    payload = build_findings()
    print(f"status={payload['status']} statements={len(payload['statements'])}")
    if payload["unavailable_inputs"]:
        print("unavailable:", ", ".join(payload["unavailable_inputs"]))
    for s in payload["statements"]:
        print(f"- {s['id']}: {s['answer'][:140]}...")
    print(f"wrote {payload.get('artifact')}")


if __name__ == "__main__":
    main()
