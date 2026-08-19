"""Empirical-Bayes player PA outcome probabilities by platoon context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from .transitions import OUTCOME_CLASSES

# Prior strength (pseudo-counts) for EB shrinkage toward league
PRIOR_PA = 200.0


def _rates_from_counts(counts: pd.Series, prior: pd.Series, prior_n: float = PRIOR_PA) -> np.ndarray:
    """Shrink observed multinomial counts toward prior rates."""
    obs_n = float(counts.sum())
    prior_rates = prior.values.astype(np.float64)
    prior_rates = prior_rates / prior_rates.sum()
    if obs_n <= 0:
        return prior_rates
    post = counts.values.astype(np.float64) + prior_n * prior_rates
    return post / post.sum()


def build_pa_probability_tables(
    pa_path: Path | None = None,
    train_seasons: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build player outcome probability tables.

    Returns dict of dataframes keyed by context: neutral, vs_R, vs_L.
    Each has columns: player_id, n_pa, + one col per outcome class.
    """
    pa_path = pa_path or (settings.processed_dir / "plate_appearances.parquet")
    train_seasons = train_seasons or list(settings.train_seasons)
    df = pd.read_parquet(pa_path)
    df = df[df["season"].isin(train_seasons)].copy()
    df = df[df["outcome_class"].isin(OUTCOME_CLASSES)]

    league_counts = df["outcome_class"].value_counts().reindex(OUTCOME_CLASSES, fill_value=0)
    league_rates = league_counts / league_counts.sum()

    def table_for(mask: pd.Series) -> pd.DataFrame:
        sub = df.loc[mask]
        # league prior within this context
        ctx_counts = sub["outcome_class"].value_counts().reindex(OUTCOME_CLASSES, fill_value=0)
        ctx_rates = ctx_counts / max(ctx_counts.sum(), 1)
        rows = []
        for pid, g in sub.groupby("batter_id"):
            counts = g["outcome_class"].value_counts().reindex(OUTCOME_CLASSES, fill_value=0)
            rates = _rates_from_counts(counts, ctx_rates)
            row = {"player_id": int(pid), "n_pa": int(counts.sum())}
            for i, c in enumerate(OUTCOME_CLASSES):
                row[c] = float(rates[i])
            rows.append(row)
        # Always include league fallback row
        league_row = {"player_id": -1, "n_pa": int(ctx_counts.sum())}
        for i, c in enumerate(OUTCOME_CLASSES):
            league_row[c] = float(ctx_rates.iloc[i]) if hasattr(ctx_rates, "iloc") else float(ctx_rates[i])
        # ctx_rates may be Series
        for c in OUTCOME_CLASSES:
            league_row[c] = float(ctx_rates[c]) if isinstance(ctx_rates, pd.Series) else float(ctx_rates[OUTCOME_CLASSES.index(c)])
        rows.append(league_row)
        return pd.DataFrame(rows)

    tables = {
        "neutral": table_for(pd.Series(True, index=df.index)),
        "vs_R": table_for(df["pitcher_hand"] == "R"),
        "vs_L": table_for(df["pitcher_hand"] == "L"),
    }
    out = settings.models_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_parquet(out / f"pa_probs_{name}.parquet", index=False)
    meta = {
        "train_seasons": train_seasons,
        "prior_pa": PRIOR_PA,
        "outcome_classes": OUTCOME_CLASSES,
        "league_rates": {c: float(league_rates[c]) for c in OUTCOME_CLASSES},
        "n_pa_train": int(len(df)),
    }
    import json
    (out / "pa_probs_meta.json").write_text(json.dumps(meta, indent=2))
    return tables


class PAProbabilityStore:
    def __init__(self, models_dir: Path | None = None):
        models_dir = models_dir or settings.models_dir
        self.tables: dict[str, dict[int, dict[str, float]]] = {}
        self.league: dict[str, dict[str, float]] = {}
        for ctx in ("neutral", "vs_R", "vs_L"):
            path = models_dir / f"pa_probs_{ctx}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_parquet(path)
            by_id: dict[int, dict[str, float]] = {}
            for rec in df.to_dict(orient="records"):
                pid = int(rec["player_id"])
                by_id[pid] = {
                    "n_pa": float(rec.get("n_pa", 0)),
                    **{c: float(rec[c]) for c in OUTCOME_CLASSES},
                }
            self.tables[ctx] = by_id
            if -1 not in by_id:
                raise KeyError(f"league fallback row (player_id=-1) missing in {path}")
            self.league[ctx] = by_id[-1]

    def probs_for(self, player_id: int, context: str = "neutral") -> np.ndarray:
        ctx = context if context in self.tables else "neutral"
        row = self.tables[ctx].get(int(player_id)) or self.league[ctx]
        return np.array([row[c] for c in OUTCOME_CLASSES], dtype=np.float64)

    def probs_matrix(self, player_ids: list[int], context: str = "neutral") -> np.ndarray:
        return np.vstack([self.probs_for(pid, context) for pid in player_ids])

    def resolve_context(self, opp_sp_hand: str | None) -> str:
        if opp_sp_hand == "R":
            return "vs_R"
        if opp_sp_hand == "L":
            return "vs_L"
        return "neutral"


if __name__ == "__main__":
    tables = build_pa_probability_tables()
    for k, v in tables.items():
        print(k, len(v), "players")
