"""Out-of-time validation for the Markov expected-run engine.

Separates noisy single-game observed runs from aggregate expected-run calibration.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..engine.markov import LineupEngine
from ..engine.pa_probs import PAProbabilityStore
from ..teams import normalize_abbrev

SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def _holdout_mask(dates: pd.Series, holdout_frac: float = 0.25) -> pd.Series:
    """Last chronological fraction of dates as holdout."""
    uniq = sorted(dates.dropna().astype(str).unique())
    if not uniq:
        return pd.Series([False] * len(dates), index=dates.index)
    cut = uniq[max(0, int(len(uniq) * (1.0 - holdout_frac)))]
    return dates.astype(str) >= cut


def build_markov_validation(season: int | None = None) -> dict[str, Any]:
    season = season or settings.target_season
    lu_path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    if not lu_path.exists():
        return {"available": False, "reason": f"missing {lu_path.name}"}

    needed = [settings.models_dir / f"pa_probs_{c}.parquet" for c in ("neutral", "vs_R", "vs_L")]
    if any(not p.exists() for p in needed):
        return {"available": False, "reason": "pa_probs artifacts missing"}

    lu = pd.read_parquet(lu_path)
    lu = lu[lu["runs_scored"].notna()].copy()
    if lu.empty:
        return {"available": False, "reason": "no lineups with observed runs"}

    holdout = _holdout_mask(lu["game_date"])
    test = lu.loc[holdout].copy()
    if len(test) < 50:
        test = lu.copy()
        holdout_label = "full_season_fallback"
    else:
        holdout_label = f"last_25pct_dates_from_{test['game_date'].min()}"

    pa = PAProbabilityStore(settings.models_dir)
    engine = LineupEngine()

    preds = []
    actuals = []
    teams = []
    for _, row in test.iterrows():
        try:
            ids = [int(row[c]) for c in SLOT_COLS]
        except Exception:
            continue
        hand = str(row.get("opp_sp_hand") or "").upper()
        ctx = "vs_L" if hand.startswith("L") else ("vs_R" if hand.startswith("R") else "neutral")
        try:
            probs = pa.probs_matrix(ids, ctx)
            er = float(engine.expected_runs(probs))
        except Exception:
            continue
        preds.append(er)
        actuals.append(float(row["runs_scored"]))
        teams.append(normalize_abbrev(row["team"]) or str(row["team"]))

    if len(preds) < 30:
        return {"available": False, "reason": "insufficient scored holdout rows"}

    yhat = np.asarray(preds, dtype=float)
    y = np.asarray(actuals, dtype=float)
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Aggregate to team-game is already the row; also team-level means
    df = pd.DataFrame({"pred": yhat, "actual": y, "team": teams})
    by_team = (
        df.groupby("team")
        .agg(pred_rg=("pred", "mean"), actual_rg=("actual", "mean"), n=("actual", "size"))
        .reset_index()
    )
    team_mae = float(np.mean(np.abs(by_team["pred_rg"] - by_team["actual_rg"])))

    # Calibration by predicted-run buckets
    bins = np.quantile(yhat, np.linspace(0, 1, 6))
    bins = np.unique(bins)
    if len(bins) < 3:
        buckets = []
    else:
        cats = pd.cut(yhat, bins=bins, include_lowest=True)
        buckets = []
        for cat, g in df.groupby(cats, observed=False):
            if len(g) == 0:
                continue
            buckets.append({
                "bucket": str(cat),
                "n": int(len(g)),
                "mean_predicted": float(g["pred"].mean()),
                "mean_actual": float(g["actual"].mean()),
            })

    payload = {
        "available": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "season": season,
        "holdout": holdout_label,
        "n_team_games": int(len(df)),
        "aggregate": {
            "mean_predicted_rg": float(yhat.mean()),
            "mean_actual_rg": float(y.mean()),
            "bias_predicted_minus_actual": float(yhat.mean() - y.mean()),
            "mae": mae,
            "rmse": rmse,
            "corr": float(np.corrcoef(yhat, y)[0, 1]) if yhat.std() > 0 and y.std() > 0 else None,
        },
        "team_level": {
            "n_teams": int(len(by_team)),
            "mae_of_team_means": team_mae,
            "teams": by_team.sort_values("team").to_dict(orient="records"),
        },
        "calibration_by_predicted_bucket": buckets,
        "notes": [
            "Single-game observed runs are noisy; prefer aggregate MAE/RMSE and team-level means.",
            "This validates the expected-run engine as a level/calibration check, not lineup-order ranking alone.",
        ],
    }

    out = settings.artifacts_dir / "research" / "markov_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["written"] = str(out)
    return payload


def main() -> None:
    print(json.dumps(build_markov_validation(), indent=2, default=str))


if __name__ == "__main__":
    main()
