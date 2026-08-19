"""Data-quality report after lineup extraction / rebuild."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings
from ..teams import CANONICAL_ABBREVS

SLOT_COLS = [f"slot{i}" for i in range(1, 10)]
# Known pure pitchers that must never appear as starting hitters (DH era).
KNOWN_PITCHER_IDS = {
    547973,  # Aroldis Chapman
}


def build_data_quality_report(season: int | None = None) -> dict[str, Any]:
    season = season or settings.target_season
    lu_path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    pl_path = settings.processed_dir / f"players_{season}.parquet"
    evals_path = settings.artifacts_dir / "lineup_evaluations.parquet"
    pairs_path = settings.artifacts_dir / "research" / "player_pair_effects.parquet"

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "season": season,
        "available": True,
    }

    if not lu_path.exists():
        report["available"] = False
        report["reason"] = f"missing {lu_path.name}"
        return report

    lu = pd.read_parquet(lu_path)
    players = pd.read_parquet(pl_path) if pl_path.exists() else pd.DataFrame()

    n = len(lu)
    invalid = []
    dup_players = []
    pitcher_as_hitter = []
    unresolved = []

    name_by_id = {}
    if len(players):
        name_by_id = {
            int(r["player_id"]): str(r.get("name") or r["player_id"])
            for r in players.to_dict(orient="records")
        }

    for _, row in lu.iterrows():
        try:
            ids = [int(row[c]) for c in SLOT_COLS]
        except Exception:
            invalid.append({"game_pk": int(row.get("game_pk") or 0), "team": row.get("team"), "reason": "slot_cast"})
            continue
        if len(ids) != 9 or len(set(ids)) != 9:
            dup_players.append({
                "game_pk": int(row["game_pk"]),
                "team": row["team"],
                "ids": ids,
            })
            invalid.append({
                "game_pk": int(row["game_pk"]),
                "team": row["team"],
                "reason": "not_nine_unique",
            })
        for pid in ids:
            if pid in KNOWN_PITCHER_IDS:
                pitcher_as_hitter.append({
                    "game_pk": int(row["game_pk"]),
                    "team": row["team"],
                    "player_id": pid,
                    "name": name_by_id.get(pid, str(pid)),
                })
            if pid not in name_by_id:
                unresolved.append(pid)

    # Position==P counts (exploratory; can mislabel two-way / utility)
    pos_p = 0
    if "batter_positions" in lu.columns:
        for pos_str in lu["batter_positions"].fillna(""):
            parts = str(pos_str).split("|")
            pos_p += sum(1 for p in parts if p.upper() == "P")

    evals_n = 0
    eval_coverage = None
    if evals_path.exists():
        ev = pd.read_parquet(evals_path)
        evals_n = len(ev)
        keys_lu = set(zip(lu["game_pk"].astype(int), lu["team"].astype(str)))
        keys_ev = set(zip(ev["game_pk"].astype(int), ev["team"].astype(str)))
        eval_coverage = len(keys_lu & keys_ev) / max(len(keys_lu), 1)

    pair_n = None
    if pairs_path.exists():
        pair_n = int(len(pd.read_parquet(pairs_path)))

    report.update({
        "n_teams": int(lu["team"].nunique()),
        "expected_teams": 30,
        "teams_ok": int(lu["team"].nunique()) == 30,
        "canonical_missing": sorted(set(CANONICAL_ABBREVS) - set(lu["team"].unique())),
        "n_team_games": n,
        "n_games": int(lu["game_pk"].nunique()),
        "date_min": str(lu["game_date"].min()) if n else None,
        "date_max": str(lu["game_date"].max()) if n else None,
        "n_invalid_starting_lineups": len(invalid),
        "invalid_examples": invalid[:20],
        "n_duplicate_player_lineups": len(dup_players),
        "n_pitcher_as_starting_hitter": len(pitcher_as_hitter),
        "pitcher_as_starting_hitter_cases": pitcher_as_hitter[:50],
        "n_position_labeled_P_slots": pos_p,
        "n_unresolved_player_ids": len(set(unresolved)),
        "unresolved_player_id_examples": sorted(set(unresolved))[:30],
        "n_players_table": int(len(players)),
        "optimizer_n_evaluations": evals_n,
        "optimizer_coverage": eval_coverage,
        "interaction_pair_sample_size": pair_n,
        "extraction_method": "battingOrder_codes_100_900",
        "notes": [
            "Starting lineups use GUMBO/live battingOrder codes 100–900, not the live slot array.",
            "Position==P counts are exploratory; do not treat them as pitcher-as-hitter without identity checks.",
        ],
    })

    out = settings.artifacts_dir / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written"] = str(out)
    return report


def main() -> None:
    r = build_data_quality_report()
    print(json.dumps({k: r[k] for k in r if k != "invalid_examples"}, indent=2, default=str))


if __name__ == "__main__":
    main()
