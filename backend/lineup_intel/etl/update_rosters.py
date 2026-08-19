"""Fetch MLB Stats API transactions/rosters and write canonical roster-history artifacts."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..config import settings
from ..db.store import SLOT_COLS
from ..roster_history import (
    build_status_intervals,
    build_tenures,
    classify_transaction,
    infer_tenure_fallbacks_from_lineups,
    overlapping_tenure_violations,
    tenures_to_frame,
    validate_lineups_against_tenures,
)
from ..teams import CANONICAL_ABBREVS, TEAMS

CACHE = settings.data_dir / "cache" / "mlb_api"
TX_CACHE = CACHE / "transactions"
ROSTER_CACHE = CACHE / "rosters"


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = httpx.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def fetch_transactions(
    *,
    start: date,
    end: date,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Cache MLB transactions in monthly chunks."""
    TX_CACHE.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            month_end = date(cursor.year, 12, 31)
        else:
            month_end = date(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
        chunk_start = max(cursor, start)
        chunk_end = min(month_end, end)
        fname = TX_CACHE / f"{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        if fname.exists() and not force:
            payload = json.loads(fname.read_text(encoding="utf-8"))
            rows = payload.get("transactions") or payload
        else:
            payload = _http_get_json(
                "https://statsapi.mlb.com/api/v1/transactions",
                params={
                    "sportId": 1,
                    "startDate": chunk_start.isoformat(),
                    "endDate": chunk_end.isoformat(),
                },
            )
            _write_json(fname, payload)
            rows = payload.get("transactions") or []
        if isinstance(rows, list):
            out.extend(rows)
        cursor = month_end + timedelta(days=1)
    return out


def fetch_40man_roster(team_id: int, as_of: date, *, force: bool = False) -> list[dict[str, Any]]:
    ROSTER_CACHE.mkdir(parents=True, exist_ok=True)
    path = ROSTER_CACHE / f"{int(team_id)}_{as_of.isoformat()}_40Man.json"
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _http_get_json(
            f"https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster",
            params={
                "rosterType": "40Man",
                "season": as_of.year,
                "date": as_of.isoformat(),
            },
        )
        _write_json(path, payload)
    rows = []
    for entry in payload.get("roster") or []:
        person = entry.get("person") or {}
        pid = person.get("id")
        if pid is None:
            continue
        status = entry.get("status") or {}
        pos = (entry.get("position") or {}).get("abbreviation")
        rows.append({
            "player_id": int(pid),
            "name": person.get("fullName"),
            "position": pos,
            "status_code": status.get("code"),
            "status": status.get("description"),
            "team_id": int(team_id),
        })
    return rows


def opening_day(season: int, lineups: pd.DataFrame | None) -> date:
    if lineups is not None and not lineups.empty and "game_date" in lineups.columns:
        mx = pd.to_datetime(lineups["game_date"], errors="coerce").min()
        if pd.notna(mx):
            return mx.date()
    return date(season, 3, 26)


def build_roster_history(
    *,
    season: int | None = None,
    through: str | date | None = None,
    force_fetch: bool = False,
) -> dict[str, Any]:
    season = int(season or settings.target_season)
    through_d = (
        date.fromisoformat(str(through)[:10])
        if through
        else date.today()
    )
    lu_path = settings.processed_dir / f"starting_lineups_{season}.parquet"
    lineups = pd.read_parquet(lu_path) if lu_path.exists() else None
    open_d = opening_day(season, lineups)
    # Include prior offseason so late signings before opening day are in the feed
    # (opening 40-man is still the membership baseline).
    tx_start = date(season - 1, 11, 1)

    raw_tx = fetch_transactions(start=tx_start, end=through_d, force=force_fetch)
    events = [classify_transaction(t) for t in raw_tx]
    events = [e for e in events if e.get("player_id") and e.get("effective_at")]

    opening: dict[int, list[dict[str, Any]]] = {}
    current: dict[int, list[dict[str, Any]]] = {}
    for abbr in CANONICAL_ABBREVS:
        tid = TEAMS[abbr]["id"]
        opening[tid] = fetch_40man_roster(tid, open_d, force=force_fetch)
        current[tid] = fetch_40man_roster(tid, through_d, force=force_fetch)

    tenures = build_tenures(opening, events, opening_date=open_d, through=through_d)
    if lineups is not None and not lineups.empty:
        tenures.extend(
            infer_tenure_fallbacks_from_lineups(lineups, tenures, slot_cols=SLOT_COLS)
        )
    intervals = build_status_intervals(
        opening, events, tenures, opening_date=open_d, through=through_d
    )

    processed = settings.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    tenure_df = tenures_to_frame(tenures)
    events_df = pd.DataFrame(events)
    intervals_df = pd.DataFrame(intervals)
    tenure_df.to_parquet(processed / "player_team_tenure.parquet", index=False)
    events_df.to_parquet(processed / "roster_status_events.parquet", index=False)
    intervals_df.to_parquet(processed / "player_roster_intervals.parquet", index=False)

    # Flatten current 40-man for current-mode lookups
    current_rows = []
    for tid, players in current.items():
        abbr = next((a for a in CANONICAL_ABBREVS if TEAMS[a]["id"] == tid), None)
        for p in players:
            current_rows.append({**p, "team": abbr, "as_of": through_d.isoformat()})
    pd.DataFrame(current_rows).to_parquet(
        processed / "current_40man_roster.parquet", index=False
    )

    overlaps = overlapping_tenure_violations(tenures)
    validation: dict[str, Any] = {}
    if lineups is not None and not lineups.empty:
        validation = validate_lineups_against_tenures(
            lineups, tenures, slot_cols=SLOT_COLS
        )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "season": season,
        "opening_date": open_d.isoformat(),
        "through": through_d.isoformat(),
        "sources": [
            "https://statsapi.mlb.com/api/v1/transactions",
            "https://statsapi.mlb.com/api/v1/teams/{id}/roster?rosterType=40Man",
        ],
        "n_raw_transactions": len(raw_tx),
        "n_classified_events": len(events),
        "n_tenure_intervals": int(len(tenure_df)),
        "n_status_intervals": int(len(intervals_df)),
        "n_players_with_tenure": int(tenure_df["player_id"].nunique()) if len(tenure_df) else 0,
        "overlapping_tenures": overlaps[:20],
        "n_overlapping_tenures": len(overlaps),
        "lineup_validation": validation,
        "assumptions": [
            "Intervals are half-open [start_at, end_at).",
            "Opening-day 40-man is the membership baseline.",
            "IL / option / recall / DFA do not end organizational tenure.",
            "Trade/release/claim/sign/select change tenure.",
            "Rehab/option assignments to affiliates keep MLB org membership.",
            "Lineup-appearance fallback tenures are labeled low-confidence.",
            "Explorer as-of uses reconstructed membership; hitter PA models remain 2024–2025 trained rates.",
        ],
        "known_limitations": [
            "MLB transaction descriptions are free text; status-change parsing can miss rare list types.",
            "Same-day multi-move sequences use API effectiveDate (day resolution).",
            "40-man snapshots include pitchers; Explorer still filters to hitters via lineup history / position.",
        ],
    }
    art = settings.artifacts_dir / "data_quality"
    art.mkdir(parents=True, exist_ok=True)
    _write_json(art / "roster_lineup_validation.json", report)

    docs = settings.root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    lv = validation or {}
    md = [
        "# Roster history validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Sources",
        "",
        "- MLB Stats API `transactions` (cached monthly under `data/cache/mlb_api/transactions`)",
        "- MLB Stats API team `rosterType=40Man` snapshots (opening day + through date)",
        "- 2026 starting lineups for validation and last-resort tenure fallbacks",
        "",
        "## Coverage",
        "",
        f"- Season: {season}",
        f"- Opening date: {open_d.isoformat()}",
        f"- Through: {through_d.isoformat()}",
        f"- Classified events: {len(events)}",
        f"- Tenure intervals: {len(tenure_df)}",
        f"- Status intervals: {len(intervals_df)}",
        f"- Players with tenure: {report['n_players_with_tenure']}",
        f"- Overlapping tenure pairs: {len(overlaps)}",
        "",
        "## Lineup membership validation",
        "",
        f"- Starting-player observations: {lv.get('total_starting_player_observations')}",
        f"- Validated: {lv.get('validated')}",
        f"- Mismatched: {lv.get('mismatched')}",
        f"- Rate: {lv.get('validation_rate')}",
        "",
        "## Assumptions",
        "",
    ]
    md.extend(f"- {a}" for a in report["assumptions"])
    md += ["", "## Known limitations", ""]
    md.extend(f"- {a}" for a in report["known_limitations"])
    if lv.get("examples"):
        md += ["", "## Example mismatches", ""]
        for ex in lv["examples"][:10]:
            md.append(
                f"- player {ex.get('player_id')} {ex.get('game_date')} "
                f"lineup {ex.get('lineup_team')} tenure {ex.get('tenure_team')} "
                f"({ex.get('probable_cause')})"
            )
    (docs / "roster_history_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Rebuild roster-history artifacts")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--through", type=str, default=None)
    p.add_argument("--force-fetch", action="store_true")
    args = p.parse_args(argv)
    report = build_roster_history(
        season=args.season, through=args.through, force_fetch=args.force_fetch
    )
    lv = report.get("lineup_validation") or {}
    print(
        f"[rosters] tenures={report['n_tenure_intervals']} "
        f"events={report['n_classified_events']} "
        f"lineup_ok={lv.get('validated')}/{lv.get('total_starting_player_observations')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
