# Roster history validation

Generated: 2026-08-20T10:46:13

## Sources

- MLB Stats API `transactions` (cached monthly under `data/cache/mlb_api/transactions`)
- MLB Stats API team `rosterType=40Man` snapshots (opening day + through date)
- 2026 starting lineups for validation and last-resort tenure fallbacks

## Coverage

- Season: 2026
- Opening date: 2026-03-25
- Through: 2026-08-20
- Classified events: 15523
- Tenure intervals: 4597
- Status intervals: 12203
- Players with tenure: 4115
- Overlapping tenure pairs: 0

## Lineup membership validation

- Starting-player observations: 34452
- Validated: 34452
- Mismatched: 0
- Rate: 1.0

## Assumptions

- Intervals are half-open [start_at, end_at).
- Opening-day 40-man is the membership baseline.
- IL / option / recall / DFA do not end organizational tenure.
- Trade/release/claim/sign/select change tenure.
- Rehab/option assignments to affiliates keep MLB org membership.
- Lineup-appearance fallback tenures are labeled low-confidence.
- Explorer as-of uses reconstructed membership; hitter PA models remain 2024–2025 trained rates.

## Known limitations

- MLB transaction descriptions are free text; status-change parsing can miss rare list types.
- Same-day multi-move sequences use API effectiveDate (day resolution).
- 40-man snapshots include pitchers; Explorer still filters to hitters via lineup history / position.
