# MLB Lineup Intelligence — Status

Updated: 2026-08-12

## Release readiness

**Core product is functional end-to-end with real 2026 data.**

| Area | Status |
|------|--------|
| 30-team 2026 starting lineups | Done — 3,600 lineups (2026-03-25 → 2026-08-11) |
| PA probabilities (neutral / vs R / vs L) | Done — EB shrinkage, train 2024–2025 |
| Markov run engine + transitions | Done — DiamondIQ transitions reused |
| Same-nine optimization (9!) | Done — fast exhaustive + Markov refine |
| Precomputed evaluations | Done — 3,600 / 30 teams |
| Archetypes (unsupervised) | Done — k=4 (silhouette), labels post-hoc |
| Pair / archetype interactions | Done — EB shrinkage + reliability tiers |
| Incremental predictive value | Done — 2024→2025; interaction does **not** lift RMSE |
| FastAPI | Done — `/api/*` on `:8200` |
| React frontend | Done — all primary pages; `:5173` |
| Tests | Done — 13 core tests passing |
| README + Docker | Done |
| Research / findings page | Done — 12 computed statements |

## Key computed results (honest)

- League mean optimization gap ≈ **0.017 runs/game**
- ≈ **65%** of lineups within 0.02 runs of same-nine optimum (operationally equivalent)
- Temporal validation: archetype / prev-hitter interaction models **do not** improve out-of-sample RMSE vs talent-only baseline
- Conclusion language uses **estimated association**, not chemistry claims

## Local run

```bash
# API
PYTHONPATH=backend .venv/bin/uvicorn lineup_intel.api.main:app --host 127.0.0.1 --port 8200

# UI
cd frontend && npm run dev -- --host 127.0.0.1 --port 5173
```

## Reused from DiamondIQ (read-only / copied)

- `pa_transitions_v1.json`, baselines parquet
- GUMBO cache starting `battingOrder`
- `plate_appearances` from `diamondiq.db` (read-only)

DiamondIQ was not modified.

## Known limitations

- Fast ranking score is kernel-derived (not full VI per perm); reported runs are Markov-refined
- Statcast / as-of features lag late 2026 games where GUMBO exceeds Statcast cutoff
- Today’s live posted lineups depend on schedule + extracted DB; use `/api/today`
- Negative gaps from refine race conditions were clamped post-hoc (`best >= actual`)
- Player-selection optimization (bench/defense) intentionally deferred

## Remaining polish (non-blocking)

- Deploy to a public host (Docker compose ready)
- Optional: park / specific-pitcher context modes
- Optional: GIF/screenshot for README
- Broader API integration tests
