# MLB Lineup Intelligence

**How much does batting order actually matter?**

Every 2026 MLB starting lineup, evaluated through run expectancy, same-nine optimization, simulation, offensive archetypes, and interaction modeling — with honest uncertainty and out-of-sample tests.

This is a baseball research app. It separates:

- individual hitter quality
- batting-order / sequencing effects
- lineup personnel choice
- residual player or archetype “adjacency” associations
- observed game noise

It does **not** assume lineup chemistry exists. It tests whether residual interaction information improves future prediction.

---

## Run locally

You need **Python 3.12+**, **Node 20+**, and two terminals.

```bash
git clone https://github.com/shirinalapati/lineup-intelligence.git
cd lineup-intelligence

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Terminal 1 — API**

```bash
cd lineup-intelligence
source .venv/bin/activate
PYTHONPATH=backend uvicorn lineup_intel.api.main:app --host 127.0.0.1 --port 8200
```

**Terminal 2 — UI**

```bash
cd lineup-intelligence/frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api` to port 8200.

This clone already includes the processed 2026 lineups, models, and precomputed artifacts. You do **not** need to rebuild the pipeline just to use the app.

---

## What’s inside

| Area | What you get |
|------|----------------|
| **2026 lineup database** | Starting batting orders for all 30 teams |
| **Same-nine optimizer** | Actual order vs best order of the *same nine* hitters (9! search) |
| **Markov run engine** | Base/out state transitions + player PA outcome probabilities |
| **Monte Carlo** | Run distributions that validate against deterministic expectations |
| **Team / player pages** | Slot heatmaps, timelines, modeled slot fit |
| **Lineup Explorer** | Build a nine, then Optimize / Evaluate / Simulate |
| **Research** | Findings, Player Pair Explorer, methodology |

---

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=backend pytest tests/ -q
```

---

## Optional: rebuild research artifacts

Only if you are changing the ETL or models. Needs DiamondIQ-derived vendor files already in `vendor/diamondiq_models/`.

```bash
source .venv/bin/activate
PYTHONPATH=backend python scripts/run_pipeline.py
```

### Daily updates

A GitHub Action runs every day at **14:30 UTC**, refreshes completed games, and commits updated `data/` files to `main`. Pull to pick those up:

```bash
git pull
```

To also refresh this machine automatically (10:30am local):

```bash
bash scripts/install_daily_refresh_launchd.sh
```

Manual one-off:

```bash
PYTHONPATH=backend python scripts/daily_refresh.py
```

---

## Architecture

```
DiamondIQ GUMBO cache + plate_appearances (read-only)
        ↓
Python ETL → Parquet artifacts
        ↓
Markov engine + fast exhaustive ranking + Markov refine
        ↓
Precomputed evaluations (Parquet/JSON)
        ↓
FastAPI (/api) → React + TypeScript (Vite)
```

**Reused from DiamondIQ (copied / read-only, not coupled):**

- PA transition model (`pa_transitions_v1.json`)
- Run expectancy tables
- As-of batter/pitcher baselines
- 2024–2026 GUMBO feeds for starting `battingOrder`
- Plate appearance event store for training PA rates & interaction research

DiamondIQ itself is never modified by this project.

---

## Core research question

> Does lineup construction create measurable offensive value beyond the individual talent of the hitters, and how should a team order its available hitters to maximize expected run production?

Secondary:

> After controlling for talent and context, is there reliable residual evidence that adjacent hitters or archetypes outperform expectations?

A finding that residual interaction adds little out-of-sample value is still a valid baseball conclusion — and is reported honestly when that is what the data show.

---

## Methodology (short)

1. **PA probabilities** — Empirical-Bayes multinomial rates by batter × pitcher handedness, trained on 2024–2025 plate appearances; league priors for small samples.
2. **Transitions** — DiamondIQ empirical `P(next state, runs | outcome, outs, bases)` with backoff.
3. **Expected runs** — Exact-ish Markov value iteration over `(base/out, batter index)` for 9 innings.
4. **Same-nine optimization** — Exhaustive ranking of 362,880 orders via a kernel-derived fast score; reported actual/best/worst runs refined with the full Markov engine; operational equivalence band of 0.02 runs/game.
5. **Archetypes** — Unsupervised clustering on season offensive features; labels assigned after inspecting centers.
6. **Interactions** — Regularized residual models with temporal validation (train earlier seasons → test later). Shrinkage toward zero for sparse pairs.

See the in-app **Research** page for windows, metrics, and limitations.

---

## Project layout

```
backend/lineup_intel/   # API, engine, ETL, research
frontend/               # React app
data/processed/         # Lineups, PAs, rosters
data/models/            # PA probs, archetypes
data/artifacts/         # Precomputed evaluations + research outputs
vendor/diamondiq_models/# Copied frozen DiamondIQ artifacts
tests/
```

---

## Important interpretation notes

- **Observed runs ≠ modeled quality.** One 12-run game does not make a lineup “good.”
- **Order value ≠ selection value.** Same-nine optimization isolates ordering only.
- **Tiny gaps are noise.** Differences inside ~0.02 expected runs/game are treated as operationally equivalent.
- **Adjacency effects are associations**, not causal “chemistry.”
- Managers also optimize defense, rest, matchups, and information not in the public model.

---

## License / data

Public MLB Stats API / GUMBO-derived historical feeds and Statcast-derived features via existing research pipelines. Not affiliated with Major League Baseball or any club.
