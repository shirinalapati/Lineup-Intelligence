import { useEffect, useState } from 'react'
import { useFindings, useLeagueOverview, useMethodology, useModelCards } from '../api/hooks'
import { apiGet } from '../api/client'
import { useQuery } from '@tanstack/react-query'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { Unavailable } from '../components/Unavailable'
import { PlayerPairExplorer } from '../components/PlayerPairExplorer'
import { InteractionTransparency } from '../components/InteractionTransparency'
import { fmtInt, fmtNum, pickNumber } from '../lib/format'
import { isUnavailable } from '../api/types'

type ResearchTab = 'findings' | 'pairs'

function tabFromHash(): ResearchTab {
  const hash = window.location.hash
  if (hash === '#adjacent-hitter-research' || hash === '#pairs') return 'pairs'
  return 'findings'
}

function methodologyOpenFromHash(): boolean {
  return window.location.hash === '#methodology'
}

type MarkovValidation = {
  available?: boolean
  reason?: string
  aggregate?: Record<string, unknown>
  holdout?: string
  n_team_games?: number
  calibration_by_predicted_bucket?: unknown
  notes?: string[]
}

function asShare(n: number): number {
  return n >= 0 && n <= 1 ? n : n / 100
}

function placementGapDigits(gap: number): number {
  if (gap === 0) return 2
  for (let d = 3; d <= 6; d++) {
    if (Number(Math.abs(gap).toFixed(d)) > 0) return d
  }
  return 6
}

function ValidationCopy({ data }: { data: MarkovValidation }) {
  const agg = data.aggregate || {}
  const nGames = pickNumber(data as Record<string, unknown>, ['n_team_games'])
  const projected = pickNumber(agg, ['mean_predicted_rg'])
  const actual = pickNumber(agg, ['mean_actual_rg'])
  const mae = pickNumber(agg, ['mae'])
  const rmse = pickNumber(agg, ['rmse'])
  const bias =
    projected != null && actual != null ? projected - actual : undefined
  const holdout = data.holdout ? String(data.holdout) : undefined

  return (
    <div className="space-y-5">
      <p
        className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]"
        title={holdout ? `Holdout split: ${holdout}` : undefined}
      >
        We tested the model on {nGames != null ? fmtInt(nGames) : '—'} team-games
        it was not calibrated on. Individual baseball games are extremely noisy,
        so the most useful check is whether the model is well calibrated across
        many games.
        {holdout ? (
          <span className="ml-1 cursor-help text-xs text-[var(--color-muted-light)]">
            (how this sample was chosen)
          </span>
        ) : null}
      </p>
      <div className="grid gap-5 sm:grid-cols-2">
        <Metric
          label="Mean projected R/G"
          value={projected}
          hint={
            projected != null
              ? `The model expected teams to score ${fmtNum(projected, 2)} runs per game on average.`
              : undefined
          }
        />
        <Metric
          label="Mean actual R/G"
          value={actual}
          hint={
            actual != null
              ? `Those teams actually scored ${fmtNum(actual, 2)} runs per game on average.`
              : undefined
          }
        />
      </div>
      {bias != null ? (
        <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-ink)]">
          {Math.abs(bias) < 0.005
            ? 'Across the full test sample, the model’s average projection matched observed scoring.'
            : `Across the full test sample, the model projected about ${fmtNum(Math.abs(bias), 2)} ${bias > 0 ? 'more' : 'fewer'} runs per game than were actually scored.`}
        </p>
      ) : null}
      <div className="grid gap-5 sm:grid-cols-2">
        <Metric
          label="Average game error"
          value={mae}
          tech="MAE"
          hint={
            mae != null
              ? `On a typical individual game, the prediction missed the actual score by about ${fmtNum(mae, 1)} runs.`
              : undefined
          }
        />
        <Metric
          label="Error with big misses weighted more"
          value={rmse}
          tech="RMSE"
          hint="This penalizes unusually large misses more heavily. Baseball scores vary a lot from game to game, so this is expected to be larger than average error."
        />
      </div>
      <div className="panel p-5">
        <div className="eyebrow mb-2">What should I take from this?</div>
        <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
          The model is designed to estimate long-run scoring ability, not
          predict the exact score of one game. Its average projection was
          reasonably close to observed scoring across the holdout sample, while
          individual-game errors remained large because baseball scoring is
          highly variable.
        </p>
      </div>
    </div>
  )
}

function SlotPlacementCopy({ data }: { data: Record<string, unknown> }) {
  const gap = pickNumber(data, ['mean_opportunity_gap'])
  const top1 = pickNumber(data, ['pct_starts_top1_slot'])
  const top3 = pickNumber(data, ['pct_starts_top3_slot'])
  const within02 = pickNumber(data, ['pct_starts_within_02'])
  const tinyGap = gap != null && Math.abs(gap) < 0.01
  const top1Share = top1 != null ? asShare(top1) : undefined
  const top3Share = top3 != null ? asShare(top3) : undefined
  const tiedShare = within02 != null ? asShare(within02) : undefined
  const top1Pct =
    top1Share != null ? Math.round(top1Share * 100) : undefined
  const tiedPct =
    tiedShare != null ? Math.round(tiedShare * 100) : undefined
  const oneIn =
    top1Share != null && top1Share > 0
      ? Math.max(1, Math.round(1 / top1Share))
      : undefined

  return (
    <div className="space-y-5">
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Average placement gap"
          value={gap}
          digits={gap != null ? placementGapDigits(gap) : 3}
          tech="mean opportunity gap"
          hint={
            gap != null
              ? `Average runs per game left between a hitter’s actual slot and the model’s best slot. Smaller is better.${tinyGap ? ' On average, the practical difference was tiny.' : ''}`
              : undefined
          }
        />
        <Metric
          label="Started in model’s #1 slot"
          value={top1}
          format="pct"
          digits={0}
          tech="% starts in top modeled slot"
          hint={
            oneIn != null
              ? `About 1 in ${oneIn} starts placed the hitter in the single highest-projected slot.`
              : undefined
          }
        />
        <Metric
          label="Started in a top-3 modeled slot"
          value={top3}
          format="pct"
          digits={0}
          tech="% starts in top-3 slots"
          hint={
            top3Share == null
              ? undefined
              : top3Share >= 0.5 && top3Share < 0.6
                ? 'Just over half of starts placed the hitter in one of the model’s three highest-ranked slots.'
                : `${fmtNum(top3Share * 100, 0)}% of starts placed the hitter in one of the model’s three highest-ranked slots.`
          }
        />
        <Metric
          label="Within 0.02 R/G of best slot"
          value={within02}
          format="pct"
          digits={0}
          tech="% essentially tied with best slot"
          hint={
            tiedShare != null && tiedShare >= 0.9
              ? 'Even when the actual slot was not ranked #1, almost every start was within 0.02 runs per game of the best modeled placement.'
              : 'Share of starts where the actual slot was within 0.02 runs per game of the best modeled placement.'
          }
        />
      </div>
      <div className="panel p-5">
        <div className="eyebrow mb-2">
          {top1Pct != null && tiedPct != null
            ? `Why can ${top1Pct}% and ${tiedPct}% both be true?`
            : 'Why can these percentages both be true?'}
        </div>
        <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
          The model may rank one slot first, but several slots often project
          almost identically. A hitter can therefore be outside the model&apos;s
          exact #1 slot while still being effectively tied with it.
        </p>
      </div>
    </div>
  )
}

function QualityCopy({ data }: { data: Record<string, unknown> }) {
  const nTeams = pickNumber(data, ['n_teams'])
  const expectedTeams = pickNumber(data, ['expected_teams']) ?? 30
  const nInvalid = pickNumber(data, ['n_invalid_starting_lineups'])
  const coverage = pickNumber(data, ['optimizer_coverage'])
  const coverageShare = coverage != null ? asShare(coverage) : undefined

  return (
    <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <Metric
        label="Teams"
        value={nTeams}
        format="int"
        hint={
          nTeams === expectedTeams
            ? 'All MLB teams are represented.'
            : nTeams != null
              ? `${fmtInt(nTeams)} MLB teams are represented.`
              : undefined
        }
      />
      <Metric
        label="Team-games"
        value={data.n_team_games}
        format="int"
        hint="Each team in each game counts once, so one MLB game contributes two team-games."
      />
      <Metric
        label="Invalid lineups"
        value={nInvalid}
        format="int"
        hint={
          nInvalid === 0
            ? 'No malformed starting lineups remained in the final analysis.'
            : nInvalid != null
              ? `${fmtInt(nInvalid)} malformed starting lineups were excluded from the final analysis.`
              : undefined
        }
      />
      <Metric
        label="Optimizer coverage"
        value={coverage}
        format="pct"
        digits={1}
        hint={
          coverageShare != null && coverageShare >= 0.999
            ? 'Every valid lineup in the final dataset could be evaluated by the same-nine optimizer.'
            : coverageShare != null
              ? `${fmtNum(coverageShare * 100, 1)}% of valid lineups in the final dataset could be evaluated by the same-nine optimizer.`
              : undefined
        }
      />
      <Metric
        label="Interaction pairs"
        value={data.interaction_pair_sample_size}
        format="int"
        hint="Number of distinct adjacent-hitter pairings for which the research pipeline produced an estimate. Most of those pairings have limited samples."
      />
    </div>
  )
}

function findingById(
  list: Array<Record<string, unknown>>,
  id: string,
): Record<string, unknown> | undefined {
  return list.find((s) => String(s.id ?? '') === id)
}

function supportOf(stmt: Record<string, unknown> | undefined): Record<string, unknown> {
  const s = stmt?.support
  return s && typeof s === 'object' ? (s as Record<string, unknown>) : {}
}

/** Shared-PA floor implied by reliability = n / (n + prior). */
function minPaFromReliability(minRel: unknown, n0: unknown): number | undefined {
  const r = typeof minRel === 'number' ? minRel : Number(minRel)
  const prior = typeof n0 === 'number' ? n0 : Number(n0)
  if (!Number.isFinite(r) || !Number.isFinite(prior) || r <= 0 || r >= 1 || prior <= 0) {
    return undefined
  }
  return Math.round((r * prior) / (1 - r))
}

export function ResearchPage() {
  const methodology = useMethodology()
  const findings = useFindings()
  const cards = useModelCards()
  const overview = useLeagueOverview()
  const markov = useQuery({
    queryKey: ['research', 'markov-validation'],
    queryFn: () => apiGet<MarkovValidation>('/api/research/markov-validation'),
  })
  const quality = useQuery({
    queryKey: ['research', 'data-quality'],
    queryFn: () => apiGet<Record<string, unknown>>('/api/research/data-quality'),
  })
  const slotIntel = useQuery({
    queryKey: ['research', 'player-slot-intelligence'],
    queryFn: () =>
      apiGet<Record<string, unknown>>('/api/research/player-slot-intelligence'),
  })
  const [tab, setTab] = useState<ResearchTab>(tabFromHash)
  const [showMethodology, setShowMethodology] = useState(methodologyOpenFromHash)

  useEffect(() => {
    const onHash = () => {
      setTab(tabFromHash())
      setShowMethodology(methodologyOpenFromHash())
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  function selectTab(next: ResearchTab) {
    setTab(next)
    const hash = next === 'pairs' ? '#adjacent-hitter-research' : ''
    window.history.replaceState(null, '', `${window.location.pathname}${hash}`)
    if (next !== 'findings') setShowMethodology(false)
  }

  function openMethodology() {
    setTab('findings')
    setShowMethodology(true)
    window.history.replaceState(null, '', `${window.location.pathname}#methodology`)
    requestAnimationFrame(() => {
      document.getElementById('methodology')?.scrollIntoView({ behavior: 'smooth' })
    })
  }

  const ov =
    overview.data?.available && overview.data
      ? (overview.data as Record<string, unknown>)
      : null
  const metrics =
    ov && ov.metrics && !isUnavailable(ov.metrics)
      ? (ov.metrics as Record<string, unknown>)
      : ov

  const sections =
    methodology.data?.available &&
    methodology.data.methodology &&
    Array.isArray(
      (methodology.data.methodology as { sections?: unknown }).sections,
    )
      ? (
          methodology.data.methodology as {
            sections: Array<{ id: string; title: string; body: string }>
            title?: string
          }
        ).sections
      : []

  const findingList = (() => {
    const f = findings.data
    if (!f || !f.available) return []
    const payload = (f as { findings?: unknown }).findings ?? f
    if (Array.isArray(payload)) return payload as Array<Record<string, unknown>>
    if (payload && typeof payload === 'object') {
      const o = payload as Record<string, unknown>
      if (Array.isArray(o.statements)) return o.statements as Array<Record<string, unknown>>
      if (Array.isArray(o.items)) return o.items as Array<Record<string, unknown>>
    }
    return []
  })()

  const pairSupport = supportOf(findingById(findingList, 'pair_reliability'))
  const nPairs = pickNumber(pairSupport, ['n_player_pairs'])
  const nStrong = pickNumber(pairSupport, ['n_strong', 'n_player_pairs_strong'])
  const strongMinPa = minPaFromReliability(
    pairSupport.tier_strong_min,
    pairSupport.prior_n0,
  )
  const moderateMinPa = minPaFromReliability(
    pairSupport.tier_moderate_min,
    pairSupport.prior_n0,
  )

  const avgGap = pickNumber(metrics ?? undefined, ['league_avg_gap', 'avg_gap'])
  const within02 = pickNumber(metrics ?? undefined, [
    'pct_lineups_within_02',
    'pct_operationally_equivalent',
  ])
  const spread = pickNumber(metrics ?? undefined, ['league_avg_best_worst_spread'])
  const opp162 = pickNumber(metrics ?? undefined, ['ordering_opportunity_162'])

  return (
    <div>
      <PageHeader
        title="Research"
        description="How much does batting order actually matter?"
      />

      <div className="mb-8 flex flex-wrap gap-2 border-b border-[var(--color-border)] pb-4">
        {(
          [
            ['findings', 'Findings'],
            ['pairs', 'Player Pair Explorer'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`btn ${tab === id ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => selectTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'pairs' ? (
        <section id="adjacent-hitter-research" className="mb-14">
          <h2 className="font-display mb-2 text-2xl tracking-tight">
            Player Pair Explorer
          </h2>
          <p className="mb-6 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
            Pair associations are interesting exploratory context. We did not
            find evidence strong enough to use them when optimizing lineups, so
            they are <strong className="text-[var(--color-ink)]">not</strong>{' '}
            used by the optimizer.
          </p>
          <PlayerPairExplorer />
        </section>
      ) : (
        <>
          <section className="mb-14">
            <p className="mb-8 max-w-3xl text-lg leading-relaxed text-[var(--color-ink)]">
              The short answer: usually less than the names in the lineup.
            </p>
            {overview.isLoading ? <Loading label="Loading league results…" /> : null}
            {overview.data && !overview.data.available ? (
              <Unavailable data={overview.data} title="League results unavailable" />
            ) : null}
            <div className="mb-8 grid gap-4 sm:grid-cols-2">
              <article className="panel p-5">
                <Metric
                  label="Average order gap"
                  value={avgGap}
                  digits={2}
                  hint="Average difference between the posted lineup and the model's best ordering of those same nine hitters."
                />
                <p className="mt-2 mb-0 text-xs text-[var(--color-muted)]">
                  runs/game
                </p>
              </article>
              <article className="panel p-5">
                <Metric
                  label="Within 0.02 of best"
                  value={within02}
                  format="pct"
                  digits={0}
                  hint="Share of games where the posted lineup was within 0.02 runs/game of the best same-nine order."
                />
              </article>
              <article className="panel p-5">
                <Metric
                  label="Best-to-worst spread"
                  value={spread}
                  digits={2}
                  hint="Average difference between the best and worst possible ordering of the same nine hitters."
                />
                <p className="mt-2 mb-0 text-xs text-[var(--color-muted)]">
                  runs/game
                </p>
              </article>
              <article className="panel p-5">
                <Metric
                  label="Full-season ordering opportunity"
                  value={
                    opp162 != null
                      ? `~${fmtNum(opp162, 1)} runs / 162 games`
                      : undefined
                  }
                  format="raw"
                  hint="The average modeled ordering gap extended across a 162-game season."
                />
              </article>
            </div>
            <div className="max-w-3xl">
              <h3 className="font-display m-0 text-xl tracking-tight">
                What does that mean?
              </h3>
              <div className="mt-3 space-y-3 text-sm leading-relaxed text-[var(--color-muted)]">
                <p className="m-0">
                  Changing batting order does matter, but for most MLB lineups
                  the differences between reasonable orders are very small.
                </p>
                <p className="m-0">
                  A lineup can rank poorly among all 362,880 possible
                  arrangements and still project almost identically to the best
                  one because thousands of orders may be essentially tied.
                </p>
                <p className="m-0">
                  The bigger offensive question is generally who the nine
                  hitters are, rather than the exact order in which those nine
                  hitters bat.
                </p>
              </div>
            </div>
          </section>

          <section className="mb-14">
            <h2 className="font-display mb-4 text-2xl tracking-tight">
              What actually drives offense?
            </h2>
            <div className="grid gap-4 md:grid-cols-3">
              <article className="panel p-5">
                <div className="eyebrow mb-2">Who is playing</div>
                <p className="m-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  The quality of the nine hitters creates most of the
                  lineup&apos;s offensive potential. Stronger personnel
                  generally matters much more than moving the same hitters
                  between nearby batting slots.
                </p>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-2">Where they bat</div>
                <p className="m-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  Batting order changes how often hitters come to the plate and
                  which base/out situations they tend to encounter. That creates
                  real value, but usually a relatively small amount.
                </p>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-2">Who bats next to whom</div>
                <p className="m-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  We also tested whether certain hitters or offensive styles
                  perform unusually well when placed next to each other. That
                  signal was much weaker.
                </p>
              </article>
            </div>
          </section>

          <section className="mb-14">
            <h2 className="font-display mb-3 text-2xl tracking-tight">
              Do certain hitters have &ldquo;chemistry&rdquo;?
            </h2>
            <p className="mb-6 max-w-3xl text-lg leading-relaxed text-[var(--color-ink)]">
              We did not find strong predictive evidence for it.
            </p>
            <InteractionTransparency />
            <button
              type="button"
              className="btn btn-primary mt-6"
              onClick={() => selectTab('pairs')}
            >
              Explore Player Pairs →
            </button>
          </section>

          <section className="mb-14">
            <h2 className="font-display mb-3 text-2xl tracking-tight">
              What are hitter styles?
            </h2>
            <p className="mb-5 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
              We grouped hitters with similar offensive approaches using their
              previous-season statistics. These groups are descriptive shortcuts
              — not grades and not predictions of chemistry.
            </p>
            <div className="mb-4 grid gap-4 sm:grid-cols-2">
              <article className="panel p-5">
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Spray Contact
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  More balls in play, less extra-base power, and more
                  opposite-field contact.
                </p>
              </article>
              <article className="panel p-5">
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Three True Outcomes
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  More walks, strikeouts, barrels, and extra-base power — more
                  of the classic walk/strikeout/home-run style.
                </p>
              </article>
              <article className="panel p-5">
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Balanced
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  A profile closer to the typical MLB hitter without one
                  characteristic dominating.
                </p>
              </article>
              <article className="panel p-5">
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Power
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  Harder contact and more extra-base ability without the extreme
                  strikeout/walk profile of the Three True Outcomes group.
                </p>
              </article>
            </div>
            <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
              The groups were learned from 2025 statistics and applied to 2026
              hitters. The names are shorthand for similar offensive profiles,
              not hard identities.
            </p>
          </section>

          <section className="mb-14">
            <h2 className="font-display mb-3 text-2xl tracking-tight">
              How much should we trust a player pairing?
            </h2>
            <p className="mb-5 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
              The more often two hitters actually batted back-to-back, the more
              evidence we have about that particular pairing.
            </p>
            <div className="mb-5 grid gap-4 sm:grid-cols-3">
              <article className="panel p-5">
                <div className="eyebrow mb-1">Strong</div>
                <div className="font-display text-2xl tracking-tight">
                  {strongMinPa != null ? `~${strongMinPa}+ shared PA` : '—'}
                </div>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-1">Moderate</div>
                <div className="font-display text-2xl tracking-tight">
                  {moderateMinPa != null && strongMinPa != null
                    ? `~${moderateMinPa}–${strongMinPa - 1} shared PA`
                    : '—'}
                </div>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-1">Limited</div>
                <div className="font-display text-2xl tracking-tight">
                  {moderateMinPa != null ? `<${moderateMinPa} shared PA` : '—'}
                </div>
              </article>
            </div>
            <p className="mb-3 max-w-3xl text-sm leading-relaxed text-[var(--color-ink)]">
              {nPairs != null && nStrong != null
                ? `Of ${fmtInt(nPairs)} estimated player pairings, ${fmtInt(nStrong)} reached the strong-reliability group.`
                : findings.data && !findings.data.available
                  ? 'Pairing counts are unavailable until research artifacts are generated.'
                  : findings.isLoading
                    ? 'Loading pairing counts…'
                    : 'Pairing counts are not available yet.'}
            </p>
            <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
              Most pairings have limited samples because MLB batting orders
              change frequently. Even among higher-sample pairs, the remaining
              effects after accounting for hitter ability and game situation
              were generally small.
            </p>
          </section>

          <section className="mb-14">
            <h2 className="font-display mb-4 text-2xl tracking-tight">
              How the model works
            </h2>
            <div className="grid gap-4 md:grid-cols-2">
              <article className="panel p-5">
                <div className="eyebrow mb-2">Step 1</div>
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Estimate each hitter
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  We estimate the likelihood of outcomes such as a walk, single,
                  double, home run, or out, including pitcher-handedness
                  context.
                </p>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-2">Step 2</div>
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Evaluate the batting order
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  The model tracks outs, baserunners, scoring, and how often
                  each lineup position comes to the plate.
                </p>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-2">Step 3</div>
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Test every arrangement
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  For the same nine hitters, the optimizer evaluates all 362,880
                  possible batting orders.
                </p>
              </article>
              <article className="panel p-5">
                <div className="eyebrow mb-2">Step 4</div>
                <h3 className="font-display m-0 text-lg tracking-tight">
                  Compare the alternatives
                </h3>
                <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                  By keeping the same nine players fixed, we can isolate how
                  much value comes from batting order rather than player talent.
                </p>
              </article>
            </div>
          </section>

          <section id="methodology" className="mb-14">
            <h2 className="font-display mb-3 text-2xl tracking-tight">
              Want the technical details?
            </h2>
            <p className="mb-5 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
              See how player probabilities, the run-expectancy model, exhaustive
              lineup optimization, validation, and interaction testing are
              constructed.
            </p>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() =>
                showMethodology
                  ? document
                      .getElementById('methodology-detail')
                      ?.scrollIntoView({ behavior: 'smooth' })
                  : openMethodology()
              }
            >
              Full methodology →
            </button>

            {showMethodology ? (
              <div id="methodology-detail" className="mt-10 space-y-14">
                <section>
                  <h3 className="font-display mb-3 text-xl tracking-tight">
                    Methodology
                  </h3>
                  {methodology.isLoading ? <Loading /> : null}
                  {sections.length > 0 ? (
                    <div className="space-y-6">
                      {sections.map((s) => (
                        <article
                          key={s.id}
                          id={s.id}
                          className="border-b border-[var(--color-border)] pb-6"
                        >
                          <h4 className="font-display m-0 text-lg tracking-tight">
                            {s.title}
                          </h4>
                          <p className="mt-2 mb-0 max-w-3xl leading-relaxed text-[var(--color-muted)]">
                            {s.body}
                          </p>
                        </article>
                      ))}
                    </div>
                  ) : methodology.data && !methodology.data.available ? (
                    <Unavailable
                      data={methodology.data}
                      title="Methodology unavailable"
                    />
                  ) : null}
                </section>

                <section>
                  <h3 className="font-display mb-3 text-xl tracking-tight">
                    How well does the run model match real scoring?
                  </h3>
                  {markov.isLoading ? <Loading /> : null}
                  {markov.data && markov.data.available === false ? (
                    <Unavailable
                      data={markov.data}
                      title="Validation unavailable"
                    />
                  ) : null}
                  {markov.data?.available !== false && markov.data?.aggregate ? (
                    <ValidationCopy data={markov.data} />
                  ) : null}
                </section>

                <section>
                  <h3 className="font-display mb-3 text-xl tracking-tight">
                    How much data was successfully processed?
                  </h3>
                  {quality.isLoading ? <Loading /> : null}
                  {quality.data?.available === false ? (
                    <Unavailable
                      data={quality.data}
                      title="Data quality unavailable"
                    />
                  ) : null}
                  {quality.data?.available !== false && quality.data ? (
                    <QualityCopy data={quality.data} />
                  ) : null}
                </section>

                <section>
                  <h3 className="font-display mb-2 text-xl tracking-tight">
                    How often were hitters placed near the model&apos;s preferred
                    slot?
                  </h3>
                  <p className="mb-4 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
                    For each hitter, we moved him through batting slots 1–9
                    while holding the other eight hitters fixed. This measures
                    modeled placement value, not managerial quality.
                  </p>
                  {slotIntel.isLoading ? <Loading /> : null}
                  {slotIntel.data?.available === false ? (
                    <Unavailable
                      data={slotIntel.data}
                      title="Player slot intelligence unavailable"
                    />
                  ) : null}
                  {slotIntel.data?.available !== false && slotIntel.data ? (
                    <SlotPlacementCopy data={slotIntel.data} />
                  ) : null}
                </section>

                <section>
                  <h3 className="font-display mb-4 text-xl tracking-tight">
                    Model cards
                  </h3>
                  {cards.isLoading ? <Loading /> : null}
                  {cards.data?.available && Array.isArray(cards.data.cards) ? (
                    <div className="space-y-4">
                      {cards.data.cards.map((c) => {
                        const card = c as Record<string, unknown>
                        return (
                          <article
                            key={String(card.id ?? card.name)}
                            className="panel p-5"
                          >
                            <h4 className="font-display m-0 text-lg tracking-tight">
                              {String(card.name ?? card.id)}
                            </h4>
                            {card.role ? (
                              <p className="mt-2 mb-0 text-sm text-[var(--color-muted)]">
                                {String(card.role)}
                              </p>
                            ) : null}
                            {card.limitations ? (
                              <p className="mt-2 mb-0 text-xs text-[var(--color-muted)]">
                                Limits: {JSON.stringify(card.limitations)}
                              </p>
                            ) : null}
                          </article>
                        )
                      })}
                    </div>
                  ) : cards.data && !cards.data.available ? (
                    <Unavailable
                      data={cards.data}
                      title="Model cards unavailable"
                    />
                  ) : null}
                </section>
              </div>
            ) : null}
          </section>
        </>
      )}
    </div>
  )
}
