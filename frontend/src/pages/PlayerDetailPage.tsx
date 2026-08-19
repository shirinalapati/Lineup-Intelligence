import { Link, useParams } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../api/client'
import { usePlayer } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { RankedMetric, SlotFitStrip } from '../components/RankedMetric'
import { TeamBadge } from '../components/TeamBadge'
import { Unavailable } from '../components/Unavailable'
import { fmtNum, pickNumber } from '../lib/format'

type LineupProfile = {
  available?: boolean
  reason?: string
  name?: string
  bat_side?: string
  position?: string
  data_cutoff?: string
  offensive_profile?: { label?: string; tooltip?: string }
  summary?: Record<string, unknown>
  season_metrics?: {
    metrics?: Record<string, { mlb?: Record<string, unknown>; team?: Record<string, unknown> }>
    pa?: number
    team?: string
  }
  modeled_slot_fit?: Record<
    string,
    {
      slots?: Array<Record<string, number>>
      best_slot?: number
      near_equivalent_slots?: number[]
      primary_slot?: number
      placement_opportunity?: number
      primary_fit_rank?: number
      actual_usage_fit?: Record<string, number>
      n_starts?: number
    }
  >
  run_opportunity_profile?: { slots?: Array<Record<string, number>>; note?: string }
  why_this_slot?: Array<{ id?: string; text?: string }>
  observed_slot_splits?: Array<Record<string, unknown>>
  platoon_slot_fit?: Record<string, { best_slot?: number; near_equivalent_slots?: number[] }>
  lineup_neighbors?: {
    before?: Array<Record<string, unknown>>
    after?: Array<Record<string, unknown>>
  }
  qualification?: Record<string, unknown>
}

function mlbOf(block: unknown) {
  if (!block || typeof block !== 'object') return null
  const b = block as { mlb?: Record<string, unknown> }
  return (b.mlb as import('../components/RankedMetric').RankPayload) || null
}
function teamOf(block: unknown) {
  if (!block || typeof block !== 'object') return null
  const b = block as { team?: Record<string, unknown> }
  return (b.team as import('../components/RankedMetric').RankPayload) || null
}

export function PlayerDetailPage() {
  const { id = '' } = useParams()
  const { data, isLoading, isError, error } = usePlayer(id)
  const lineupQ = useQuery({
    queryKey: ['players', id, 'lineup-profile'],
    queryFn: () => apiGet<LineupProfile>(`/api/players/${id}/lineup-profile`),
    enabled: Boolean(id),
  })

  if (isLoading) return <Loading label="Loading player…" />
  if (isError) {
    return <Unavailable title="Player not found" reason={String(error)} />
  }
  if (data && !data.available) {
    return <Unavailable data={data} title="Player unavailable" />
  }
  if (!data?.available) return null

  const li =
    lineupQ.data && lineupQ.data.available !== false
      ? lineupQ.data
      : data.lineup_intelligence && !isUnavailable(data.lineup_intelligence)
        ? (data.lineup_intelligence as LineupProfile)
        : null

  const profile =
    data.profile && !isUnavailable(data.profile)
      ? (data.profile as Record<string, unknown>)
      : null

  const summary = li?.summary || {}
  const metrics = li?.season_metrics?.metrics || {}
  const neu = li?.modeled_slot_fit?.neutral
  const slotRows = (neu?.slots || []).map((s) => ({
    slot: Number(s.slot),
    expected_runs: Number(s.expected_runs),
    delta_vs_avg: Number(s.delta_vs_avg ?? 0),
    expected_pa: Number(s.expected_pa ?? 0),
    prob_runners_on: Number(s.prob_runners_on ?? 0),
    prob_risp: Number(s.prob_risp ?? 0),
    fit_rank: Number(s.fit_rank ?? 0),
  }))

  const usageChart = data.appearances
    ? Object.entries(data.appearances.slot_counts).map(([slot, count]) => ({
        slot: `#${slot}`,
        count,
      }))
    : []

  const teams = data.appearances?.teams ?? []

  return (
    <div>
      <PageHeader
        eyebrow="Player profile"
        title={data.name}
        description={[
          teams[0] || null,
          data.position,
          data.bat_side ? `Bats ${data.bat_side}` : null,
        ]
          .filter(Boolean)
          .join(' · ')}
      />

      {li?.data_cutoff ? (
        <p className="mb-6 text-xs text-[var(--color-muted)]">
          Data through {li.data_cutoff}
          {li.qualification?.overall_min_pa != null
            ? ` · Overall ranks require ≥${String(li.qualification.overall_min_pa)} PA`
            : ''}
        </p>
      ) : null}

      {/* LINEUP ROLE summary */}
      <section className="panel mb-10 p-5">
        <div className="eyebrow mb-2">Lineup role</div>
        <p className="mb-4 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
          <span className="text-[var(--color-ink)]">Placement opportunity</span>{' '}
          is the expected-runs gap between this player&apos;s{' '}
          <span className="text-[var(--color-ink)]">best modeled slot</span> and
          their{' '}
          <span className="text-[var(--color-ink)]">primary actual slot</span>,
          holding the other eight hitters fixed (same-nine Markov insertion).
          Units are expected runs per game for that lineup. Higher means more
          modeled value left on the table by usual placement;{' '}
          <span className="tabular-nums">0</span> means they are already in (or
          tied for) the best slot among the nine.
        </p>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          <Metric
            label="Primary actual slot"
            value={
              summary.primary_actual_slot != null
                ? `#${summary.primary_actual_slot}`
                : undefined
            }
            format="raw"
            hint="Most common starting slot this season"
          />
          <Metric
            label="Best modeled slot"
            value={
              summary.best_modeled_slot != null
                ? `#${summary.best_modeled_slot}`
                : undefined
            }
            format="raw"
            accent
            hint="Slot with highest projected R/G in same-nine tests"
          />
          <Metric
            label="Placement opportunity"
            value={pickNumber(summary, ['placement_opportunity'])}
            digits={3}
            hint={
              summary.primary_actual_slot != null &&
              summary.best_modeled_slot != null
                ? `Projected R/G (#${summary.best_modeled_slot}) − projected R/G (#${summary.primary_actual_slot})`
                : 'Best modeled projected R/G − primary actual projected R/G'
            }
          />
          <Metric
            label="Offensive profile"
            value={
              li?.offensive_profile?.label ||
              (profile
                ? String(profile.archetype_label ?? profile.archetype ?? '—')
                : '—')
            }
            format="raw"
            hint={li?.offensive_profile?.tooltip}
          />
        </div>
        {(() => {
          const opp = pickNumber(summary, ['placement_opportunity'])
          if (opp === undefined) return null
          const primary = summary.primary_actual_slot
          const best = summary.best_modeled_slot
          let reading: string
          if (opp < 0.005) {
            reading =
              'Reading: essentially no placement gap — usual slot is already near optimal for this same nine.'
          } else if (opp < 0.02) {
            reading =
              primary != null && best != null
                ? `Reading: small gap — moving from #${primary} to #${best} adds about ${fmtNum(opp, 3)} expected runs per game in the model.`
                : `Reading: small gap — about ${fmtNum(opp, 3)} expected runs per game left vs the best modeled slot.`
          } else {
            reading =
              primary != null && best != null
                ? `Reading: meaningful gap — moving from #${primary} to #${best} adds about ${fmtNum(opp, 3)} expected runs per game in the model.`
                : `Reading: meaningful gap — about ${fmtNum(opp, 3)} expected runs per game vs the best modeled slot.`
          }
          return (
            <p className="mt-4 mb-0 text-sm text-[var(--color-ink)]">{reading}</p>
          )
        })()}
        {Array.isArray(summary.near_equivalent_slots) &&
        (summary.near_equivalent_slots as number[]).length > 1 ? (
          <p className="mt-2 mb-0 text-sm text-[var(--color-muted)]">
            Near-equivalent slots:{' '}
            {(summary.near_equivalent_slots as number[])
              .map((s) => `#${s}`)
              .join(', ')}{' '}
            (within operational equivalence of the best slot).
          </p>
        ) : null}
        {summary.placement_opportunity_rank &&
        typeof summary.placement_opportunity_rank === 'object' ? (
          <div className="mt-4">
            <RankedMetric
              label="League placement-opportunity rank"
              mlb={
                summary.placement_opportunity_rank as import('../components/RankedMetric').RankPayload
              }
              format="runs"
            />
            <p className="mt-1 mb-0 text-xs text-[var(--color-muted)]">
              Same gap ranked across qualifying MLB hitters — higher percentile
              means a larger same-nine slot gap than most peers (not that the
              player is poorly managed).
            </p>
          </div>
        ) : null}
      </section>

      {teams.length || (data.team_history && data.team_history.length) ? (
        <section className="mb-10">
          <h2 className="font-display mb-3 text-xl tracking-tight">
            2026 team history
          </h2>
          {data.team_history && data.team_history.length ? (
            <ul className="panel m-0 list-none p-0">
              {data.team_history.map((stint, i) => {
                const start = stint.start_at
                  ? new Date(`${stint.start_at}T12:00:00`).toLocaleDateString(
                      'en-US',
                      { month: 'short', day: 'numeric' },
                    )
                  : '—'
                const end = stint.end_at
                  ? new Date(`${stint.end_at}T12:00:00`).toLocaleDateString(
                      'en-US',
                      { month: 'short', day: 'numeric' },
                    )
                  : 'Present'
                return (
                  <li
                    key={`${stint.team}-${stint.start_at}-${i}`}
                    className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-3 py-2 last:border-b-0"
                  >
                    {stint.team ? (
                      <TeamBadge abbr={stint.team} size="sm" />
                    ) : (
                      <span>—</span>
                    )}
                    <span className="text-sm text-[var(--color-muted)]">
                      {start} – {end}
                    </span>
                  </li>
                )
              })}
            </ul>
          ) : (
            <div className="flex flex-wrap gap-2">
              {teams.map((t) => (
                <TeamBadge key={t} abbr={t} size="sm" />
              ))}
            </div>
          )}
        </section>
      ) : null}

      {/* MODELED HITTER PROFILE with ranks */}
      <section className="mb-12">
        <h2 className="font-display mb-2 text-2xl tracking-tight">
          Modeled hitter profile
        </h2>
        <p className="mb-2 max-w-3xl text-sm text-[var(--color-muted)]">
          Season performance with MLB rank / denominator / percentile. No
          salary or undervalued-hitter composites.
        </p>
        <p className="mb-6 max-w-3xl text-sm text-[var(--color-muted)]">
          Rank and percentile are among hitters with at least{' '}
          {li?.qualification?.overall_min_pa != null
            ? String(li.qualification.overall_min_pa)
            : '100'}{' '}
          PA this season — that is the MLB denominator. The team rank uses the
          same PA gate among this player&apos;s teammates, so it is not the
          full roster.
        </p>

        <h3 className="eyebrow mb-3">Core offense</h3>
        <div className="mb-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {(
            [
              ['xwoba', 'xwOBA', 'woba'],
              ['woba', 'wOBA', 'woba'],
              ['obp', 'OBP', 'rate'],
              ['slg', 'SLG', 'rate'],
              ['iso', 'ISO', 'woba'],
              ['wrc_plus', 'wRC+', 'count'],
            ] as const
          ).map(([key, label, fmt]) => (
            <RankedMetric
              key={key}
              label={label}
              mlb={mlbOf(metrics[key])}
              team={teamOf(metrics[key])}
              format={fmt}
            />
          ))}
        </div>

        <h3 className="eyebrow mb-3">Plate discipline</h3>
        <div className="mb-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {(
            [
              ['k_pct', 'K%'],
              ['bb_pct', 'BB%'],
              ['chase_pct', 'Chase%'],
              ['contact_pct', 'Contact%'],
              ['z_contact_pct', 'Z-Contact%'],
            ] as const
          ).map(([key, label]) => (
            <RankedMetric
              key={key}
              label={label}
              mlb={mlbOf(metrics[key])}
              team={teamOf(metrics[key])}
              format="rate"
            />
          ))}
        </div>

        <h3 className="eyebrow mb-3">Contact quality</h3>
        <div className="mb-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <RankedMetric
            label="Avg EV"
            mlb={mlbOf(metrics.avg_exit_velocity)}
            team={teamOf(metrics.avg_exit_velocity)}
            format="raw"
          />
          <RankedMetric
            label="HardHit%"
            mlb={mlbOf(metrics.hardhit_pct)}
            team={teamOf(metrics.hardhit_pct)}
            format="rate"
          />
          <RankedMetric
            label="Barrel%"
            mlb={mlbOf(metrics.barrel_pct)}
            team={teamOf(metrics.barrel_pct)}
            format="rate"
          />
          <RankedMetric
            label="xSLG"
            mlb={mlbOf(metrics.xslg)}
            team={teamOf(metrics.xslg)}
            format="woba"
          />
        </div>

        <h3 className="eyebrow mb-3">Batted ball</h3>
        <div className="mb-4 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {(
            [
              ['gb_pct', 'GB%'],
              ['ld_pct', 'LD%'],
              ['fb_pct', 'FB%'],
              ['pull_pct', 'Pull%'],
              ['center_pct', 'Center%'],
              ['oppo_pct', 'Oppo%'],
            ] as const
          ).map(([key, label]) => (
            <RankedMetric
              key={key}
              label={label}
              mlb={mlbOf(metrics[key])}
              team={teamOf(metrics[key])}
              format="rate"
            />
          ))}
        </div>
        <p className="text-xs text-[var(--color-muted)]">
          Spray and ground/fly rates are descriptive — percentiles are not
          quality judgments.
        </p>

        {profile?.vs_R || profile?.vs_L || metrics.woba_vs_R || metrics.woba_vs_L ? (
          <div className="mt-8">
            <h3 className="eyebrow mb-3">Platoon (model PA rates)</h3>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
              {profile?.vs_R || metrics.woba_vs_R ? (
                <RankedMetric
                  label="wOBA vs RHP"
                  mlb={mlbOf(metrics.woba_vs_R)}
                  team={teamOf(metrics.woba_vs_R)}
                  value={pickNumber(
                    (profile?.vs_R as Record<string, unknown> | undefined) ??
                      undefined,
                    ['woba'],
                  )}
                  format="woba"
                />
              ) : null}
              {profile?.vs_L || metrics.woba_vs_L ? (
                <RankedMetric
                  label="wOBA vs LHP"
                  mlb={mlbOf(metrics.woba_vs_L)}
                  team={teamOf(metrics.woba_vs_L)}
                  value={pickNumber(
                    (profile?.vs_L as Record<string, unknown> | undefined) ??
                      undefined,
                    ['woba'],
                  )}
                  format="woba"
                />
              ) : null}
            </div>
            <p className="mt-3 mb-0 text-xs text-[var(--color-muted)]">
              Ranked among modeled PA rates vs that pitching hand. The
              denominator is hitters with at least{' '}
              {li?.qualification?.platoon_min_pa != null
                ? String(li.qualification.platoon_min_pa)
                : '50'}{' '}
              PA in that split — not the overall 100-PA pool.
            </p>
          </div>
        ) : null}
      </section>

      {/* BATTING SLOT INTELLIGENCE */}
      <section className="mb-12">
        <h2 className="font-display mb-2 text-2xl tracking-tight">
          Batting slot intelligence
        </h2>
        <p className="mb-4 max-w-3xl text-sm text-[var(--color-muted)]">
          Where does this hitter create the most modeled team value? Same-nine
          personnel held constant; other eight hitters keep relative order while
          this player moves through slots 1–9.
        </p>
        {lineupQ.isLoading ? <Loading label="Loading slot fit…" /> : null}
        {lineupQ.data && lineupQ.data.available === false ? (
          <Unavailable
            data={lineupQ.data}
            title="Slot intelligence not precomputed yet"
          />
        ) : null}
        {slotRows.length > 0 ? (
          <div className="space-y-6">
            <SlotFitStrip
              slots={slotRows}
              bestSlot={neu?.best_slot}
              primarySlot={neu?.primary_slot}
              nearSlots={neu?.near_equivalent_slots}
            />
            <div className="panel overflow-x-auto">
              <table className="table-dense min-w-[720px]">
                <thead>
                  <tr>
                    <th>Slot</th>
                    <th>Exp R/G</th>
                    <th>Δ vs avg</th>
                    <th>Exp PA/G</th>
                    <th>Runner on %</th>
                    <th>RISP %</th>
                    <th>Fit rank</th>
                  </tr>
                </thead>
                <tbody>
                  {slotRows.map((r) => (
                    <tr
                      key={r.slot}
                      className={
                        r.slot === neu?.best_slot
                          ? 'bg-[color-mix(in_srgb,var(--color-accent)_8%,transparent)]'
                          : undefined
                      }
                    >
                      <td>
                        #{r.slot}
                        {r.slot === neu?.primary_slot ? ' · actual' : ''}
                        {r.slot === neu?.best_slot ? ' · best' : ''}
                      </td>
                      <td>{fmtNum(r.expected_runs, 3)}</td>
                      <td>
                        {r.delta_vs_avg >= 0 ? '+' : ''}
                        {fmtNum(r.delta_vs_avg, 3)}
                      </td>
                      <td>{fmtNum(r.expected_pa, 2)}</td>
                      <td>{fmtNum(r.prob_runners_on * 100, 0)}%</td>
                      <td>{fmtNum(r.prob_risp * 100, 0)}%</td>
                      <td>{r.fit_rank || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {neu?.actual_usage_fit ? (
              <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                <Metric
                  label="% starts top modeled slot"
                  value={neu.actual_usage_fit.pct_top1}
                  format="pct"
                  digits={0}
                />
                <Metric
                  label="% starts top-3 slots"
                  value={neu.actual_usage_fit.pct_top3}
                  format="pct"
                  digits={0}
                />
                <Metric
                  label="% within 0.01 R/G of best"
                  value={neu.actual_usage_fit.pct_within_01}
                  format="pct"
                  digits={0}
                />
                <Metric
                  label="% within 0.02 R/G of best"
                  value={neu.actual_usage_fit.pct_within_02}
                  format="pct"
                  digits={0}
                />
              </div>
            ) : null}
            <p className="text-xs text-[var(--color-muted)]">
              Tiny differences (e.g. 0.002 R/G) are labeled essentially equivalent.
              Placement opportunity is not a managerial grade.
            </p>
          </div>
        ) : null}
      </section>

      {/* WHY THIS SLOT */}
      {li?.why_this_slot?.length ? (
        <section className="mb-12">
          <h2 className="font-display mb-3 text-2xl tracking-tight">
            Why this slot?
          </h2>
          <ul className="m-0 max-w-3xl list-disc space-y-2 pl-5 text-sm text-[var(--color-muted)]">
            {li.why_this_slot.map((b, i) => (
              <li key={b.id || i}>{b.text}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* RUN OPPORTUNITY */}
      {slotRows.length > 0 ? (
        <section className="mb-12">
          <h2 className="font-display mb-2 text-2xl tracking-tight">
            Run opportunity profile
          </h2>
          <p className="mb-4 max-w-3xl text-sm text-[var(--color-muted)]">
            {li?.run_opportunity_profile?.note ||
              'Batting order changes both the game states a hitter encounters and the number of plate appearances he receives.'}
          </p>
          <div className="panel overflow-x-auto">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Slot</th>
                  <th>Exp PA/G</th>
                  <th>Exp PA/162</th>
                  <th>Runners on</th>
                  <th>RISP</th>
                </tr>
              </thead>
              <tbody>
                {slotRows.map((r) => (
                  <tr key={`opp-${r.slot}`}>
                    <td>#{r.slot}</td>
                    <td>{fmtNum(r.expected_pa, 2)}</td>
                    <td>{fmtNum(r.expected_pa * 162, 0)}</td>
                    <td>{fmtNum(r.prob_runners_on * 100, 0)}%</td>
                    <td>{fmtNum(r.prob_risp * 100, 0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {/* ACTUAL SLOT USAGE */}
      {usageChart.length > 0 ? (
        <section className="mb-12">
          <h2 className="font-display mb-3 text-2xl tracking-tight">
            Actual slot usage
          </h2>
          <div className="panel h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={usageChart}>
                <CartesianGrid stroke="#d5dce4" strokeDasharray="3 3" />
                <XAxis dataKey="slot" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} width={32} />
                <Tooltip />
                <Bar dataKey="count" fill="#132337" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}

      {/* OBSERVED SLOT SPLITS */}
      <section className="mb-12">
        <h2 className="font-display mb-2 text-2xl tracking-tight">
          Observed 2026 slot splits
        </h2>
        <p className="mb-4 max-w-3xl text-sm text-[var(--color-muted)]">
          Observed performance — descriptive, not estimated causal slot effect.
          Ranks compare hitters with meaningful PA <em>in the same slot</em>.
        </p>
        {(li?.observed_slot_splits || []).length ? (
          <div className="space-y-4">
            {(li!.observed_slot_splits || []).map((row) => {
              const raw = (row.raw || {}) as Record<string, number>
              const mets = (row.metrics || {}) as Record<
                string,
                { mlb?: import('../components/RankedMetric').RankPayload }
              >
              const slotMinPa =
                li?.qualification?.slot_min_pa != null
                  ? Number(li.qualification.slot_min_pa)
                  : 30
              const slotPop = mets.woba?.mlb?.population_n
              const slotNum = String(row.slot)
              return (
                <div key={slotNum} className="panel p-4">
                  <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                    <h3 className="font-display m-0 text-lg">
                      Slot #{slotNum}
                    </h3>
                    <span className="text-sm text-[var(--color-muted)]">
                      {String(row.starts)} starts · {String(row.pa)} PA
                      {Number(row.pa) < slotMinPa ? ' · LIMITED SAMPLE' : ''}
                    </span>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    <RankedMetric
                      label="wOBA"
                      mlb={mets.woba?.mlb}
                      value={raw.woba}
                      format="woba"
                    />
                    <RankedMetric
                      label="OBP"
                      mlb={mets.obp?.mlb}
                      value={raw.obp}
                      format="rate"
                    />
                    <RankedMetric
                      label="SLG"
                      mlb={mets.slg?.mlb}
                      value={raw.slg}
                      format="rate"
                    />
                    <RankedMetric
                      label="ISO"
                      mlb={mets.iso?.mlb}
                      value={raw.iso}
                      format="woba"
                    />
                    <RankedMetric
                      label="K%"
                      mlb={mets.k_pct?.mlb}
                      value={raw.k_pct}
                      format="rate"
                    />
                    <RankedMetric
                      label="BB%"
                      mlb={mets.bb_pct?.mlb}
                      value={raw.bb_pct}
                      format="rate"
                    />
                  </div>
                  {slotPop != null ? (
                    <p className="mt-3 mb-0 text-xs text-[var(--color-muted)]">
                      {slotPop} hitters had at least {slotMinPa} PA as a starter
                      in slot #{slotNum} this season. That is the denominator
                      for this slot only — not the overall 100-PA league pool,
                      and not the same group as other slots.
                    </p>
                  ) : null}
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-sm text-[var(--color-muted)]">
            No observed slot splits available yet.
          </p>
        )}
      </section>

      {/* NEIGHBORS */}
      {li?.lineup_neighbors ? (
        <section className="mb-12">
          <h2 className="font-display mb-2 text-2xl tracking-tight">
            Common lineup neighbors
          </h2>
          <p className="mb-3 max-w-3xl text-sm text-[var(--color-muted)]">
            Starting-lineup adjacency. These are exploratory residual
            associations — not chemistry — and are not used by the optimizer.
          </p>
          <p className="mb-4 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
            Residual Effect measures whether the trailing hitter performed
            better or worse than expected after accounting for individual talent
            and game situation. Positive values indicate slightly
            better-than-expected outcomes and negative values indicate slightly
            worse-than-expected outcomes. In <em>Bats before him</em>, this
            player is the trailing hitter. In <em>Bats after him</em>, the
            listed teammate is the trailing hitter. A dash means the pair did
            not have enough shared plate appearances for an estimate.
          </p>
          <div className="grid gap-6 md:grid-cols-2">
            {(
              [
                ['before', 'Bats before him'],
                ['after', 'Bats after him'],
              ] as const
            ).map(([key, title]) => (
              <div key={key}>
                <h3 className="eyebrow mb-2">{title}</h3>
                <div className="panel overflow-hidden">
                  <table className="table-dense">
                    <thead>
                      <tr>
                        <th>Player</th>
                        <th>Starts</th>
                        <th>Profile</th>
                        <th>Residual Effect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(li.lineup_neighbors?.[key] || []).map((n, i) => {
                        const assoc = n.association as
                          | Record<string, unknown>
                          | null
                          | undefined
                        return (
                          <tr key={`${key}-${String(n.player_id)}-${i}`}>
                            <td>
                              <Link
                                to={`/players/${n.player_id}`}
                                className="font-semibold hover:underline"
                              >
                                {String(n.name)}
                              </Link>
                            </td>
                            <td>{String(n.n_adjacent_starts ?? '—')}</td>
                            <td className="text-sm text-[var(--color-muted)]">
                              {String(n.offensive_profile ?? '—')}
                            </td>
                            <td className="text-sm text-[var(--color-muted)]">
                              {assoc?.effect != null
                                ? `${fmtNum(assoc.effect, 4)}${
                                    assoc.reliability_tier
                                      ? ` · ${String(assoc.reliability_tier)}`
                                      : ''
                                  }`
                                : 'Too few shared PA'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <Link to="/players" className="text-sm font-semibold hover:underline">
        ← All players
      </Link>
    </div>
  )
}
