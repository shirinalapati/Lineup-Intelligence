import { Link, useParams } from 'react-router-dom'
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import {
  useTeam,
  useTeamHeatmap,
  useTeamLineups,
  useTeamMostUsed,
  useTeamMostUsedByUsage,
  useTeamTimeline,
} from '../api/hooks'
import { isUnavailable, type LineupRow, type TeamMetricRank } from '../api/types'
import { Heatmap } from '../components/Heatmap'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { TeamBadge } from '../components/TeamBadge'
import { Unavailable } from '../components/Unavailable'
import { fmtDate, fmtNum, handLabel, pickNumber } from '../lib/format'

function rankBits(ranks: Record<string, TeamMetricRank> | undefined, key: string) {
  const r = ranks?.[key]
  if (!r || r.rank == null || r.population_n == null) return {}
  return { rank: r.rank, populationN: r.population_n }
}

function parseIdList(raw: unknown): number[] | undefined {
  if (Array.isArray(raw)) {
    const ids = raw.map(Number).filter((n) => Number.isFinite(n))
    return ids.length ? ids : undefined
  }
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const p = JSON.parse(raw)
      return Array.isArray(p) ? parseIdList(p) : undefined
    } catch {
      return undefined
    }
  }
  return undefined
}

function orderNames(
  ids: number[] | undefined,
  names: string[] | null | undefined,
  battingOrder: number[] | undefined,
  batterNames: string[] | null | undefined,
): string {
  if (names && names.length) return names.join(' · ')
  if (!ids?.length) return '—'
  const map = new Map<number, string>()
  if (battingOrder && batterNames) {
    battingOrder.forEach((id, i) => {
      if (batterNames[i]) map.set(Number(id), batterNames[i])
    })
  }
  return ids.map((id) => map.get(Number(id)) ?? String(id)).join(' · ')
}

function gameScore(row: LineupRow): string {
  const scored =
    typeof row.runs_scored === 'number' && Number.isFinite(row.runs_scored)
      ? row.runs_scored
      : null
  const allowed =
    typeof row.runs_allowed === 'number' && Number.isFinite(row.runs_allowed)
      ? row.runs_allowed
      : null
  if (scored == null && allowed == null) return '—'
  const core =
    scored != null && allowed != null
      ? `${scored}–${allowed}`
      : scored != null
        ? String(scored)
        : `–${allowed}`
  return row.result ? `${core} ${row.result}` : core
}

function timelinePoints(data: unknown): Array<Record<string, unknown>> {
  if (!data || typeof data !== 'object' || isUnavailable(data)) return []
  const d = data as Record<string, unknown>
  if (Array.isArray(d.points)) return d.points as Array<Record<string, unknown>>
  if (Array.isArray(d.lineups)) return d.lineups as Array<Record<string, unknown>>
  if (Array.isArray(d.data)) return d.data as Array<Record<string, unknown>>
  if (Array.isArray(d.timeline)) return d.timeline as Array<Record<string, unknown>>
  return []
}

function mostUsedRows(data: unknown): Array<Record<string, unknown>> {
  if (!data || typeof data !== 'object' || isUnavailable(data)) return []
  const d = data as Record<string, unknown>
  if (Array.isArray(d.orders)) return d.orders as Array<Record<string, unknown>>
  if (Array.isArray(d.lineups)) return d.lineups as Array<Record<string, unknown>>
  if (Array.isArray(d.data)) return d.data as Array<Record<string, unknown>>
  return []
}

export function TeamDetailPage() {
  const { abbr = '' } = useParams()
  const team = useTeam(abbr)
  const lineups = useTeamLineups(abbr, 2000)
  const heatmap = useTeamHeatmap(abbr)
  const timeline = useTeamTimeline(abbr)
  const mostUsed = useTeamMostUsed(abbr)
  const mostUsedFreq = useTeamMostUsedByUsage(abbr)

  if (team.isLoading) return <Loading label="Loading team…" />
  if (team.isError) {
    return <Unavailable title="Team not found" reason={String(team.error)} />
  }
  if (team.data && !team.data.available) {
    return <Unavailable data={team.data} title="Team unavailable" />
  }

  const t = team.data
  if (!t || !t.available) return null

  const summary =
    t.summary && !isUnavailable(t.summary)
      ? (t.summary as Record<string, unknown>)
      : null

  const avgExp = pickNumber(summary ?? undefined, [
    'avg_actual_runs',
    'avg_expected_runs',
    'mean_expected_runs',
    'avg_actual_expected_runs',
  ])
  const avgGap = pickNumber(summary ?? undefined, ['avg_gap', 'mean_gap', 'gap'])
  const avgPct = pickNumber(summary ?? undefined, [
    'avg_percentile',
    'mean_percentile',
    'percentile',
  ])

  const lineupRows = lineups.data?.available ? lineups.data.lineups : []

  const largestGaps = [...lineupRows]
    .map((row) => {
      const ev =
        row.evaluation && !isUnavailable(row.evaluation)
          ? (row.evaluation as Record<string, unknown>)
          : null
      const gap = pickNumber(ev ?? undefined, ['gap', 'order_gap'])
      return { row, ev, gap }
    })
    .filter((item) => item.gap != null)
    .sort((a, b) => {
      const gapDiff = (b.gap ?? 0) - (a.gap ?? 0)
      if (gapDiff !== 0) return gapDiff
      return String(b.row.game_date ?? '').localeCompare(String(a.row.game_date ?? ''))
    })
    .slice(0, 15)

  const points = timelinePoints(timeline.data).map((p) => ({
    ...p,
    label: fmtDate(p.game_date),
    expected: pickNumber(p, [
      'expected_runs',
      'actual_runs',
      'actual_expected_runs',
      'exp_runs',
    ]),
    gap: pickNumber(p, ['gap']),
    observed: pickNumber(p, ['runs_scored', 'observed_runs']),
  }))

  const used = mostUsedRows(mostUsed.data)
  const usedFreq = mostUsedRows(mostUsedFreq.data)

  return (
    <div>
      <PageHeader
        eyebrow={t.division}
        title={t.name}
        description={`Modeled batting-order efficiency and personnel patterns for ${t.abbr}.`}
        actions={<TeamBadge abbr={t.abbr} name={t.name} link={false} />}
      />

      <section className="mb-10 grid gap-6 border-b border-[var(--color-border)] pb-8 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Games logged" value={t.games} format="int" />
        <Metric
          label="Avg order gap"
          value={avgGap}
          hint="Same-nine optimal minus posted (runs/game)"
          {...rankBits(t.ranks, 'avg_gap')}
        />
        <Metric
          label="Avg percentile"
          value={avgPct}
          digits={0}
          hint="Dramatic when many orders are near-equivalent — prefer gap"
          {...rankBits(t.ranks, 'avg_percentile')}
        />
        <Metric
          label="Lineup talent (projected R/G)"
          value={avgExp}
          hint="Context only — personnel strength, not ordering skill"
          {...rankBits(t.ranks, 'avg_actual_runs')}
        />
        <Metric
          label="Unique orders"
          value={t.unique_orders}
          format="int"
          {...rankBits(t.ranks, 'unique_orders')}
        />
        <Metric
          label="Unique personnel"
          value={t.unique_personnel}
          format="int"
          {...rankBits(t.ranks, 'unique_personnel')}
        />
        {t.summary && isUnavailable(t.summary) ? (
          <div className="sm:col-span-2">
            <Unavailable data={t.summary} title="Team summary metrics unavailable" />
          </div>
        ) : null}
      </section>

      <section className="mb-12">
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Batting-slot heatmap
        </h2>
        {heatmap.isLoading ? <Loading /> : null}
        {heatmap.data ? <Heatmap data={heatmap.data} /> : null}
      </section>

      <section className="mb-12">
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Efficiency timeline
        </h2>
        <p className="mb-4 text-sm text-[var(--color-muted)]">
          Expected runs are model estimates. Observed runs (when shown) are
          game outcomes and are plotted separately.
        </p>
        {timeline.isLoading ? <Loading /> : null}
        {timeline.data && isUnavailable(timeline.data) ? (
          <Unavailable data={timeline.data} title="Timeline unavailable" />
        ) : null}
        {points.length > 0 ? (
          <div className="panel h-72 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points}>
                <CartesianGrid stroke="#d5dce4" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11 }} width={36} />
                <Tooltip
                  contentStyle={{
                    borderRadius: 2,
                    borderColor: '#d5dce4',
                    fontSize: 12,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="expected"
                  name="Expected runs"
                  stroke="#132337"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="observed"
                  name="Observed runs"
                  stroke="#b91c1c"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : timeline.data && !isUnavailable(timeline.data) ? (
          <p className="text-sm text-[var(--color-muted)]">No timeline points returned.</p>
        ) : null}
      </section>

      <section className="mb-12">
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Highest-projected lineups used
        </h2>
        <p className="mb-4 text-sm text-[var(--color-muted)]">
          Unique starting orders used this season, ranked by average modeled
          expected runs. Avg runs scored is the actual game outcome average for
          those same appearances.
        </p>
        {mostUsed.isLoading ? <Loading /> : null}
        {mostUsed.data && isUnavailable(mostUsed.data) ? (
          <Unavailable data={mostUsed.data} title="Lineup rankings unavailable" />
        ) : null}
        {used.length > 0 ? (
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Avg exp. runs</th>
                  <th>Avg runs scored</th>
                  <th>Avg gap</th>
                  <th>Avg %ile</th>
                  <th>Times used</th>
                  <th>Order</th>
                </tr>
              </thead>
              <tbody>
                {used.map((row, i) => (
                  <tr key={String(row.order_id ?? i)}>
                    <td>{String(row.rank ?? i + 1)}</td>
                    <td>
                      {fmtNum(
                        pickNumber(row, [
                          'avg_expected_runs',
                          'mean_expected_runs',
                          'expected_runs',
                        ]),
                      )}
                    </td>
                    <td>
                      {fmtNum(
                        pickNumber(row, [
                          'avg_runs_scored',
                          'mean_runs_scored',
                          'avg_observed_runs',
                        ]),
                      )}
                    </td>
                    <td>
                      {fmtNum(pickNumber(row, ['avg_gap', 'mean_gap', 'gap']))}
                    </td>
                    <td>
                      {fmtNum(
                        pickNumber(row, ['avg_percentile', 'mean_percentile', 'percentile']),
                        0,
                      )}
                    </td>
                    <td>{String(row.n ?? row.count ?? '—')}</td>
                    <td className="text-sm">
                      {Array.isArray(row.batter_names)
                        ? (row.batter_names as string[]).join(' · ')
                        : String(row.order_id ?? row.personnel_id ?? '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="px-3 py-2 text-xs text-[var(--color-muted)]">
              {used.length} unique order{used.length === 1 ? '' : 's'}
            </div>
          </div>
        ) : mostUsed.data && !isUnavailable(mostUsed.data) ? (
          <p className="text-sm text-[var(--color-muted)]">
            No evaluated lineups available to rank yet.
          </p>
        ) : null}
      </section>

      <section className="mb-12">
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Most frequently used orders
        </h2>
        {mostUsedFreq.isLoading ? <Loading /> : null}
        {usedFreq.length > 0 ? (
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Times used</th>
                  <th>Avg exp. runs</th>
                  <th>Avg gap</th>
                  <th>Order</th>
                </tr>
              </thead>
              <tbody>
                {usedFreq.map((row, i) => (
                  <tr key={`freq-${String(row.order_id ?? i)}`}>
                    <td>{String(row.rank ?? i + 1)}</td>
                    <td>{String(row.n ?? row.count ?? '—')}</td>
                    <td>
                      {fmtNum(
                        pickNumber(row, [
                          'avg_expected_runs',
                          'mean_expected_runs',
                          'expected_runs',
                        ]),
                      )}
                    </td>
                    <td>
                      {fmtNum(pickNumber(row, ['avg_gap', 'mean_gap', 'gap']))}
                    </td>
                    <td className="text-sm">
                      {Array.isArray(row.batter_names)
                        ? (row.batter_names as string[]).join(' · ')
                        : String(row.order_id ?? '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      <section className="mb-12">
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Largest optimization gaps
        </h2>
        <p className="mb-4 text-sm text-[var(--color-muted)]">
          Games where the posted order left the most modeled expected runs on
          the table versus the best permutation of that same nine.
        </p>
        {largestGaps.length > 0 ? (
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Date</th>
                  <th>Opp</th>
                  <th>Score</th>
                  <th>Gap</th>
                  <th>Posted</th>
                  <th>Optimal</th>
                </tr>
              </thead>
              <tbody>
                {largestGaps.map(({ row, ev, gap }, i) => {
                  const bestIds = parseIdList(
                    ev?.best_order_ids ?? ev?.best_order,
                  )
                  const bestNames = Array.isArray(ev?.best_order_names)
                    ? (ev.best_order_names as string[])
                    : undefined
                  const posted =
                    Array.isArray(row.batter_names) && row.batter_names.length
                      ? row.batter_names.join(' · ')
                      : '—'
                  const optimal = orderNames(
                    bestIds,
                    bestNames,
                    row.batting_order,
                    row.batter_names,
                  )
                  return (
                    <tr key={`gap-${row.game_pk}-${row.order_id ?? i}`}>
                      <td>{i + 1}</td>
                      <td>
                        <Link
                          to={`/lineups/${row.game_pk}/${abbr.toUpperCase()}`}
                          className="font-semibold hover:underline"
                        >
                          {fmtDate(row.game_date)}
                        </Link>
                      </td>
                      <td>
                        {row.opponent ? (
                          <TeamBadge abbr={row.opponent} size="sm" />
                        ) : (
                          '—'
                        )}
                      </td>
                      <td>{gameScore(row)}</td>
                      <td>{fmtNum(gap)}</td>
                      <td className="text-sm">{posted}</td>
                      <td className="text-sm">{optimal}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      <section>
        <h2 className="font-display mb-3 text-xl tracking-tight">
          Lineups
        </h2>
        <p className="mb-4 text-sm text-[var(--color-muted)]">
          Every starting lineup this season. Expected runs are model estimates;
          runs scored are the actual game total.
        </p>
        {lineups.isLoading ? <Loading /> : null}
        {lineups.data && !lineups.data.available ? (
          <Unavailable data={lineups.data} title="Lineups unavailable" />
        ) : null}
        {lineupRows.length > 0 ? (
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th>Starter</th>
                  <th>Hand</th>
                  <th>Exp. runs</th>
                  <th>Runs scored</th>
                  <th>Gap</th>
                  <th>%ile</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {lineupRows.map((row) => {
                  const ev =
                    row.evaluation && !isUnavailable(row.evaluation)
                      ? (row.evaluation as Record<string, unknown>)
                      : null
                  return (
                    <tr key={`${row.game_pk}-${row.order_id}`}>
                      <td>{fmtDate(row.game_date)}</td>
                      <td>
                        {row.opponent ? (
                          <TeamBadge abbr={row.opponent} size="sm" />
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="text-sm">
                        {row.opp_sp_name ?? '—'}
                      </td>
                      <td>{handLabel(row.opp_sp_hand)}</td>
                      <td>
                        {fmtNum(
                          pickNumber(ev ?? undefined, [
                            'actual_runs',
                            'actual_expected_runs',
                            'expected_runs',
                          ]),
                        )}
                      </td>
                      <td>
                        {row.runs_scored != null ? row.runs_scored : '—'}
                        {row.result ? (
                          <span className="ml-1 text-xs text-[var(--color-muted)]">
                            {row.result}
                          </span>
                        ) : null}
                      </td>
                      <td>{fmtNum(pickNumber(ev ?? undefined, ['gap']))}</td>
                      <td>
                        {fmtNum(
                          pickNumber(ev ?? undefined, ['percentile']),
                          0,
                        )}
                      </td>
                      <td className="text-right">
                        <Link
                          to={`/lineups/${row.game_pk}/${abbr.toUpperCase()}`}
                          className="text-sm font-semibold hover:underline"
                        >
                          Detail
                        </Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="px-3 py-2 text-xs text-[var(--color-muted)]">
              {lineupRows.length}
              {lineups.data?.available &&
              typeof lineups.data.total === 'number' &&
              lineups.data.total !== lineupRows.length
                ? ` of ${lineups.data.total}`
                : ''}{' '}
              lineup{lineupRows.length === 1 ? '' : 's'}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  )
}
