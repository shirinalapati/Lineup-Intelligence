import { Link, useParams } from 'react-router-dom'
import { useLineupDetail } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { LineupOrder } from '../components/LineupOrder'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { TeamBadge } from '../components/TeamBadge'
import { Unavailable } from '../components/Unavailable'
import { fmtDate, fmtNum, handLabel, pickNumber } from '../lib/format'

function asList(v: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(v)) return v as Array<Record<string, unknown>>
  return []
}

export function LineupDetailPage() {
  const { gamePk = '', team = '' } = useParams()
  const { data, isLoading, isError, error } = useLineupDetail(gamePk, team)
  const opponentAbbr =
    data?.available && data.lineup?.opponent
      ? String(data.lineup.opponent).toUpperCase()
      : ''
  useLineupDetail(gamePk, opponentAbbr)

  if (isLoading) return <Loading label="Loading lineup…" />
  if (isError) {
    return <Unavailable title="Lineup not found" reason={String(error)} />
  }
  if (data && !data.available) {
    return <Unavailable data={data} title="Lineup unavailable" />
  }
  if (!data?.available) return null

  const lu = data.lineup
  const thisTeam = String(lu.team ?? team).toUpperCase()
  const oppTeam = lu.opponent ? String(lu.opponent).toUpperCase() : ''
  const ev =
    lu.evaluation && !isUnavailable(lu.evaluation)
      ? (lu.evaluation as Record<string, unknown>)
      : null

  const actualExp = pickNumber(ev ?? undefined, [
    'actual_expected_runs',
    'actual_runs',
    'expected_runs',
  ])
  const bestExp = pickNumber(ev ?? undefined, [
    'best_expected_runs',
    'best_runs',
    'optimal_expected_runs',
  ])
  const gap = pickNumber(ev ?? undefined, ['gap', 'order_gap'])
  const percentile = pickNumber(ev ?? undefined, ['percentile', 'pct'])

  const batters = lu.batters
  const bestOrderIdsRaw = ev?.best_order_ids ?? ev?.best_order
  const bestOrderIds: number[] | undefined = Array.isArray(bestOrderIdsRaw)
    ? (bestOrderIdsRaw as number[])
    : typeof bestOrderIdsRaw === 'string'
      ? (() => {
          try {
            const p = JSON.parse(bestOrderIdsRaw)
            return Array.isArray(p) ? (p as number[]) : undefined
          } catch {
            return undefined
          }
        })()
      : undefined
  const bestOrderNames = (ev?.best_order_names as string[] | undefined) ?? undefined
  const namesMap: Record<string, string> = {
    ...((ev?.player_names ?? {}) as Record<string, string>),
  }
  for (const b of batters ?? []) {
    if (b?.player_id != null && b?.name) namesMap[String(b.player_id)] = String(b.name)
  }

  const nearOrders = asList(
    ev?.near_optimal_orders ?? lu.near_optimal_orders ?? [],
  )
  const explanations = asList(ev?.explanations ?? lu.explanations ?? [])
  const adjacent = asList(ev?.adjacent ?? lu.adjacent ?? lu.adjacent_connections ?? [])
  const sim =
    ev?.simulation && typeof ev.simulation === 'object'
      ? (ev.simulation as Record<string, unknown>)
      : null

  return (
    <div>
      <PageHeader
        eyebrow={`Game ${lu.game_pk}`}
        title={`${lu.team} lineup`}
        description={`${fmtDate(lu.game_date)} · ${handLabel(lu.opp_sp_hand)}${
          lu.opp_sp_name ? ` · vs ${lu.opp_sp_name}` : ''
        }`}
        actions={
          oppTeam ? (
            <div>
              <div className="eyebrow mb-2">Same game</div>
              <div className="flex flex-wrap gap-2">
                <Link
                  to={`/lineups/${lu.game_pk}/${thisTeam}`}
                  className="btn btn-primary"
                  aria-current="page"
                >
                  {thisTeam}
                </Link>
                <Link
                  to={`/lineups/${lu.game_pk}/${oppTeam}`}
                  className="btn btn-secondary"
                >
                  {oppTeam}
                </Link>
              </div>
              <p className="mt-2 mb-0 text-xs text-[var(--color-muted)]">
                Switch clubs to see the other batting order.
              </p>
            </div>
          ) : (
            <TeamBadge abbr={thisTeam} />
          )
        }
      />

      <div className="mb-10 grid gap-8 lg:grid-cols-[1fr_1.1fr]">
        <section className="panel p-5">
          <h2 className="font-display mb-3 text-lg tracking-tight">
            Actual batting order
          </h2>
          <LineupOrder batters={batters} />
        </section>

        <section>
          <div className="mb-6 grid gap-5 sm:grid-cols-2">
            <Metric
              label="Your order — projected R/G"
              value={actualExp}
              hint="Expected runs per 9 innings using this batting order."
            />
            <Metric
              label="Best same-nine order — projected R/G"
              value={bestExp}
              hint="Highest projection using these exact nine hitters, only reordered."
              accent
            />
            <Metric
              label="Optimization gap — R/G"
              value={gap}
              hint="How far this order trails the model’s best order. Smaller is better."
            />
            <Metric
              label="Percentile"
              value={percentile}
              digits={0}
              hint={
                percentile != null
                  ? `This order beats about ${Number(percentile).toFixed(0)}% of all batting orders of these same nine. Higher is better.`
                  : 'Share of same-nine batting orders this lineup beats. Higher is better.'
              }
            />
          </div>

          {lu.evaluation && isUnavailable(lu.evaluation) ? (
            <Unavailable
              data={lu.evaluation}
              title="Lineup evaluation unavailable"
              className="mb-4"
            />
          ) : null}

          {bestOrderIds && bestOrderIds.length === 9 ? (
            <div className="panel p-5">
              <h3 className="font-display mb-3 text-lg tracking-tight">
                Modeled optimal order
              </h3>
              <LineupOrder
                ids={bestOrderIds}
                names={
                  bestOrderNames ??
                  bestOrderIds.map(
                    (id) => namesMap[String(id)] ?? String(id),
                  )
                }
                linkPlayers
              />
            </div>
          ) : null}
        </section>
      </div>

      {nearOrders.length > 0 ? (
        <section className="mb-10">
          <h2 className="font-display mb-3 text-xl tracking-tight">
            Near-optimal alternatives
          </h2>
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Expected runs</th>
                  <th>Gap to best</th>
                  <th>Order</th>
                </tr>
              </thead>
              <tbody>
                {nearOrders.slice(0, 12).map((row, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      {fmtNum(
                        pickNumber(row, [
                          'expected_runs',
                          'runs',
                          'expected_runs_9',
                        ]),
                      )}
                    </td>
                    <td>{fmtNum(pickNumber(row, ['gap', 'delta']))}</td>
                    <td className="text-sm text-[var(--color-muted)]">
                      {Array.isArray(row.order_names)
                        ? (row.order_names as string[]).join(' · ')
                        : Array.isArray(row.order_ids)
                          ? (row.order_ids as number[]).join('-')
                          : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {explanations.length > 0 ? (
        <section className="mb-10">
          <h2 className="font-display mb-3 text-xl tracking-tight">
            Why the gap?
          </h2>
          <ul className="m-0 list-none space-y-3 p-0">
            {explanations.map((ex, i) => (
              <li key={i} className="panel px-4 py-3 text-sm leading-relaxed">
                {String(
                  ex.text ??
                    ex.explanation ??
                    ex.reason ??
                    ex.summary ??
                    JSON.stringify(ex),
                )}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {sim ? (
        <section className="mb-10">
          <h2 className="font-display mb-3 text-xl tracking-tight">
            Simulation summary
          </h2>
          <div className="grid gap-5 sm:grid-cols-3">
            <Metric
              label="Sim mean runs"
              value={pickNumber(sim, ['mean_runs', 'mean'])}
            />
            <Metric
              label="Sim p50"
              value={pickNumber(sim, ['p50', 'median'])}
            />
            <Metric
              label="Games simulated"
              value={pickNumber(sim, ['n_games'])}
              format="int"
            />
          </div>
        </section>
      ) : null}

      {adjacent.length > 0 ? (
        <section className="mb-10">
          <h2 className="font-display mb-3 text-xl tracking-tight">
            Adjacent connections
          </h2>
          <p className="mb-3 text-sm text-[var(--color-muted)]">
            Estimated associations between consecutive hitters — not causal
            claims. Sample size shown when provided.
          </p>
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Pair</th>
                  <th>Estimated effect</th>
                  <th>n</th>
                  <th>Reliability</th>
                </tr>
              </thead>
              <tbody>
                {adjacent.map((row, i) => (
                  <tr key={i}>
                    <td>
                      {String(
                        row.pair_label ??
                          `${row.prev_name ?? row.player_a ?? '—'} → ${
                            row.name ?? row.player_b ?? '—'
                          }`,
                      )}
                    </td>
                    <td>
                      {fmtNum(
                        pickNumber(row, [
                          'effect',
                          'shrunk_effect',
                          'estimate',
                          'delta',
                        ]),
                      )}
                    </td>
                    <td>
                      {pickNumber(row, ['n', 'n_pa', 'sample_size']) ?? '—'}
                    </td>
                    <td className="text-sm text-[var(--color-muted)]">
                      {String(row.reliability ?? row.tier ?? '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="border-t border-[var(--color-border)] pt-8">
        <h2 className="font-display mb-2 text-xl tracking-tight">
          Observed game result
        </h2>
        <p className="mb-4 max-w-2xl text-sm text-[var(--color-muted)]">
          Observed runs are the actual game outcome and are shown separately
          from expected runs. Do not treat them as a direct measure of
          batting-order quality for a single game.
        </p>
        <div className="grid gap-5 sm:grid-cols-3">
          <Metric
            label="Observed runs scored"
            value={lu.runs_scored}
            format="int"
            accent
          />
          <Metric label="Result" value={lu.result ?? '—'} format="raw" />
          <Metric
            label="Venue"
            value={lu.venue ?? (lu.is_home ? 'Home' : lu.is_home === false ? 'Away' : '—')}
            format="raw"
          />
        </div>
        <div className="mt-6">
          <Link
            to={`/teams/${lu.team}`}
            className="text-sm font-semibold hover:underline"
          >
            ← Back to {lu.team}
          </Link>
        </div>
      </section>
    </div>
  )
}
