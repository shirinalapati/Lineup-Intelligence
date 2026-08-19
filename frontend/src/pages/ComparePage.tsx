import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTeam, useTeams } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { Unavailable } from '../components/Unavailable'
import { pickNumber } from '../lib/format'

function summaryMetrics(summary: unknown) {
  if (!summary || isUnavailable(summary)) return null
  const s = summary as Record<string, unknown>
  return {
    games: pickNumber(s, ['games', 'n_games']),
    expected: pickNumber(s, [
      'avg_actual_runs',
      'avg_expected_runs',
      'mean_expected_runs',
      'avg_actual_expected_runs',
    ]),
    gap: pickNumber(s, ['avg_gap', 'mean_gap', 'gap']),
    percentile: pickNumber(s, [
      'avg_percentile',
      'mean_percentile',
      'percentile',
    ]),
  }
}

export function ComparePage() {
  const teamsQ = useTeams()
  const [a, setA] = useState('NYY')
  const [b, setB] = useState('LAD')
  const teamA = useTeam(a)
  const teamB = useTeam(b)

  const options = useMemo(() => {
    if (!teamsQ.data?.available) return []
    return teamsQ.data.teams.map((t) => ({
      abbr: t.abbr,
      name: t.name,
    }))
  }, [teamsQ.data])

  const ma =
    teamA.data?.available
      ? summaryMetrics(teamA.data.summary) ?? {
          games: teamA.data.games,
          expected: undefined,
          gap: undefined,
          percentile: undefined,
        }
      : null
  const mb =
    teamB.data?.available
      ? summaryMetrics(teamB.data.summary) ?? {
          games: teamB.data.games,
          expected: undefined,
          gap: undefined,
          percentile: undefined,
        }
      : null

  return (
    <div>
      <PageHeader
        eyebrow="Side by side"
        title="Compare teams"
        description="Compare modeled batting-order efficiency summaries for two clubs. Missing metrics stay blank — never invented."
      />

      <div className="mb-8 grid gap-4 sm:grid-cols-2">
        <label className="text-sm">
          <span className="eyebrow mb-1 block">Team A</span>
          <select
            className="input"
            value={a}
            onChange={(e) => setA(e.target.value)}
          >
            {options.map((t) => (
              <option key={t.abbr} value={t.abbr}>
                {t.abbr} — {t.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm">
          <span className="eyebrow mb-1 block">Team B</span>
          <select
            className="input"
            value={b}
            onChange={(e) => setB(e.target.value)}
          >
            {options.map((t) => (
              <option key={t.abbr} value={t.abbr}>
                {t.abbr} — {t.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {(teamA.isLoading || teamB.isLoading) && <Loading />}

      <div className="grid gap-6 lg:grid-cols-2">
        <TeamCompareCard
          label="Team A"
          abbr={a}
          loading={teamA.isLoading}
          data={teamA.data}
          metrics={ma}
        />
        <TeamCompareCard
          label="Team B"
          abbr={b}
          loading={teamB.isLoading}
          data={teamB.data}
          metrics={mb}
        />
      </div>
    </div>
  )
}

function TeamCompareCard({
  label,
  abbr,
  loading,
  data,
  metrics,
}: {
  label: string
  abbr: string
  loading: boolean
  data: ReturnType<typeof useTeam>['data']
  metrics: ReturnType<typeof summaryMetrics> | {
    games?: number
    expected?: number
    gap?: number
    percentile?: number
  } | null
}) {
  if (loading) return <Loading />
  if (data && !data.available) {
    return <Unavailable data={data} title={`${abbr} unavailable`} />
  }
  if (!data?.available) return null

  return (
    <section className="panel p-5">
      <div className="eyebrow mb-1">{label}</div>
      <h2 className="font-display m-0 text-2xl tracking-tight">
        <Link to={`/teams/${abbr}`} className="hover:underline">
          {data.name}
        </Link>
      </h2>
      <p className="mt-1 text-sm text-[var(--color-muted)]">{data.division}</p>

      {data.summary && isUnavailable(data.summary) ? (
        <Unavailable
          className="mt-4"
          data={data.summary}
          title="Summary metrics unavailable"
        />
      ) : null}

      <div className="mt-6 grid gap-5 sm:grid-cols-2">
        <Metric label="Games" value={metrics?.games ?? data.games} format="int" />
        <Metric label="Unique orders" value={data.unique_orders} format="int" />
        <Metric
          label="Avg expected runs"
          value={metrics?.expected}
          hint="Model estimate"
        />
        <Metric label="Avg order gap" value={metrics?.gap} />
        <Metric
          label="Avg percentile"
          value={metrics?.percentile}
          digits={0}
        />
        <Metric
          label="Unique personnel"
          value={data.unique_personnel}
          format="int"
        />
      </div>
    </section>
  )
}
