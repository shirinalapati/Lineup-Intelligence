import { Link } from 'react-router-dom'
import { useTeams } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { Loading } from '../components/Loading'
import { PageHeader } from '../components/PageHeader'
import { Unavailable } from '../components/Unavailable'
import { pickNumber } from '../lib/format'
import { DIVISIONS, teamColor } from '../lib/teams'

export function TeamsPage() {
  const { data, isLoading, isError, error } = useTeams()
  const teams = data?.available ? data.teams : []

  return (
    <div>
      <PageHeader
        eyebrow="Clubs"
        title="Teams"
        description="Browse all 30 clubs. Open a team for batting-slot heatmaps, modeled efficiency, and lineup history."
      />

      {isLoading ? <Loading /> : null}
      {isError ? <Unavailable title="Failed to load teams" reason={String(error)} /> : null}
      {data && !data.available ? <Unavailable data={data} /> : null}

      <div className="grid gap-8">
        {DIVISIONS.map((div) => {
          const rows = teams.filter((t) => t.division === div)
          if (!rows.length) return null
          return (
            <section key={div}>
              <h2 className="eyebrow mb-3">{div}</h2>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {rows.map((t) => {
                  const summary =
                    t.summary && !isUnavailable(t.summary)
                      ? (t.summary as Record<string, unknown>)
                      : null
                  const games =
                    t.games_games ??
                    pickNumber(summary ?? undefined, ['games', 'n_games']) ??
                    '—'
                  return (
                    <Link
                      key={t.abbr}
                      to={`/teams/${t.abbr}`}
                      className="panel flex items-center gap-3 px-4 py-3 transition-colors hover:border-[var(--color-border-strong)] hover:bg-white"
                    >
                      <span
                        className="h-8 w-1.5 rounded-full"
                        style={{ background: teamColor(t.abbr) }}
                        aria-hidden
                      />
                      <div className="min-w-0">
                        <div className="font-semibold tracking-wide">{t.abbr}</div>
                        <div className="truncate text-sm text-[var(--color-muted)]">
                          {t.name}
                        </div>
                      </div>
                      <div className="ml-auto text-right text-xs text-[var(--color-muted)]">
                        <div className="font-semibold text-[var(--color-ink)]">
                          {games}
                        </div>
                        games
                      </div>
                    </Link>
                  )
                })}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}
