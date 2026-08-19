import { Link } from 'react-router-dom'
import { useFindings, useLeagueOverview } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { Loading } from '../components/Loading'
import { Unavailable } from '../components/Unavailable'
import { ArchetypeGuide } from '../components/ArchetypeGuide'
import { fmtNum, pickNumber } from '../lib/format'
import { DIVISIONS } from '../lib/teams'

function findingItems(findings: unknown): Array<{ title: string; body: string }> {
  if (!findings) return []
  if (Array.isArray(findings)) {
    return findings.slice(0, 4).map((f, i) => {
      if (typeof f === 'string') return { title: `Finding ${i + 1}`, body: f }
      const o = f as Record<string, unknown>
      return {
        title: String(o.question ?? o.title ?? o.name ?? `Finding ${i + 1}`),
        body: String(
          o.answer ?? o.body ?? o.summary ?? o.text ?? o.description ?? JSON.stringify(o),
        ),
      }
    })
  }
  if (typeof findings === 'object') {
    const o = findings as Record<string, unknown>
    if (Array.isArray(o.statements)) return findingItems(o.statements)
    if (Array.isArray(o.items)) return findingItems(o.items)
    if (Array.isArray(o.findings)) return findingItems(o.findings)
    return Object.entries(o)
      .filter(([k]) => !['available', 'reason', 'source', 'status', 'title', 'generated_from', 'unavailable_inputs'].includes(k))
      .slice(0, 4)
      .map(([k, v]) => ({
        title: k.replace(/_/g, ' '),
        body: typeof v === 'string' ? v : JSON.stringify(v),
      }))
  }
  return [{ title: 'Findings', body: String(findings) }]
}

export function HomePage() {
  const overview = useLeagueOverview()
  const findings = useFindings()

  const teams =
    overview.data && overview.data.available && Array.isArray(overview.data.teams)
      ? overview.data.teams
      : []

  return (
    <div>
      {/* Hero — brand first, compact so league content is visible without scrolling past empty space */}
      <section className="relative mb-10 overflow-hidden border-b border-[var(--color-border)] pb-8 pt-2 sm:pb-10 sm:pt-4">
        <div
          className="pointer-events-none absolute inset-0 -z-10 opacity-90"
          aria-hidden
          style={{
            background:
              'linear-gradient(135deg, rgba(19,35,55,0.08) 0%, transparent 42%), linear-gradient(180deg, transparent 60%, rgba(246,247,249,0.9) 100%), repeating-linear-gradient(-12deg, transparent, transparent 18px, rgba(19,35,55,0.03) 18px, rgba(19,35,55,0.03) 19px)',
          }}
        />
        <p className="eyebrow fade-in mb-3 text-[var(--color-accent)]">
          2026 MLB season
        </p>
        <h1 className="hero-brand fade-in text-[clamp(2.2rem,6vw,3.75rem)] leading-[1.05] text-[var(--color-ink)]">
          MLB Lineup Intelligence
        </h1>
        <p className="fade-in-delay font-display mt-4 max-w-2xl text-xl font-medium tracking-tight text-[var(--color-navy)] sm:text-2xl">
          How much does batting order actually matter?
        </p>
        <p className="fade-in-delay-2 mt-3 max-w-xl text-base leading-relaxed text-[var(--color-muted)]">
          Every 2026 MLB lineup, evaluated through run expectancy, simulation,
          lineup optimization, and interaction modeling.
        </p>
        <div className="fade-in-delay-2 mt-6 flex flex-wrap gap-3">
          <Link to="/explorer" className="btn btn-primary">
            Open explorer
          </Link>
          <Link to="/teams" className="btn btn-secondary">
            Teams
          </Link>
          <Link to="/research" className="btn btn-secondary">
            Methodology
          </Link>
        </div>
      </section>

      {/* 30-team table */}
      <section className="mb-14">
        <div className="mb-4">
          <div className="eyebrow mb-1">League</div>
          <h2 className="font-display m-0 text-2xl tracking-tight">
            30-team overview
          </h2>
          <p className="mt-1 mb-0 text-sm text-[var(--color-muted)]">
            Each starting lineup is scored on its own game, then averaged for
            the team. Different nines across the season all count — this is not
            one fixed lineup.
          </p>
          <div className="mt-3 space-y-1.5 text-sm leading-relaxed text-[var(--color-muted)]">
            <p className="m-0">
              <span className="font-semibold text-[var(--color-ink)]">Order gap</span>
              {' '}— for each game: expected runs left vs the best reordering of{' '}
              <em>that game&apos;s</em> nine, then the team average. Smaller is
              better.
            </p>
            <p className="m-0">
              <span className="font-semibold text-[var(--color-ink)]">% ≤0.02</span>
              {' '}— share of that team&apos;s games where the order gap was within
              0.02 runs of same-nine optimum.
            </p>
            <p className="m-0">
              <span className="font-semibold text-[var(--color-ink)]">Talent (projected R/G)</span>
              {' '}— average modeled expected runs of the actual batting order each
              game (how strong the nine was, not how well they were ordered).
            </p>
          </div>
        </div>
        {overview.isLoading ? <Loading label="Loading league overview…" /> : null}
        {overview.data && !overview.data.available ? (
          <Unavailable data={overview.data} title="League overview unavailable" />
        ) : null}
        {teams.length > 0 ? (
          <div className="space-y-8">
            {DIVISIONS.map((div) => {
              const rows = teams.filter((t) => t.division === div)
              if (!rows.length) return null
              return (
                <div key={div}>
                  <h3 className="eyebrow mb-2">{div}</h3>
                  <div className="panel overflow-hidden">
                    <table className="table-dense">
                      <thead>
                        <tr>
                          <th>Team</th>
                          <th>Games</th>
                          <th>Order gap</th>
                          <th>% ≤0.02</th>
                          <th>Talent (projected R/G)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((t) => {
                          const summary =
                            t.summary && !isUnavailable(t.summary)
                              ? (t.summary as Record<string, unknown>)
                              : t
                          const games =
                            t.games ??
                            t.games_games ??
                            pickNumber(summary as Record<string, unknown>, [
                              'games',
                              'n_games',
                              'games_games',
                            ])
                          const exp = pickNumber(summary as Record<string, unknown>, [
                            'avg_actual_runs',
                            'avg_expected_runs',
                            'mean_expected_runs',
                            'expected_runs',
                            'avg_actual_expected_runs',
                          ])
                          const gap = pickNumber(summary as Record<string, unknown>, [
                            'avg_gap',
                            'mean_gap',
                            'gap',
                          ])
                          const within02 = pickNumber(summary as Record<string, unknown>, [
                            'pct_within_02',
                            'pct_operationally_equivalent',
                          ])
                          const metricsOk =
                            t.metrics_available !== false &&
                            (exp !== undefined || gap !== undefined || within02 !== undefined)
                          return (
                            <tr key={t.abbr}>
                              <td>
                                <Link
                                  to={`/teams/${t.abbr}`}
                                  className="font-semibold hover:underline"
                                >
                                  {t.abbr}
                                </Link>
                                <span className="ml-2 text-[var(--color-muted)]">
                                  {t.name}
                                </span>
                              </td>
                              <td>{games ?? '—'}</td>
                              <td>
                                {metricsOk && gap !== undefined ? fmtNum(gap) : '—'}
                              </td>
                              <td>
                                {metricsOk && within02 !== undefined
                                  ? `${(within02 <= 1 ? within02 * 100 : within02).toFixed(0)}%`
                                  : '—'}
                              </td>
                              <td>
                                {metricsOk && exp !== undefined ? fmtNum(exp) : '—'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )
            })}
          </div>
        ) : null}
        {overview.data?.available && overview.data.metrics && isUnavailable(overview.data.metrics) ? (
          <p className="mt-3 text-sm text-[var(--color-muted)]">
            Summary metrics not yet precomputed — showing structural counts only.
          </p>
        ) : null}
      </section>

      {/* Research findings */}
      <section className="mb-6">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <div className="eyebrow mb-1">Research</div>
            <h2 className="font-display m-0 text-2xl tracking-tight">
              Findings
            </h2>
          </div>
          <Link to="/research" className="text-sm font-semibold text-[var(--color-accent)] hover:underline">
            Full methodology →
          </Link>
        </div>
        <ArchetypeGuide />
        {findings.isLoading ? <Loading label="Loading findings…" /> : null}
        {findings.data && !findings.data.available ? (
          <Unavailable data={findings.data} title="Research findings unavailable" />
        ) : null}
        {findings.data?.available ? (
          <div className="grid gap-4 md:grid-cols-2">
            {findingItems(findings.data.findings).map((f) => (
              <article key={f.title} className="panel p-5">
                <h3 className="font-display m-0 text-lg tracking-tight">{f.title}</h3>
                <p className="mt-2 mb-0 whitespace-pre-line text-sm leading-relaxed text-[var(--color-muted)]">
                  {f.body}
                </p>
              </article>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  )
}
