import { NavLink, Outlet, Link } from 'react-router-dom'
import { GlobalSearch } from './GlobalSearch'

const LINKS = [
  { to: '/', label: 'League', end: true },
  { to: '/teams', label: 'Teams' },
  { to: '/explorer', label: 'Explorer' },
  { to: '/players', label: 'Players' },
  { to: '/research', label: 'Research' },
]

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-[var(--color-border)] bg-[color-mix(in_srgb,var(--color-paper)_88%,white)] backdrop-blur-md">
        <div className="page-shell flex flex-col gap-3 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center justify-between gap-4">
            <Link to="/" className="group flex items-baseline gap-2">
              <span className="font-display text-lg font-bold tracking-tight text-[var(--color-ink)] sm:text-xl">
                MLB Lineup Intelligence
              </span>
            </Link>
          </div>
          <nav
            className="flex flex-wrap items-center gap-x-1 gap-y-1 text-sm"
            aria-label="Primary"
          >
            {LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) =>
                  `rounded-[2px] px-2.5 py-1.5 font-medium transition-colors ${
                    isActive
                      ? 'bg-[var(--color-ink)] text-[var(--color-paper)]'
                      : 'text-[var(--color-navy-mid)] hover:bg-white/70 hover:text-[var(--color-ink)]'
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
          <GlobalSearch />
        </div>
      </header>

      <main id="main" className="flex-1 py-8 sm:py-10">
        <div className="page-shell">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-[var(--color-border)] py-8 text-sm text-[var(--color-muted)]">
        <div className="page-shell flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <span className="font-display font-semibold text-[var(--color-ink)]">
              MLB Lineup Intelligence
            </span>
            <span className="mx-2">·</span>
            2026 season research
          </div>
          <p className="m-0 max-w-xl text-xs leading-relaxed">
            Expected runs are model estimates. Observed runs are game outcomes.
            Interaction effects are reported as estimated associations with
            sample-size context when available.
          </p>
        </div>
      </footer>
    </div>
  )
}
