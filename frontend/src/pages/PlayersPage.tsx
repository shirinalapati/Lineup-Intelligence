import { useState } from 'react'
import { Link } from 'react-router-dom'
import { usePlayers } from '../api/hooks'
import { isUnavailable } from '../api/types'
import { Loading } from '../components/Loading'
import { PageHeader } from '../components/PageHeader'
import { TeamBadge } from '../components/TeamBadge'
import { Unavailable } from '../components/Unavailable'
import { pickNumber } from '../lib/format'

export function PlayersPage() {
  const [q, setQ] = useState('')
  const [offset, setOffset] = useState(0)
  const limit = 200
  const { data, isLoading, isError, error } = usePlayers(q, limit, offset)

  return (
    <div>
      <PageHeader
        eyebrow="Directory"
        title="Players"
        description="All 2026 starting hitters with slot usage, offensive profile, and modeled slot fit when available."
      />

      <div className="mb-6 max-w-md">
        <label className="eyebrow mb-1 block" htmlFor="players-q">
          Search
        </label>
        <input
          id="players-q"
          className="input"
          placeholder="Player name…"
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setOffset(0)
          }}
        />
      </div>

      <p className="mb-4 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
        <strong className="text-[var(--color-ink)]">Primary actual slot</strong>{' '}
        is this hitter&apos;s most common starting slot this season.{' '}
        <strong className="text-[var(--color-ink)]">Best modeled slot</strong>{' '}
        is the slot with the highest projected runs per game in same-nine tests.
      </p>

      {isLoading ? <Loading label="Loading players…" /> : null}
      {isError ? (
        <Unavailable title="Failed to load players" reason={String(error)} />
      ) : null}
      {data && !data.available ? (
        <Unavailable data={data} title="Player directory unavailable" />
      ) : null}

      {data?.available ? (
        <>
          <div className="panel overflow-hidden">
            <table className="table-dense">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Team</th>
                  <th>Bat</th>
                  <th>Pos</th>
                  <th>Games</th>
                  <th>Archetype</th>
                  <th>Primary actual slot</th>
                  <th>Best modeled slot</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {data.players.map((p) => {
                  const profile =
                    p.profile && !isUnavailable(p.profile)
                      ? (p.profile as Record<string, unknown>)
                      : null
                  const team =
                    (p as { team?: string }).team ??
                    (Array.isArray(profile?.teams)
                      ? String((profile?.teams as string[])[0] ?? '')
                      : '')
                  const games =
                    pickNumber(p as Record<string, unknown>, ['games']) ??
                    pickNumber(profile ?? undefined, ['games'])
                  const primary =
                    pickNumber(p as Record<string, unknown>, [
                      'primary_actual_slot',
                      'primary_slot',
                    ]) ?? pickNumber(profile ?? undefined, ['primary_slot'])
                  const best = pickNumber(p as Record<string, unknown>, [
                    'best_modeled_slot',
                    'best_slot',
                  ])
                  const archetype =
                    (p as { archetype?: string }).archetype ??
                    profile?.archetype_label ??
                    profile?.archetype
                  const slotDiffers =
                    primary != null && best != null && Number(primary) !== Number(best)
                  return (
                    <tr key={p.player_id}>
                      <td className="font-medium">{p.name}</td>
                      <td>
                        {team ? <TeamBadge abbr={String(team)} size="sm" /> : '—'}
                      </td>
                      <td>{p.bat_side ?? '—'}</td>
                      <td>{p.position ?? '—'}</td>
                      <td>{games ?? '—'}</td>
                      <td className="text-sm">{String(archetype ?? '—')}</td>
                      <td className="tabular-nums">
                        {primary != null ? `#${primary}` : '—'}
                      </td>
                      <td
                        className={`tabular-nums ${
                          slotDiffers ? 'font-semibold text-[var(--color-accent)]' : ''
                        }`}
                      >
                        {best != null ? `#${best}` : '—'}
                      </td>
                      <td className="text-right">
                        <Link
                          to={`/players/${p.player_id}`}
                          className="text-sm font-semibold hover:underline"
                        >
                          Profile
                        </Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {data.players.length === 0 ? (
              <p className="px-4 py-3 text-sm text-[var(--color-muted)]">
                No players match that search.
              </p>
            ) : null}
          </div>
          <div className="mt-4 flex items-center justify-between text-sm text-[var(--color-muted)]">
            <span>
              Showing {data.total === 0 ? 0 : data.offset + 1}–
              {Math.min(data.offset + data.players.length, data.total)} of{' '}
              {data.total.toLocaleString()} hitters
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                className="btn btn-secondary !min-h-0 !px-3 !py-1.5"
                disabled={offset <= 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
              >
                Prev
              </button>
              <button
                type="button"
                className="btn btn-secondary !min-h-0 !px-3 !py-1.5"
                disabled={offset + limit >= data.total}
                onClick={() => setOffset(offset + limit)}
              >
                Next
              </button>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
