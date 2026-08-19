import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useSearch } from '../api/hooks'
import type { SearchResult } from '../api/types'

function resultPath(r: SearchResult): string | null {
  if (r.type === 'team' && r.abbr) return `/teams/${r.abbr}`
  if (r.type === 'player' && r.player_id) return `/players/${r.player_id}`
  if ((r.type === 'game' || r.type === 'lineup') && r.game_pk && r.team) {
    return `/lineups/${r.game_pk}/${r.team}`
  }
  return null
}

function resultLabel(r: SearchResult): string {
  if (r.type === 'team') return `${r.abbr} — ${r.name}`
  if (r.type === 'player') return r.name ?? `Player ${r.player_id}`
  if (r.type === 'game' || r.type === 'lineup') {
    return `${r.team} vs ${r.opponent ?? '—'} · ${r.game_date ?? r.game_pk}`
  }
  return r.name ?? r.type
}

export function GlobalSearch() {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const nav = useNavigate()
  const wrap = useRef<HTMLDivElement>(null)
  const { data, isFetching } = useSearch(q)

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  const results = data && data.available ? data.results : []

  return (
    <div ref={wrap} className="relative w-full max-w-xs">
      <label className="sr-only" htmlFor="global-search">
        Search teams, players, games
      </label>
      <input
        id="global-search"
        className="input text-sm"
        placeholder="Search teams, players, games…"
        value={q}
        onChange={(e) => {
          setQ(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && results[0]) {
            const path = resultPath(results[0])
            if (path) {
              nav(path)
              setOpen(false)
              setQ('')
            }
          }
          if (e.key === 'Escape') setOpen(false)
        }}
        autoComplete="off"
      />
      {open && q.trim() ? (
        <div className="absolute z-40 mt-1 w-full overflow-hidden rounded-[2px] border border-[var(--color-border-strong)] bg-white shadow-lg">
          {isFetching ? (
            <div className="px-3 py-2 text-sm text-[var(--color-muted)]">
              Searching…
            </div>
          ) : null}
          {!isFetching && data && !data.available ? (
            <div className="px-3 py-2 text-sm text-[var(--color-muted)]">
              Search unavailable
            </div>
          ) : null}
          {!isFetching && results.length === 0 ? (
            <div className="px-3 py-2 text-sm text-[var(--color-muted)]">
              No matches
            </div>
          ) : null}
          <ul className="m-0 max-h-72 list-none overflow-auto p-0">
            {results.map((r, i) => {
              const path = resultPath(r)
              const label = resultLabel(r)
              const key = `${r.type}-${i}-${label}`
              if (!path) {
                return (
                  <li
                    key={key}
                    className="border-b border-[var(--color-border)] px-3 py-2 text-sm"
                  >
                    <span className="eyebrow mr-2">{r.type}</span>
                    {label}
                  </li>
                )
              }
              return (
                <li key={key} className="border-b border-[var(--color-border)]">
                  <Link
                    to={path}
                    className="block px-3 py-2 text-sm hover:bg-[var(--color-paper-warm)]"
                    onClick={() => {
                      setOpen(false)
                      setQ('')
                    }}
                  >
                    <span className="eyebrow mr-2">{r.type}</span>
                    {label}
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
      ) : null}
    </div>
  )
}
