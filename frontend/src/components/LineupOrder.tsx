import { Link } from 'react-router-dom'

export type LineupBatter = {
  slot: number
  player_id: number
  name: string
  position?: string | null
  bat_side?: string | null
}

type Props = {
  batters?: LineupBatter[]
  names?: string[]
  ids?: number[]
  highlightSlots?: number[]
  compact?: boolean
  linkPlayers?: boolean
  className?: string
}

export function LineupOrder({
  batters,
  names,
  ids,
  highlightSlots = [],
  compact = false,
  linkPlayers = true,
  className = '',
}: Props) {
  const rows: LineupBatter[] =
    batters ??
    Array.from({ length: 9 }, (_, i) => ({
      slot: i + 1,
      player_id: ids?.[i] ?? 0,
      name: names?.[i] ?? (ids?.[i] ? String(ids[i]) : '—'),
    }))

  return (
    <ol
      className={`m-0 list-none p-0 ${className}`}
      aria-label="Batting order"
    >
      {rows.map((b) => {
        const hi = highlightSlots.includes(b.slot)
        const nameEl =
          linkPlayers && b.player_id ? (
            <Link
              to={`/players/${b.player_id}`}
              className="font-medium hover:underline"
            >
              {b.name}
            </Link>
          ) : (
            <span className="font-medium">{b.name}</span>
          )
        const meta = [b.position, b.bat_side].filter(
          (x) => x && x !== '?' && x !== 'P',
        )

        return (
          <li
            key={`${b.slot}-${b.player_id}`}
            className={`flex items-center gap-3 border-b border-[var(--color-border)] ${
              compact ? 'py-1.5' : 'py-2.5'
            } ${hi ? 'bg-[var(--color-accent-soft)]' : ''}`}
          >
            <span
              className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-[2px] font-display text-sm font-semibold ${
                hi
                  ? 'bg-[var(--color-accent)] text-white'
                  : 'bg-[var(--color-navy)] text-[var(--color-paper)]'
              }`}
            >
              {b.slot}
            </span>
            <div className="min-w-0 flex-1 truncate">
              {nameEl}
              {meta.length ? (
                <span className="ml-2 text-xs text-[var(--color-muted)]">
                  {meta.join(' · ')}
                </span>
              ) : null}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
