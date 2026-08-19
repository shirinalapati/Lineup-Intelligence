import { Link } from 'react-router-dom'
import { teamColor } from '../lib/teams'

type Props = {
  abbr: string
  name?: string
  size?: 'sm' | 'md'
  link?: boolean
  className?: string
}

export function TeamBadge({
  abbr,
  name,
  size = 'md',
  link = true,
  className = '',
}: Props) {
  const a = abbr?.toUpperCase?.() ?? String(abbr)
  const color = teamColor(a)
  const pad = size === 'sm' ? 'px-1.5 py-0.5 text-xs' : 'px-2 py-1 text-sm'
  const inner = (
    <span
      className={`inline-flex items-center gap-1.5 rounded-[2px] border border-[var(--color-border)] bg-white font-semibold tracking-wide ${pad} ${className}`}
      title={name ?? a}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: color }}
        aria-hidden
      />
      <span>{a}</span>
      {name && size !== 'sm' ? (
        <span className="hidden font-normal text-[var(--color-muted)] sm:inline">
          {name}
        </span>
      ) : null}
    </span>
  )

  if (!link) return inner
  return (
    <Link to={`/teams/${a}`} className="hover:opacity-90">
      {inner}
    </Link>
  )
}
