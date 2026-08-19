import { fmtNum, fmtPct, fmtInt, fmtYear, fmtOrdinal } from '../lib/format'

type Format = 'number' | 'int' | 'pct' | 'year' | 'raw'

type Props = {
  label: string
  value: unknown
  format?: Format
  digits?: number
  hint?: string
  /** Technical name, shown beside the public label and as a tooltip. */
  tech?: string
  accent?: boolean
  className?: string
  rank?: number | null
  populationN?: number | null
}

export function Metric({
  label,
  value,
  format = 'number',
  digits = 2,
  hint,
  tech,
  accent = false,
  className = '',
  rank,
  populationN,
}: Props) {
  let display = '—'
  if (value !== null && value !== undefined && value !== '') {
    if (format === 'number') display = fmtNum(value, digits)
    else if (format === 'int') display = fmtInt(value)
    else if (format === 'pct') display = fmtPct(value, digits)
    else if (format === 'year') display = fmtYear(value)
    else display = String(value)
  }

  return (
    <div className={`min-w-0 ${className}`}>
      <div className="eyebrow mb-1" title={tech || undefined}>
        {label}
        {tech ? (
          <span className="ml-1 font-sans text-[10px] font-normal normal-case tracking-normal text-[var(--color-muted-light)]">
            {tech}
          </span>
        ) : null}
      </div>
      <div
        className={`font-display text-2xl tracking-tight ${
          accent ? 'text-[var(--color-accent)]' : 'text-[var(--color-ink)]'
        }`}
      >
        {display}
      </div>
      {rank != null && populationN != null ? (
        <div className="mt-1 text-xs text-[var(--color-muted)]">
          {fmtOrdinal(rank)} / {populationN}
        </div>
      ) : null}
      {hint ? (
        <div className="mt-1.5 max-w-[22rem] text-xs leading-relaxed text-[var(--color-muted)]">
          {hint}
        </div>
      ) : null}
    </div>
  )
}
