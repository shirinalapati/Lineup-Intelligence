import { fmtNum } from '../lib/format'

export type RankPayload = {
  value?: number | null
  rank?: number | null
  population_n?: number | null
  percentile?: number | null
  qualifying_threshold?: number | null
  direction?: string
  qualified?: boolean
  metric?: string
  display?: string | null
  note?: string | null
  sample_size?: number | null
}

type Props = {
  label: string
  mlb?: RankPayload | null
  team?: RankPayload | null
  /** Fallback raw value when ranks unavailable */
  value?: number | null
  format?: 'rate' | 'woba' | 'count' | 'raw' | 'runs'
  className?: string
}

function ordinal(n: number): string {
  const v = Math.abs(n) % 100
  const suf =
    v >= 11 && v <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[v % 10] || 'th'
  return `${n}${suf}`
}

function formatValue(p: RankPayload | null | undefined, fallback: number | null | undefined, format: Props['format']): string {
  if (p?.display) return p.display
  const v = p?.value ?? fallback
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  const n = Number(v)
  if (format === 'rate') return n <= 1 ? `${(n * 100).toFixed(1)}%` : `${n.toFixed(1)}%`
  if (format === 'woba') return n.toFixed(3)
  if (format === 'count') return fmtNum(n, 0)
  if (format === 'runs') return fmtNum(n, 3)
  return fmtNum(n, 3)
}

export function RankedMetric({
  label,
  mlb,
  team,
  value,
  format = 'woba',
  className = '',
}: Props) {
  const hasMlb = mlb != null && (mlb.value != null || mlb.rank != null || mlb.note != null)
  const qualified = Boolean(
    mlb?.qualified && mlb?.rank != null && mlb?.population_n != null,
  )
  const title = [
    mlb?.note,
    mlb?.qualifying_threshold != null
      ? `Min sample: ${mlb.qualifying_threshold}`
      : null,
    mlb?.direction ? `Direction: ${mlb.direction.replace('_', ' ')}` : null,
    team?.rank != null && team?.population_n != null
      ? `Team: ${ordinal(team.rank)} / ${team.population_n}`
      : null,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className={`min-w-0 ${className}`} title={title || undefined}>
      <div className="eyebrow mb-1">{label}</div>
      <div className="font-display text-2xl tracking-tight text-[var(--color-ink)]">
        {formatValue(mlb, value, format)}
      </div>
      {qualified ? (
        <div className="mt-1 space-y-0.5 text-xs text-[var(--color-muted)]">
          <div>
            {ordinal(Number(mlb!.rank))} / {mlb!.population_n}
          </div>
          {mlb!.percentile != null ? (
            <div>{mlb!.percentile}th %ile</div>
          ) : null}
          {team?.qualified && team.rank != null && team.population_n != null ? (
            <div className="text-[11px] opacity-80">
              Team {ordinal(team.rank)} / {team.population_n}
            </div>
          ) : null}
        </div>
      ) : hasMlb && mlb?.note ? (
        <div className="mt-1 text-xs text-[var(--color-muted)]">{mlb.note}</div>
      ) : hasMlb && mlb?.qualified === false ? (
        <div className="mt-1 text-xs text-[var(--color-muted)]">
          Limited sample — rank unavailable
        </div>
      ) : null}
    </div>
  )
}

/** Compact 1–9 slot fit strip */
export function SlotFitStrip({
  slots,
  bestSlot,
  primarySlot,
  nearSlots,
}: {
  slots: Array<{ slot: number; expected_runs: number; delta_vs_avg?: number }>
  bestSlot?: number | null
  primarySlot?: number | null
  nearSlots?: number[]
}) {
  if (!slots?.length) return null
  const deltas = slots.map((s) => Number(s.delta_vs_avg ?? 0))
  const maxAbs = Math.max(0.02, ...deltas.map((d) => Math.abs(d)))

  return (
    <div className="grid grid-cols-9 gap-1">
      {slots.map((s) => {
        const d = Number(s.delta_vs_avg ?? 0)
        const intensity = Math.min(1, Math.abs(d) / maxAbs)
        const positive = d >= 0
        const isBest = bestSlot === s.slot
        const isPrimary = primarySlot === s.slot
        const isNear = nearSlots?.includes(s.slot)
        const bg = positive
          ? `rgba(19, 35, 55, ${0.08 + intensity * 0.45})`
          : `rgba(120, 40, 40, ${0.05 + intensity * 0.25})`
        return (
          <div
            key={s.slot}
            className="relative border border-[var(--color-border)] px-1 py-2 text-center"
            style={{ background: bg }}
            title={`Slot ${s.slot}: ${s.expected_runs.toFixed(3)} projected R/G (${d >= 0 ? '+' : ''}${d.toFixed(3)} vs avg)`}
          >
            <div className="font-display text-sm font-semibold">{s.slot}</div>
            <div className="text-[10px] text-[var(--color-muted)]">
              {d >= 0 ? '+' : ''}
              {d.toFixed(3)}
            </div>
            {isBest ? (
              <div className="mt-1 text-[9px] font-semibold uppercase tracking-wide text-[var(--color-accent)]">
                Best
              </div>
            ) : null}
            {isPrimary && !isBest ? (
              <div className="mt-1 text-[9px] font-semibold uppercase tracking-wide">
                Actual
              </div>
            ) : null}
            {isNear && !isBest ? (
              <div className="text-[9px] text-[var(--color-muted)]">≈</div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
