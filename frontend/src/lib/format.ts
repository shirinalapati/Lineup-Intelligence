export function fmtNum(v: unknown, digits = 2): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return '—'
  return n.toFixed(digits)
}

export function fmtOrdinal(n: number): string {
  const v = Math.abs(n) % 100
  const suf =
    v >= 11 && v <= 13
      ? 'th'
      : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[v % 10] || 'th'
  return `${n}${suf}`
}

export function fmtInt(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return '—'
  return Math.round(n).toLocaleString()
}

/** Calendar years must not use thousand separators (2024, not 2,024). */
export function fmtYear(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return String(v)
  return String(Math.round(n))
}

export function fmtPct(v: unknown, digits = 0): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return '—'
  const pct = n <= 1 && n >= 0 ? n * 100 : n
  return `${pct.toFixed(digits)}%`
}

export function fmtDate(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  const s = String(v)
  const d = new Date(s.includes('T') ? s : `${s}T12:00:00`)
  if (Number.isNaN(d.getTime())) return s
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function pickNumber(
  obj: Record<string, unknown> | null | undefined,
  keys: string[],
): number | undefined {
  if (!obj) return undefined
  for (const k of keys) {
    const v = obj[k]
    if (typeof v === 'number' && Number.isFinite(v)) return v
    if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) {
      return Number(v)
    }
  }
  return undefined
}

export function handLabel(hand: unknown): string {
  const h = String(hand ?? '').toUpperCase()
  if (h === 'R' || h === 'RHP' || h === 'VS_R') return 'vs RHP'
  if (h === 'L' || h === 'LHP' || h === 'VS_L') return 'vs LHP'
  if (h === 'N' || h === 'NEUTRAL') return 'Neutral'
  if (!h) return '—'
  return String(hand)
}

/** Platoon context for schedule rows — missing hand is not Neutral. */
export function platoonContextLabel(hand: unknown): string {
  if (hand === null || hand === undefined || String(hand).trim() === '') {
    return 'Hand TBD'
  }
  return handLabel(hand)
}
