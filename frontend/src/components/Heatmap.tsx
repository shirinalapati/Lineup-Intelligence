import { isUnavailable } from '../api/types'
import { Unavailable } from './Unavailable'

type Props = {
  data: unknown
  className?: string
}

type Cell = {
  player: string
  playerId?: number
  slot: number
  value: number
}

function playerLabel(raw: unknown, fallback = '—'): string {
  if (raw == null) return fallback
  if (typeof raw === 'string' || typeof raw === 'number') return String(raw)
  if (typeof raw === 'object') {
    const o = raw as Record<string, unknown>
    const name = o.name ?? o.player_name ?? o.fullName ?? o.player_id
    if (name != null && (typeof name === 'string' || typeof name === 'number')) {
      return String(name)
    }
  }
  return fallback
}

function extractCells(data: Record<string, unknown>): {
  players: string[]
  cells: Cell[]
  max: number
} {
  const cells: Cell[] = []

  if (Array.isArray(data.cells)) {
    for (const c of data.cells as Array<Record<string, unknown>>) {
      const slot = Number(c.slot ?? 0)
      const value = Number(c.count ?? c.n ?? c.share ?? 0)
      const player = playerLabel(
        c.player_name ?? c.name ?? c.player ?? c.player_id,
      )
      cells.push({
        player,
        playerId: typeof c.player_id === 'number' ? c.player_id : undefined,
        slot,
        value: Number.isFinite(value) ? value : 0,
      })
    }
  } else if (Array.isArray(data.matrix) && Array.isArray(data.players)) {
    const players = data.players as unknown[]
    const matrix = data.matrix as number[][]
    matrix.forEach((row, pi) => {
      row.forEach((val, si) => {
        cells.push({
          player: playerLabel(players[pi], String(pi)),
          slot: si + 1,
          value: Number(val) || 0,
        })
      })
    })
  } else if (data.data && typeof data.data === 'object') {
    return extractCells(data.data as Record<string, unknown>)
  }

  const players = [...new Set(cells.map((c) => c.player))]
  const max = Math.max(1, ...cells.map((c) => c.value))
  return { players, cells, max }
}

function heatColor(t: number): string {
  // navy ink → stadium green soft scale (readable, not neon)
  const clamped = Math.max(0, Math.min(1, t))
  const r = Math.round(246 - clamped * (246 - 26))
  const g = Math.round(247 - clamped * (247 - 95))
  const b = Math.round(249 - clamped * (249 - 74))
  const a = 0.12 + clamped * 0.78
  return `rgba(${r}, ${g}, ${b}, ${a})`
}

export function Heatmap({ data, className = '' }: Props) {
  if (!data || isUnavailable(data)) {
    return <Unavailable data={data} title="Batting-slot heatmap unavailable" />
  }

  const { players, cells, max } = extractCells(data as Record<string, unknown>)
  if (!players.length || !cells.length) {
    return (
      <Unavailable
        title="Batting-slot heatmap unavailable"
        reason="No heatmap cells were returned for this team."
      />
    )
  }

  const slots = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  const lookup = new Map(cells.map((c) => [`${c.player}|${c.slot}`, c.value]))

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="table-dense min-w-[640px]">
        <thead>
          <tr>
            <th>Player</th>
            {slots.map((s) => (
              <th key={s} className="text-center">
                {s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {players.map((p) => (
            <tr key={p}>
              <td className="whitespace-nowrap font-medium">{p}</td>
              {slots.map((s) => {
                const v = lookup.get(`${p}|${s}`) ?? 0
                return (
                  <td key={s} className="p-0 text-center">
                    <div
                      className="mx-auto flex h-9 w-9 items-center justify-center text-xs font-semibold"
                      style={{ background: heatColor(v / max) }}
                      title={`${p} slot ${s}: ${v}`}
                    >
                      {v || ''}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
