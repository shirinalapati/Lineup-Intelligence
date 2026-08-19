import { useEffect, useMemo, useRef, useState } from 'react'
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  useSortable,
  verticalListSortingStrategy,
  sortableKeyboardCoordinates,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import {
  useEvaluate,
  useOptimize,
  useSimulate,
  useTeamRoster,
  useTeams,
} from '../api/hooks'
import { isUnavailable, type OptimizeResult } from '../api/types'
import { apiPost } from '../api/client'
import { LineupOrder } from '../components/LineupOrder'
import { Loading } from '../components/Loading'
import { Metric } from '../components/Metric'
import { PageHeader } from '../components/PageHeader'
import { TeamBadge } from '../components/TeamBadge'
import { Unavailable } from '../components/Unavailable'
import { fmtInt, fmtNum, fmtOrdinal, fmtPct, pickNumber } from '../lib/format'
import { DIVISIONS, teamColor } from '../lib/teams'

type SlotPlayer = {
  player_id: number
  name: string
  bat_side?: string | null
  position?: string | null
}

type PlatoonContext = 'neutral' | 'vs_R' | 'vs_L'
type ResultView = 'optimize' | 'evaluate' | 'simulate'

function SortableRow({
  id,
  slot,
  name,
  batSide,
}: {
  id: string
  slot: number
  name: string
  batSide?: string | null
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id })
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.85 : 1,
  }
  return (
    <li
      ref={setNodeRef}
      style={style}
      className="flex items-center gap-3 border-b border-[var(--color-border)] bg-white py-2.5 pr-2"
      {...attributes}
      {...listeners}
    >
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[2px] bg-[var(--color-navy)] font-display text-sm font-semibold text-[var(--color-paper)]">
        {slot}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">{name}</span>
        {batSide ? (
          <span className="text-xs text-[var(--color-muted)]">
            Bats {batSide}
          </span>
        ) : null}
      </span>
      <span className="cursor-grab text-xs text-[var(--color-muted-light)] select-none">
        drag
      </span>
    </li>
  )
}

function contextLabel(ctx: PlatoonContext): string {
  if (ctx === 'vs_R') return 'vs RHP'
  if (ctx === 'vs_L') return 'vs LHP'
  return 'Neutral'
}

const RISP_BASES = new Set(['-2-', '--3', '12-', '1-3', '-23', '123'])

function numOrNull(v: unknown): number | null {
  if (v === null || v === undefined || v === '') return null
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : null
}

function flowRowMetrics(f: Record<string, unknown>): {
  expectedPa: number | null
  runnersOn: number | null
  risp: number | null
  avgRunners: number | null
  leadoff: number | null
  twoOut: number | null
} {
  let runnersOn = numOrNull(f.runners_on_pct) ?? numOrNull(f.prob_runners_on)
  let risp = numOrNull(f.risp_pct)
  let avgRunners = numOrNull(f.avg_runners_on)
  let leadoff = numOrNull(f.leadoff_pct)
  let twoOut = numOrNull(f.two_out_pct)
  const dist = f.base_state_distribution
  if (dist && typeof dist === 'object') {
    let rOn = 0
    let rRisp = 0
    let rAvg = 0
    let rLead = 0
    let rTwo = 0
    for (const [key, mass] of Object.entries(dist as Record<string, unknown>)) {
      const m = numOrNull(mass)
      if (m == null) continue
      const [outsS, bases = ''] = String(key).split('|')
      const outs = Number(outsS)
      const nRun = [...bases].filter((ch) => ch !== '-').length
      rAvg += m * nRun
      if (bases && bases !== '---') rOn += m
      if (RISP_BASES.has(bases)) rRisp += m
      if (outs === 2) rTwo += m
      if (outs === 0 && bases === '---') rLead += m
    }
    runnersOn ??= rOn
    risp ??= rRisp
    avgRunners ??= rAvg
    leadoff ??= rLead
    twoOut ??= rTwo
  }
  return {
    expectedPa: numOrNull(f.expected_pa),
    runnersOn,
    risp,
    avgRunners,
    leadoff,
    twoOut,
  }
}

function parseHistogram(
  raw: unknown,
): Array<{ runs: number; count: number }> {
  if (!raw || typeof raw !== 'object') return []
  const out: Array<{ runs: number; count: number }> = []
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    const runs = Number(k)
    const count = Number(v)
    if (Number.isFinite(runs) && Number.isFinite(count) && count > 0) {
      out.push({ runs, count })
    }
  }
  return out.sort((a, b) => a.runs - b.runs)
}

function simDisplayStats(result: Record<string, unknown>): {
  p05: number | null
  p95: number | null
  p02: number | null
  p35: number | null
  p6: number | null
  bars: Array<{ runs: number; count: number }>
} {
  const bars = parseHistogram(result.histogram)
  const total = bars.reduce((s, b) => s + b.count, 0)
  const quantile = (p: number): number | null => {
    if (!total) return null
    const target = (p / 100) * total
    let acc = 0
    for (const b of bars) {
      acc += b.count
      if (acc >= target) return b.runs
    }
    return bars[bars.length - 1]?.runs ?? null
  }
  const share = (lo: number, hi?: number): number | null => {
    if (!total) return null
    let c = 0
    for (const b of bars) {
      if (b.runs >= lo && (hi == null || b.runs <= hi)) c += b.count
    }
    return c / total
  }
  return {
    p05: numOrNull(result.p05) ?? quantile(5),
    p95: numOrNull(result.p95) ?? quantile(95),
    p02: numOrNull(result.p0_2) ?? share(0, 2),
    p35: numOrNull(result.p3_5) ?? share(3, 5),
    p6: numOrNull(result.p6_plus) ?? share(6),
    bars,
  }
}

function orderIsBest(
  result: { best_order_ids?: number[]; gap?: number },
  orderIds: number[],
): boolean {
  const best = result.best_order_ids
  if (
    Array.isArray(best) &&
    best.length === orderIds.length &&
    best.every((id, i) => id === orderIds[i])
  ) {
    return true
  }
  const gap = Number(result.gap)
  return Number.isFinite(gap) && gap <= 1e-6
}

function optimizeInterpretation(opt: {
  rank?: number
  n_perms?: number
  gap?: number
}): string {
  const n = Number(opt.n_perms) || 362880
  const rank = Number(opt.rank)
  const gap = Number(opt.gap)
  const nFmt = n.toLocaleString()
  if (!Number.isFinite(gap) || !Number.isFinite(rank)) {
    return `There are ${nFmt} ways to arrange nine hitters, and many produce almost identical run expectations.`
  }
  if (gap <= 1e-6) {
    return `Your order matches the best of all ${nFmt} possible arrangements of these nine hitters.`
  }
  const rankWord = rank / n > 0.5 ? 'low' : rank / n > 0.2 ? 'in the middle' : 'near the top'
  const practical = gap < 0.05 ? 'small' : 'meaningful'
  return `Your order ranks ${rankWord} among all ${nFmt} possible arrangements, but the practical difference is ${practical}: it projects only ${fmtNum(gap)} fewer runs per game than the best order. Many batting orders are nearly tied, so rank can look dramatic even when the expected-run difference is tiny.`
}

export function ExplorerPage() {
  const [team, setTeam] = useState('')
  const [q, setQ] = useState('')
  const [context, setContext] = useState<PlatoonContext>('neutral')
  const [rosterMode, setRosterMode] = useState<'season' | 'current'>('season')
  const [slots, setSlots] = useState<(SlotPlayer | null)[]>(Array(9).fill(null))
  const [showResults, setShowResults] = useState(false)
  const [resultView, setResultView] = useState<ResultView | null>(null)
  const [fillError, setFillError] = useState<string | null>(null)
  const [rosterWarning, setRosterWarning] = useState<string | null>(null)
  const [fillingCtx, setFillingCtx] = useState<PlatoonContext | null>(null)
  const [alreadyOptimal, setAlreadyOptimal] = useState(false)
  const skipOptimalCheck = useRef(false)
  const optimalCheckGen = useRef(0)

  const teamsQ = useTeams()
  const rosterQ = useTeamRoster(team, {
    mode: rosterMode,
    includeUnavailable: false,
  })
  const optimize = useOptimize()
  const evaluate = useEvaluate()
  const simulate = useSimulate()

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const filled = slots.every(Boolean)
  const playerIds = useMemo(
    () => slots.map((s) => s?.player_id ?? 0),
    [slots],
  )

  const rosterPlayers = useMemo(() => {
    if (!rosterQ.data?.available) return []
    const rows = rosterQ.data.players ?? []
    const ql = q.trim().toLowerCase()
    if (!ql) return rows
    return rows.filter((p) => p.name.toLowerCase().includes(ql))
  }, [rosterQ.data, q])

  const ineligibleSlots = useMemo(() => {
    if (rosterMode === 'season') return [] as SlotPlayer[]
    if (!rosterQ.data?.available) return [] as SlotPlayer[]
    const byId = new Map(
      (rosterQ.data.players ?? []).map((p) => [p.player_id, p]),
    )
    return slots.filter((s): s is SlotPlayer => {
      if (!s) return false
      const row = byId.get(s.player_id)
      if (!row) return true
      return row.selectable === false
    })
  }, [slots, rosterQ.data, rosterMode])

  useEffect(() => {
    if (!ineligibleSlots.length) {
      setRosterWarning(null)
      return
    }
    const names = ineligibleSlots.map((s) => s.name).join(', ')
    setRosterWarning(
      `${names} ${ineligibleSlots.length === 1 ? 'is' : 'are'} not eligible for this roster mode.`,
    )
  }, [ineligibleSlots])

  useEffect(() => {
    if (!filled || ineligibleSlots.length > 0) {
      setAlreadyOptimal(false)
      return
    }
    if (skipOptimalCheck.current) {
      skipOptimalCheck.current = false
      setAlreadyOptimal(true)
      return
    }
    const gen = ++optimalCheckGen.current
    const ids = [...playerIds]
    const ctx = context
    setAlreadyOptimal(false)
    void apiPost<OptimizeResult>('/api/optimize', {
      player_ids: ids,
      order: ids,
      context: ctx,
    })
      .then((res) => {
        if (gen !== optimalCheckGen.current) return
        if (
          !isUnavailable(res) &&
          res.available &&
          res.result &&
          orderIsBest(res.result, ids)
        ) {
          setAlreadyOptimal(true)
        }
      })
      .catch(() => {
        /* keep the card usable if the silent check fails */
      })
  }, [filled, playerIds, context, ineligibleSlots.length])

  function hideResults() {
    setShowResults(false)
    setResultView(null)
  }

  function changeRosterMode(next: 'season' | 'current') {
    setRosterMode(next)
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  function removeIneligible() {
    const bad = new Set(ineligibleSlots.map((s) => s.player_id))
    setSlots((prev) => prev.map((s) => (s && bad.has(s.player_id) ? null : s)))
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  function selectTeam(abbr: string) {
    setTeam(abbr)
    setSlots(Array(9).fill(null))
    setQ('')
    hideResults()
    setFillError(null)
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  function clearTeam() {
    setTeam('')
    setSlots(Array(9).fill(null))
    setQ('')
    hideResults()
    setFillError(null)
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  function changeContext(next: PlatoonContext) {
    setContext(next)
    // Platoon split changes the model — hide prior results until re-run.
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  function addPlayer(p: SlotPlayer) {
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
    setSlots((prev) => {
      if (prev.some((s) => s?.player_id === p.player_id)) return prev
      const next = [...prev]
      const idx = next.findIndex((s) => s === null)
      if (idx === -1) return prev
      next[idx] = p
      return next
    })
  }

  function clearSlot(i: number) {
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
    setSlots((prev) => {
      const next = [...prev]
      next[i] = null
      return next
    })
  }

  function loadLatest() {
    const latest = rosterQ.data?.available ? rosterQ.data.latest_lineup : null
    if (!latest?.batting_order?.length) return
    const names = latest.batter_names ?? []
    const byId = new Map(
      (rosterQ.data?.available ? rosterQ.data.players : []).map((p) => [
        p.player_id,
        p,
      ]),
    )
    setSlots(
      latest.batting_order.slice(0, 9).map((pid, i) => {
        const meta = byId.get(pid)
        return {
          player_id: pid,
          name: meta?.name ?? names[i] ?? String(pid),
          bat_side: meta?.bat_side,
          position: meta?.position,
        }
      }),
    )
    hideResults()
    setFillError(null)
    optimize.reset()
    evaluate.reset()
    simulate.reset()
  }

  /** Nine-player pool: current filled order, else latest posted, else top 9 by GS. */
  function personnelPool(): SlotPlayer[] | null {
    if (slots.every(Boolean)) {
      return slots as SlotPlayer[]
    }
    const roster = (rosterQ.data?.available ? rosterQ.data.players : []).filter(
      (p) => p.selectable !== false,
    )
    const byId = new Map(roster.map((p) => [p.player_id, p]))
    const latest = rosterQ.data?.available ? rosterQ.data.latest_lineup : null
    if (latest?.batting_order?.length === 9) {
      const fromLatest = latest.batting_order.map((pid, i) => {
        const meta = byId.get(pid)
        return {
          player_id: pid,
          name: meta?.name ?? latest.batter_names?.[i] ?? String(pid),
          bat_side: meta?.bat_side,
          position: meta?.position,
          eligible: Boolean(meta),
        }
      })
      if (
        rosterMode === 'season' ||
        fromLatest.every((p) => p.eligible)
      ) {
        return fromLatest.map(({ eligible: _e, ...p }) => p)
      }
    }
    if (roster.length >= 9) {
      return roster.slice(0, 9).map((p) => ({
        player_id: p.player_id,
        name: p.name,
        bat_side: p.bat_side,
        position: p.position,
      }))
    }
    return null
  }

  async function fillMostOptimal(ctx: PlatoonContext) {
    setFillError(null)
    if (ineligibleSlots.length) {
      setFillError(
        rosterWarning ||
          'Selected players are not eligible for this roster mode.',
      )
      return
    }
    const pool = personnelPool()
    if (!pool || pool.length !== 9) {
      setFillError(
        'Need nine hitters from this roster (load latest posted, or add nine) before filling an optimal order.',
      )
      return
    }
    const ids = pool.map((p) => p.player_id)
    setFillingCtx(ctx)
    try {
      const res = await optimize.mutateAsync({
        player_ids: ids,
        order: ids,
        context: ctx,
      })
      if (!res.available || !res.result?.best_order_ids) {
        setFillError(
          isUnavailable(res)
            ? String(res.reason ?? 'Optimization unavailable')
            : 'Optimization returned no best order.',
        )
        return
      }
      const names = res.result.player_names ?? {}
      const metaById = new Map(pool.map((p) => [p.player_id, p]))
      skipOptimalCheck.current = true
      optimalCheckGen.current += 1
      setAlreadyOptimal(true)
      setContext(ctx)
      setSlots(
        res.result.best_order_ids.map((pid) => {
          const meta = metaById.get(pid)
          return {
            player_id: pid,
            name: meta?.name ?? names[String(pid)] ?? String(pid),
            bat_side: meta?.bat_side,
            position: meta?.position,
          }
        }),
      )
      // The optimize call compared the *previous* nine's order. After fill,
      // the card is already the best permutation — drop that stale gap.
      optimize.reset()
      evaluate.reset()
      simulate.reset()
      hideResults()
    } catch (err) {
      setFillError(String(err))
    } finally {
      setFillingCtx(null)
    }
  }

  function onDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (!over || active.id === over.id) return
    hideResults()
    optimize.reset()
    evaluate.reset()
    simulate.reset()
    setSlots((prev) => {
      const ids = prev.map((s, i) => String(s?.player_id ?? `empty-${i}`))
      const oldIndex = ids.indexOf(String(active.id))
      const newIndex = ids.indexOf(String(over.id))
      if (oldIndex < 0 || newIndex < 0) return prev
      return arrayMove(prev, oldIndex, newIndex)
    })
  }

  async function runOptimize() {
    if (!filled) return
    if (ineligibleSlots.length) {
      setFillError(rosterWarning || 'Selected players are not eligible for this roster mode.')
      return
    }
    const res = await optimize.mutateAsync({
      player_ids: playerIds,
      order: playerIds,
      context,
    })
    setAlreadyOptimal(
      Boolean(
        res.available &&
          res.result &&
          orderIsBest(res.result, playerIds),
      ),
    )
    setResultView('optimize')
    setShowResults(true)
  }

  async function runEvaluate() {
    if (!filled) return
    if (ineligibleSlots.length) {
      setFillError(rosterWarning || 'Selected players are not eligible for this roster mode.')
      return
    }
    await evaluate.mutateAsync({ player_ids: playerIds, context })
    setResultView('evaluate')
    setShowResults(true)
  }

  async function runSimulate() {
    if (!filled) return
    if (ineligibleSlots.length) {
      setFillError(rosterWarning || 'Selected players are not eligible for this roster mode.')
      return
    }
    await simulate.mutateAsync({
      player_ids: playerIds,
      context,
      n_games: 2000,
      seed: 42,
    })
    setResultView('simulate')
    setShowResults(true)
  }

  const optData = optimize.data
  const optResult = optData && optData.available ? optData.result : null
  const evalData = evaluate.data
  const simData = simulate.data
  const sortableIds = slots.map((s, i) => String(s?.player_id ?? `empty-${i}`))
  const teams = teamsQ.data?.available ? teamsQ.data.teams : []

  return (
    <div>
      <PageHeader
        eyebrow="Flagship"
        title="Lineup Explorer"
        description="Build a nine from the 2026 season pool or current available roster, then compare your order to the optimal same-nine order under Neutral / vs RHP / vs LHP."
      />

      {/* Step 1 — team */}
      <section className="mb-10">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="eyebrow mb-1">Step 1</div>
            <h2 className="font-display m-0 text-xl tracking-tight">
              Pick a team
            </h2>
          </div>
          {team ? (
            <button
              type="button"
              className="btn btn-secondary !min-h-0 !px-3 !py-1.5 text-xs"
              onClick={clearTeam}
            >
              Change team
            </button>
          ) : null}
        </div>

        {teamsQ.isLoading ? <Loading label="Loading teams…" /> : null}
        {teamsQ.data && !teamsQ.data.available ? (
          <Unavailable data={teamsQ.data} title="Teams unavailable" />
        ) : null}

        {!team ? (
          <div className="space-y-6">
            {DIVISIONS.map((div) => {
              const rows = teams.filter((t) => t.division === div)
              if (!rows.length) return null
              return (
                <div key={div}>
                  <h3 className="eyebrow mb-2">{div}</h3>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
                    {rows.map((t) => (
                      <button
                        key={t.abbr}
                        type="button"
                        onClick={() => selectTeam(t.abbr)}
                        className="panel flex items-center gap-3 px-3 py-2.5 text-left transition-colors hover:border-[var(--color-border-strong)] hover:bg-white"
                      >
                        <span
                          className="h-7 w-1.5 rounded-full"
                          style={{ background: teamColor(t.abbr) }}
                          aria-hidden
                        />
                        <span className="min-w-0">
                          <span className="block font-semibold tracking-wide">
                            {t.abbr}
                          </span>
                          <span className="block truncate text-xs text-[var(--color-muted)]">
                            {t.name}
                          </span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="panel flex flex-wrap items-center gap-3 px-4 py-3">
            <TeamBadge abbr={team} size="md" />
            <div className="min-w-0">
              <div className="font-semibold">
                {teams.find((t) => t.abbr === team)?.name ?? team}
              </div>
              <div className="text-sm text-[var(--color-muted)]">
                Build a lineup from this club&apos;s 2026 starting hitters
              </div>
            </div>
          </div>
        )}
      </section>

      {team ? (
        <>
          <div className="grid gap-8 lg:grid-cols-[1fr_1.1fr]">
            <section>
              <div className="mb-3 space-y-3 rounded-[2px] border border-[var(--color-border)] bg-white/60 p-3">
                <div className="eyebrow">Roster pool</div>
                <div className="flex flex-wrap gap-2">
                  {(
                    [
                      ['season', '2026 Season Pool'],
                      ['current', 'Current Available'],
                    ] as const
                  ).map(([val, label]) => (
                    <button
                      key={val}
                      type="button"
                      aria-pressed={rosterMode === val}
                      className={`btn !min-h-0 !px-2.5 !py-1.5 text-xs ${
                        rosterMode === val
                          ? 'btn-primary ring-2 ring-[var(--color-navy)] ring-offset-1'
                          : 'btn-secondary'
                      }`}
                      onClick={() => changeRosterMode(val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <p className="m-0 text-sm font-medium">
                  {rosterMode === 'season'
                    ? 'Active mode: 2026 Season Pool'
                    : 'Active mode: Current Available'}
                </p>
                <p className="m-0 text-xs leading-relaxed text-[var(--color-muted)]">
                  {rosterMode === 'season'
                    ? 'Every hitter who appeared for this club in 2026. Players who later left the team remain available for historical/hypothetical season analysis.'
                    : 'Players currently belonging to and eligible for this MLB roster.'}
                </p>
              </div>
              <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                <div>
                  <div className="eyebrow mb-1">Step 2</div>
                  <h2 className="font-display m-0 text-xl tracking-tight">
                    Build your lineup
                  </h2>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary !min-h-0 !px-2 !py-1 text-xs"
                  disabled={!rosterQ.data?.available || !rosterQ.data.latest_lineup}
                  onClick={loadLatest}
                >
                  Load latest posted
                </button>
              </div>
              <div className="mb-3 flex flex-wrap gap-2">
                {(
                  [
                    ['neutral', 'Fill most optimal · Neutral'],
                    ['vs_R', 'Fill most optimal · vs RHP'],
                    ['vs_L', 'Fill most optimal · vs LHP'],
                  ] as const
                ).map(([ctx, label]) => (
                  <button
                    key={ctx}
                    type="button"
                    className="btn btn-secondary !min-h-0 !px-2.5 !py-1.5 text-xs"
                    disabled={
                      rosterQ.isLoading ||
                      fillingCtx !== null ||
                      optimize.isPending ||
                      ineligibleSlots.length > 0 ||
                      !(rosterQ.data?.available && (rosterQ.data.players?.length ?? 0) >= 9)
                    }
                    onClick={() => void fillMostOptimal(ctx)}
                  >
                    {fillingCtx === ctx ? 'Filling…' : label}
                  </button>
                ))}
              </div>
              <p className="mb-3 text-xs text-[var(--color-muted)]">
                Puts the best batting order for these nine on the card. Uses
                your current nine if filled; otherwise the latest posted lineup
                (or top nine by games started).
              </p>
              {fillError ? (
                <p className="mb-3 text-sm text-[var(--color-accent)]">{fillError}</p>
              ) : null}
              {rosterWarning ? (
                <div className="mb-3 rounded-[2px] border border-[var(--color-border)] bg-white px-3 py-2 text-sm">
                  <p className="m-0">{rosterWarning}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      className="btn btn-secondary !min-h-0 !px-2 !py-1 text-xs"
                      onClick={removeIneligible}
                    >
                      Remove invalid players
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary !min-h-0 !px-2 !py-1 text-xs"
                      onClick={() => changeRosterMode('season')}
                    >
                      Switch to 2026 Season Pool
                    </button>
                  </div>
                </div>
              ) : null}
              <label className="sr-only" htmlFor="roster-search">
                Filter roster
              </label>
              <input
                id="roster-search"
                className="input mb-3"
                placeholder="Filter roster by name…"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
              {rosterQ.isLoading ? <Loading label="Loading roster…" /> : null}
              {rosterQ.isError ? (
                <Unavailable
                  title="Could not load roster"
                  reason={String(rosterQ.error)}
                />
              ) : null}
              {rosterQ.data && !rosterQ.data.available ? (
                <Unavailable data={rosterQ.data} title="Roster unavailable" />
              ) : null}
              {rosterQ.data?.available ? (
                <ul className="panel m-0 max-h-[28rem] list-none overflow-auto p-0">
                  {rosterPlayers.map((p) => {
                    const taken = slots.some((s) => s?.player_id === p.player_id)
                    const disabledAdd =
                      taken || slots.every(Boolean) || p.selectable === false
                    const badge = p.transaction_badge || p.badge
                    return (
                      <li
                        key={p.player_id}
                        className={`flex items-center justify-between gap-2 border-b border-[var(--color-border)] px-3 py-2 text-sm ${
                          p.selectable === false ? 'opacity-60' : ''
                        }`}
                      >
                        <div className="min-w-0">
                          <div className="font-medium">{p.name}</div>
                          <div className="text-xs text-[var(--color-muted)]">
                            {[
                              p.position,
                              p.bat_side ? `Bats ${p.bat_side}` : null,
                              `${p.games} GS`,
                              badge || null,
                            ]
                              .filter(Boolean)
                              .join(' · ')}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="btn btn-secondary !min-h-0 !px-2 !py-1 text-xs"
                          disabled={disabledAdd}
                          onClick={() =>
                            addPlayer({
                              player_id: p.player_id,
                              name: p.name,
                              bat_side: p.bat_side,
                              position: p.position,
                            })
                          }
                        >
                          {taken ? 'Added' : p.selectable === false ? 'Unavailable' : 'Add'}
                        </button>
                      </li>
                    )
                  })}
                  {rosterPlayers.length === 0 ? (
                    <li className="px-3 py-4 text-sm text-[var(--color-muted)]">
                      No roster matches for that filter.
                    </li>
                  ) : null}
                </ul>
              ) : null}
            </section>

            <section>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h2 className="font-display m-0 text-xl tracking-tight">
                  Your order
                </h2>
                <div className="flex flex-wrap items-center gap-2">
                  {(
                    [
                      ['neutral', 'Neutral'],
                      ['vs_R', 'vs RHP'],
                      ['vs_L', 'vs LHP'],
                    ] as const
                  ).map(([val, label]) => (
                    <button
                      key={val}
                      type="button"
                      className={`btn !min-h-0 !px-2 !py-1 text-xs ${
                        context === val ? 'btn-primary' : 'btn-secondary'
                      }`}
                      onClick={() => changeContext(val)}
                    >
                      {label}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="btn btn-secondary !min-h-0 !px-2 !py-1 text-xs"
                    onClick={() => {
                      setSlots(Array(9).fill(null))
                      setAlreadyOptimal(false)
                      hideResults()
                      optimize.reset()
                      evaluate.reset()
                      simulate.reset()
                    }}
                  >
                    Clear
                  </button>
                </div>
              </div>

              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={onDragEnd}
              >
                <SortableContext
                  items={sortableIds}
                  strategy={verticalListSortingStrategy}
                >
                  <ol className="panel m-0 list-none p-0 px-3">
                    {slots.map((s, i) =>
                      s ? (
                        <div key={sortableIds[i]} className="relative">
                          <SortableRow
                            id={sortableIds[i]}
                            slot={i + 1}
                            name={s.name}
                            batSide={s.bat_side}
                          />
                          <button
                            type="button"
                            className="absolute right-12 top-1/2 -translate-y-1/2 text-xs text-[var(--color-muted)] hover:text-[var(--color-accent)]"
                            onClick={() => clearSlot(i)}
                          >
                            remove
                          </button>
                        </div>
                      ) : (
                        <li
                          key={sortableIds[i]}
                          className="flex items-center gap-3 border-b border-[var(--color-border)] py-2.5 text-[var(--color-muted-light)]"
                        >
                          <span className="flex h-7 w-7 items-center justify-center rounded-[2px] border border-dashed border-[var(--color-border-strong)] font-display text-sm">
                            {i + 1}
                          </span>
                          Empty slot
                        </li>
                      ),
                    )}
                  </ol>
                </SortableContext>
              </DndContext>

              {alreadyOptimal ? (
                <p className="mt-3 mb-0 rounded-[2px] border border-[var(--color-border)] bg-white px-3 py-2 text-sm">
                  You already picked the most optimal lineup for{' '}
                  {contextLabel(context)}.
                </p>
              ) : null}

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!filled || optimize.isPending || ineligibleSlots.length > 0}
                  onClick={() => void runOptimize()}
                >
                  {optimize.isPending ? 'Optimizing…' : 'Optimize'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={!filled || evaluate.isPending || ineligibleSlots.length > 0}
                  onClick={() => void runEvaluate()}
                >
                  {evaluate.isPending ? 'Evaluating…' : 'Evaluate'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={!filled || simulate.isPending || ineligibleSlots.length > 0}
                  onClick={() => void runSimulate()}
                >
                  {simulate.isPending ? 'Simulating…' : 'Simulate'}
                </button>
              </div>
              {!filled ? (
                <p className="mt-2 text-sm text-[var(--color-muted)]">
                  Fill all nine slots from the {team} roster, then run a model.
                  Results stay hidden until you do.
                </p>
              ) : alreadyOptimal ? null : !showResults ? (
                <p className="mt-2 text-sm text-[var(--color-muted)]">
                  Optimize asks whether this order can be arranged better.
                  Evaluate explains how the order works. Simulate shows the
                  range of scoring outcomes.
                </p>
              ) : null}
            </section>
          </div>

          {/* Step 3 — results only after an action */}
          {showResults ? (
            <div className="mt-10 space-y-8">
              <div>
                <div className="eyebrow mb-1">Step 3</div>
                <h2 className="font-display m-0 text-xl tracking-tight">
                  {resultView === 'evaluate'
                    ? 'Lineup breakdown'
                    : resultView === 'simulate'
                      ? 'Run distribution'
                      : 'User order vs optimal same-nine'}
                  <span className="ml-2 text-sm font-sans font-normal text-[var(--color-muted)]">
                    {contextLabel(context)}
                  </span>
                </h2>
              </div>

              {resultView === 'optimize' && optimize.isError ? (
                <Unavailable
                  title="Optimize failed"
                  reason={String(optimize.error)}
                />
              ) : null}
              {resultView === 'optimize' && optData && !optData.available ? (
                <Unavailable data={optData} title="Optimization unavailable" />
              ) : null}
              {resultView === 'optimize' && optResult ? (
                <section className="panel p-5">
                  <h3 className="font-display mb-2 text-lg tracking-tight">
                    User order vs optimal same-nine
                  </h3>
                  <p className="mb-6 text-sm leading-relaxed text-[var(--color-ink)]">
                    {optimizeInterpretation(optResult)}
                  </p>
                  <div className="mb-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    <Metric
                      label="Your order — projected R/G"
                      value={optResult.actual_runs}
                      hint="Expected runs per 9 innings using your batting order."
                    />
                    <Metric
                      label="Best same-nine order — projected R/G"
                      value={optResult.best_runs}
                      accent
                      hint="Highest projection using these exact nine hitters, only reordered."
                    />
                    <Metric
                      label="Optimization gap — R/G"
                      value={optResult.gap}
                      hint="How far your order trails the model’s best order. Smaller is better."
                    />
                    <Metric
                      label="Vs. median order — R/G"
                      value={
                        (optResult as { value_vs_median?: number })
                          .value_vs_median ??
                        (optResult.actual_runs != null &&
                        optResult.median_runs != null
                          ? Number(optResult.actual_runs) -
                            Number(optResult.median_runs)
                          : null)
                      }
                      hint={(() => {
                        const v =
                          Number(
                            (optResult as { value_vs_median?: number })
                              .value_vs_median ??
                              (optResult.actual_runs != null &&
                              optResult.median_runs != null
                                ? Number(optResult.actual_runs) -
                                  Number(optResult.median_runs)
                                : NaN),
                          )
                        if (!Number.isFinite(v) || Math.abs(v) <= 1e-9) {
                          return 'Your order matches the middle-ranked permutation’s projection.'
                        }
                        return `Your order projects ${fmtNum(Math.abs(v))} runs ${
                          v < 0 ? 'below' : 'above'
                        } the middle-ranked permutation.`
                      })()}
                    />
                    <Metric
                      label="Rank"
                      value={`${fmtInt(optResult.rank)} / ${fmtInt(
                        (optResult as { n_perms?: number }).n_perms ?? 362880,
                      )}`}
                      format="raw"
                      hint="Position among every possible batting order of these nine hitters. #1 is best."
                    />
                    <Metric
                      label="Percentile"
                      value={
                        optResult.percentile != null
                          ? fmtOrdinal(Math.round(Number(optResult.percentile)))
                          : null
                      }
                      format="raw"
                      hint={(() => {
                        const p = Math.round(Number(optResult.percentile))
                        if (!Number.isFinite(p)) {
                          return 'Share of possible orders your projection beats.'
                        }
                        return `Your order projects better than about ${p}% of possible orders and worse than about ${Math.max(0, 100 - p)}%.`
                      })()}
                    />
                    <Metric
                      label="Best–worst spread — R/G"
                      value={
                        (optResult as { best_worst_spread?: number })
                          .best_worst_spread ??
                        (optResult.best_runs != null &&
                        optResult.worst_runs != null
                          ? Number(optResult.best_runs) -
                            Number(optResult.worst_runs)
                          : null)
                      }
                      hint="Total difference between the best and worst possible orders. This shows how much batting order matters for this specific group of hitters."
                    />
                    {(() => {
                      const nPerms =
                        Number((optResult as { n_perms?: number }).n_perms) ||
                        362880
                      const nNear = Number(
                        (optResult as { n_near_optimal_02?: number })
                          .n_near_optimal_02,
                      )
                      const nearPct = Number.isFinite(nNear)
                        ? (100 * nNear) / nPerms
                        : null
                      const pctLabel =
                        nearPct == null ? '—' : `${fmtNum(nearPct, 1)}%`
                      return (
                        <Metric
                          label="Orders basically tied with the best"
                          value={pctLabel}
                          format="raw"
                          hint={
                            nearPct == null
                              ? 'Share of batting orders that score almost as well as the best one.'
                              : `About ${fmtNum(nearPct, 0)}% of all ${fmtInt(nPerms)} possible batting orders project within 0.02 runs per game of the best order. That gap is tiny, so the model treats those lineups as essentially tied.`
                          }
                        />
                      )
                    })()}
                  </div>
                  <div className="mb-6 rounded-[2px] border border-[var(--color-border)] bg-white px-3 py-3 text-sm leading-relaxed">
                    <div className="font-medium">
                      Why can a low-ranked lineup still be close to optimal?
                    </div>
                    <p className="mt-1 mb-0 text-[var(--color-muted)]">
                      There are{' '}
                      {fmtInt(
                        (optResult as { n_perms?: number }).n_perms ?? 362880,
                      )}{' '}
                      ways to arrange nine hitters, and many produce almost
                      identical run expectations. A large difference in rank
                      does not necessarily mean a large difference in expected
                      offense.
                    </p>
                  </div>
                  <div className="mb-6 grid gap-4 text-sm sm:grid-cols-3">
                    <div>
                      <div className="text-[var(--color-ink)]">
                        Median order:{' '}
                        <strong>{fmtNum(optResult.median_runs)} R/G</strong>
                      </div>
                      <div className="mt-1 text-xs text-[var(--color-muted)]">
                        The projection of the middle-ranked batting order.
                      </div>
                    </div>
                    <div>
                      <div className="text-[var(--color-ink)]">
                        Worst order:{' '}
                        <strong>{fmtNum(optResult.worst_runs)} R/G</strong>
                      </div>
                      <div className="mt-1 text-xs text-[var(--color-muted)]">
                        The lowest projection among all{' '}
                        {fmtInt(
                          (optResult as { n_perms?: number }).n_perms ?? 362880,
                        )}{' '}
                        arrangements.
                      </div>
                    </div>
                    <div>
                      <div className="text-[var(--color-ink)]">
                        How many orders are separated from the optimum by no
                        more than one-hundredth of a run per game?
                      </div>
                      <div className="mt-1 text-sm">
                        <strong>
                          {fmtInt(
                            (optResult as { n_near_optimal_01?: number })
                              .n_near_optimal_01,
                          )}{' '}
                          orders
                        </strong>
                      </div>
                    </div>
                  </div>
                  {optResult.best_order_ids ? (
                    <div className="grid gap-6 md:grid-cols-2">
                      <div>
                        <h4 className="eyebrow mb-2">Your order</h4>
                        <LineupOrder
                          ids={optResult.actual_order_ids}
                          names={(optResult.actual_order_ids ?? []).map(
                            (id) =>
                              optResult.player_names?.[String(id)] ??
                              String(id),
                          )}
                          linkPlayers
                        />
                      </div>
                      <div>
                        <h4 className="eyebrow mb-2">
                          Optimal same-nine ({contextLabel(context)})
                        </h4>
                        <LineupOrder
                          ids={optResult.best_order_ids}
                          names={(optResult.best_order_ids ?? []).map(
                            (id) =>
                              optResult.player_names?.[String(id)] ??
                              String(id),
                          )}
                          linkPlayers
                        />
                      </div>
                    </div>
                  ) : null}
                  {Array.isArray(optResult.explanations) &&
                  optResult.explanations.length > 0 ? (
                    <div className="mt-6">
                      <h4 className="eyebrow mb-2">
                        Your order vs the best order
                      </h4>
                      <p className="mb-3 text-sm text-[var(--color-muted)]">
                        These notes compare the batting order on your card with
                        the model’s highest-projected order of the same nine
                        hitters.
                      </p>
                      <ul className="space-y-2">
                        {(
                          optResult.explanations as Array<
                            Record<string, unknown>
                          >
                        ).map((ex, i) => (
                          <li
                            key={i}
                            className="border-t border-[var(--color-border)] pt-2 text-sm text-[var(--color-muted)]"
                          >
                            {String(
                              ex.text ?? ex.explanation ?? JSON.stringify(ex),
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </section>
              ) : null}

              {resultView === 'evaluate' && evaluate.isError ? (
                <Unavailable
                  title="Evaluate failed"
                  reason={String(evaluate.error)}
                />
              ) : null}
              {resultView === 'evaluate' && evalData && isUnavailable(evalData) ? (
                <Unavailable data={evalData} title="Evaluation unavailable" />
              ) : null}
              {resultView === 'evaluate' && evalData && evalData.available ? (
                <section className="panel p-5">
                  <h3 className="font-display mb-2 text-lg tracking-tight">
                    Lineup breakdown
                  </h3>
                  <p className="mb-4 text-2xl font-display tracking-tight">
                    Projected runs/game:{' '}
                    <span className="text-[var(--color-accent)]">
                      {fmtNum(evalData.expected_runs_9)}
                    </span>
                  </p>
                  <p className="mb-6 text-sm leading-relaxed text-[var(--color-ink)]">
                    {evalData.summary_sentence ||
                      `This order projects ${fmtNum(evalData.expected_runs_9)} runs per game.`}
                  </p>
                  <h4 className="eyebrow mb-2">Lineup flow</h4>
                  <div className="overflow-x-auto">
                    <table className="table-dense">
                      <thead>
                        <tr>
                          <th>Slot</th>
                          <th>Hitter</th>
                          <th>Expected PA/Game</th>
                          <th>Runners On %</th>
                          <th>RISP %</th>
                          <th>Average Runners On</th>
                          <th>Leads Off Inning %</th>
                          <th>Two-Out %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(evalData.lineup_flow ?? []).map((raw) => {
                          const f = raw as Record<string, unknown>
                          const slot = Number(f.slot)
                          const hitter = slots[slot - 1]
                          const m = flowRowMetrics(f)
                          return (
                            <tr key={slot}>
                              <td>{slot}</td>
                              <td>{hitter?.name ?? `Slot ${slot}`}</td>
                              <td>{fmtNum(m.expectedPa)}</td>
                              <td>{fmtPct(m.runnersOn, 0)}</td>
                              <td>{fmtPct(m.risp, 0)}</td>
                              <td>{fmtNum(m.avgRunners)}</td>
                              <td>{fmtPct(m.leadoff, 0)}</td>
                              <td>{fmtPct(m.twoOut, 0)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  {(() => {
                    const notes =
                      evalData.observations && evalData.observations.length > 0
                        ? evalData.observations
                        : (() => {
                            const rows = (evalData.lineup_flow ?? []).map((raw) =>
                              flowRowMetrics(raw as Record<string, unknown>),
                            )
                            if (rows.length !== 9) return [] as string[]
                            const out: string[] = []
                            const iPa = rows.reduce(
                              (best, r, i) =>
                                (r.expectedPa ?? -1) > (rows[best].expectedPa ?? -1)
                                  ? i
                                  : best,
                              0,
                            )
                            if (rows[iPa].expectedPa != null) {
                              out.push(
                                `Slot ${iPa + 1} receives the most plate appearances at ${fmtNum(rows[iPa].expectedPa)} per game.`,
                              )
                            }
                            const iOn = rows.reduce(
                              (best, r, i) =>
                                (r.runnersOn ?? -1) > (rows[best].runnersOn ?? -1)
                                  ? i
                                  : best,
                              0,
                            )
                            if (rows[iOn].runnersOn != null) {
                              out.push(
                                `Slot ${iOn + 1} has the highest runners-on probability at ${fmtPct(rows[iOn].runnersOn, 0)}.`,
                              )
                            }
                            return out
                          })()
                    return notes.length ? (
                      <ul className="mt-5 mb-0 list-disc space-y-1 pl-5 text-sm text-[var(--color-muted)]">
                        {notes.map((note) => (
                          <li key={note}>{note}</li>
                        ))}
                      </ul>
                    ) : null
                  })()}
                </section>
              ) : null}

              {resultView === 'simulate' && simulate.isError ? (
                <Unavailable
                  title="Simulate failed"
                  reason={String(simulate.error)}
                />
              ) : null}
              {resultView === 'simulate' && simData && isUnavailable(simData) ? (
                <Unavailable data={simData} title="Simulation unavailable" />
              ) : null}
              {resultView === 'simulate' && simData && simData.available ? (
                <section className="panel p-5">
                  <h3 className="font-display mb-2 text-lg tracking-tight">
                    Run distribution
                  </h3>
                  <div className="mb-5 rounded-[2px] border border-[var(--color-border)] bg-white px-3 py-3 text-sm leading-relaxed text-[var(--color-muted)]">
                    <p className="m-0">
                      <strong className="text-[var(--color-ink)]">
                        Monte Carlo simulation
                      </strong>{' '}
                      plays this batting order thousands of times, randomly
                      sampling each plate appearance, to show how scoring can
                      bounce around from game to game.
                    </p>
                    <p className="mt-2 mb-0">
                      <strong className="text-[var(--color-ink)]">
                        The Markov model
                      </strong>{' '}
                      calculates the average scoring of this order in one shot,
                      without random noise. That is the “Markov projection”
                      below. The two should be close; they will not match
                      exactly.
                    </p>
                  </div>
                  {(() => {
                    const stats = simDisplayStats(
                      simData.result as Record<string, unknown>,
                    )
                    const maxC = Math.max(1, ...stats.bars.map((b) => b.count))
                    const maxRun = Math.min(
                      15,
                      Math.max(0, ...stats.bars.map((b) => b.runs), 8),
                    )
                    const p05p95 =
                      stats.p05 != null && stats.p95 != null
                        ? `${fmtNum(stats.p05, 0)} – ${fmtNum(stats.p95, 0)} runs`
                        : '—'
                    return (
                      <>
                  <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    <Metric
                      label="Simulated games"
                      value={pickNumber(simData.result, ['n_games'])}
                      format="int"
                    />
                    <Metric
                      label="Mean runs"
                      value={pickNumber(simData.result, ['mean_runs', 'mean'])}
                    />
                    <Metric
                      label="Median runs"
                      value={pickNumber(simData.result, [
                        'median_runs',
                        'median',
                      ])}
                    />
                    <Metric
                      label="P05–P95"
                      value={p05p95}
                      format="raw"
                      hint="In 90% of simulated games, scoring landed in this range."
                    />
                    <Metric
                      label="P(0–2 runs)"
                      value={stats.p02}
                      format="pct"
                      digits={0}
                    />
                    <Metric
                      label="P(3–5 runs)"
                      value={stats.p35}
                      format="pct"
                      digits={0}
                    />
                    <Metric
                      label="P(6+ runs)"
                      value={stats.p6}
                      format="pct"
                      digits={0}
                    />
                  </div>
                  {stats.bars.length > 0 ? (
                    <div className="mt-6">
                      <h4 className="eyebrow mb-2">Runs per game</h4>
                      <div className="flex h-36 items-stretch gap-px">
                        {Array.from({ length: maxRun + 1 }, (_, runs) => {
                          const c =
                            stats.bars.find((b) => b.runs === runs)?.count ?? 0
                          const px = Math.round((c / maxC) * 112)
                          return (
                            <div
                              key={runs}
                              className="flex h-full min-w-0 flex-1 flex-col justify-end"
                              title={`${runs} runs: ${c} games`}
                            >
                              <div
                                className="w-full bg-[var(--color-navy)]"
                                style={{ height: `${px}px` }}
                              />
                              {runs % 2 === 0 ? (
                                <div className="mt-1 text-center text-[10px] text-[var(--color-muted)]">
                                  {runs}
                                </div>
                              ) : (
                                <div className="mt-1 h-3" />
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-[var(--color-muted)]">
                      No per-game scores were returned, so a chart cannot be
                      drawn. Restart the API and run Simulate again.
                    </p>
                  )}
                  <p className="mt-5 mb-0 text-xs text-[var(--color-muted)]">
                    Markov projection:{' '}
                    {fmtNum(
                      pickNumber(simData.result, [
                        'deterministic_expected',
                        'deterministic',
                      ]),
                    )}{' '}
                    R/G.
                  </p>
                      </>
                    )
                  })()}
                </section>
              ) : null}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
