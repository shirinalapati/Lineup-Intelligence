import { useDeferredValue, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useSynergyPairs } from '../api/hooks'
import { Loading } from './Loading'
import { Unavailable } from './Unavailable'
import { fmtNum, pickNumber } from '../lib/format'

function profileSequence(row: Record<string, unknown>): string {
  const prev = String(
    row.prev_arch_label ?? row.prev_archetype ?? row.archetype_a ?? '',
  ).trim()
  const next = String(
    row.batter_arch_label ?? row.batter_archetype ?? row.archetype_b ?? '',
  ).trim()
  const left = prev && prev.toLowerCase() !== 'nan' ? prev : ''
  const right = next && next.toLowerCase() !== 'nan' ? next : ''
  if (left && right) return `${left} → ${right}`
  if (left) return `${left} → Unassigned`
  if (right) return `Unassigned → ${right}`
  return 'Unassigned'
}

function pairTier(row: Record<string, unknown>): 'strong' | 'moderate' | 'limited' | 'unknown' {
  const raw = String(row.reliability ?? row.reliability_tier ?? row.tier ?? '')
    .toLowerCase()
  if (raw.includes('strong')) return 'strong'
  if (raw.includes('moderate')) return 'moderate'
  if (raw.includes('limited')) return 'limited'
  return 'unknown'
}

export function PlayerPairExplorer() {
  const [includeWeaker, setIncludeWeaker] = useState(false)
  const [minN, setMinN] = useState(100)
  const [pairSearch, setPairSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const limit = 100
  const deferredSearch = useDeferredValue(pairSearch.trim())
  const minTier = includeWeaker ? 'all' : 'strong'
  const sampleFloor = includeWeaker ? minN : 0
  const pairs = useSynergyPairs(
    sampleFloor,
    limit,
    offset,
    deferredSearch,
    minTier,
  )

  const emptyHint = useMemo(() => {
    if (deferredSearch) {
      return `No pairs match “${deferredSearch}” with the current sample filter.`
    }
    return includeWeaker
      ? 'No pairs meet the current sample filter.'
      : 'No strong-reliability pairs match this filter.'
  }, [deferredSearch, includeWeaker])

  return (
    <section>
      <h3 className="font-display mb-2 text-xl tracking-tight">
        Explore Player Pairs
      </h3>
      <p className="mb-5 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
        Residual Effect measures whether the trailing hitter performed better or
        worse than expected after accounting for individual talent and game
        situation. Positive values indicate slightly better-than-expected
        outcomes and negative values indicate slightly worse-than-expected
        outcomes. These are exploratory associations, not causal effects, and
        are not used by the lineup optimizer.
      </p>
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="min-w-[16rem] flex-1 text-sm sm:max-w-sm">
          <span className="eyebrow mb-1 block">Search players</span>
          <input
            type="search"
            className="input w-full"
            placeholder="Name"
            value={pairSearch}
            onChange={(e) => {
              setPairSearch(e.target.value)
              setOffset(0)
            }}
            autoComplete="off"
            spellCheck={false}
          />
        </label>
        <label className="flex items-center gap-2 pb-2 text-sm">
          <input
            type="checkbox"
            checked={includeWeaker}
            onChange={(e) => {
              setIncludeWeaker(e.target.checked)
              setOffset(0)
            }}
          />
          Include moderate and limited samples
        </label>
        {includeWeaker ? (
          <label className="text-sm">
            <span className="eyebrow mb-1 block">Minimum shared PA</span>
            <input
              type="number"
              className="input w-28"
              min={0}
              value={minN}
              onChange={(e) => {
                setMinN(Number(e.target.value) || 0)
                setOffset(0)
              }}
            />
          </label>
        ) : null}
        {pairs.data?.available ? (
          <p className="mb-1 text-sm text-[var(--color-muted)]">
            {pairs.data.total?.toLocaleString?.() ?? pairs.data.total} pairs
            meet this filter
            {deferredSearch ? (
              <>
                {' '}
                for &ldquo;{deferredSearch}&rdquo;
              </>
            ) : null}
          </p>
        ) : null}
      </div>
      {includeWeaker ? (
        <p className="mb-3 text-xs text-[var(--color-muted)]">
          Limited-sample rows are marked. Default floor is 100 shared plate
          appearances so tiny samples stay hidden unless you lower that number.
        </p>
      ) : (
        <p className="mb-3 text-xs text-[var(--color-muted)]">
          Showing strong-reliability pairs only.
        </p>
      )}
      {pairs.isLoading ? <Loading label="Loading pair effects…" /> : null}
      {pairs.isError ? (
        <Unavailable title="Failed to load pairs" reason={String(pairs.error)} />
      ) : null}
      {pairs.data && !pairs.data.available ? (
        <Unavailable data={pairs.data} title="Player-pair effects unavailable" />
      ) : null}
      {pairs.data?.available ? (
        <div className="panel overflow-x-auto">
          <table className="table-dense">
            <thead>
              <tr>
                <th>Preceding hitter</th>
                <th>Batter</th>
                <th>Residual Effect</th>
                <th>Shared PA</th>
                <th>Reliability</th>
                <th>Profile Sequence</th>
              </tr>
            </thead>
            <tbody>
              {(pairs.data.pairs ?? []).map((row, i) => {
                const tier = pairTier(row)
                return (
                  <tr
                    key={`${row.player_a}-${row.player_b}-${i}`}
                    className={
                      tier === 'limited'
                        ? 'bg-[color-mix(in_srgb,var(--color-accent)_6%,transparent)]'
                        : undefined
                    }
                  >
                    <td>
                      {row.player_a ? (
                        <Link
                          to={`/players/${row.player_a}`}
                          className="font-medium hover:underline"
                        >
                          {String(
                            row.player_a_name ??
                              row.prev_name ??
                              row.player_a ??
                              '—',
                          )}
                        </Link>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>
                      {row.player_b ? (
                        <Link
                          to={`/players/${row.player_b}`}
                          className="font-medium hover:underline"
                        >
                          {String(
                            row.player_b_name ??
                              row.name ??
                              row.player_b ??
                              '—',
                          )}
                        </Link>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>
                      {(() => {
                        const v = pickNumber(row, [
                          'effect',
                          'shrunk_effect',
                          'estimate',
                          'delta',
                        ])
                        if (v === undefined || v === null || Number.isNaN(Number(v))) {
                          return '—'
                        }
                        const n = Number(v)
                        return (
                          <span className="tabular-nums">
                            {n > 0 ? '+' : ''}
                            {fmtNum(n, 4)}
                          </span>
                        )
                      })()}
                    </td>
                    <td>
                      {pickNumber(row, ['n', 'n_pa', 'sample_size']) ?? '—'}
                    </td>
                    <td className="text-sm">
                      <span className="text-[var(--color-muted)]">
                        {String(row.reliability ?? row.tier ?? '—')}
                      </span>
                      {tier === 'limited' ? (
                        <span className="ml-2 text-xs font-medium text-[var(--color-accent)]">
                          Limited sample
                        </span>
                      ) : null}
                      {tier === 'moderate' ? (
                        <span className="ml-2 text-xs text-[var(--color-muted)]">
                          Moderate sample
                        </span>
                      ) : null}
                    </td>
                    <td className="whitespace-nowrap text-sm text-[var(--color-muted)]">
                      {profileSequence(row)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {(pairs.data.pairs ?? []).length === 0 ? (
            <p className="px-4 py-3 text-sm text-[var(--color-muted)]">
              {emptyHint}
            </p>
          ) : (
            <div className="flex items-center justify-between px-3 py-2 text-sm text-[var(--color-muted)]">
              <span>
                Showing {offset + 1}–
                {Math.min(
                  offset + (pairs.data.pairs?.length ?? 0),
                  pairs.data.total ?? 0,
                )}{' '}
                of {pairs.data.total?.toLocaleString?.() ?? pairs.data.total}
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
                  disabled={offset + limit >= (pairs.data.total ?? 0)}
                  onClick={() => setOffset(offset + limit)}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </section>
  )
}
