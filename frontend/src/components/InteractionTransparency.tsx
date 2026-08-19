import { useSynergyIncremental } from '../api/hooks'
import { pickNumber } from '../lib/format'
import { Loading } from './Loading'
import { Metric } from './Metric'
import { Unavailable } from './Unavailable'

type ModelRow = {
  key: string
  rmse?: number
  logloss?: number
}

const LADDER: Array<{
  ids: string[]
  name: string
  description: string
}> = [
  {
    ids: ['m1_talent', 'm1'],
    name: 'Hitter ability',
    description:
      'How good is the hitter, including pitcher-handedness context?',
  },
  {
    ids: ['m2_slot', 'm2'],
    name: 'Batting-order position',
    description: 'Where is the hitter batting?',
  },
  {
    ids: ['m3_state', 'm3'],
    name: 'Game situation',
    description: 'How many outs are there and which bases are occupied?',
  },
  {
    ids: ['m4_prev_feats', 'm4'],
    name: 'Previous hitter information',
    description: 'What do we know about the hitter who just batted?',
  },
  {
    ids: ['m5_arch_interact', 'm5'],
    name: 'Hitter-style pairing',
    description:
      'Does the combination of offensive styles batting next to each other add anything?',
  },
]

const RMSE_MATERIAL = 0.001
const LL_MATERIAL = 0.003

function parseModels(payload: Record<string, unknown>): ModelRow[] {
  const models = Array.isArray(payload.models)
    ? (payload.models as Array<Record<string, unknown>>)
    : []
  return models.map((m, i) => ({
    key: String(m.model ?? `m${i + 1}`),
    rmse: pickNumber(m, ['rmse']),
    logloss: pickNumber(m, ['logloss_reached', 'logloss']),
  }))
}

function findModel(rows: ModelRow[], ids: string[]): ModelRow | undefined {
  return rows.find((r) => ids.some((id) => r.key === id || r.key.startsWith(id)))
}

function metricShift(
  prev: number | undefined,
  next: number | undefined,
  material: number,
): 'better' | 'worse' | 'same' | 'unknown' {
  if (prev === undefined || next === undefined) return 'unknown'
  const d = prev - next
  if (!Number.isFinite(d)) return 'unknown'
  if (Math.abs(d) < material) return 'same'
  return d > 0 ? 'better' : 'worse'
}

function stepVerdict(prev: ModelRow | undefined, curr: ModelRow): string {
  if (!prev) return 'Baseline'
  const rmse = metricShift(prev.rmse, curr.rmse, RMSE_MATERIAL)
  const ll = metricShift(prev.logloss, curr.logloss, LL_MATERIAL)
  if (rmse === 'unknown' || ll === 'unknown') return '—'
  if (rmse === 'better' && ll === 'better') return 'Improved prediction'
  if (rmse === 'same' && ll === 'same') return 'About the same overall'
  if (rmse === 'better' && ll === 'same') return 'Improved some predictions'
  if (rmse === 'same' && ll === 'better') return 'Improved some predictions'
  if (
    (rmse === 'better' && ll === 'worse') ||
    (rmse === 'worse' && ll === 'better')
  ) {
    return 'Mixed improvement'
  }
  return 'Did not improve prediction'
}

function pairingVsPriorCopy(prior: ModelRow, pairing: ModelRow): string {
  const rmse = metricShift(prior.rmse, pairing.rmse, 1e-6)
  const ll = metricShift(prior.logloss, pairing.logloss, 1e-6)
  if (rmse === 'same' && ll === 'worse') {
    return 'Adding the hitter-style pairing on top of the richer prior-batter model produced no RMSE improvement and worse log-loss.'
  }
  if (rmse === 'same' && ll === 'better') {
    return 'Adding the hitter-style pairing on top of the richer prior-batter model produced no RMSE change and better log-loss.'
  }
  if (rmse === 'worse' && ll === 'worse') {
    return 'Adding the hitter-style pairing on top of the richer prior-batter model made both RMSE and log-loss worse.'
  }
  if (rmse === 'better' && ll === 'better') {
    return 'Adding the hitter-style pairing on top of the richer prior-batter model improved both RMSE and log-loss.'
  }
  if (rmse === 'same' && ll === 'same') {
    return 'Adding the hitter-style pairing on top of the richer prior-batter model did not change RMSE or log-loss.'
  }
  return 'Adding the hitter-style pairing on top of the richer prior-batter model did not improve prediction in this historical test.'
}

export function InteractionTransparency() {
  const incremental = useSynergyIncremental()
  const data = incremental.data

  return (
    <section>
      <h3 className="font-display mb-3 text-xl tracking-tight">
        Did lineup chemistry help predict future games?
      </h3>
      <p className="mb-6 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
        We trained the models using 2024 data, then tested them on 2025 games
        they had never seen. Each step gave the model one additional piece of
        information so we could see whether it actually improved prediction.
      </p>
      {incremental.isLoading ? (
        <Loading label="Loading prediction comparison…" />
      ) : null}
      {incremental.isError ? (
        <Unavailable
          title="Failed to load prediction comparison"
          reason={String(incremental.error)}
        />
      ) : null}
      {data && !data.available ? (
        <Unavailable data={data} title="Prediction comparison unavailable" />
      ) : null}
      {data?.available ? <ComparisonBody data={data} /> : null}
    </section>
  )
}

function ComparisonBody({ data }: { data: Record<string, unknown> }) {
  const payload = (data.data ?? data) as Record<string, unknown>
  const rows = parseModels(payload)
  const learned = payload.train_season
  const tested = payload.valid_season
  const m4 = findModel(rows, ['m4_prev_feats', 'm4'])
  const m5 = findModel(rows, ['m5_arch_interact', 'm5'])

  return (
    <div className="space-y-8">
      <div className="grid gap-5 sm:grid-cols-2 max-w-md">
        <Metric label="Learned from" value={learned} format="year" />
        <Metric label="Tested on" value={tested} format="year" />
      </div>

      <div className="space-y-3">
        {LADDER.map((step, i) => {
          const curr = findModel(rows, step.ids)
          const prev =
            i === 0 ? undefined : findModel(rows, LADDER[i - 1].ids)
          const verdict = curr ? stepVerdict(prev, curr) : '—'
          return (
            <article key={step.name} className="panel p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="font-display m-0 text-lg tracking-tight">
                  {i === 0 ? step.name : `+ ${step.name}`}
                </h3>
                <div className="text-sm font-semibold text-[var(--color-ink)]">
                  {verdict}
                </div>
              </div>
              <p className="mt-2 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
                {step.description}
              </p>
            </article>
          )
        })}
      </div>

      <div className="panel border-[var(--color-border-strong)] p-5">
        <p className="m-0 max-w-3xl text-base leading-relaxed text-[var(--color-ink)]">
          Knowing the hitters and the game situation was useful. But once that
          information was already included, adding the specific combination of
          offensive styles batting next to each other did not make prediction
          more accurate.
        </p>
        <p className="mt-3 mb-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
          Because of that, hitter-style &ldquo;chemistry&rdquo; is treated as
          exploratory and is not used by the lineup optimizer.
        </p>
        {m4 && m5 ? (
          <p className="mt-3 mb-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
            {pairingVsPriorCopy(m4, m5)}
          </p>
        ) : null}
        <p className="mt-3 mb-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
          We tested whether adjacent hitter information added predictive value.
          Offensive-style pairings did not improve prediction in this historical
          test.
        </p>
      </div>
    </div>
  )
}
