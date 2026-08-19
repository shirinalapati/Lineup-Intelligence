import { isUnavailable, type Unavailable } from '../api/types'
import { reasonOf } from '../api/client'

type Props = {
  reason?: string
  data?: unknown
  title?: string
  className?: string
}

export function Unavailable({
  reason,
  data,
  title = 'Data not available',
  className = '',
}: Props) {
  const msg =
    reason ??
    (isUnavailable(data) ? reasonOf(data as Unavailable) : undefined) ??
    'This artifact has not been computed yet. Run the research pipeline to populate it.'

  return (
    <div
      className={`panel px-4 py-5 text-sm text-[var(--color-muted)] ${className}`}
      role="status"
    >
      <div className="eyebrow mb-1 text-[var(--color-warn)]">Unavailable</div>
      <p className="m-0 font-semibold text-[var(--color-ink)]">{title}</p>
      <p className="mt-1 mb-0 leading-relaxed">{msg}</p>
    </div>
  )
}
