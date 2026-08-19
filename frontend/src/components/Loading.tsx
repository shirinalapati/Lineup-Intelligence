type Props = {
  label?: string
  className?: string
}

export function Loading({ label = 'Loading…', className = '' }: Props) {
  return (
    <div
      className={`panel flex items-center gap-3 px-4 py-5 text-sm text-[var(--color-muted)] ${className}`}
      role="status"
    >
      <span
        className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--color-border-strong)] border-t-[var(--color-navy)]"
        aria-hidden
      />
      {label}
    </div>
  )
}
