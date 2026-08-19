/** Plain-language offensive style groups used across the app. */

const GROUPS = [
  {
    name: 'Spray Contact',
    when: 'Puts the ball in play more, hits for less extra-base power, sprays the ball the other way more often.',
    not: 'Not a power profile. Think contact-first hitters.',
  },
  {
    name: 'Three True Outcomes',
    when: 'Walks more, strikes out more, and hits for more extra-base power / barrels.',
    not: 'Walk, strikeout, or homer — less in-between contact.',
  },
  {
    name: 'Balanced',
    when: 'Walks, strikeouts, and power are close to a typical MLB hitter. No one trait sticks out.',
    not: 'A leftover “average mix” group, not a special skill.',
  },
  {
    name: 'Power',
    when: 'Harder contact than average. More thump than the contact group, without the extreme walk/K mix of Three True Outcomes.',
    not: 'Does not require a home-run crown — it is relative style, not a ranking.',
  },
]

export function ArchetypeGuide() {
  return (
    <div className="panel mb-4 space-y-3 p-5">
      <div className="eyebrow">How a hitter gets a style label</div>
      <p className="m-0 max-w-3xl text-sm leading-relaxed text-[var(--color-muted)]">
        There is no rule like “ISO above .200, therefore Power.” We look at a
        hitter&apos;s own season (walks, strikeouts, extra-base power, how hard
        and where they hit the ball), find four groups of similar hitters, and
        put each player in the group they look most like. The groups were
        learned from 2025 so 2026 labels are not fit on this season&apos;s
        lineup outcomes. The name is a nickname for that group — not chemistry
        with the next batter, and not a grade.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {GROUPS.map((g) => (
          <div key={g.name}>
            <div className="font-medium text-[var(--color-ink)]">{g.name}</div>
            <p className="mt-1 mb-0 text-sm leading-relaxed text-[var(--color-muted)]">
              {g.when}{' '}
              <span className="text-[var(--color-muted-light)]">{g.not}</span>
            </p>
          </div>
        ))}
      </div>
      <p className="m-0 text-xs text-[var(--color-muted)]">
        Pair reliability is separate: it only asks how often two specific
        hitters batted back-to-back. Strong ≈ 117+ shared PAs, moderate ≈
        33–116, limited = fewer. Most pairs are limited because that exact
        order is rare.
      </p>
    </div>
  )
}
