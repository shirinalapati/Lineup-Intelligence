/** Shared API types matching backend response shapes. */

export type Unavailable = {
  available: false
  reason: string
}

export type Available<T> = { available: true } & T

export type MaybeAvailable<T> = Available<T> | Unavailable

export function isUnavailable(v: unknown): v is Unavailable {
  return (
    typeof v === 'object' &&
    v !== null &&
    'available' in v &&
    (v as { available: unknown }).available === false
  )
}

export type TeamMeta = {
  abbr: string
  name: string
  division: string
  id?: number
  games?: number
  games_games?: number
  metrics_available?: boolean
  summary?: Record<string, unknown> | Unavailable
}

export type LeagueOverview = MaybeAvailable<{
  source?: string
  reason?: string
  n_teams?: number
  teams?: TeamMeta[]
  metrics?: Record<string, unknown> | Unavailable
  findings_preview?: unknown
  [key: string]: unknown
}>

export type TeamsList = MaybeAvailable<{
  teams: TeamMeta[]
}>

export type TeamMetricRank = {
  value?: number | null
  rank?: number | null
  population_n?: number | null
  percentile?: number | null
  qualified?: boolean
  metric?: string
  direction?: string
}

export type TeamDetail = MaybeAvailable<{
  abbr: string
  name: string
  division: string
  id: number
  summary?: Record<string, unknown> | Unavailable
  games?: number
  unique_orders?: number
  unique_personnel?: number
  ranks?: Record<string, TeamMetricRank>
}>

export type LineupEvaluation = {
  expected_runs?: number
  actual_expected_runs?: number
  best_expected_runs?: number
  worst_expected_runs?: number
  gap?: number
  percentile?: number
  near_optimal?: boolean
  context?: string
  [key: string]: unknown
}

export type LineupRow = {
  game_pk: number
  game_date?: string
  team?: string
  opponent?: string
  is_home?: boolean
  venue?: string
  order_id?: string
  personnel_id?: string
  opp_sp_hand?: string
  opp_sp_name?: string
  runs_scored?: number | null
  runs_allowed?: number | null
  result?: string | null
  batting_order?: number[]
  batter_names?: string[] | null
  evaluation?: LineupEvaluation | Unavailable
}

export type TeamLineups = MaybeAvailable<{
  team: string
  total: number
  offset: number
  limit: number
  lineups: LineupRow[]
}>

export type TeamRosterPlayer = {
  player_id: number
  name: string
  bat_side?: string | null
  position?: string | null
  games: number
  season_games_started?: number
  primary_slot?: number
  last_date?: string | null
  belongs_to_team?: boolean
  available_for_mlb_lineup?: boolean
  available?: boolean
  selectable?: boolean
  roster_status?: string
  badge?: string | null
  transaction_badge?: string | null
  team_tenure_start?: string | null
  team_tenure_end?: string | null
  current_team?: string | null
  source_confidence?: string
}

export type TeamRosterResponse = MaybeAvailable<{
  team: string
  mode?: string
  as_of?: string | null
  n_players?: number
  players: TeamRosterPlayer[]
  label?: string
  evaluation_note?: string | null
  toggle_note?: string
  latest_lineup?: {
    game_pk: number
    game_date?: string
    opponent?: string
    batting_order: number[]
    batter_names: string[]
    opp_sp_hand?: string | null
    opp_sp_name?: string | null
  } | null
}>

export type HeatmapCell = {
  player_id?: number
  player_name?: string
  name?: string
  slot?: number
  count?: number
  n?: number
  share?: number
  [key: string]: unknown
}

export type HeatmapResponse = MaybeAvailable<{
  team?: string
  players?: string[]
  slots?: number[]
  matrix?: number[][]
  cells?: HeatmapCell[]
  data?: unknown
  [key: string]: unknown
}>

export type TimelinePoint = {
  game_date?: string
  game_pk?: number
  expected_runs?: number
  gap?: number
  percentile?: number
  runs_scored?: number | null
  [key: string]: unknown
}

export type TimelineResponse = MaybeAvailable<{
  team?: string
  points?: TimelinePoint[]
  data?: TimelinePoint[] | unknown
  [key: string]: unknown
}>

export type MostUsedResponse = MaybeAvailable<{
  team?: string
  lineups?: Array<{
    order_id?: string
    personnel_id?: string
    n?: number
    count?: number
    batting_order?: number[]
    batter_names?: string[]
    avg_expected_runs?: number
    [key: string]: unknown
  }>
  data?: unknown
  [key: string]: unknown
}>

export type BatterSlot = {
  slot: number
  player_id: number
  name: string
}

export type LineupDetail = MaybeAvailable<{
  lineup: {
    game_pk: number
    game_date?: string
    team: string
    opponent?: string
    is_home?: boolean
    venue?: string
    batting_order?: number[]
    batters?: BatterSlot[]
    runs_scored?: number | null
    result?: string | null
    opp_sp_hand?: string
    opp_sp_name?: string
    evaluation?: LineupEvaluation | Unavailable
    near_optimal_orders?: unknown
    explanations?: unknown
    adjacent?: unknown
    [key: string]: unknown
  }
}>

export type PlayerListItem = {
  player_id: number
  name: string
  bat_side?: string
  position?: string
  team?: string
  teams?: string[]
  games?: number
  primary_slot?: string | number
  primary_actual_slot?: number
  best_modeled_slot?: number
  archetype?: string
  profile?: Record<string, unknown> | Unavailable
}

export type PlayersList = MaybeAvailable<{
  total: number
  offset: number
  limit: number
  players: PlayerListItem[]
}>

export type PlayerDetail = MaybeAvailable<{
  player_id: number
  name: string
  bat_side?: string
  position?: string
  profile?: Record<string, unknown> | Unavailable
  appearances?: {
    games: number
    teams: string[]
    slot_counts: Record<string, number>
  }
  lineup_intelligence?: Record<string, unknown> | Unavailable
  team_history?: Array<{
    team?: string
    start_at?: string | null
    end_at?: string | null
    start_reason?: string | null
    end_reason?: string | null
  }>
}>

export type OptimizeResult = MaybeAvailable<{
  result: {
    actual_runs?: number
    best_runs?: number
    worst_runs?: number
    gap?: number
    percentile?: number
    actual_order?: number[]
    best_order?: number[]
    worst_order?: number[]
    actual_order_ids?: number[]
    best_order_ids?: number[]
    worst_order_ids?: number[]
    near_optimal_orders?: unknown
    explanations?: unknown
    player_names?: Record<string, string>
    context?: string
    [key: string]: unknown
  }
}>

export type EvaluateResult = MaybeAvailable<{
  context: string
  player_ids: number[]
  expected_runs_9: number
  expected_runs_per_inning: number
  expected_pa_by_slot: number[]
  slot_start_share: number[]
  lineup_flow?: Array<{
    slot: number
    expected_pa?: number
    runners_on_pct?: number
    risp_pct?: number
    avg_runners_on?: number
    leadoff_pct?: number
    two_out_pct?: number
  }>
  summary_sentence?: string
  observations?: string[]
}>

export type SimulateResult = MaybeAvailable<{
  context: string
  player_ids: number[]
  result: {
    mean?: number
    mean_runs?: number
    median?: number
    median_runs?: number
    std?: number
    std_runs?: number
    p05?: number
    p95?: number
    p0_2?: number
    p3_5?: number
    p6_plus?: number
    deterministic_expected?: number
    n_games?: number
    histogram?: Record<string, number>
    [key: string]: unknown
  }
}>

export type SynergyPairs = MaybeAvailable<{
  source?: string
  total?: number
  offset?: number
  limit?: number
  pairs: Array<Record<string, unknown>>
}>

export type SynergyArchetypes = MaybeAvailable<{
  source?: string
  data?: unknown
  pairs?: unknown
  matrix?: unknown | Unavailable
  archetypes?: unknown
}>

export type SynergyIncremental = MaybeAvailable<{
  source?: string
  data?: unknown
  [key: string]: unknown
}>

export type ResearchMethodology = MaybeAvailable<{
  source?: string
  methodology: {
    title?: string
    season?: number
    sections?: Array<{ id: string; title: string; body: string }>
    [key: string]: unknown
  }
}>

export type ResearchFindings = MaybeAvailable<{
  findings: unknown
}>

export type ModelCards = MaybeAvailable<{
  source?: string
  reason?: string
  cards: Array<Record<string, unknown>>
}>

export type SearchResult = {
  type: string
  abbr?: string
  name?: string
  division?: string
  player_id?: number
  bat_side?: string
  position?: string
  game_pk?: number
  game_date?: string
  team?: string
  opponent?: string
  order_id?: string
  match?: string
}

export type SearchResponse = MaybeAvailable<{
  query: string
  results: SearchResult[]
  lineups_index?: boolean
  players_index?: boolean
}>
