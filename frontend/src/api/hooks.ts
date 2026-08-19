import { useQuery, useMutation } from '@tanstack/react-query'
import { apiGet, apiPost } from './client'
import type {
  LeagueOverview,
  TeamsList,
  TeamDetail,
  TeamLineups,
  TeamRosterResponse,
  HeatmapResponse,
  TimelineResponse,
  MostUsedResponse,
  LineupDetail,
  PlayersList,
  PlayerDetail,
  OptimizeResult,
  EvaluateResult,
  SimulateResult,
  SynergyPairs,
  SynergyArchetypes,
  SynergyIncremental,
  ResearchMethodology,
  ResearchFindings,
  ModelCards,
  SearchResponse,
} from './types'

export function useLeagueOverview() {
  return useQuery({
    queryKey: ['league', 'overview'],
    queryFn: () => apiGet<LeagueOverview>('/api/league/overview'),
  })
}

export function useTeams() {
  return useQuery({
    queryKey: ['teams'],
    queryFn: () => apiGet<TeamsList>('/api/teams'),
  })
}

export function useTeam(abbr: string) {
  return useQuery({
    queryKey: ['teams', abbr],
    queryFn: () => apiGet<TeamDetail>(`/api/teams/${abbr}`),
    enabled: Boolean(abbr),
  })
}

export function useTeamLineups(abbr: string, limit = 500) {
  return useQuery({
    queryKey: ['teams', abbr, 'lineups', limit],
    queryFn: () =>
      apiGet<TeamLineups>(`/api/teams/${abbr}/lineups?limit=${limit}`),
    enabled: Boolean(abbr),
  })
}

export function useTeamHeatmap(abbr: string) {
  return useQuery({
    queryKey: ['teams', abbr, 'heatmap'],
    queryFn: () => apiGet<HeatmapResponse>(`/api/teams/${abbr}/heatmap`),
    enabled: Boolean(abbr),
  })
}

export function useTeamTimeline(abbr: string) {
  return useQuery({
    queryKey: ['teams', abbr, 'timeline'],
    queryFn: () => apiGet<TimelineResponse>(`/api/teams/${abbr}/timeline`),
    enabled: Boolean(abbr),
  })
}

export function useTeamMostUsed(abbr: string) {
  return useQuery({
    queryKey: ['teams', abbr, 'most-used', 'all', 'effectiveness'],
    queryFn: () =>
      apiGet<MostUsedResponse>(
        `/api/teams/${abbr}/most-used?top_n=0&rank_by=effectiveness`,
      ),
    enabled: Boolean(abbr),
  })
}

export function useTeamMostUsedByUsage(abbr: string) {
  return useQuery({
    queryKey: ['teams', abbr, 'most-used', 'all', 'usage'],
    queryFn: () =>
      apiGet<MostUsedResponse>(
        `/api/teams/${abbr}/most-used?top_n=25&rank_by=usage`,
      ),
    enabled: Boolean(abbr),
  })
}

export function useLineupDetail(gamePk: string, team: string) {
  return useQuery({
    queryKey: ['lineups', gamePk, team],
    queryFn: () =>
      apiGet<LineupDetail>(`/api/lineups/${gamePk}/${team}`),
    enabled: Boolean(gamePk && team),
  })
}

export function useTeamRoster(
  abbr: string,
  opts?: {
    mode?: 'season' | 'current' | 'as_of'
    asOf?: string
    includeUnavailable?: boolean
  },
) {
  const mode = opts?.mode ?? 'season'
  const asOf = opts?.asOf
  const includeUnavailable = opts?.includeUnavailable ?? false
  const params = new URLSearchParams({
    mode,
    include_unavailable: String(includeUnavailable),
  })
  if (mode === 'as_of' && asOf) params.set('as_of', asOf)
  return useQuery({
    queryKey: ['teams', abbr, 'roster', mode, asOf, includeUnavailable],
    queryFn: () =>
      apiGet<TeamRosterResponse>(`/api/teams/${abbr}/roster?${params}`),
    enabled: Boolean(abbr) && (mode !== 'as_of' || Boolean(asOf)),
  })
}

export function usePlayers(q: string, limit = 200, offset = 0) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })
  if (q.trim()) params.set('q', q.trim())
  return useQuery({
    queryKey: ['players', q, limit, offset],
    queryFn: () => apiGet<PlayersList>(`/api/players?${params}`),
  })
}

export function usePlayer(id: string) {
  return useQuery({
    queryKey: ['players', id],
    queryFn: () => apiGet<PlayerDetail>(`/api/players/${id}`),
    enabled: Boolean(id),
  })
}

export function useSearch(q: string) {
  return useQuery({
    queryKey: ['search', q],
    queryFn: () =>
      apiGet<SearchResponse>(`/api/search?q=${encodeURIComponent(q)}&limit=20`),
    enabled: q.trim().length >= 1,
  })
}

export function useSynergyPairs(
  minN = 0,
  limit = 100,
  offset = 0,
  q = '',
  minTier = 'strong',
) {
  const params = new URLSearchParams({
    min_n: String(minN),
    limit: String(limit),
    offset: String(offset),
    min_tier: minTier,
  })
  const query = q.trim()
  if (query) params.set('q', query)
  return useQuery({
    queryKey: ['synergy', 'pairs', minN, minTier, limit, offset, query],
    queryFn: () => apiGet<SynergyPairs>(`/api/synergy/pairs?${params}`),
  })
}

export function useSynergyArchetypes() {
  return useQuery({
    queryKey: ['synergy', 'archetypes'],
    queryFn: () => apiGet<SynergyArchetypes>('/api/synergy/archetypes'),
  })
}

export function useSynergyIncremental() {
  return useQuery({
    queryKey: ['synergy', 'incremental'],
    queryFn: () => apiGet<SynergyIncremental>('/api/synergy/incremental'),
  })
}

export function useMethodology() {
  return useQuery({
    queryKey: ['research', 'methodology'],
    queryFn: () => apiGet<ResearchMethodology>('/api/research/methodology'),
  })
}

export function useFindings() {
  return useQuery({
    queryKey: ['research', 'findings'],
    queryFn: () => apiGet<ResearchFindings>('/api/research/findings'),
  })
}

export function useModelCards() {
  return useQuery({
    queryKey: ['research', 'model-cards'],
    queryFn: () => apiGet<ModelCards>('/api/research/model-cards'),
  })
}

export function useOptimize() {
  return useMutation({
    mutationFn: (body: {
      player_ids: number[]
      order?: number[]
      context: string
    }) => apiPost<OptimizeResult>('/api/optimize', body),
  })
}

export function useEvaluate() {
  return useMutation({
    mutationFn: (body: { player_ids: number[]; context: string }) =>
      apiPost<EvaluateResult>('/api/evaluate', body),
  })
}

export function useSimulate() {
  return useMutation({
    mutationFn: (body: {
      player_ids: number[]
      context: string
      n_games?: number
      seed?: number
    }) => apiPost<SimulateResult>('/api/simulate', body),
  })
}
