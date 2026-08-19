export const DIVISIONS = [
  'AL East',
  'AL Central',
  'AL West',
  'NL East',
  'NL Central',
  'NL West',
] as const

export const TEAM_COLORS: Record<string, string> = {
  ARI: '#A71930',
  ATL: '#CE1141',
  ATH: '#003831',
  BAL: '#DF4601',
  BOS: '#BD3039',
  CHC: '#0E3386',
  CWS: '#27251F',
  CIN: '#C6011F',
  CLE: '#00385D',
  COL: '#33006F',
  DET: '#0C2340',
  HOU: '#EB6E1F',
  KC: '#004687',
  LAA: '#BA0021',
  LAD: '#005A9C',
  MIA: '#00A3E0',
  MIL: '#FFC52F',
  MIN: '#002B5C',
  NYM: '#FF5910',
  NYY: '#0C2340',
  PHI: '#E81828',
  PIT: '#FDB827',
  SD: '#2F241D',
  SF: '#FD5A1E',
  SEA: '#0C2C56',
  STL: '#C41E3A',
  TB: '#8FBCE6',
  TEX: '#C0111F',
  TOR: '#134A8E',
  WSH: '#AB0003',
}

export function teamColor(abbr: string): string {
  return TEAM_COLORS[abbr.toUpperCase()] ?? '#132337'
}
