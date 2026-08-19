"""Canonical MLB team metadata."""

from __future__ import annotations

TEAMS: dict[str, dict] = {
    "ARI": {"id": 109, "name": "Arizona Diamondbacks", "division": "NL West"},
    "ATL": {"id": 144, "name": "Atlanta Braves", "division": "NL East"},
    "BAL": {"id": 110, "name": "Baltimore Orioles", "division": "AL East"},
    "BOS": {"id": 111, "name": "Boston Red Sox", "division": "AL East"},
    "CHC": {"id": 112, "name": "Chicago Cubs", "division": "NL Central"},
    "CWS": {"id": 145, "name": "Chicago White Sox", "division": "AL Central"},
    "CIN": {"id": 113, "name": "Cincinnati Reds", "division": "NL Central"},
    "CLE": {"id": 114, "name": "Cleveland Guardians", "division": "AL Central"},
    "COL": {"id": 115, "name": "Colorado Rockies", "division": "NL West"},
    "DET": {"id": 116, "name": "Detroit Tigers", "division": "AL Central"},
    "HOU": {"id": 117, "name": "Houston Astros", "division": "AL West"},
    "KC": {"id": 118, "name": "Kansas City Royals", "division": "AL Central"},
    "LAA": {"id": 108, "name": "Los Angeles Angels", "division": "AL West"},
    "LAD": {"id": 119, "name": "Los Angeles Dodgers", "division": "NL West"},
    "MIA": {"id": 146, "name": "Miami Marlins", "division": "NL East"},
    "MIL": {"id": 158, "name": "Milwaukee Brewers", "division": "NL Central"},
    "MIN": {"id": 142, "name": "Minnesota Twins", "division": "AL Central"},
    "NYM": {"id": 121, "name": "New York Mets", "division": "NL East"},
    "NYY": {"id": 147, "name": "New York Yankees", "division": "AL East"},
    "ATH": {"id": 133, "name": "Athletics", "division": "AL West"},  # Oakland/Sacramento
    "OAK": {"id": 133, "name": "Athletics", "division": "AL West"},
    "PHI": {"id": 143, "name": "Philadelphia Phillies", "division": "NL East"},
    "PIT": {"id": 134, "name": "Pittsburgh Pirates", "division": "NL Central"},
    "SD": {"id": 135, "name": "San Diego Padres", "division": "NL West"},
    "SF": {"id": 137, "name": "San Francisco Giants", "division": "NL West"},
    "SEA": {"id": 136, "name": "Seattle Mariners", "division": "AL West"},
    "STL": {"id": 138, "name": "St. Louis Cardinals", "division": "NL Central"},
    "TB": {"id": 139, "name": "Tampa Bay Rays", "division": "AL East"},
    "TEX": {"id": 140, "name": "Texas Rangers", "division": "AL West"},
    "TOR": {"id": 141, "name": "Toronto Blue Jays", "division": "AL East"},
    "WSH": {"id": 120, "name": "Washington Nationals", "division": "NL East"},
}

# Canonical 30 abbrevs used in the app (prefer ATH over OAK)
CANONICAL_ABBREVS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
]

ID_TO_ABBREV = {v["id"]: k for k, v in TEAMS.items() if k != "OAK"}


def normalize_abbrev(abbr: str | None) -> str | None:
    if not abbr:
        return None
    a = abbr.upper()
    if a == "OAK":
        return "ATH"
    if a == "AZ":
        return "ARI"
    if a == "WSN":
        return "WSH"
    if a == "CHA":
        return "CWS"
    if a == "CHW":
        return "CWS"
    if a == "KCA":
        return "KC"
    if a == "TBR":
        return "TB"
    if a == "SDP":
        return "SD"
    if a == "SFG":
        return "SF"
    if a == "WAS":
        return "WSH"
    return a
