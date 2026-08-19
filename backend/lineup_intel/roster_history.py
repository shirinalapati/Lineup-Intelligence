"""Time-aware MLB roster membership vs lineup availability.

Membership (organizational tenure) and MLB-lineup availability are separate.
Intervals are half-open [start_at, end_at). ``end_at`` None means still open.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from .config import settings
from .teams import CANONICAL_ABBREVS, ID_TO_ABBREV, TEAMS, normalize_abbrev

MLB_TEAM_IDS: frozenset[int] = frozenset(TEAMS[a]["id"] for a in CANONICAL_ABBREVS)

RosterMode = Literal["season", "current", "as_of"]

# Membership-changing MLB Stats API typeCodes (when involving an MLB club).
MEMBERSHIP_TYPE_CODES = {
    "TR": "TRADE",
    "REL": "RELEASE",
    "CLW": "WAIVER_CLAIM",
    "SFA": "FREE_AGENT_SIGNING",
    "SGN": "SIGN",
    "SE": "CONTRACT_SELECTED",
    "ACQ": "ACQUIRED",
    "DFA": "DECLARED_FREE_AGENCY",  # free agency, not designated-for-assignment
}

# Availability-changing (org membership usually persists).
AVAILABILITY_TYPE_CODES = {
    "OPT": "OPTIONED",
    "CU": "RECALLED",
    "DES": "DFA",
    "OUT": "OUTRIGHT",
    "ASG": "ASSIGNED",
    "SC": "STATUS_CHANGE",
}

UNAVAILABLE_STATUSES = {
    "IL",
    "IL_7",
    "IL_10",
    "IL_15",
    "IL_60",
    "OPTIONED",
    "SUSPENDED",
    "RESTRICTED",
    "DFA",
    "REHAB",
    "INACTIVE",
    "OTHER_UNAVAILABLE",
    "MINORS",
}

ACTIVE_STATUSES = {"ACTIVE", "RECALLED"}

_IL_PATTERNS = (
    (re.compile(r"\b60-day injured list\b", re.I), "IL_60"),
    (re.compile(r"\b15-day injured list\b", re.I), "IL_15"),
    (re.compile(r"\b10-day injured list\b", re.I), "IL_10"),
    (re.compile(r"\b7-day injured list\b", re.I), "IL_7"),
    (re.compile(r"\binjured list\b", re.I), "IL"),
)


def fmt_short_date(d: date) -> str:
    return f"{d.strftime('%b')} {d.day}"


def parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()[:10]
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def mlb_id_to_abbr(team_id: int | None) -> str | None:
    if team_id is None:
        return None
    return ID_TO_ABBREV.get(int(team_id))


def is_mlb_team_id(team_id: Any) -> bool:
    try:
        return int(team_id) in MLB_TEAM_IDS
    except (TypeError, ValueError):
        return False


def snapshot_status_to_canonical(code: str | None, description: str | None) -> str:
    c = str(code or "").upper()
    desc = str(description or "")
    mapping = {
        "A": "ACTIVE",
        "D7": "IL_7",
        "D10": "IL_10",
        "D15": "IL_15",
        "D60": "IL_60",
        "ILF": "IL_60",
        "RM": "OPTIONED",
        "MIN": "MINORS",
        "RA": "REHAB",
        "RST": "RESTRICTED",
        "SUS": "SUSPENDED",
        "DFA": "DFA",
        "DES": "DFA",
        "TR": "TRADED",
        "FA": "RELEASED",
        "CL": "WAIVER_CLAIM",
        "NYR": "INACTIVE",
        "DEV": "INACTIVE",
    }
    if c in mapping:
        return mapping[c]
    low = desc.lower()
    if "60-day" in low:
        return "IL_60"
    if "15-day" in low:
        return "IL_15"
    if "10-day" in low:
        return "IL_10"
    if "7-day" in low:
        return "IL_7"
    if "injured" in low:
        return "IL"
    if "active" in low:
        return "ACTIVE"
    if "option" in low or "minors" in low or "reassigned" in low:
        return "OPTIONED"
    return "OTHER_UNAVAILABLE" if desc else "ACTIVE"


def mlb_lineup_available(status: str | None) -> bool:
    st = str(status or "").upper()
    if st in ACTIVE_STATUSES:
        return True
    if st in UNAVAILABLE_STATUSES or st in {"TRADED", "RELEASED"}:
        return False
    return st == "ACTIVE"


def classify_transaction(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one MLB Stats API transaction into membership/availability enums."""
    code = str(raw.get("typeCode") or "").upper()
    desc = str(raw.get("typeDesc") or "")
    text = str(raw.get("description") or "")
    person = raw.get("person") or raw.get("player") or {}
    pid = person.get("id")
    from_team = raw.get("fromTeam") or {}
    to_team = raw.get("toTeam") or {}
    from_id = from_team.get("id")
    to_id = to_team.get("id")
    eff = parse_date(raw.get("effectiveDate") or raw.get("date"))
    from_mlb = is_mlb_team_id(from_id)
    to_mlb = is_mlb_team_id(to_id)

    membership_type: str | None = None
    availability_type: str | None = None
    status_after: str | None = None

    if code in MEMBERSHIP_TYPE_CODES and (from_mlb or to_mlb):
        membership_type = MEMBERSHIP_TYPE_CODES[code]
        if code == "TR" and from_mlb and to_mlb:
            status_after = "ACTIVE"
        elif code in {"REL", "DFA"}:
            status_after = "RELEASED"
        elif code in {"CLW", "SFA", "SGN", "SE", "ACQ"} and to_mlb:
            status_after = "ACTIVE"

    if code == "OPT":
        availability_type = "OPTIONED"
        status_after = "OPTIONED"
    elif code == "CU":
        availability_type = "RECALLED"
        status_after = "ACTIVE"
    elif code == "DES":
        availability_type = "DFA"
        status_after = "DFA"
    elif code == "OUT":
        availability_type = "OUTRIGHT"
        status_after = "MINORS"
        # Outright keeps organizational affiliation unless later released.
    elif code == "ASG":
        availability_type = "ASSIGNED"
        low = text.lower()
        if "rehab" in low:
            status_after = "REHAB"
        elif to_mlb:
            status_after = "ACTIVE"
        else:
            status_after = "MINORS"
    elif code == "SC":
        availability_type = "STATUS_CHANGE"
        status_after = _status_from_description(text)

    # Rehab assignment to an affiliate is not a membership change.
    if membership_type == "TRADE" and not (from_mlb and to_mlb):
        membership_type = None
        availability_type = availability_type or "ASSIGNED"
        status_after = status_after or ("REHAB" if "rehab" in text.lower() else "MINORS")

    return {
        "source_event_id": raw.get("id"),
        "player_id": int(pid) if pid is not None else None,
        "player_name": person.get("fullName"),
        "from_team_id": int(from_id) if from_id is not None else None,
        "to_team_id": int(to_id) if to_id is not None else None,
        "from_team": mlb_id_to_abbr(int(from_id)) if from_id is not None and from_mlb else None,
        "to_team": mlb_id_to_abbr(int(to_id)) if to_id is not None and to_mlb else None,
        "effective_at": eff.isoformat() if eff else None,
        "type_code": code,
        "type_desc": desc,
        "raw_description": text,
        "membership_type": membership_type,
        "availability_type": availability_type,
        "status_after": status_after,
        "source": "mlb_stats_api.transactions",
    }


def _status_from_description(text: str) -> str:
    low = text.lower()
    if "activated" in low and "injured list" in low:
        return "ACTIVE"
    if "transferred" in low and "60-day" in low:
        return "IL_60"
    for pat, label in _IL_PATTERNS:
        if pat.search(text) and "activated" not in low:
            return label
    if "suspended" in low:
        return "SUSPENDED"
    if "restricted" in low:
        return "RESTRICTED"
    if "paternity" in low or "bereavement" in low or "family medical" in low:
        return "INACTIVE"
    if "rehab" in low:
        return "REHAB"
    return "ACTIVE" if "activated" in low else "OTHER_UNAVAILABLE"


@dataclass
class Tenure:
    player_id: int
    team_id: int
    team: str
    start_at: date
    end_at: date | None
    start_reason: str
    end_reason: str | None
    source: str
    source_event_id: Any = None
    confidence: str = "high"


def build_tenures(
    opening_rosters: dict[int, list[dict[str, Any]]],
    events: list[dict[str, Any]],
    *,
    opening_date: date,
    through: date | None = None,
) -> list[Tenure]:
    """Construct non-overlapping MLB org tenures from opening 40-man + transactions."""
    open_tenures: dict[int, Tenure] = {}
    closed: list[Tenure] = []

    def close(pid: int, at: date, reason: str, event_id: Any = None) -> Tenure | None:
        cur = open_tenures.pop(pid, None)
        if cur is None:
            return None
        if at < cur.start_at:
            at = cur.start_at
        cur.end_at = at
        cur.end_reason = reason
        if event_id is not None:
            cur.source_event_id = event_id
        if cur.end_at != cur.start_at:
            closed.append(cur)
        return cur

    def open_new(pid: int, team_id: int, at: date, reason: str, event_id: Any = None, source: str = "mlb_stats_api.transactions") -> None:
        abbr = mlb_id_to_abbr(team_id)
        if not abbr:
            return
        existing = open_tenures.get(pid)
        if existing and existing.team_id == team_id:
            return
        if existing:
            close(pid, at, reason, event_id)
        open_tenures[pid] = Tenure(
            player_id=pid,
            team_id=int(team_id),
            team=abbr,
            start_at=at,
            end_at=None,
            start_reason=reason,
            end_reason=None,
            source=source,
            source_event_id=event_id,
            confidence="high",
        )

    for team_id, players in opening_rosters.items():
        if not is_mlb_team_id(team_id):
            continue
        for p in players:
            pid = p.get("player_id")
            if pid is None:
                continue
            open_new(int(pid), int(team_id), opening_date, "OPENING_ROSTER", source="mlb_stats_api.roster.40Man")

    ordered = sorted(
        (e for e in events if e.get("player_id") and e.get("effective_at")),
        key=lambda e: (str(e["effective_at"]), int(e.get("source_event_id") or 0)),
    )
    for ev in ordered:
        pid = int(ev["player_id"])
        at = parse_date(ev["effective_at"])
        if at is None:
            continue
        if through is not None and at > through:
            continue
        mtype = ev.get("membership_type")
        from_id = ev.get("from_team_id")
        to_id = ev.get("to_team_id")
        eid = ev.get("source_event_id")

        if mtype == "TRADE" and is_mlb_team_id(from_id) and is_mlb_team_id(to_id):
            close(pid, at, "TRADE", eid)
            open_new(pid, int(to_id), at, "TRADE", eid)
        elif mtype in {"RELEASE", "DECLARED_FREE_AGENCY"}:
            close(pid, at, mtype, eid)
        elif mtype in {"WAIVER_CLAIM", "SIGN", "FREE_AGENT_SIGNING", "CONTRACT_SELECTED", "ACQUIRED"}:
            if is_mlb_team_id(to_id):
                if is_mlb_team_id(from_id) and int(from_id) != int(to_id):
                    close(pid, at, mtype, eid)
                open_new(pid, int(to_id), at, mtype, eid)

    closed.extend(open_tenures.values())
    closed.sort(key=lambda t: (t.player_id, t.start_at, t.team))
    return closed


def build_status_intervals(
    opening_rosters: dict[int, list[dict[str, Any]]],
    events: list[dict[str, Any]],
    tenures: list[Tenure],
    *,
    opening_date: date,
    through: date | None = None,
) -> list[dict[str, Any]]:
    """Roster-status intervals per (player, team). IL/option do not end tenure."""
    by_player_team: dict[tuple[int, int], list[dict[str, Any]]] = {}

    def push(pid: int, team_id: int, at: date, status: str, reason: str, event_id: Any = None) -> None:
        abbr = mlb_id_to_abbr(team_id)
        if not abbr:
            return
        key = (pid, int(team_id))
        rows = by_player_team.setdefault(key, [])
        if rows:
            prev = rows[-1]
            if prev["end_at"] is None:
                if at < parse_date(prev["start_at"]):  # type: ignore[arg-type]
                    at = parse_date(prev["start_at"])  # type: ignore[assignment]
                prev["end_at"] = at.isoformat()
                if prev["start_at"] == prev["end_at"]:
                    rows.pop()
        rows.append({
            "player_id": pid,
            "team_id": int(team_id),
            "team": abbr,
            "start_at": at.isoformat(),
            "end_at": None,
            "roster_status": status,
            "mlb_lineup_available": mlb_lineup_available(status),
            "start_reason": reason,
            "source_event_id": event_id,
            "source": "mlb_stats_api",
        })

    for team_id, players in opening_rosters.items():
        if not is_mlb_team_id(team_id):
            continue
        for p in players:
            pid = p.get("player_id")
            if pid is None:
                continue
            st = snapshot_status_to_canonical(p.get("status_code"), p.get("status"))
            push(int(pid), int(team_id), opening_date, st, "OPENING_ROSTER")

    tenure_index: dict[int, list[Tenure]] = {}
    for t in tenures:
        tenure_index.setdefault(t.player_id, []).append(t)

    def team_at(pid: int, at: date) -> int | None:
        for t in tenure_index.get(pid, []):
            if t.start_at <= at and (t.end_at is None or at < t.end_at):
                return t.team_id
        return None

    ordered = sorted(
        (e for e in events if e.get("player_id") and e.get("effective_at")),
        key=lambda e: (str(e["effective_at"]), int(e.get("source_event_id") or 0)),
    )
    for ev in ordered:
        pid = int(ev["player_id"])
        at = parse_date(ev["effective_at"])
        if at is None:
            continue
        if through is not None and at > through:
            continue
        status = ev.get("status_after")
        mtype = ev.get("membership_type")
        atype = ev.get("availability_type")
        eid = ev.get("source_event_id")
        if mtype == "TRADE" and is_mlb_team_id(ev.get("to_team_id")):
            push(pid, int(ev["to_team_id"]), at, status or "ACTIVE", "TRADE", eid)
            continue
        if mtype in {"WAIVER_CLAIM", "SIGN", "FREE_AGENT_SIGNING", "CONTRACT_SELECTED", "ACQUIRED"}:
            if is_mlb_team_id(ev.get("to_team_id")):
                push(pid, int(ev["to_team_id"]), at, status or "ACTIVE", mtype, eid)
            continue
        if mtype in {"RELEASE", "DECLARED_FREE_AGENCY"}:
            from_id = ev.get("from_team_id") or team_at(pid, at - timedelta(days=1) if at > opening_date else at)
            if from_id:
                push(pid, int(from_id), at, "RELEASED", mtype, eid)
            continue
        if status and (atype or mtype):
            tid = ev.get("to_team_id") if is_mlb_team_id(ev.get("to_team_id")) else None
            if tid is None:
                tid = ev.get("from_team_id") if is_mlb_team_id(ev.get("from_team_id")) else None
            if tid is None:
                tid = team_at(pid, at)
            if tid:
                push(pid, int(tid), at, status, atype or mtype or "STATUS_CHANGE", eid)

    out: list[dict[str, Any]] = []
    for rows in by_player_team.values():
        out.extend(rows)
    out.sort(key=lambda r: (r["player_id"], r["start_at"], r["team"]))
    return out


def index_tenures(tenures: pd.DataFrame | list[Tenure]) -> dict[int, list[Tenure]]:
    rows: list[Tenure]
    if isinstance(tenures, pd.DataFrame):
        rows = [_tenure_from_row(r) for r in tenures.to_dict(orient="records")]
    else:
        rows = list(tenures)
    out: dict[int, list[Tenure]] = {}
    for t in rows:
        out.setdefault(t.player_id, []).append(t)
    return out


def tenure_at(tenures: Iterable[Tenure] | pd.DataFrame, player_id: int, at: date) -> Tenure | None:
    rows: list[Tenure]
    if isinstance(tenures, pd.DataFrame):
        rows = [_tenure_from_row(r) for r in tenures.to_dict(orient="records")]
    else:
        rows = list(tenures)
    hits = [
        t
        for t in rows
        if t.player_id == int(player_id)
        and t.start_at <= at
        and (t.end_at is None or at < t.end_at)
    ]
    if not hits:
        return None
    hits.sort(key=lambda t: t.start_at, reverse=True)
    return hits[0]


def status_at(
    intervals: pd.DataFrame | list[dict[str, Any]],
    player_id: int,
    team_id: int,
    at: date,
) -> dict[str, Any] | None:
    if isinstance(intervals, pd.DataFrame):
        rows = intervals.to_dict(orient="records")
    else:
        rows = intervals
    hits = []
    for r in rows:
        if int(r["player_id"]) != int(player_id) or int(r["team_id"]) != int(team_id):
            continue
        start = parse_date(r["start_at"])
        end = parse_date(r.get("end_at"))
        if start and start <= at and (end is None or at < end):
            hits.append(r)
    if not hits:
        return None
    hits.sort(key=lambda r: str(r["start_at"]), reverse=True)
    return hits[0]


def _tenure_from_row(r: dict[str, Any]) -> Tenure:
    return Tenure(
        player_id=int(r["player_id"]),
        team_id=int(r["team_id"]),
        team=str(r["team"]),
        start_at=parse_date(r["start_at"]) or date.min,
        end_at=parse_date(r.get("end_at")),
        start_reason=str(r.get("start_reason") or ""),
        end_reason=r.get("end_reason"),
        source=str(r.get("source") or ""),
        source_event_id=r.get("source_event_id"),
        confidence=str(r.get("confidence") or "high"),
    )


def tenures_to_frame(tenures: list[Tenure]) -> pd.DataFrame:
    rows = []
    for t in tenures:
        d = asdict(t)
        d["start_at"] = t.start_at.isoformat()
        d["end_at"] = t.end_at.isoformat() if t.end_at else None
        rows.append(d)
    return pd.DataFrame(rows)


def overlapping_tenure_violations(tenures: list[Tenure]) -> list[dict[str, Any]]:
    """Same player, two MLB tenures overlapping. Empty unless data is inconsistent."""
    by: dict[int, list[Tenure]] = {}
    for t in tenures:
        by.setdefault(t.player_id, []).append(t)
    bad: list[dict[str, Any]] = []
    for pid, rows in by.items():
        rows = sorted(rows, key=lambda x: (x.start_at, x.team))
        for a, b in zip(rows, rows[1:]):
            a_end = a.end_at or date.max
            if a.start_at < b.start_at < a_end:
                bad.append({
                    "player_id": pid,
                    "a_team": a.team,
                    "a_start": a.start_at.isoformat(),
                    "a_end": a.end_at.isoformat() if a.end_at else None,
                    "b_team": b.team,
                    "b_start": b.start_at.isoformat(),
                })
    return bad


def transaction_badge(
    *,
    team: str,
    tenure: Tenure | None,
    status: str | None,
    mode: str,
    as_of: date | None,
) -> str | None:
    if tenure is None:
        return None
    today = as_of or date.today()
    if tenure.end_at and tenure.end_at <= today:
        reason = str(tenure.end_reason or "")
        other = None
        # Find counterpart team from reason TRADE — caller may pass end_reason only.
        if reason == "TRADE":
            return f"Traded · {fmt_short_date(tenure.end_at)}"
        if reason in {"RELEASE", "DECLARED_FREE_AGENCY"}:
            return f"Released · {fmt_short_date(tenure.end_at)}"
        return f"{reason.title()} · {fmt_short_date(tenure.end_at)}" if reason else None
    if tenure.start_reason in {"TRADE", "WAIVER_CLAIM", "SIGN", "FREE_AGENT_SIGNING", "ACQUIRED", "CONTRACT_SELECTED"}:
        if tenure.start_at > date(today.year, 3, 1):
            verb = {
                "TRADE": "Acquired",
                "WAIVER_CLAIM": "Claimed",
                "SIGN": "Signed",
                "FREE_AGENT_SIGNING": "Signed",
                "ACQUIRED": "Acquired",
                "CONTRACT_SELECTED": "Selected",
            }.get(tenure.start_reason, tenure.start_reason.title())
            return f"{verb} · {fmt_short_date(tenure.start_at)}"
    st = str(status or "")
    labels = {
        "IL": "IL",
        "IL_7": "7-DAY IL",
        "IL_10": "10-DAY IL",
        "IL_15": "15-DAY IL",
        "IL_60": "60-DAY IL",
        "OPTIONED": "OPTIONED",
        "RECALLED": "RECALLED",
        "DFA": "DFA",
        "REHAB": "REHAB",
        "SUSPENDED": "SUSPENDED",
        "RESTRICTED": "RESTRICTED",
        "MINORS": "MINORS",
        "INACTIVE": "INACTIVE",
    }
    if st in labels and st != "ACTIVE":
        return labels[st]
    return None


def badge_with_counterparty(
    tenures: list[Tenure] | pd.DataFrame,
    player_id: int,
    team: str,
    *,
    as_of: date | None = None,
    status: str | None = None,
) -> str | None:
    if isinstance(tenures, pd.DataFrame):
        rows = [_tenure_from_row(r) for r in tenures.to_dict(orient="records") if int(r["player_id"]) == int(player_id)]
    else:
        rows = [t for t in tenures if t.player_id == int(player_id)]
    rows.sort(key=lambda t: t.start_at)
    team_t = next((t for t in rows if t.team == team), None)
    if team_t is None:
        return transaction_badge(team=team, tenure=None, status=status, mode="", as_of=as_of)
    today = as_of or date.today()
    if team_t.end_at and team_t.end_reason == "TRADE":
        nxt = next((t for t in rows if t.start_at == team_t.end_at and t.team != team), None)
        dest = nxt.team if nxt else None
        label = f"Traded to {dest} · {fmt_short_date(team_t.end_at)}" if dest else f"Traded · {fmt_short_date(team_t.end_at)}"
        return label
    if team_t.end_at and team_t.end_reason in {"RELEASE", "DECLARED_FREE_AGENCY"}:
        return f"Released · {fmt_short_date(team_t.end_at)}"
    if team_t.start_reason == "TRADE" and team_t.start_at > date(team_t.start_at.year, 3, 15):
        prev = next((t for t in reversed(rows) if t.end_at == team_t.start_at and t.team != team), None)
        src = prev.team if prev else None
        if src:
            return f"Acquired from {src} · {fmt_short_date(team_t.start_at)}"
        return f"Acquired · {fmt_short_date(team_t.start_at)}"
    if team_t.end_at and team_t.end_at <= today:
        return transaction_badge(team=team, tenure=team_t, status=status, mode="", as_of=as_of)
    return transaction_badge(team=team, tenure=team_t, status=status, mode="", as_of=as_of)


def infer_tenure_fallbacks_from_lineups(
    lineups: pd.DataFrame,
    tenures: list[Tenure],
    *,
    slot_cols: list[str],
) -> list[Tenure]:
    """Last-resort: cover starting-lineup observations missing from transaction tenure.

    Does not replace transaction history. Labeled ``lineup_appearance_fallback``.
    """
    extra: list[Tenure] = []
    covered: dict[tuple[int, str], list[Tenure]] = {}
    for t in tenures:
        covered.setdefault((t.player_id, t.team), []).append(t)

    first_last: dict[tuple[int, str, int], list[date]] = {}
    team_ids = {a: TEAMS[a]["id"] for a in CANONICAL_ABBREVS}
    for rec in lineups.to_dict(orient="records"):
        team = normalize_abbrev(rec.get("team")) or rec.get("team")
        if not team or team not in team_ids:
            continue
        gdate = parse_date(rec.get("game_date"))
        if not gdate:
            continue
        tid = team_ids[team]
        for col in slot_cols:
            if col not in rec or pd.isna(rec[col]):
                continue
            pid = int(rec[col])
            first_last.setdefault((pid, team, tid), []).append(gdate)

    for (pid, team, tid), dates in first_last.items():
        dates.sort()
        start, end = dates[0], dates[-1]
        spans = covered.get((pid, team), [])
        if spans:
            # Transaction tenure exists; uncovered games are validation mismatches, not new tenures.
            continue
        extra.append(
            Tenure(
                player_id=pid,
                team_id=tid,
                team=team,
                start_at=start,
                end_at=end + timedelta(days=1),
                start_reason="LINEUP_APPEARANCE",
                end_reason="LINEUP_APPEARANCE",
                source="lineup_appearance_fallback",
                confidence="low",
            )
        )
    return extra


def validate_lineups_against_tenures(
    lineups: pd.DataFrame,
    tenures: list[Tenure],
    *,
    slot_cols: list[str],
    max_examples: int = 25,
) -> dict[str, Any]:
    by_player: dict[int, list[Tenure]] = {}
    for t in tenures:
        by_player.setdefault(t.player_id, []).append(t)
    total = 0
    validated = 0
    mismatched = 0
    examples: list[dict[str, Any]] = []
    for rec in lineups.to_dict(orient="records"):
        team = normalize_abbrev(rec.get("team")) or rec.get("team")
        gdate = parse_date(rec.get("game_date"))
        if not team or not gdate:
            continue
        for col in slot_cols:
            if col not in rec or pd.isna(rec[col]):
                continue
            pid = int(rec[col])
            total += 1
            t = tenure_at(by_player.get(pid, []), pid, gdate)
            if t is not None and t.team == team:
                validated += 1
            else:
                mismatched += 1
                if len(examples) < max_examples:
                    examples.append({
                        "player_id": pid,
                        "lineup_team": team,
                        "game_date": gdate.isoformat(),
                        "game_pk": rec.get("game_pk"),
                        "tenure_team": t.team if t else None,
                        "probable_cause": (
                            "no_tenure_covering_game_date"
                            if t is None
                            else "tenure_team_mismatch"
                        ),
                    })
    return {
        "total_starting_player_observations": total,
        "validated": validated,
        "mismatched": mismatched,
        "unresolved": mismatched,
        "validation_rate": float(validated / total) if total else None,
        "examples": examples,
    }


def load_tenure_frame(processed_dir: Path | None = None) -> pd.DataFrame | None:
    path = Path(processed_dir or settings.processed_dir) / "player_team_tenure.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_status_frame(processed_dir: Path | None = None) -> pd.DataFrame | None:
    path = Path(processed_dir or settings.processed_dir) / "player_roster_intervals.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_events_frame(processed_dir: Path | None = None) -> pd.DataFrame | None:
    path = Path(processed_dir or settings.processed_dir) / "roster_status_events.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
