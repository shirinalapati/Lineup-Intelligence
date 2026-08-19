"""Parquet/JSON artifact store with mtime-based in-memory cache.

Reads from ``data/processed`` and ``data/artifacts``. SQLite is optional later;
parquet-first keeps local iteration fast.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings
from ..teams import CANONICAL_ABBREVS, normalize_abbrev

SLOT_COLS = [f"slot{i}" for i in range(1, 10)]


def unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


class ArtifactStore:
    """Load processed tables and precomputed artifacts with mtime invalidation."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        models_dir: Path | None = None,
        season: int | None = None,
    ):
        self.processed_dir = Path(processed_dir or settings.processed_dir)
        self.artifacts_dir = Path(artifacts_dir or settings.artifacts_dir)
        self.models_dir = Path(models_dir or settings.models_dir)
        self.season = int(season or settings.target_season)
        self._cache: dict[str, tuple[float | None, Any]] = {}

    # ------------------------------------------------------------------ cache
    def _mtime(self, path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return None

    def _cached(self, key: str, path: Path, loader):
        mtime = self._mtime(path)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        if mtime is None:
            data = None
        else:
            data = loader(path)
        self._cache[key] = (mtime, data)
        return data

    def clear_cache(self) -> None:
        self._cache.clear()

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def path_exists(self, path: Path) -> bool:
        return path.is_file()

    # --------------------------------------------------------------- processed
    def lineups_path(self, season: int | None = None) -> Path:
        return self.processed_dir / f"starting_lineups_{season or self.season}.parquet"

    def players_path(self, season: int | None = None) -> Path:
        return self.processed_dir / f"players_{season or self.season}.parquet"

    def load_lineups(self, season: int | None = None) -> pd.DataFrame | None:
        season = season or self.season
        path = self.lineups_path(season)
        df = self._cached(f"lineups:{season}", path, self._load_parquet)
        if df is None:
            return None
        out = df.copy()
        if "team" in out.columns:
            out["team"] = out["team"].map(lambda x: normalize_abbrev(x) or x)
        if "opponent" in out.columns:
            out["opponent"] = out["opponent"].map(lambda x: normalize_abbrev(x) or x)
        return out

    def load_players(self, season: int | None = None) -> pd.DataFrame | None:
        season = season or self.season
        path = self.players_path(season)
        return self._cached(f"players:{season}", path, self._load_parquet)

    def load_tenures(self) -> pd.DataFrame | None:
        path = self.processed_dir / "player_team_tenure.parquet"
        return self._cached("player_team_tenure", path, self._load_parquet)

    def load_roster_intervals(self) -> pd.DataFrame | None:
        path = self.processed_dir / "player_roster_intervals.parquet"
        return self._cached("player_roster_intervals", path, self._load_parquet)

    def load_current_40man(self) -> pd.DataFrame | None:
        path = self.processed_dir / "current_40man_roster.parquet"
        return self._cached("current_40man", path, self._load_parquet)

    def player_name_map(self, season: int | None = None) -> dict[int, str]:
        season = season or self.season
        out: dict[int, str] = {}
        # Prefer full roster (includes rare pitcher ABs) then hitters table
        for path in (
            self.processed_dir / f"players_{season}_all.parquet",
            self.players_path(season),
        ):
            if not path.exists():
                continue
            df = self._cached(f"players_file:{path.name}", path, self._load_parquet)
            if df is None or df.empty:
                continue
            for r in df.itertuples(index=False):
                if pd.notna(r.player_id):
                    out[int(r.player_id)] = str(r.name)
        return out

    # --------------------------------------------------------------- artifacts
    def artifact_path(self, *parts: str) -> Path:
        return self.artifacts_dir.joinpath(*parts)

    def load_json_artifact(self, *parts: str) -> Any | None:
        path = self.artifact_path(*parts)
        key = f"json:{path}"
        return self._cached(key, path, self._load_json)

    def load_parquet_artifact(self, *parts: str) -> pd.DataFrame | None:
        path = self.artifact_path(*parts)
        key = f"parquet:{path}"
        return self._cached(key, path, self._load_parquet)

    def load_lineup_evaluations(self) -> pd.DataFrame | None:
        return self.load_parquet_artifact("lineup_evaluations.parquet")

    def load_team_summaries(self) -> dict | None:
        data = self.load_json_artifact("team_summaries.json")
        return data if isinstance(data, dict) else None

    def load_league_overview(self) -> dict | None:
        data = self.load_json_artifact("league_overview.json")
        return data if isinstance(data, dict) else None

    def load_player_profiles(self) -> Any | None:
        json_path = self.artifact_path("player_profiles.json")
        if json_path.exists():
            return self.load_json_artifact("player_profiles.json")
        parquet_path = self.artifact_path("player_profiles.parquet")
        if parquet_path.exists():
            df = self.load_parquet_artifact("player_profiles.parquet")
            if df is None:
                return None
            return df.to_dict(orient="records")
        return None

    def load_research_artifact(self, name: str) -> Any | None:
        """Load a research artifact by basename from artifacts/research/ (or models/)."""
        stem = Path(name).stem
        suffix = Path(name).suffix
        research_dir = self.artifacts_dir / "research"
        candidates: list[Path] = []
        if suffix in {".json", ".parquet"}:
            candidates.append(research_dir / name)
            candidates.append(self.artifacts_dir / name)
            candidates.append(self.models_dir / name)
        else:
            candidates.extend([
                research_dir / f"{stem}.json",
                research_dir / f"{stem}.parquet",
                self.artifacts_dir / f"{stem}.json",
                self.artifacts_dir / f"{stem}.parquet",
                self.models_dir / f"{stem}.json",
                self.models_dir / f"{stem}.parquet",
            ])
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".parquet":
                return self._cached(f"parquet:{path}", path, self._load_parquet)
            if path.suffix == ".json":
                return self._cached(f"json:{path}", path, self._load_json)
        return None

    def load_findings(self) -> Any | None:
        for name in ("findings.json", "research/findings.json", "what_we_learned.json"):
            data = self.load_json_artifact(*name.split("/"))
            if data is not None:
                return data
        return None

    def load_methodology(self) -> Any | None:
        data = self.load_json_artifact("methodology.json")
        if data is not None:
            return data
        data = self.load_json_artifact("research", "methodology.json")
        return data

    def load_model_cards(self) -> Any | None:
        for parts in (
            ("model_cards.json",),
            ("research", "model_cards.json"),
            ("models", "model_cards.json"),
        ):
            # models/ is under data/, not artifacts
            if parts[0] == "models":
                path = settings.models_dir / "model_cards.json"
                if path.exists():
                    return self._cached(f"json:{path}", path, self._load_json)
                continue
            data = self.load_json_artifact(*parts)
            if data is not None:
                return data
        return None

    # ---------------------------------------------------------- derivations
    def team_lineups(self, abbr: str, season: int | None = None) -> pd.DataFrame | None:
        abbr = normalize_abbrev(abbr) or abbr.upper()
        lineups = self.load_lineups(season)
        if lineups is None:
            return None
        return lineups[lineups["team"] == abbr].copy()

    def team_roster(
        self,
        abbr: str,
        *,
        mode: str = "season",
        as_of: str | None = None,
        include_unavailable: bool = False,
    ) -> dict[str, Any]:
        """Hitter pool for Explorer.

        Modes:
        - ``season``: every 2026 starter for the club (includes later trades/IL)
        - ``current``: belong to the org now AND (optionally) MLB-lineup available
        - ``as_of``: membership + availability reconstructed for ``as_of``
        """
        from datetime import date as _date

        from ..roster_history import (
            badge_with_counterparty,
            index_tenures,
            mlb_lineup_available,
            parse_date,
            snapshot_status_to_canonical,
            status_at,
            tenure_at,
        )
        from ..teams import TEAMS

        abbr = normalize_abbrev(abbr) or abbr.upper()
        mode = (mode or "season").strip().lower()
        if mode not in ("season", "current", "as_of"):
            mode = "season"

        df = self.team_lineups(abbr)
        if df is None:
            return unavailable(f"starting_lineups_{self.season}.parquet not found")

        names = self.player_name_map()
        players_df = self.load_players()
        meta: dict[int, dict] = {}
        if players_df is not None and not players_df.empty:
            for rec in players_df.to_dict(orient="records"):
                meta[int(rec["player_id"])] = {
                    "name": rec.get("name"),
                    "bat_side": rec.get("bat_side"),
                    "position": rec.get("position"),
                }

        as_of_d = parse_date(as_of) if as_of else None
        if mode == "as_of" and as_of_d is None:
            return unavailable("as_of mode requires date=YYYY-MM-DD")
        query_date = as_of_d or _date.today()
        if mode == "current":
            query_date = _date.today()

        # Season GS on this club (full season for badges; as-of truncates appearances)
        appear_df = df
        if mode == "as_of" and as_of_d is not None:
            appear_df = df[df["game_date"].astype(str) <= as_of_d.isoformat()]

        counts: dict[int, int] = {}
        last_date: dict[int, str] = {}
        primary_slot: dict[int, int] = {}
        slot_hist: dict[int, list[int]] = {}
        season_counts: dict[int, int] = {}
        for rec in df.to_dict(orient="records"):
            for col in SLOT_COLS:
                if col not in rec or pd.isna(rec[col]):
                    continue
                season_counts[int(rec[col])] = season_counts.get(int(rec[col]), 0) + 1
        for rec in appear_df.to_dict(orient="records"):
            gdate = str(rec.get("game_date") or "")
            for slot, col in enumerate(SLOT_COLS, start=1):
                if col not in rec or pd.isna(rec[col]):
                    continue
                pid = int(rec[col])
                counts[pid] = counts.get(pid, 0) + 1
                slot_hist.setdefault(pid, []).append(slot)
                prev = last_date.get(pid)
                if prev is None or gdate >= prev:
                    last_date[pid] = gdate
        for pid, slots in slot_hist.items():
            primary_slot[pid] = max(set(slots), key=slots.count)

        tenures_df = self.load_tenures()
        intervals_df = self.load_roster_intervals()
        current_40 = self.load_current_40man()
        team_id = int((TEAMS.get(abbr) or {}).get("id") or 0)
        tenure_index = index_tenures(tenures_df) if tenures_df is not None and not tenures_df.empty else {}
        interval_index: dict[tuple[int, int], list] = {}
        if intervals_df is not None and not intervals_df.empty:
            for rec in intervals_df.to_dict(orient="records"):
                interval_index.setdefault(
                    (int(rec["player_id"]), int(rec["team_id"])), []
                ).append(rec)

        def _player_row(pid: int, extra: dict | None = None) -> dict:
            m = meta.get(pid, {})
            row = {
                "player_id": pid,
                "name": m.get("name") or names.get(pid, str(pid)),
                "bat_side": m.get("bat_side"),
                "position": m.get("position"),
                "games": int(counts.get(pid, 0)),
                "season_games_started": int(season_counts.get(pid, 0)),
                "primary_slot": int(primary_slot.get(pid, 0) or 0),
                "last_date": last_date.get(pid),
                "belongs_to_team": True,
                "available_for_mlb_lineup": True,
                "available": True,
                "selectable": True,
                "roster_status": "ACTIVE",
                "source_confidence": "high",
            }
            if extra:
                row.update(extra)
            return row

        def _is_pitcher_only(pid: int, pos: str | None) -> bool:
            p = str(pos or meta.get(pid, {}).get("position") or "").upper()
            return p == "P" and int(season_counts.get(pid, 0)) == 0

        def _state_for(pid: int) -> dict[str, Any]:
            belongs = False
            status = "ACTIVE"
            tenure_row = None
            available = True
            confidence = "high"
            all_t = tenure_index.get(pid, [])
            team_t = [t for t in all_t if t.team == abbr]
            if team_t:
                tenure_row = tenure_at(team_t, pid, query_date)
                belongs = tenure_row is not None and tenure_row.team == abbr
                if tenure_row is not None:
                    confidence = tenure_row.confidence
            if team_id:
                st = status_at(interval_index.get((pid, team_id), []), pid, team_id, query_date)
                if st:
                    status = str(st.get("roster_status") or status)
                    available = bool(st.get("mlb_lineup_available"))
                else:
                    available = mlb_lineup_available(status)
            badge = badge_with_counterparty(
                all_t, pid, abbr, as_of=query_date, status=status
            ) if all_t else None
            t_start = tenure_row.start_at.isoformat() if tenure_row else None
            t_end = tenure_row.end_at.isoformat() if tenure_row and tenure_row.end_at else None
            current_team = tenure_row.team if tenure_row else None
            now_t = tenure_at(all_t, pid, _date.today()) if all_t else None
            if now_t:
                current_team = now_t.team
            return {
                "belongs_to_team": belongs,
                "available_for_mlb_lineup": available,
                "roster_status": status,
                "transaction_badge": badge,
                "team_tenure_start": t_start,
                "team_tenure_end": t_end,
                "current_team": current_team,
                "source_confidence": confidence,
            }

        players: list[dict] = []

        if mode == "season":
            # Historical lineup universe — leavers remain.
            for pid in counts:
                st = _state_for(pid)
                st["belongs_to_team"] = True  # appeared for this club this season
                players.append(
                    _player_row(
                        pid,
                        {
                            "status": st["roster_status"],
                            "available": True,
                            "selectable": True,
                            "available_for_mlb_lineup": st["available_for_mlb_lineup"],
                            "badge": st["transaction_badge"],
                            **st,
                        },
                    )
                )
        elif mode == "current":
            src = current_40
            ids: list[int] = []
            snap_meta: dict[int, dict] = {}
            if src is not None and not src.empty:
                sub = src[src["team"] == abbr] if "team" in src.columns else src
                for rec in sub.to_dict(orient="records"):
                    pid = int(rec["player_id"])
                    ids.append(pid)
                    snap_meta[pid] = rec
            else:
                # Reconstruct from tenure if snapshot missing
                for pid, rows in tenure_index.items():
                    t = tenure_at(rows, pid, query_date)
                    if t is not None and t.team == abbr:
                        ids.append(pid)
            seen: set[int] = set()
            for pid in ids:
                if pid in seen:
                    continue
                seen.add(pid)
                snap = snap_meta.get(pid, {})
                pos = snap.get("position") or meta.get(pid, {}).get("position")
                if _is_pitcher_only(pid, pos):
                    continue
                st = _state_for(pid)
                if snap:
                    status = snapshot_status_to_canonical(
                        snap.get("status_code"), snap.get("status")
                    )
                    st["roster_status"] = status
                    st["available_for_mlb_lineup"] = mlb_lineup_available(status)
                    st["belongs_to_team"] = True
                    if not st.get("transaction_badge"):
                        st["transaction_badge"] = badge_with_counterparty(
                            tenure_index.get(pid, []),
                            pid,
                            abbr,
                            as_of=query_date,
                            status=status,
                        )
                if not st["belongs_to_team"] and not snap:
                    continue
                avail = bool(st["available_for_mlb_lineup"])
                if not include_unavailable and not avail:
                    continue
                name = snap.get("name") or meta.get(pid, {}).get("name")
                players.append(
                    _player_row(
                        pid,
                        {
                            "name": name or names.get(pid, str(pid)),
                            "position": pos,
                            "status": st["roster_status"],
                            "available": avail,
                            "selectable": bool(avail),
                            "badge": st["transaction_badge"],
                            **st,
                            "belongs_to_team": True,
                            "available_for_mlb_lineup": avail,
                        },
                    )
                )
        else:  # as_of
            ids = set()
            for pid, rows in tenure_index.items():
                t = tenure_at(rows, pid, query_date)
                if t is not None and t.team == abbr:
                    ids.add(pid)
            # Hitters who started here on/before as_of and still belonged that day
            for pid in list(ids):
                pos = meta.get(pid, {}).get("position")
                if _is_pitcher_only(pid, pos) and pid not in counts:
                    ids.discard(pid)
            for pid in sorted(ids):
                st = _state_for(pid)
                if not st["belongs_to_team"]:
                    continue
                avail = bool(st["available_for_mlb_lineup"])
                if not include_unavailable and not avail:
                    continue
                players.append(
                    _player_row(
                        pid,
                        {
                            "status": st["roster_status"],
                            "available": avail,
                            "selectable": bool(avail),
                            "badge": st["transaction_badge"],
                            **st,
                        },
                    )
                )

        players.sort(
            key=lambda p: (
                0 if p.get("selectable") else 1,
                -int(p.get("games") or 0),
                str(p.get("name") or ""),
            )
        )

        latest_lineup = None
        if not df.empty:
            latest = df.sort_values(["game_date", "game_pk"], ascending=[False, False]).iloc[0]
            latest_ids = [int(latest[c]) for c in SLOT_COLS]
            latest_lineup = {
                "game_pk": int(latest["game_pk"]),
                "game_date": str(latest["game_date"]),
                "opponent": latest.get("opponent"),
                "batting_order": latest_ids,
                "batter_names": [names.get(pid, str(pid)) for pid in latest_ids],
                "opp_sp_hand": latest.get("opp_sp_hand"),
                "opp_sp_name": latest.get("opp_sp_name"),
            }

        labels = {
            "season": f"{self.season} season starting hitters",
            "current": "Current available roster",
            "as_of": f"As-of {as_of}" if as_of else "As-of date",
        }
        note = None
        if mode == "as_of":
            note = (
                "Historical roster reconstruction using 2024–2025 trained hitter "
                "probabilities — not 2026 performance through the selected date."
            )
        return _jsonable({
            "available": True,
            "team": abbr,
            "mode": mode,
            "as_of": as_of,
            "include_unavailable": include_unavailable,
            "n_players": len(players),
            "players": players,
            "latest_lineup": latest_lineup,
            "label": labels.get(mode, mode),
            "evaluation_note": note,
            "toggle_note": (
                "Show unavailable only includes players who still belong to this "
                "club; traded/released players stay excluded."
            ),
        })

    def player_team_history(self, player_id: int) -> list[dict[str, Any]]:
        tenures = self.load_tenures()
        if tenures is None or tenures.empty:
            return []
        sub = tenures[tenures["player_id"] == int(player_id)].sort_values("start_at")
        out = []
        for rec in sub.to_dict(orient="records"):
            start = rec.get("start_at")
            end = rec.get("end_at")
            if pd.isna(end) if end is not None else True:
                end = None
            out.append({
                "team": rec.get("team"),
                "team_id": rec.get("team_id"),
                "start_at": str(start)[:10] if start is not None and not pd.isna(start) else None,
                "end_at": str(end)[:10] if end is not None else None,
                "start_reason": rec.get("start_reason"),
                "end_reason": rec.get("end_reason"),
                "source": rec.get("source"),
                "confidence": rec.get("confidence"),
            })
        return out

    def lineup_row(self, game_pk: int, team: str) -> dict | None:
        team = normalize_abbrev(team) or team.upper()
        game_pk = int(game_pk)
        lineups = self.load_lineups()
        if lineups is not None:
            m = lineups[(lineups["game_pk"] == game_pk) & (lineups["team"] == team)]
            if not m.empty:
                rec = m.iloc[0].to_dict()
                rec["batting_order"] = [
                    int(rec[c]) for c in SLOT_COLS if c in rec and pd.notna(rec[c])
                ]
                evals = self.load_lineup_evaluations()
                if evals is not None and not evals.empty:
                    em = evals[
                        (evals["game_pk"] == game_pk)
                        & (evals["team"].map(lambda x: normalize_abbrev(x) or x) == team)
                    ]
                    if len(em):
                        rec["evaluation"] = _jsonable(em.iloc[0].to_dict())
                return _jsonable(rec)

        return None

    def slot_heatmap(self, abbr: str) -> dict[str, Any]:
        """Player × batting-slot appearance counts for a team."""
        abbr = normalize_abbrev(abbr) or abbr.upper()
        df = self.team_lineups(abbr)
        if df is None:
            return unavailable(f"starting_lineups_{self.season}.parquet not found")
        if df.empty:
            return {"available": True, "team": abbr, "slots": list(range(1, 10)), "players": [], "matrix": []}
        names = self.player_name_map()
        counts: dict[int, list[int]] = {}
        for _, row in df.iterrows():
            for slot, col in enumerate(SLOT_COLS, start=1):
                pid = int(row[col])
                if pid not in counts:
                    counts[pid] = [0] * 9
                counts[pid][slot - 1] += 1
        players = []
        matrix = []
        for pid, arr in sorted(counts.items(), key=lambda kv: (-sum(kv[1]), kv[0])):
            players.append({"player_id": pid, "name": names.get(pid, str(pid))})
            matrix.append(arr)
        return {
            "available": True,
            "team": abbr,
            "games": int(len(df)),
            "slots": list(range(1, 10)),
            "players": players,
            "matrix": matrix,
        }

    def lineup_timeline(self, abbr: str) -> dict[str, Any]:
        abbr = normalize_abbrev(abbr) or abbr.upper()
        df = self.team_lineups(abbr)
        if df is None:
            return unavailable(f"starting_lineups_{self.season}.parquet not found")
        evals = self.load_lineup_evaluations()
        rows = []
        for rec in df.sort_values("game_date").to_dict(orient="records"):
            item = {
                "game_pk": int(rec["game_pk"]),
                "game_date": rec.get("game_date"),
                "opponent": normalize_abbrev(rec.get("opponent")) or rec.get("opponent"),
                "is_home": bool(rec.get("is_home")),
                "order_id": rec.get("order_id"),
                "personnel_id": rec.get("personnel_id"),
                "runs_scored": rec.get("runs_scored"),
                "result": rec.get("result"),
                "batting_order": [int(rec[c]) for c in SLOT_COLS],
            }
            if evals is not None and not evals.empty:
                em = evals[
                    (evals["game_pk"] == rec["game_pk"])
                    & (evals["team"].map(lambda x: normalize_abbrev(x) or x) == abbr)
                ]
                if len(em):
                    e = em.iloc[0]
                    for col in (
                        "actual_runs",
                        "best_runs",
                        "gap",
                        "percentile",
                        "rank",
                        "ordering_value",
                        "operationally_equivalent",
                    ):
                        if col in e.index:
                            item[col] = _jsonable(e[col])
            rows.append(item)
        # Alias `points` for chart consumers; map actual_runs → expected_runs
        points = []
        for item in rows:
            points.append({
                **item,
                "expected_runs": item.get("actual_runs"),
                "observed_runs": item.get("runs_scored"),
            })
        return {"available": True, "team": abbr, "lineups": rows, "points": points}

    def most_used_lineups(
        self, abbr: str, top_n: int = 25, *, rank_by: str = "effectiveness"
    ) -> dict[str, Any]:
        """Unique batting orders for a team, ranked by modeled effectiveness or usage.

        Effectiveness = mean modeled expected runs (actual_runs in evaluations)
        for games that used that exact order. Higher is better.
        """
        abbr = normalize_abbrev(abbr) or abbr.upper()
        df = self.team_lineups(abbr)
        if df is None:
            return unavailable(f"starting_lineups_{self.season}.parquet not found")
        if df.empty:
            return {"available": True, "team": abbr, "orders": [], "personnel": []}
        names = self.player_name_map()
        agg_kwargs: dict[str, Any] = {
            "n": ("game_pk", "count"),
            "last_date": ("game_date", "max"),
            **{c: (c, "first") for c in SLOT_COLS},
        }
        if "runs_scored" in df.columns:
            agg_kwargs["avg_runs_scored"] = ("runs_scored", "mean")
            agg_kwargs["total_runs_scored"] = ("runs_scored", "sum")
        order_counts = (
            df.groupby("order_id", dropna=False)
            .agg(**agg_kwargs)
            .reset_index()
        )
        orders: list[dict[str, Any]] = []
        evals = self.load_lineup_evaluations()
        for rec in order_counts.to_dict(orient="records"):
            order = [int(rec[c]) for c in SLOT_COLS]
            item: dict[str, Any] = {
                "order_id": rec["order_id"],
                "n": int(rec["n"]),
                "count": int(rec["n"]),
                "last_date": rec["last_date"],
                "batting_order": order,
                "batter_names": [names.get(pid, str(pid)) for pid in order],
            }
            if rec.get("avg_runs_scored") is not None and pd.notna(rec.get("avg_runs_scored")):
                item["avg_runs_scored"] = float(rec["avg_runs_scored"])
            if rec.get("total_runs_scored") is not None and pd.notna(rec.get("total_runs_scored")):
                item["total_runs_scored"] = float(rec["total_runs_scored"])
            if evals is not None and not evals.empty and "order_id" in evals.columns:
                em = evals[
                    (evals["team"].map(lambda x: normalize_abbrev(x) or x) == abbr)
                    & (evals["order_id"] == rec["order_id"])
                ]
                if len(em):
                    item["avg_expected_runs"] = float(em["actual_runs"].mean())
                    item["avg_gap"] = float(em["gap"].mean())
                    item["avg_percentile"] = float(em["percentile"].mean())
                    item["avg_best_runs"] = float(em["best_runs"].mean())
            orders.append(item)

        if rank_by == "usage":
            orders.sort(
                key=lambda o: (int(o.get("n") or 0), str(o.get("last_date") or "")),
                reverse=True,
            )
        else:
            # Most effective first: highest avg expected runs; require ≥1 evaluated use.
            orders.sort(
                key=lambda o: (
                    0 if o.get("avg_expected_runs") is None else 1,
                    float(o.get("avg_expected_runs") or 0.0),
                    int(o.get("n") or 0),
                ),
                reverse=True,
            )

        # top_n <= 0 means return every unique order
        ranked = orders if top_n <= 0 else orders[:top_n]
        for i, item in enumerate(ranked, start=1):
            item["rank"] = i

        pers = (
            df.groupby("personnel_id", dropna=False)
            .agg(n=("game_pk", "count"), last_date=("game_date", "max"))
            .reset_index()
            .sort_values(["n", "last_date"], ascending=[False, False])
        )
        if top_n > 0:
            pers = pers.head(top_n)
        personnel = [
            {"personnel_id": r["personnel_id"], "n": int(r["n"]), "last_date": r["last_date"]}
            for r in pers.to_dict(orient="records")
        ]
        return {
            "available": True,
            "team": abbr,
            "rank_by": rank_by,
            "n_unique_orders": len(orders),
            "orders": ranked,
            "lineups": ranked,
            "personnel": personnel,
        }

    def data_health_report(self) -> dict[str, Any]:
        """Validate processed lineup data; never invent missing metrics."""
        checks: list[dict[str, Any]] = []
        lineups = self.load_lineups()
        players = self.load_players()

        def add(name: str, ok: bool, detail: Any = None):
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        if lineups is None:
            add("lineups_present", False, f"missing {self.lineups_path().name}")
            return {
                "available": True,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "season": self.season,
                "ok": False,
                "checks": checks,
            }

        add("lineups_present", True, {"rows": int(len(lineups))})
        teams_present = sorted({normalize_abbrev(t) or t for t in lineups["team"].dropna().unique()})
        missing_teams = [a for a in CANONICAL_ABBREVS if a not in teams_present]
        add("all_30_teams", len(missing_teams) == 0, {
            "n_teams": len(teams_present),
            "missing": missing_teams,
            "extra": [t for t in teams_present if t not in CANONICAL_ABBREVS],
        })

        # two lineups per game
        per_game = lineups.groupby("game_pk").size()
        bad_games = int((per_game != 2).sum())
        add("two_lineups_per_game", bad_games == 0, {"games_not_two": bad_games})

        # 9 unique slots
        bad_slots = 0
        for _, row in lineups.iterrows():
            ids = [int(row[c]) for c in SLOT_COLS]
            if len(ids) != 9 or len(set(ids)) != 9:
                bad_slots += 1
        add("nine_unique_batters", bad_slots == 0, {"bad_rows": bad_slots})

        # player resolve
        if players is None:
            add("players_table_present", False, f"missing {self.players_path().name}")
            unresolved = None
        else:
            add("players_table_present", True, {"rows": int(len(players))})
            known = set(int(x) for x in players["player_id"].dropna().astype(int))
            used = set()
            for c in SLOT_COLS:
                used.update(int(x) for x in lineups[c].dropna().astype(int))
            unresolved = sorted(used - known)
            add("player_ids_resolve", len(unresolved) == 0, {"unresolved_n": len(unresolved)})

        # duplicate game/team
        dup = int(lineups.duplicated(subset=["game_pk", "team"]).sum())
        add("no_duplicate_game_team", dup == 0, {"duplicates": dup})

        # order identity consistency
        id_mismatch = 0
        try:
            from ..identity import order_id, personnel_id

            for _, row in lineups.iterrows():
                ids = [int(row[c]) for c in SLOT_COLS]
                if row.get("order_id") != order_id(ids) or row.get("personnel_id") != personnel_id(ids):
                    id_mismatch += 1
            add("deterministic_identities", id_mismatch == 0, {"mismatches": id_mismatch})
        except Exception as e:  # noqa: BLE001
            add("deterministic_identities", False, {"error": str(e)})

        hand_known = int(lineups["opp_sp_hand"].notna().sum()) if "opp_sp_hand" in lineups.columns else 0
        add("opp_sp_hand_coverage", True, {
            "known": hand_known,
            "total": int(len(lineups)),
            "pct": float(hand_known / max(len(lineups), 1)),
        })

        evals = self.load_lineup_evaluations()
        add(
            "lineup_evaluations_artifact",
            evals is not None,
            {"rows": int(len(evals)) if evals is not None else 0},
        )

        ok = all(c["ok"] for c in checks if c["name"] not in ("opp_sp_hand_coverage", "lineup_evaluations_artifact"))
        return {
            "available": True,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "season": self.season,
            "ok": ok,
            "checks": checks,
        }


def _jsonable(obj: Any) -> Any:
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


_STORE: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    global _STORE
    if _STORE is None:
        _STORE = ArtifactStore()
    return _STORE
