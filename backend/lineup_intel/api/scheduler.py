"""Optional in-process daily refresh for deployed API containers.

Enable with:
  LI_DAILY_REFRESH=1
  LI_REFRESH_TOKEN=...   # also used by POST /api/ops/daily-refresh
  LI_DAILY_REFRESH_HOUR=14   # UTC hour (default 14 ≈ 10am ET during EDT)

Runs once per calendar day while the process is up, through the configured
season end date.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime, timezone

log = logging.getLogger("lineup_intel.daily_refresh")

# Cover regular season + World Series window for 2026.
_SEASON_END = date(2026, 11, 5)


def _enabled() -> bool:
    return (os.environ.get("LI_DAILY_REFRESH") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _hour_utc() -> int:
    raw = (os.environ.get("LI_DAILY_REFRESH_HOUR") or "14").strip()
    try:
        h = int(raw)
    except ValueError:
        return 14
    return max(0, min(23, h))


def start_daily_refresh_scheduler() -> None:
    if not _enabled():
        log.info("LI_DAILY_REFRESH not set — skipping in-process daily scheduler")
        return

    hour = _hour_utc()
    last_run_day: str | None = None

    def loop() -> None:
        nonlocal last_run_day
        log.info(
            "Daily refresh scheduler started (UTC hour=%s, through %s)",
            hour,
            _SEASON_END.isoformat(),
        )
        while True:
            try:
                now = datetime.now(timezone.utc)
                today = now.date()
                if today > _SEASON_END:
                    log.info("Past season end %s — daily scheduler idle", _SEASON_END)
                    time.sleep(6 * 3600)
                    continue
                day_s = today.isoformat()
                if now.hour >= hour and last_run_day != day_s:
                    log.info("Running scheduled daily refresh for %s", day_s)
                    from .routers.ops import run_daily_refresh

                    result = run_daily_refresh(skip_precompute=False)
                    last_run_day = day_s
                    log.info(
                        "Daily refresh finished: +%s lineups",
                        (result.get("refresh") or {}).get("n_new_lineups"),
                    )
            except Exception as exc:  # noqa: BLE001
                log.exception("Daily refresh failed: %s", exc)
            time.sleep(15 * 60)

    thread = threading.Thread(
        target=loop,
        name="li-daily-refresh",
        daemon=True,
    )
    thread.start()
