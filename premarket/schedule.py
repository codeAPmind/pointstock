"""When the morning run should go ahead (DESIGN.md, "When it runs").

The job is scheduled in New York time: 08:30 ET. Local schedulers on this
machine run in Singapore time, where 08:30 ET is 21:30 (EST, winter) or
20:30 (EDT, summer), so cron fires at both and this guard decides:

- skip non-sessions (weekends, NYSE holidays via exchange_calendars XNYS),
- run only inside the pre-market window (default 08:15-09:25 ET; a late
  wake-up still produces a report before the open),
- run once per session (a stamp file marks success).
"""

from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
WINDOW = (dtime(8, 15), dtime(9, 25))


@lru_cache(maxsize=1)
def _calendar():
    import exchange_calendars as xcals

    return xcals.get_calendar("XNYS")


def is_trading_day(day) -> bool:
    import pandas as pd

    try:
        return bool(_calendar().is_session(pd.Timestamp(day)))
    except Exception:
        # Outside the calendar's range or library missing: fall back to weekdays.
        return pd.Timestamp(day).weekday() < 5


def session_date(now_utc: datetime | None = None) -> str:
    """NYSE session date for `now` (the calendar day in New York)."""
    now_ny = (now_utc or datetime.now(timezone.utc)).astimezone(NY)
    return now_ny.date().isoformat()


def stamp_path(data_dir: str | Path, session: str) -> Path:
    return Path(data_dir) / "runs" / session / "DONE"


def should_run(data_dir: str | Path, now_utc: datetime | None = None,
               window: tuple[dtime, dtime] = WINDOW, force: bool = False) -> tuple[bool, str]:
    now_ny = (now_utc or datetime.now(timezone.utc)).astimezone(NY)
    day = now_ny.date().isoformat()
    if not is_trading_day(now_ny.date()):
        return False, f"market closed on {day} (NYSE calendar)"
    if force:
        return True, "forced"
    if not (window[0] <= now_ny.time() <= window[1]):
        return False, f"outside the pre-market window ({now_ny:%H:%M} ET; window {window[0]:%H:%M}-{window[1]:%H:%M})"
    if stamp_path(data_dir, day).exists():
        return False, f"already ran for the {day} session"
    return True, f"pre-market run for {day}"


def mark_done(data_dir: str | Path, session: str, note: str = "") -> None:
    p = stamp_path(data_dir, session)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{datetime.now(timezone.utc).isoformat()} {note}\n")
