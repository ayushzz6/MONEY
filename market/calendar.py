"""NSE equity session calendar, in India Standard Time.

IST is a fixed UTC+05:30 offset (no daylight saving), so no tz database is needed.
Regular session 09:15 to 15:30. Pre-open 09:00 to 09:08 is not traded by the engine.
"""
import datetime as dt
from typing import Optional, Tuple

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")

OPEN_MIN = 9 * 60 + 15      # 09:15
CLOSE_MIN = 15 * 60 + 30    # 15:30
BAR_MINUTES = 5
SESSION_BARS = (CLOSE_MIN - OPEN_MIN) // BAR_MINUTES   # 75 five-minute bars

# NSE equity segment trading holidays (weekday closures), from NSE's 2026 circular as
# reported by brokers (read 2026-10-23). Jan 15 2026 was an extra closure (Maharashtra
# municipal elections). Muhurat trading (Sun Nov 8 2026) is a special evening session
# and is not traded by the engine.
NSE_HOLIDAYS = {
    2026: {
        "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
        "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
        "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
        "2026-12-25",
    },
}


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def to_ist(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, IST)


def parse_hhmm(text: str) -> int:
    """'09:30' -> minutes after midnight."""
    hh, mm = str(text).strip().split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad clock time {text!r}")
    return h * 60 + m


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def minute_of(t: dt.datetime) -> int:
    return t.hour * 60 + t.minute


def holiday_list_known(day: dt.date) -> bool:
    return day.year in NSE_HOLIDAYS


def is_trading_day(day: dt.date) -> bool:
    if day.weekday() >= 5:
        return False
    return day.isoformat() not in NSE_HOLIDAYS.get(day.year, set())


def session_state(t: Optional[dt.datetime] = None) -> str:
    """WEEKEND, HOLIDAY, PRE_OPEN (before 09:15), OPEN, or CLOSED (after 15:30)."""
    t = (t or now_ist()).astimezone(IST)
    d = t.date()
    if d.weekday() >= 5:
        return "WEEKEND"
    if not is_trading_day(d):
        return "HOLIDAY"
    m = t.hour * 60 + t.minute + t.second / 60.0
    if m < OPEN_MIN:
        return "PRE_OPEN"
    if m < CLOSE_MIN:
        return "OPEN"
    return "CLOSED"


def next_open(t: Optional[dt.datetime] = None) -> dt.datetime:
    t = (t or now_ist()).astimezone(IST)
    d = t.date()
    if is_trading_day(d) and minute_of(t) < OPEN_MIN:
        return dt.datetime.combine(d, dt.time(9, 15), IST)
    d += dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            return dt.datetime.combine(d, dt.time(9, 15), IST)
        d += dt.timedelta(days=1)
    raise RuntimeError("no trading day in the next 30 days")


def previous_trading_day(day: dt.date) -> dt.date:
    d = day - dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            return d
        d -= dt.timedelta(days=1)
    raise RuntimeError("no trading day in the previous 30 days")


def session_bounds(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    return (dt.datetime.combine(day, dt.time(9, 15), IST),
            dt.datetime.combine(day, dt.time(15, 30), IST))


def slot_of_minute(minute: int) -> int:
    """Index of the 5-minute bar that starts at or contains this minute (0 = 09:15 bar)."""
    return (minute - OPEN_MIN) // BAR_MINUTES


def bar_close_minute(slot: int) -> int:
    """Clock minute at which bar `slot` closes (bar 0 closes at 09:20)."""
    return OPEN_MIN + BAR_MINUTES * (slot + 1)


def last_closed_slot(t: dt.datetime) -> int:
    """Highest bar index whose close time has passed, -1 before the first bar closes."""
    m = minute_of(t.astimezone(IST))
    return min(SESSION_BARS - 1, (m - OPEN_MIN) // BAR_MINUTES - 1)


def describe(t: Optional[dt.datetime] = None) -> dict:
    t = (t or now_ist()).astimezone(IST)
    state = session_state(t)
    out = {
        "now": t.strftime("%Y-%m-%d %H:%M:%S"),
        "date": t.date().isoformat(),
        "state": state,
        "holiday_list_known": holiday_list_known(t.date()),
    }
    if state == "OPEN":
        close = dt.datetime.combine(t.date(), dt.time(15, 30), IST)
        out["minutes_to_close"] = int((close - t).total_seconds() // 60)
    else:
        nxt = next_open(t)
        out["next_open"] = nxt.strftime("%a %Y-%m-%d %H:%M")
        out["minutes_to_open"] = int((nxt - t).total_seconds() // 60)
    return out
