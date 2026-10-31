"""Bars, sessions and the intraday measures every strategy shares (VWAP, ATR, opening range)."""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from market.calendar import BAR_MINUTES, OPEN_MIN, SESSION_BARS, to_ist


@dataclass
class Bar:
    start: int          # bar start, epoch seconds (UTC)
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0
    date: str = field(init=False)
    slot: int = field(init=False)

    def __post_init__(self):
        t = to_ist(self.start)
        self.date = t.date().isoformat()
        self.slot = (t.hour * 60 + t.minute - OPEN_MIN) // BAR_MINUTES

    @property
    def close_minute(self) -> int:
        return OPEN_MIN + BAR_MINUTES * (self.slot + 1)

    def to_dict(self) -> dict:
        return {"t": self.start, "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v}


@dataclass
class DailyBar:
    date: str
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


def split_sessions(bars: List[Bar]) -> Dict[str, List[Bar]]:
    days: Dict[str, List[Bar]] = {}
    for b in bars:
        days.setdefault(b.date, []).append(b)
    for d in days:
        days[d].sort(key=lambda x: x.slot)
    return dict(sorted(days.items()))


def full_session(day_bars: List[Bar], max_missing: int = 3) -> Optional[List[Bar]]:
    """All 75 bars of a session indexed by slot, forward-filling up to `max_missing` empty
    bars with a flat bar at the previous close. None when more are missing or the first is."""
    by_slot = {b.slot: b for b in day_bars if 0 <= b.slot < SESSION_BARS}
    if 0 not in by_slot:
        return None
    missing = SESSION_BARS - len(by_slot)
    if missing > max_missing:
        return None
    out: List[Bar] = []
    for i in range(SESSION_BARS):
        b = by_slot.get(i)
        if b is None:
            p = out[-1]
            b = Bar(p.start + BAR_MINUTES * 60, p.c, p.c, p.c, p.c, 0.0)
        out.append(b)
    return out


def aggregate_daily(bars: List[Bar]) -> List[DailyBar]:
    out = []
    for d, rows in split_sessions(bars).items():
        out.append(DailyBar(d, rows[0].o, max(b.h for b in rows), min(b.l for b in rows),
                            rows[-1].c, sum(b.v for b in rows)))
    return out


def vwap_series(bars: List[Bar]) -> List[float]:
    """Session VWAP after each bar, from typical prices. An index has no volume, so it
    falls back to the running mean of typical prices."""
    out, pv, vol, tp_sum = [], 0.0, 0.0, 0.0
    for i, b in enumerate(bars):
        tp = (b.h + b.l + b.c) / 3.0
        pv += tp * b.v
        vol += b.v
        tp_sum += tp
        out.append(pv / vol if vol > 0 else tp_sum / (i + 1))
    return out


def atr(bars: List[Bar], n: int = 14) -> Optional[float]:
    """Wilder-free simple ATR over the last n bars (all of them when fewer)."""
    if not bars:
        return None
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b.h - b.l)
        else:
            pc = bars[i - 1].c
            trs.append(max(b.h - b.l, abs(b.h - pc), abs(b.l - pc)))
    tail = trs[-n:]
    return sum(tail) / len(tail)


def opening_range(bars: List[Bar], n: int):
    """(high, low, volume) of the first n bars of a session list ordered by slot, or None
    when any of slots 0..n-1 is missing. Volume skips the 09:15 bar, which free feeds
    often report as zero (the opening auction is booked separately)."""
    first = [b for b in bars if b.slot < n]
    if len(first) < n or {b.slot for b in first} != set(range(n)):
        return None
    return (max(b.h for b in first), min(b.l for b in first),
            sum(b.v for b in first if b.slot > 0))


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """Round to the instrument's tick. mode: nearest, down, up."""
    if tick <= 0:
        tick = 0.05
    q = price / tick
    if mode == "down":
        n = math.floor(q + 1e-9)
    elif mode == "up":
        n = math.ceil(q - 1e-9)
    else:
        n = math.floor(q + 0.5)
    decimals = max(0, -int(math.floor(math.log10(tick))) if tick < 1 else 0)
    return round(n * tick, decimals + 2)


class BarBuilder:
    """Builds 5-minute bars from live ticks (LTP plus the day's cumulative volume)."""

    def __init__(self):
        self.closed: List[Bar] = []
        self._cur: Optional[Bar] = None
        self._vol_at_start = None
        self._last_cum = None

    def on_tick(self, ts: float, ltp: float, cum_volume: Optional[float] = None) -> Optional[Bar]:
        t = to_ist(ts)
        slot = (t.hour * 60 + t.minute - OPEN_MIN) // BAR_MINUTES
        if slot < 0 or slot >= SESSION_BARS:
            return None
        finished = None
        if self._cur is not None and slot != self._cur.slot:
            finished = self._close()
        if self._cur is None:
            start = int(ts) - (t.minute % BAR_MINUTES) * 60 - t.second
            self._cur = Bar(start, ltp, ltp, ltp, ltp, 0.0)
            self._vol_at_start = self._last_cum if self._last_cum is not None else cum_volume
        else:
            b = self._cur
            b.h, b.l, b.c = max(b.h, ltp), min(b.l, ltp), ltp
        if cum_volume is not None:
            self._last_cum = cum_volume
            if self._vol_at_start is not None:
                self._cur.v = max(0.0, cum_volume - self._vol_at_start)
        return finished

    def roll(self, now_ts: float) -> Optional[Bar]:
        """Close the current bar once the clock passes its end, even with no new tick."""
        if self._cur is not None and now_ts >= self._cur.start + BAR_MINUTES * 60:
            return self._close()
        return None

    def _close(self) -> Bar:
        b = self._cur
        self._cur = None
        if not self.closed or self.closed[-1].start != b.start:
            self.closed.append(b)
        return b

    def seed(self, bars: List[Bar]):
        """Backfill today's closed bars (from a history call) before live ticks arrive."""
        known = {b.start for b in self.closed}
        self.closed = sorted(self.closed + [b for b in bars if b.start not in known], key=lambda x: x.start)

    @property
    def forming(self) -> Optional[Bar]:
        return self._cur
