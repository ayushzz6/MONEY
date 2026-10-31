"""Modeled index option prices for the 5-year backtest.

Real prices of expired option contracts are not available from Angel One or Yahoo, so the
option strategies are priced with Black-Scholes:
- underlying: the NIFTY / BANKNIFTY 5-minute bars;
- volatility: India VIX at that bar as the at-the-money IV, plus a simple skew (out-of-money
  puts richer, out-of-money calls slightly cheaper) and an `iv_scale` knob for stress tests;
- time: trading time (375 minutes a session, 252 sessions a year) to 15:30 on expiry day;
- expiries: NIFTY weekly on Thursday until 2025-08-28, Tuesday from 2025-09-02 (moved to the
  previous trading day on holidays); BANKNIFTY monthly on the last Thursday, last Tuesday
  from Sep 2025.

This is a model. Real fills differ: the actual skew, event IV, expiry-day behaviour and
spreads. The report shows how results move when IV is 15% higher or lower.
"""
import datetime as dt
from typing import Dict, List

import numpy as np
from scipy.special import ndtr

R = 0.065
MIN_PER_DAY = 375.0
YEAR_MIN = MIN_PER_DAY * 252
OPEN_MIN = 9 * 60 + 15
CLOSE_MIN = 15 * 60 + 30
TUESDAY_SWITCH = dt.date(2025, 9, 1)

PUT_SKEW = 6.0    # IV multiplier slope per unit log-moneyness below spot
CALL_SKEW = 1.5   # and above spot


def bs(S, K, T, sigma, call: bool):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-4)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (R + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    disc = np.exp(-R * T)
    if call:
        return S * ndtr(d1) - K * disc * ndtr(d2)
    return K * disc * ndtr(-d2) - S * ndtr(-d1)


def iv_at(K, S, atm_iv, iv_scale: float = 1.0):
    m = np.log(np.asarray(K, float) / np.asarray(S, float))
    slope = np.where(m < 0, -PUT_SKEW, -CALL_SKEW)
    return np.maximum(0.05, atm_iv * iv_scale * (1.0 + slope * m))


def price(S, K, T, atm_iv, call: bool, iv_scale: float = 1.0):
    return bs(S, K, T, iv_at(K, S, atm_iv, iv_scale), call)


# ---------------------------------------------------------------- expiries and time
def nifty_weekly_expiries(trading_days: List[dt.date]) -> List[dt.date]:
    days = sorted(trading_days)
    tset = set(days)
    out = []
    d0, d1 = days[0], days[-1] + dt.timedelta(days=14)
    d = d0
    while d <= d1:
        target = 1 if d >= TUESDAY_SWITCH else 3          # Tuesday or Thursday
        if d.weekday() == target:
            e = d
            for _ in range(7):
                if e in tset or e > days[-1]:
                    break
                e -= dt.timedelta(days=1)
            out.append(e)
        d += dt.timedelta(days=1)
    return sorted(set(out))


def banknifty_monthly_expiries(trading_days: List[dt.date]) -> List[dt.date]:
    days = sorted(trading_days)
    tset = set(days)
    out = []
    y, m = days[0].year, days[0].month
    last_y, last_m = (days[-1].year + (days[-1].month == 12), days[-1].month % 12 + 1)
    while (y, m) <= (last_y, last_m):
        last = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
        target = 1 if last >= TUESDAY_SWITCH else 3
        while last.weekday() != target:
            last -= dt.timedelta(days=1)
        e = last
        for _ in range(7):
            if e in tset or e > days[-1]:
                break
            e -= dt.timedelta(days=1)
        out.append(e)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(set(out))


class ExpiryClock:
    """Trading-time to expiry for any (session, bar) of the backtest."""

    def __init__(self, trading_days: List[dt.date], expiries: List[dt.date]):
        self.days = sorted(trading_days)
        self.index = {d: i for i, d in enumerate(self.days)}
        self.expiries = sorted(expiries)

    def expiry_for(self, day: dt.date, min_sessions_left: int = 0) -> dt.date:
        """The first expiry on or after `day` with at least `min_sessions_left` sessions after today."""
        for e in self.expiries:
            if e >= day and self.sessions_between(day, e) >= min_sessions_left:
                return e
        return self.expiries[-1]

    def sessions_between(self, day: dt.date, expiry: dt.date) -> int:
        i, j = self.index.get(day), self.index.get(expiry)
        if i is None or j is None:
            return max(0, int(np.busday_count(day, expiry)))
        return j - i

    def T(self, day: dt.date, minute_of_day: float, expiry: dt.date) -> float:
        left_today = max(0.0, CLOSE_MIN - minute_of_day)
        full = self.sessions_between(day, expiry)
        return max(5.0, left_today + MIN_PER_DAY * full) / YEAR_MIN


def round_strike(x, step):
    return np.round(np.asarray(x, float) / step) * step


def slot_minute(slot) -> float:
    """Clock minute at which 5-minute bar `slot` closes."""
    return OPEN_MIN + 5 * (np.asarray(slot) + 1)


# ---------------------------------------------------------------- futures margin (for reference)
def futures_margin(index_level: float, lot: int, pct: float = 0.12) -> float:
    return index_level * lot * pct


def lot_sizes() -> Dict[str, int]:
    return {"NIFTY": 65, "BANKNIFTY": 30}
