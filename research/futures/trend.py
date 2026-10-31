"""Positional trend following on MCX commodities and NSE currency futures (held overnight).

Systems: Donchian channel breakouts, moving-average crossovers and time-series momentum, long or
short, signal at the close, trade at the next open. Size: lots so that a 2-ATR move costs the risk
budget, capped by what Rs 1 lakh of margin carries. Costs on every trade plus a monthly roll (two
orders and slippage). MCX's continuous series is not roll-adjusted: the overnight jump on the first
session after a contract's usual expiry window is left out of P&L when it is an outlier (a crude
roll-adjustment). Currency futures pay (long) or earn (short) a forward premium of about 1.5% a year.
Chosen on data to 2024-12-31 (currencies: 2016-2022 daily, test 2023 on), judged after.
"""
import itertools

import numpy as np

from research.futures.data import SPECS, load_daily
from research.futures.factory import charges

CAPITAL = 100000.0
ROLL_WINDOWS = {"CRUDEOIL": (15, 23), "NATURALGAS": (22, 29), "SILVER": (1, 8), "GOLD": (1, 8)}
FWD_PREMIUM = 0.015


def _sma(x, n):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.r_[0, x])
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def signals(c, h, l, system):
    kind, a, b = system
    pos = np.zeros(len(c))
    if kind == "donchian":
        hi = np.r_[[np.nan], [np.max(h[max(0, i - a):i]) if i >= a else np.nan for i in range(1, len(c))]]
        lo = np.r_[[np.nan], [np.min(l[max(0, i - a):i]) if i >= a else np.nan for i in range(1, len(c))]]
        xhi = np.r_[[np.nan], [np.max(h[max(0, i - b):i]) if i >= b else np.nan for i in range(1, len(c))]]
        xlo = np.r_[[np.nan], [np.min(l[max(0, i - b):i]) if i >= b else np.nan for i in range(1, len(c))]]
        p = 0
        for i in range(len(c)):
            if p == 0:
                if c[i] > hi[i]:
                    p = 1
                elif c[i] < lo[i]:
                    p = -1
            elif p == 1 and c[i] < xlo[i]:
                p = 0
            elif p == -1 and c[i] > xhi[i]:
                p = 0
            pos[i] = p
    elif kind == "ma":
        f, s = _sma(c, a), _sma(c, b)
        pos = np.where(np.isnan(s), 0, np.sign(f - s))
    elif kind == "tsmom":
        r = np.r_[[np.nan] * a, c[a:] / c[:-a] - 1]
        raw = np.where(np.isnan(r), 0, np.sign(r))
        pos = np.zeros(len(c))
        for i in range(len(c)):           # rebalance every b sessions
            pos[i] = raw[i] if i % b == 0 else pos[i - 1] if i else 0
    return pos


SYSTEMS = [("donchian", 20, 10), ("donchian", 55, 20), ("ma", 10, 50), ("ma", 20, 100), ("ma", 50, 200),
           ("tsmom", 60, 5), ("tsmom", 120, 5), ("tsmom", 250, 5)]
RISKS = [2000.0, 4000.0]


def backtest(name, system, risk, train_end):
    sp = SPECS[name]
    dates, o, h, l, c = load_daily(name)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.r_[[np.nan], _sma(np.nan_to_num(tr), 20)[:-1]]
    sig = signals(c, h, l, system)
    gap = o - pc
    days = [d.astype(object) for d in dates]
    roll = np.zeros(len(c), bool)
    if sp.exchange == "MCX":
        lo_d, hi_d = ROLL_WINDOWS[sp.series]
        big = np.abs(gap) > 1.0 * np.nan_to_num(atr, nan=np.inf)
        inwin = np.array([lo_d <= d.day <= hi_d for d in days])
        roll = big & inwin
    pnl = np.zeros(len(c))
    lots = 0.0
    side = 0
    trades = 0
    last_month = None
    for i in range(1, len(c)):
        # P&L of the position carried into today (gap then intraday), at yesterday's size
        if side and lots:
            g = 0.0 if roll[i] else gap[i]
            pnl[i] += side * lots * sp.multiplier * (g + (c[i] - o[i]))
            if sp.exchange == "CDS":
                pnl[i] -= side * lots * sp.multiplier * c[i] * FWD_PREMIUM / 252
            m = days[i].month
            if last_month is not None and m != last_month:      # monthly roll: two orders + slippage
                v = lots * sp.multiplier * c[i]
                pnl[i] -= charges(v, True, sp) + charges(v, False, sp) + 2 * lots * sp.multiplier * max(sp.tick, 1e-4 * c[i])
            last_month = m
        # yesterday's signal sets today's position at today's open
        want = int(sig[i - 1])
        if want != side and np.isfinite(atr[i]) and atr[i] > 0:
            slip = max(sp.tick, 1e-4 * o[i])
            if side and lots:
                v = lots * sp.multiplier * o[i]
                pnl[i] -= charges(v, side < 0, sp) + lots * sp.multiplier * slip
                pnl[i] += -side * lots * sp.multiplier * (c[i] - o[i])      # closed at the open, not held all day
            if want:
                new = np.floor(risk / (2 * atr[i] * sp.multiplier))
                new = min(new, np.floor(CAPITAL / (o[i] * sp.multiplier * sp.margin_pct)))
                if new >= 1:
                    v = new * sp.multiplier * o[i]
                    pnl[i] -= charges(v, want > 0, sp) + new * sp.multiplier * slip
                    pnl[i] += want * new * sp.multiplier * (c[i] - o[i])
                    lots, side = new, want
                    trades += 1
                    last_month = days[i].month
                else:
                    lots, side = 0, 0
            else:
                lots, side = 0, 0
    train = dates <= np.datetime64(train_end)
    month = np.array([str(d)[:7] for d in dates])
    out = {"contract": name, "system": f"{system[0]} {system[1]}/{system[2]}", "risk": risk, "trades": trades,
           "from": str(dates[0]), "to": str(dates[-1]), "split": train_end}
    for nm, m in (("train", train), ("test", ~train)):
        q = pnl[m]
        eq = np.cumsum(q)
        out[f"{nm}_per_month"] = round(float(q.sum()) / max(1, len(set(month[m]))), 0)
        out[f"{nm}_maxdd"] = round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0) if len(q) else 0
    return out, dates, pnl


def run_trend(log=print):
    rows = []
    for name, sp in SPECS.items():
        split = "2024-12-31" if sp.exchange == "MCX" else "2022-12-31"
        for system, risk in itertools.product(SYSTEMS, RISKS):
            r, _, _ = backtest(name, system, risk, split)
            rows.append(r)
    log(f"  trend following: {len(rows)} variants")
    return rows
