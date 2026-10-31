"""Strategy factory: equity cash SWING trades (delivery / CNC, long only, no leverage), NIFTY 100.

Signals use the day's close; entries fill at the next session's open; stops fill at the stop
(or the open on a gap); time and condition exits fill at the close. Up to `slots` positions of
equal size (Rs 1 lakh / slots). Delivery costs: brokerage min(Rs 20, 0.1%) an order, STT 0.1% on
the buy and the sell, NSE 0.00297%, SEBI, stamp 0.015% on buys, GST 18%, DP charge Rs 20 + GST a
sell, and 5 bps slippage a side. Selection on 2021-24, judged on 2025-26.
"""
import itertools
import time
from collections import defaultdict

import numpy as np

from research import data

TRAIN_END = np.datetime64("2024-12-31")
CAPITAL = 100000.0
SLIP = 5 / 10000.0


def delivery_charges(value, buy):
    b = min(20.0, value * 0.001)
    txn = value * 0.00297 / 100
    sebi = value * 0.0001 / 100
    stt = value * 0.001
    stamp = value * 0.00015 if buy else 0.0
    dp = 0.0 if buy else 20.0 * 1.18
    return b + txn + sebi + stt + stamp + dp + 0.18 * (b + txn + sebi)


def _sma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.r_[0, x])
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au = np.full(len(c), np.nan)
    ad = np.full(len(c), np.nan)
    if len(c) > n:
        au[n], ad[n] = up[1:n + 1].mean(), dn[1:n + 1].mean()
        for i in range(n + 1, len(c)):
            au[i] = (au[i - 1] * (n - 1) + up[i]) / n
            ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def _roll(x, n, fn):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = fn(x[i - n + 1:i + 1])
    return out


class Stock:
    def __init__(self, m):
        self.sym = m.symbol
        o, h, l, c, v = m.daily()
        self.dates, self.o, self.h, self.l, self.c, self.v = m.dates, o, h, l, c, v
        self.sma5, self.sma50, self.sma100 = _sma(c, 5), _sma(c, 50), _sma(c, 100)
        self.rsi2 = _rsi(c, 2)
        pc = np.r_[np.nan, c[:-1]]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        self.atr = _sma(np.nan_to_num(tr), 14)
        self.hi20 = np.r_[np.nan, _roll(h, 20, np.max)[:-1]]
        self.lo10 = np.r_[np.nan, _roll(l, 10, np.min)[:-1]]
        self.hi250 = _roll(c, 250, np.max)
        self.vavg20 = np.r_[np.nan, _sma(v, 20)[:-1]]
        rng = h - l
        self.nr7 = np.r_[[False] * 6, [rng[i] <= rng[i - 6:i + 1].min() for i in range(6, len(c))]]
        self.ret63 = np.r_[[np.nan] * 63, c[63:] / c[:-63] - 1]
        self.pos = {d: i for i, d in enumerate(m.dates)}


def signals(st: Stock, name):
    """Per-day boolean (signal at the close) and a score (higher = taken first)."""
    c, o, l = st.c, st.o, st.l
    pc = np.r_[np.nan, c[:-1]]
    pl = np.r_[np.nan, l[:-1]]
    with np.errstate(invalid="ignore"):
        if name == "RSI(2) pullback in uptrend":
            sig = (st.rsi2 < 10) & (c > st.sma100)
            return sig, -st.rsi2
        if name == "20-day Donchian breakout":
            sig = (c > st.hi20) & (c > st.sma50)
            return sig, c / st.hi20
        if name == "Near 250-day high":
            sig = (c >= 0.98 * st.hi250) & (st.sma50 > np.r_[np.nan, st.sma50[:-1]])
            return sig, c / st.hi250
        if name == "NR7 in uptrend":
            sig = st.nr7 & (c > st.sma50)
            return sig, -(st.h - st.l) / c
        if name == "Gap-down reversal in uptrend":
            sig = (o < pl * 0.99) & (c > o) & (c > st.sma50)
            return sig, (c - o) / o
        if name == "3 down closes in uptrend":
            d1 = c < pc
            d3 = d1 & np.r_[False, d1[:-1]] & np.r_[False, False, d1[:-2]]
            return d3 & (c > st.sma100), -(c / np.r_[np.nan, np.nan, np.nan, c[:-3]] - 1)
        if name == "Volume breakout":
            sig = (c / pc - 1 > 0.03) & (st.v > 2 * st.vavg20) & (c >= 0.98 * st.hi20)
            return sig, st.v / st.vavg20
        if name == "Post-gap drift (3% gap-up, 3x volume)":
            sig = (o / pc - 1 >= 0.03) & (st.v >= 3 * st.vavg20) & (c >= (st.h + st.l) / 2)
            return sig, st.v / st.vavg20
        if name == "Weekly losers reversal (uptrend)":
            wd = np.array([d.astype(object).weekday() for d in st.dates])
            week_end = np.r_[wd[1:], [0]] < wd
            r5 = np.r_[[np.nan] * 5, c[5:] / c[:-5] - 1]
            sig = week_end & (c > st.sma100) & (r5 <= -0.04)
            return sig, -r5
        if name == "Closing strength at a 20-day high (2x volume)":
            pos = (c - st.l) / np.maximum(st.h - st.l, 1e-9)
            sig = (pos >= 0.95) & (st.v >= 2 * st.vavg20) & (c >= st.hi20)
            return sig, st.v / st.vavg20
        if name == "Weekly RS rotation (top 63-day return)":
            wd = np.array([d.astype(object).weekday() for d in st.dates])
            nxt = np.r_[wd[1:], [0]]
            week_end = nxt < wd                    # next session starts a new week
            sig = week_end & (c > st.sma50) & (st.ret63 > 0)
            return sig, st.ret63
    raise ValueError(name)


SWING_SETUPS = ["RSI(2) pullback in uptrend", "20-day Donchian breakout", "Near 250-day high", "NR7 in uptrend",
                "Gap-down reversal in uptrend", "3 down closes in uptrend", "Volume breakout",
                "Weekly RS rotation (top 63-day return)", "Post-gap drift (3% gap-up, 3x volume)",
                "Weekly losers reversal (uptrend)", "Closing strength at a 20-day high (2x volume)"]
HOLDS = [3, 5, 10, 20]
STOPS = [0, 2.0, 3.0]
SLOTS = [5, 10]
EXIT_RULES = ["time", "close>SMA5", "close<10-day low"]


def simulate(stocks, master_dates, setup, hold, stop_k, slots, exit_rule):
    sigs = [signals(s, setup) for s in stocks]
    size = CAPITAL / slots
    by_day = defaultdict(list)                    # master date index -> [(score, k, row)]
    mpos = {d: i for i, d in enumerate(master_dates)}
    for k, (st, (sig, score)) in enumerate(zip(stocks, sigs)):
        for r in np.flatnonzero(np.nan_to_num(sig, nan=0).astype(bool)):
            if r + 1 < len(st.dates):
                i = mpos.get(st.dates[r])
                if i is not None:
                    by_day[i].append((float(np.nan_to_num(score[r], nan=-1e9)), k, r))
    daily = np.zeros(len(master_dates))
    trades = []
    open_pos = []                                 # dicts
    for i in range(len(master_dates)):
        d = master_dates[i]
        # 1. manage open positions on day i
        still = []
        for p in open_pos:
            st = stocks[p["k"]]
            r = st.pos.get(d)
            if r is None:
                still.append(p)
                continue
            if r == p["entry_row"]:
                still.append(p)
                continue
            out_px, why = None, None
            if p["stop"] and st.l[r] <= p["stop"]:
                out_px, why = min(p["stop"], st.o[r]), "stop"
            elif exit_rule == "close>SMA5" and st.c[r] > st.sma5[r]:
                out_px, why = st.c[r], "sma5"
            elif exit_rule == "close<10-day low" and st.c[r] < st.lo10[r]:
                out_px, why = st.c[r], "10-day low"
            elif r - p["entry_row"] >= hold:
                out_px, why = st.c[r], "time"
            if out_px is None:
                still.append(p)
                continue
            fill = out_px * (1 - SLIP)
            gross = (fill - p["fill"]) * p["qty"]
            ch = p["ch_in"] + delivery_charges(fill * p["qty"], False)
            net = gross - ch
            daily[i] += net
            trades.append({"day": str(d), "sym": st.sym, "net": net, "why": why, "days": r - p["entry_row"]})
        open_pos = still
        # 2. yesterday's signals enter at today's open
        if i > 0 and by_day.get(i - 1):
            held = {p["k"] for p in open_pos}
            for score, k, r in sorted(by_day[i - 1], reverse=True):
                if len(open_pos) >= slots:
                    break
                if k in held:
                    continue
                st = stocks[k]
                er = r + 1
                if er >= len(st.dates) or st.dates[er] != d:
                    continue
                fill = st.o[er] * (1 + SLIP)
                qty = int(size // fill)
                if qty < 1:
                    continue
                stop = fill - stop_k * st.atr[r] if stop_k and np.isfinite(st.atr[r]) else 0.0
                ch_in = delivery_charges(fill * qty, True)
                open_pos.append({"k": k, "entry_row": er, "fill": fill, "qty": qty, "stop": stop, "ch_in": ch_in})
                held.add(k)
                # a stop can be hit on the entry day itself
                if stop and st.l[er] <= stop:
                    p = open_pos.pop()
                    out = min(stop, st.o[er]) * (1 - SLIP)
                    net = (out - fill) * qty - ch_in - delivery_charges(out * qty, False)
                    daily[i] += net
                    trades.append({"day": str(d), "sym": st.sym, "net": net, "why": "stop", "days": 0})
                    held.discard(k)
    return daily, trades


def run_swing_search(symbols, master_dates, log=print):
    t0 = time.time()
    stocks = []
    for s in symbols:
        m = data.load(s)
        if m is not None and m.n > 300:
            stocks.append(Stock(m))
    log(f"  swing: {len(stocks)} stocks loaded ({time.time() - t0:.0f}s)")
    train = master_dates <= TRAIN_END
    month = np.array([str(d)[:7] for d in master_dates])
    mtr, mte = len(set(month[train])), len(set(month[~train]))
    rows = []
    for setup, hold, stop_k, slots, exit_rule in itertools.product(SWING_SETUPS, HOLDS, STOPS, SLOTS, EXIT_RULES):
        if setup.startswith("Weekly RS") and exit_rule != "time":
            continue
        daily, trades = simulate(stocks, master_dates, setup, hold, stop_k, slots, exit_rule)
        r = {"family": "swing", "setup": setup, "hold_days": hold, "stop_atr": stop_k, "slots": slots,
             "exit_rule": exit_rule}
        for name, msk, months in (("train", train, mtr), ("test", ~train, mte)):
            q = daily[msk]
            eq = np.cumsum(q)
            n = sum(1 for t in trades if (np.datetime64(t["day"]) <= TRAIN_END) == (name == "train"))
            w = [t for t in trades if (np.datetime64(t["day"]) <= TRAIN_END) == (name == "train")]
            r[f"{name}_net"] = round(float(q.sum()), 0)
            r[f"{name}_per_month"] = round(float(q.sum()) / months, 0)
            r[f"{name}_trades"] = n
            r[f"{name}_win"] = round(100 * np.mean([t["net"] > 0 for t in w]), 1) if w else None
            r[f"{name}_maxdd"] = round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0) if len(q) else 0
        rows.append(r)
    log(f"  swing grid: {len(rows)} variants ({time.time() - t0:.0f}s)")
    return rows, stocks
