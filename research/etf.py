"""Index ETFs in the cash segment (NIFTYBEES, BANKBEES): strategies that index options and futures made
too expensive, on real Angel One 5-minute ETF bars.

Costs: intraday (MIS, up to 5x = Rs 5 lakh) brokerage min(Rs 20, 0.03%) an order, STT 0.025% on the sell,
NSE 0.00297%, stamp 0.003% on the buy; delivery (CNC, Rs 1 lakh) brokerage min(Rs 20, 0.1%), STT 0.001% on
the sell (equity ETF units), stamp 0.015%, DP Rs 23.60 a sell; SEBI fee, GST; 1 bp slippage a side.
Chosen on data to 2024-12-31, judged after.
"""
import itertools

import numpy as np

from research import data
from research.sim import slot_of_close

TRAIN_END = np.datetime64("2024-12-31")
SLIP = 1 / 10000.0


def _charges(value, buy, delivery):
    if delivery:
        b, stt, stamp = min(20.0, value * 0.001), (0 if buy else value * 0.00001), (value * 0.00015 if buy else 0)
        dp = 0 if buy else 23.6
    else:
        b, stt, stamp, dp = min(20.0, value * 0.0003), (0 if buy else value * 0.00025), (value * 0.00003 if buy else 0), 0
    txn, sebi = value * 0.0000297, value * 0.000001
    return b + stt + stamp + dp + txn + sebi + 0.18 * (b + txn + sebi)


def _trade(p_in, p_out, side, notional, delivery):
    fin, fout = p_in * (1 + SLIP * side), p_out * (1 - SLIP * side)
    q = int(notional // fin)
    if q < 1:
        return 0.0
    gross = (fout - fin) * q * side
    return gross - _charges(fin * q, side > 0, delivery) - _charges(fout * q, side < 0, delivery)


def strategies(m):
    o0 = m.o[:, 0]
    c_last = m.c[:, -1]
    prev = np.r_[np.nan, c_last[:-1]]
    vw = np.cumsum((m.h + m.l + m.c) / 3, 1) / (np.arange(m.c.shape[1]) + 1)       # TWAP (volume-free)
    out = {}
    s945 = slot_of_close("09:45")
    for at, k in itertools.product(("14:30", "14:45", "15:00"), (0.001, 0.0025, 0.004)):
        s = slot_of_close(at)
        r1 = m.c[:, s945] / prev - 1
        d = np.zeros(m.n)
        for i in range(1, m.n):
            if abs(r1[i]) >= k:
                sd = 1 if r1[i] > 0 else -1
                d[i] = _trade(m.c[i, s], m.c[i, slot_of_close("15:15")], sd, 500000, False)
        out[f"Last-half-hour momentum: 09:45 move >= {100 * k:g}%, in at {at}, out 15:15 (MIS 5x)"] = d
    # overnight: buy the 15:25 close, sell the next morning
    for exit_at, cond in itertools.product(("09:15 open", "09:20 close"), ("every day", "up days only")):
        d = np.zeros(m.n)
        for i in range(m.n - 1):
            if cond == "up days only" and not c_last[i] > o0[i]:
                continue
            p_out = m.o[i + 1, 0] if exit_at == "09:15 open" else m.c[i + 1, 0]
            d[i + 1] = _trade(c_last[i], p_out, 1, 100000, True)
        out[f"Overnight: buy 15:25, sell {exit_at} ({cond}, CNC)"] = d
    # turn of the month: buy the close 2 sessions before month end, sell the close of the 3rd session after
    months = np.array([str(x)[:7] for x in m.dates])
    d = np.zeros(m.n)
    for i in range(2, m.n - 3):
        if months[i] != months[i + 2] and months[i] == months[i + 1]:
            j = i + 2
            while j < m.n and months[j] == months[i + 2] and j - (i + 2) < 2:
                j += 1
            d[j] += _trade(c_last[i], c_last[j], 1, 100000, True)
    out["Turn of the month: last 2 sessions to the 3rd (CNC)"] = d
    # intraday trend from 11:00
    for k in (0.003, 0.005):
        s = slot_of_close("11:00")
        d = np.zeros(m.n)
        for i in range(m.n):
            mv = m.c[i, s] / o0[i] - 1
            if abs(mv) < k:
                continue
            sd = 1 if mv > 0 else -1
            if (m.c[i, s] - vw[i, s]) * sd <= 0:
                continue
            stop = vw[i, s]
            x = slot_of_close("15:15")
            px = m.c[i, x]
            for t in range(s + 1, x + 1):
                if (sd > 0 and m.l[i, t] <= stop) or (sd < 0 and m.h[i, t] >= stop):
                    px = stop
                    break
            d[i] = _trade(m.c[i, s], px, sd, 500000, False)
        out[f"Trend from 11:00: {100 * k:g}%+ move and TWAP side, stop TWAP (MIS 5x)"] = d
    # gap fade to yesterday's close
    for k in (0.004, 0.007):
        d = np.zeros(m.n)
        for i in range(1, m.n):
            g = o0[i] / prev[i] - 1
            if abs(g) < k:
                continue
            sd = -1 if g > 0 else 1
            if (m.c[i, 0] - o0[i]) * sd <= 0:                  # the first bar has to turn already
                continue
            stop = m.h[i, 0] if sd < 0 else m.l[i, 0]
            tgt = prev[i]
            x = slot_of_close("15:15")
            px = m.c[i, x]
            for t in range(1, x + 1):
                if (sd < 0 and m.h[i, t] >= stop) or (sd > 0 and m.l[i, t] <= stop):
                    px = stop
                    break
                if (sd < 0 and m.l[i, t] <= tgt) or (sd > 0 and m.h[i, t] >= tgt):
                    px = tgt
                    break
            d[i] = _trade(m.c[i, 0], px, sd, 500000, False)
        out[f"Gap fade to yesterday's close: gap >= {100 * k:g}% (MIS 5x)"] = d
    return out


def run_etf(log=print):
    rows, dailies = [], {}
    for sym in ("NIFTYBEES", "BANKBEES"):
        m = data.load(sym)
        if m is None:
            continue
        month = np.array([str(x)[:7] for x in m.dates])
        train = m.dates <= TRAIN_END
        for name, d in strategies(m).items():
            r = {"family": "etf", "setup": f"{sym}: {name}", "trades": int((d != 0).sum())}
            for nm, msk in (("train", train), ("test", ~train)):
                q = d[msk]
                eq = np.cumsum(q)
                r[f"{nm}_per_month"] = round(float(q.sum()) / len(set(month[msk])), 0)
                r[f"{nm}_maxdd"] = round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0)
            rows.append(r)
            dailies[r["setup"]] = (m.dates, d)
        log(f"  ETF {sym}: {m.n} sessions")
    return rows, dailies
