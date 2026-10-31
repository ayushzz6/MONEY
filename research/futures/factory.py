"""Strategy factory for MCX commodities and NSE currency futures (intraday, flat every session).

setups x filters x exits x risk budget, per contract. Lots: the risk budget / (stop distance x
lot multiplier), at least 1, capped by what a Rs 1 lakh account's margin allows; a trade whose
1-lot risk is over 1.5x the budget is skipped. Charges: Rs 20 an order; MCX: CTT 0.01% on sells,
exchange 0.0021%, stamp 0.002% on buys; currency: exchange 0.00035%, stamp 0.0001%; SEBI fee and
GST on both; slippage max(1 tick, 1 bp) a side. Chosen on data to 2024-12-31, judged after.
"""
import datetime as dt
import itertools
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from research.futures.data import SPECS, Sessions, Spec, load

TRAIN_END = np.datetime64("2024-12-31")
CAPITAL = 100000.0


def first_true(mask):
    a = mask.any(1)
    return np.where(a, mask.argmax(1), -1)


def _pick(u, d):
    side = np.where((u >= 0) & ((d < 0) | (u < d)), 1, np.where(d >= 0, -1, 0))
    return side, np.where(side == 1, u, d)


@dataclass
class Feat:
    prev_close: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray
    atr: np.ndarray          # daily ATR(14), known before the session
    trend_up: np.ndarray
    trend_down: np.ndarray
    vwap: np.ndarray
    vsd: np.ndarray
    atr_rank: np.ndarray     # ATR as % of price vs its trailing year: 0 low .. 1 high
    us_dst: np.ndarray       # US daylight saving on that date (shifts the US session an hour earlier in IST)
    weekday: np.ndarray


def _dst(d: dt.date) -> bool:
    def nth_sunday(y, m, n):
        x = dt.date(y, m, 1)
        x += dt.timedelta(days=(6 - x.weekday()) % 7)
        return x + dt.timedelta(weeks=n - 1)
    return nth_sunday(d.year, 3, 2) <= d < nth_sunday(d.year, 11, 1)


def features(s: Sessions) -> Feat:
    o, h, l, c = s.daily()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.full(s.n, np.nan)
    for i in range(15, s.n):
        atr[i] = np.nanmean(tr[i - 14:i])
    sma20 = np.array([np.nanmean(c[max(0, i - 20):i]) if i >= 20 else np.nan for i in range(s.n)])
    sma50 = np.array([np.nanmean(c[max(0, i - 50):i]) if i >= 50 else np.nan for i in range(s.n)])
    tp = (s.h + s.l + s.c) / 3
    vol = np.where(np.nan_to_num(s.v) > 0, s.v, 1.0)
    vol = np.where(np.isnan(tp), 0, vol)
    tpz = np.nan_to_num(tp)
    cv = np.cumsum(vol, 1)
    vw = np.cumsum(tpz * vol, 1) / np.maximum(cv, 1e-9)
    var = np.cumsum(vol * tpz * tpz, 1) / np.maximum(cv, 1e-9) - vw * vw
    atrp = atr / pc
    rank = np.full(s.n, np.nan)
    for i in range(260, s.n):
        w = atrp[i - 250:i]
        rank[i] = np.mean(w < atrp[i])
    days = [d.astype(object) for d in s.dates]
    return Feat(pc, np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]], atr, (pc > sma20) & (sma20 > sma50),
                (pc < sma20) & (sma20 < sma50), vw, np.sqrt(np.maximum(var, 0)), rank,
                np.array([_dst(d) for d in days]), np.array([d.weekday() for d in days]))


# ---------------------------------------------------------------- setups
def S_idx(s):
    return np.arange(s.c.shape[1])


def orb(minutes, until):
    def fn(s, f, sp):
        n = max(1, minutes // s.bar_min)
        oh, ol = np.nanmax(s.h[:, :n], 1), np.nanmin(s.l[:, :n], 1)
        I = S_idx(s)
        win = (I >= n) & (I <= s.slot_of_close(until))
        side, slot = _pick(first_true(win & (s.c > oh[:, None])), first_true(win & (s.c < ol[:, None])))
        return _done(s, side, slot, (oh + ol) / 2)
    return fn


def window_breakout(r_from, r_to, until, dst_shift=True):
    """Range of the bars from r_from to r_to (IST, an hour later outside US summer time), then the first
    close beyond it until `until`."""
    def fn(s, f, sp):
        I = S_idx(s)
        side = np.zeros(s.n, int)
        slot = np.full(s.n, -1)
        mid = np.full(s.n, np.nan)
        for shift in (0, 60):
            rows = np.flatnonzero(f.us_dst == (shift == 0)) if dst_shift else np.arange(s.n)
            if not dst_shift and shift:
                break
            a = _slot(s, r_from, shift if dst_shift else 0) + 1
            b = _slot(s, r_to, shift if dst_shift else 0)
            e = _slot(s, until, shift if dst_shift else 0)
            if b <= a:
                continue
            oh, ol = np.nanmax(s.h[rows, a:b + 1], 1), np.nanmin(s.l[rows, a:b + 1], 1)
            win = (I > b) & (I <= e)
            sd, sl = _pick(first_true(win & (s.c[rows] > oh[:, None])), first_true(win & (s.c[rows] < ol[:, None])))
            side[rows], slot[rows], mid[rows] = sd, sl, (oh + ol) / 2
        return _done(s, side, slot, mid)
    return fn


def _slot(s, hhmm, shift_min=0):
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m + shift_min - s.open_min) // s.bar_min - 1


def prev_day_breakout(s, f, sp):
    I = S_idx(s)
    o0 = s.o[:, 0]
    inside = (o0 < f.prev_high) & (o0 > f.prev_low)
    win = (I >= max(1, 30 // s.bar_min)) & (I <= s.slot_of_close("21:00" if sp.exchange == "MCX" else "15:30"))
    u = np.where(inside, first_true(win & (s.c > f.prev_high[:, None])), -1)
    d = np.where(inside, first_true(win & (s.c < f.prev_low[:, None])), -1)
    side, slot = _pick(u, d)
    return _done(s, side, slot, None, swing=3)


def vwap_band(sig):
    def fn(s, f, sp):
        I = S_idx(s)
        win = (I >= s.slot_of_close("10:00")) & (I <= s.slot_of_close("21:00" if sp.exchange == "MCX" else "15:30")) & (f.vsd > 0)
        u = first_true(win & (s.c < f.vwap - sig * f.vsd))
        d = first_true(win & (s.c > f.vwap + sig * f.vsd))
        side, slot = _pick(u, d)
        rows = np.flatnonzero(side != 0)
        sl, sd = slot[rows], side[rows]
        e = s.c[rows, sl]
        vw = f.vwap[rows, sl]
        return rows, sl, sd, e, e - (vw - e), False
    return fn


def time_momentum(at, k_atr):
    def fn(s, f, sp):
        sl = s.slot_of_close(at)
        if sl <= 0:
            return _empty()
        mv = s.c[:, sl] - s.o[:, 0]
        vw = f.vwap[:, sl]
        side = np.where((mv > k_atr * f.atr) & (s.c[:, sl] > vw), 1, np.where((mv < -k_atr * f.atr) & (s.c[:, sl] < vw), -1, 0))
        rows = np.flatnonzero((side != 0) & ~np.isnan(s.c[:, sl]))
        sd = side[rows]
        stop = np.where(sd > 0, np.nanmin(s.l[rows, :sl + 1], 1), np.nanmax(s.h[rows, :sl + 1], 1))
        return rows, np.full(len(rows), sl), sd, s.c[rows, sl], stop, False
    return fn


def late_reversal(at, k_atr):
    def fn(s, f, sp):
        sl = s.slot_of_close(at)
        mv = s.c[:, sl] - f.prev_close
        side = np.where(mv > k_atr * f.atr, -1, np.where(mv < -k_atr * f.atr, 1, 0))
        rows = np.flatnonzero((side != 0) & ~np.isnan(s.c[:, sl]))
        sd = side[rows]
        stop = np.where(sd < 0, np.nanmax(s.h[rows, :sl + 1], 1), np.nanmin(s.l[rows, :sl + 1], 1))
        return rows, np.full(len(rows), sl), sd, s.c[rows, sl], stop, False
    return fn


def gap_play(lo, follow):
    """Session open vs the previous session's close, in daily ATRs."""
    def fn(s, f, sp):
        o0, c0 = s.o[:, 0], s.c[:, 0]
        g = (o0 - f.prev_close) / f.atr
        big = np.abs(g) >= lo
        if follow:
            side = np.where(big & (g > 0) & (c0 > o0), 1, np.where(big & (g < 0) & (c0 < o0), -1, 0))
        else:
            side = np.where(big & (g > 0) & (c0 < o0), -1, np.where(big & (g < 0) & (c0 > o0), 1, 0))
        rows = np.flatnonzero(side != 0)
        sd = side[rows]
        stop = np.where(sd > 0, s.l[rows, 0], s.h[rows, 0])
        return rows, np.zeros(len(rows), int), sd, c0[rows], stop, False
    return fn


def ema_cross(fast=9, slow=21):
    def fn(s, f, sp):
        c = np.where(np.isnan(s.c), np.nan, s.c)
        e1, e2 = _ema(c, fast), _ema(c, slow)
        ab = e1 > e2
        cu = np.zeros_like(ab)
        cd = np.zeros_like(ab)
        cu[:, 1:] = ab[:, 1:] & ~ab[:, :-1]
        cd[:, 1:] = ~ab[:, 1:] & ab[:, :-1]
        I = S_idx(s)
        win = (I >= slow) & (I <= s.slot_of_close("21:00"))
        side, slot = _pick(first_true(win & cu), first_true(win & cd))
        return _done(s, side, slot, None, swing=5)
    return fn


def supertrend(period=10, mult=3.0):
    def fn(s, f, sp):
        H, L, C = s.h, s.l, s.c
        D, S = C.shape
        pcl = np.c_[C[:, :1], C[:, :-1]]
        tr = np.fmax(H - L, np.fmax(np.abs(H - pcl), np.abs(L - pcl)))
        atr = np.full_like(C, np.nan)
        atr[:, period - 1] = np.nanmean(tr[:, :period], 1)
        for j in range(period, S):
            atr[:, j] = (atr[:, j - 1] * (period - 1) + np.nan_to_num(tr[:, j], nan=0)) / period
        mid = (H + L) / 2
        ub, lb = mid + mult * atr, mid - mult * atr
        fub, flb = ub.copy(), lb.copy()
        trend = np.ones((D, S), int)
        line = np.full_like(C, np.nan)
        for j in range(period, S):
            fub[:, j] = np.where((ub[:, j] < fub[:, j - 1]) | (C[:, j - 1] > fub[:, j - 1]), ub[:, j], fub[:, j - 1])
            flb[:, j] = np.where((lb[:, j] > flb[:, j - 1]) | (C[:, j - 1] < flb[:, j - 1]), lb[:, j], flb[:, j - 1])
            prev = trend[:, j - 1]
            trend[:, j] = np.where((prev < 0) & (C[:, j] > fub[:, j - 1]), 1,
                                   np.where((prev > 0) & (C[:, j] < flb[:, j - 1]), -1, prev))
            line[:, j] = np.where(trend[:, j] > 0, flb[:, j], fub[:, j])
        fu = np.zeros_like(C, bool)
        fd = np.zeros_like(C, bool)
        fu[:, 1:] = (trend[:, 1:] > 0) & (trend[:, :-1] < 0)
        fd[:, 1:] = (trend[:, 1:] < 0) & (trend[:, :-1] > 0)
        I = S_idx(s)
        win = (I >= period + 2) & (I <= s.slot_of_close("21:00"))
        side, slot = _pick(first_true(win & fu), first_true(win & fd))
        rows = np.flatnonzero(side != 0)
        sl, sd = slot[rows], side[rows]
        return rows, sl, sd, s.c[rows, sl], line[rows, sl], False
    return fn


def inventory_report(weekday):
    """US weekly inventory report (crude: Wednesday, gas: Thursday) at 10:30 New York time = 20:00 IST in
    US summer time, 21:00 otherwise. Follow the direction of the report's first 5-minute bar if it moved
    more than 0.15 daily ATR. Stop: that bar's other end."""
    def fn(s, f, sp):
        rows, slots, sides, stops = [], [], [], []
        for i in np.flatnonzero(f.weekday == weekday):
            j = _slot(s, "20:05", 0 if f.us_dst[i] else 60)
            if j < 0 or j >= s.c.shape[1] or np.isnan(s.c[i, j]) or not np.isfinite(f.atr[i]):
                continue
            body = s.c[i, j] - s.o[i, j]
            if abs(body) < 0.15 * f.atr[i]:
                continue
            sd = 1 if body > 0 else -1
            rows.append(i); slots.append(j); sides.append(sd)
            stops.append(s.l[i, j] if sd > 0 else s.h[i, j])
        rows = np.array(rows, int)
        return rows, np.array(slots, int), np.array(sides, int), s.c[rows, np.array(slots, int)] if len(rows) else np.zeros(0), np.array(stops, float), False
    return fn


def _ema(x, n):
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for j in range(1, x.shape[1]):
        out[:, j] = np.where(np.isnan(x[:, j]), out[:, j - 1], a * x[:, j] + (1 - a) * out[:, j - 1])
    return out


def _empty():
    z = np.zeros(0)
    return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), z, z, False


def _done(s, side, slot, mid, swing=0):
    rows = np.flatnonzero(side != 0)
    sl, sd = slot[rows], side[rows]
    e = s.c[rows, sl]
    if swing:
        st = np.array([np.nanmin(s.l[r, max(0, k - swing + 1):k + 1]) if d > 0 else np.nanmax(s.h[r, max(0, k - swing + 1):k + 1])
                       for r, k, d in zip(rows, sl, sd)])
    else:
        st = mid[rows]
    return rows, sl, sd, e, st, False


def setups_for(sp: Spec) -> Dict[str, Callable]:
    if sp.exchange == "MCX":
        d = {
            "ORB 15 min (09:00)": orb(15, "13:00"), "ORB 30 min (09:00)": orb(30, "13:00"), "ORB 60 min (09:00)": orb(60, "15:00"),
            "Evening range 17:00-18:30 breakout": window_breakout("17:00", "18:30", "21:30"),
            "US open first hour breakout (18:30-19:30)": window_breakout("18:30", "19:30", "22:30"),
            "Previous-day high/low breakout": prev_day_breakout,
            "VWAP 2-sigma reversion": vwap_band(2.0), "VWAP 2.5-sigma reversion": vwap_band(2.5),
            "Momentum at 17:00 (0.5 ATR & VWAP)": time_momentum("17:00", 0.5),
            "Momentum at 19:30 (0.5 ATR & VWAP)": time_momentum("19:30", 0.5),
            "Reversal at 21:00 (1 ATR from prev close)": late_reversal("21:00", 1.0),
            "Gap follow (0.3 ATR)": gap_play(0.3, True), "Gap fade (0.3 ATR)": gap_play(0.3, False),
            "EMA 9/21 cross (5-min)": ema_cross(), "Supertrend(10,3) flip (5-min)": supertrend(),
        }
        if sp.series == "CRUDEOIL":
            d["EIA crude inventory (Wed) follow"] = inventory_report(2)
        if sp.series == "NATURALGAS":
            d["EIA gas storage (Thu) follow"] = inventory_report(3)
        return d
    return {
        "ORB first hour": orb(60, "14:00"), "Previous-day high/low breakout": prev_day_breakout,
        "VWAP(TWAP) 2-sigma reversion": vwap_band(2.0), "Momentum at 12:00 (0.3 ATR)": time_momentum("12:00", 0.3),
        "Reversal at 15:00 (0.7 ATR)": late_reversal("15:00", 0.7), "Gap follow (0.3 ATR)": gap_play(0.3, True),
        "Gap fade (0.3 ATR)": gap_play(0.3, False),
    }


FILTERS = ["none", "trend", "ATR high", "ATR low", "not Monday", "trend+ATR high"]


def filt_mask(name, f, rows, sides):
    m = np.ones(len(rows), bool)
    for p in name.split("+"):
        if p == "none":
            continue
        if p == "trend":
            m &= np.where(sides > 0, f.trend_up[rows], f.trend_down[rows])
        elif p == "ATR high":
            m &= np.nan_to_num(f.atr_rank[rows]) >= 0.6
        elif p == "ATR low":
            m &= np.nan_to_num(f.atr_rank[rows], nan=1) <= 0.4
        elif p == "not Monday":
            m &= f.weekday[rows] != 0
    return m


@dataclass(frozen=True)
class Exit:
    stop: str
    target: float
    breakeven: bool

    @property
    def label(self):
        return f"stop {self.stop}, {f'{self.target:g}R' if self.target else 'no target'}" + (", breakeven at 1R" if self.breakeven else "")


EXITS = [Exit(st, tg, be) for st, tg, be in itertools.product(
    ["struct", "atr0.25", "atr0.5", "atr1.0"], [1.0, 2.0, 3.0, 0], [False, True])]
RISKS = [1000.0, 2000.0]


def charges(value, buy, sp: Spec):
    b = 20.0
    if sp.exchange == "MCX":
        txn, tax, stamp = value * 0.0021 / 100, (0 if buy else value * 0.01 / 100), (value * 0.002 / 100 if buy else 0)
    else:
        txn, tax, stamp = value * 0.00035 / 100, 0.0, (value * 0.0001 / 100 if buy else 0)
    sebi = value * 0.0001 / 100
    return b + txn + tax + stamp + sebi + 0.18 * (b + txn + sebi)


def run_variant(s: Sessions, f: Feat, sp: Spec, cand, filt, ex: Exit, risk_budget):
    rows, slots, sides, entry, sstop, _ = cand
    if len(rows) == 0:
        return None
    keep = np.isfinite(f.atr[rows]) & filt_mask(filt, f, rows, sides)
    rows, slots, sides, entry, sstop = rows[keep], slots[keep], sides[keep], entry[keep], sstop[keep]
    if len(rows) == 0:
        return None
    if ex.stop == "struct":
        stop = sstop
    else:
        stop = entry - sides * float(ex.stop[3:]) * f.atr[rows]
    risk = (entry - stop) * sides
    ok = np.isfinite(risk) & (risk > 0) & (risk / entry < 0.05)
    rows, slots, sides, entry, stop, risk = rows[ok], slots[ok], sides[ok], entry[ok], stop[ok], risk[ok]
    if len(rows) == 0:
        return None
    X = s.slot_of_close(sp.flat_by)
    H, L, O, C = s.h[rows], s.l[rows], s.o[rows], s.c[rows]
    I = np.arange(s.c.shape[1])[None, :]
    after = (I > slots[:, None]) & (I <= X)
    sd = sides[:, None]
    tgt = entry + sides * ex.target * risk if ex.target else np.full(len(rows), np.nan)
    if ex.breakeven:
        one = entry + sides * risk
        reach = after & np.where(sd > 0, H >= one[:, None], L <= one[:, None])
        b1 = np.where(reach.any(1), reach.argmax(1), 999)
        stm = np.where(I > b1[:, None], entry[:, None], stop[:, None])
    else:
        stm = np.repeat(stop[:, None], s.c.shape[1], 1)
    sh = after & np.where(sd > 0, L <= stm, H >= stm)
    th = after & ~np.isnan(tgt)[:, None] & np.where(sd > 0, H >= tgt[:, None], L <= tgt[:, None])
    fs = np.where(sh.any(1), sh.argmax(1), 999)
    ft = np.where(th.any(1), th.argmax(1), 999)
    out = np.where((fs <= ft) & (fs < 999), -1, np.where(ft < 999, 1, 0))
    r = np.arange(len(rows))
    xs = np.where(out == -1, fs, np.where(out == 1, ft, X))
    xs = np.minimum(xs, s.c.shape[1] - 1)
    o_at = O[r, xs]
    stv = stm[r, xs]
    last_c = np.array([C[i, :X + 1][~np.isnan(C[i, :X + 1])][-1] for i in r])
    px = np.where(out == -1, np.where(sides > 0, np.fmin(stv, o_at), np.fmax(stv, o_at)),
                  np.where(out == 1, np.where(sides > 0, np.fmax(tgt, o_at), np.fmin(tgt, o_at)), last_c))
    slip = np.maximum(sp.tick, 1e-4 * entry)
    fin = entry + sides * slip
    fout = np.where(out == 1, px, px - sides * slip)
    per_lot_risk = (risk + slip) * sp.multiplier
    lots = np.floor(risk_budget / per_lot_risk)
    lots = np.where((lots < 1) & (per_lot_risk <= 1.5 * risk_budget), 1, lots)
    max_lots = np.floor(CAPITAL / (entry * sp.multiplier * sp.margin_pct))
    lots = np.minimum(lots, max_lots)
    gross = (fout - fin) * sides * lots * sp.multiplier
    ch = np.array([charges(a * l * sp.multiplier, sd_ > 0, sp) + charges(b * l * sp.multiplier, sd_ < 0, sp)
                   for a, b, l, sd_ in zip(fin, fout, lots, sides)])
    pnl = np.where(lots >= 1, gross - ch, 0.0)
    daily = np.zeros(s.n)
    np.add.at(daily, rows, pnl)
    return daily, int((lots >= 1).sum()), lots


def stats(daily, ntr, dates, meta):
    train = dates <= TRAIN_END
    month = np.array([str(d)[:7] for d in dates])
    out = dict(meta)
    for name, m in (("train", train), ("test", ~train)):
        q = daily[m]
        months = len(set(month[m]))
        eq = np.cumsum(q)
        out[f"{name}_per_month"] = round(float(q.sum()) / max(1, months), 0)
        out[f"{name}_maxdd"] = round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0) if len(q) else 0
        out[f"{name}_days_2k"] = round(100 * float((q >= 2000).mean()), 1) if len(q) else 0
        out[f"{name}_pos_months"] = round(100 * float(np.mean([q[month[m] == mm].sum() > 0 for mm in sorted(set(month[m]))])), 0) if months else 0
    out["trades"] = ntr
    return out


def run_search(log=print):
    rows = []
    t0 = time.time()
    for name, sp in SPECS.items():
        s = load(name)
        f = features(s)
        for sname, fn in setups_for(sp).items():
            cand = fn(s, f, sp)
            for filt, ex, rb in itertools.product(FILTERS, EXITS, RISKS):
                res = run_variant(s, f, sp, cand, filt, ex, rb)
                if res is None:
                    continue
                daily, ntr, lots = res
                rows.append(stats(daily, ntr, s.dates, {"contract": name, "exchange": sp.exchange, "setup": sname,
                                                        "filter": filt, "exit": ex.label, "risk": rb,
                                                        "avg_lots": round(float(lots[lots >= 1].mean()), 2) if (lots >= 1).any() else 0}))
        log(f"  {name}: {len(rows):,} variants so far ({time.time() - t0:.0f}s)")
    return rows
