"""Strategy factory: intraday NIFTY 100 equity variants (MIS), screened over 5 years.

entry setup (38) x filter set (11) x exit rule -> per-trade P&L with fixed risk sizing, the
first 5 trades a day across the universe, charges and slippage. Stage 1 runs every setup x
filter with 3 default exits; stage 2 runs the best stage-1 pairs (ranked on the TRAIN period
only) against the full exit grid. The TEST period is never used to choose anything.
"""
import itertools
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from portfolio.pnl import order_charges
from research import data
from research.sim import FLAT_SLOT, Features, features, first_true, slot_of_close

S_IDX = np.arange(data.SLOTS)
TRAIN_END = np.datetime64("2024-12-31")
RISK = 1000.0
MAX_POS_VALUE = 100000.0          # 5 positions x Rs 1 L = the Rs 5 L MIS allows on a Rs 1 L account
MAX_TRADES_DAY = 5
SLIP = 3 / 10000.0


# ---------------------------------------------------------------- levels known before the open
@dataclass
class Levels:
    pdh: np.ndarray
    pdl: np.ndarray
    pdc: np.ndarray
    P: np.ndarray
    TC: np.ndarray
    BC: np.ndarray
    R1: np.ndarray
    S1: np.ndarray
    H3: np.ndarray
    L3: np.ndarray
    H4: np.ndarray
    L4: np.ndarray
    wkh: np.ndarray
    wkl: np.ndarray
    cpr_w: np.ndarray       # CPR width % of price
    nr7: np.ndarray         # yesterday was the narrowest range of 7 sessions
    rvol15: np.ndarray      # 09:20-09:25 volume vs its 20-day average


def levels(m, f: Features) -> Levels:
    o, h, l, c, v = m.daily()
    H, L, C = f.prev_high, f.prev_low, f.prev_close
    P = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2 * P - BC
    rng = H - L
    wkh = np.full(m.n, np.nan)
    wkl = np.full(m.n, np.nan)
    for i in range(5, m.n):
        wkh[i], wkl[i] = h[i - 5:i].max(), l[i - 5:i].min()
    dr = h - l
    nr7 = np.zeros(m.n, bool)
    for i in range(7, m.n):
        nr7[i] = dr[i - 1] <= dr[i - 7:i].min()
    return Levels(H, L, C, P, np.maximum(TC, BC), np.minimum(TC, BC), 2 * P - L, 2 * P - H,
                  C + 1.1 * rng / 4, C - 1.1 * rng / 4, C + 1.1 * rng / 2, C - 1.1 * rng / 2, wkh, wkl,
                  100 * np.abs(TC - BC) / P, nr7, f.or_vol / f.or_vol_avg)


# ---------------------------------------------------------------- entry setups
# each returns rows, slot, side, entry, structural stop (arrays)
def _pick(up_k, dn_k):
    side = np.where((up_k >= 0) & ((dn_k < 0) | (up_k < dn_k)), 1, np.where(dn_k >= 0, -1, 0))
    return side, np.where(side == 1, up_k, dn_k)


def _swing(m, rows, slots, sides, n=3):
    out = np.empty(len(rows))
    for i, (r, s, sd) in enumerate(zip(rows, slots, sides)):
        a = max(0, s - n + 1)
        out[i] = m.l[r, a:s + 1].min() if sd > 0 else m.h[r, a:s + 1].max()
    return out


def _finish(m, side, slot, stop_fn):
    rows = np.where(side != 0)[0]
    s, sd = slot[rows], side[rows]
    e = m.c[rows, s]
    st = stop_fn(rows, s, sd, e)
    return rows, s, sd, e, st


def range_breakout(n_bars, until, touch=False):
    def fn(m, f, lv):
        oh, ol = m.h[:, :n_bars].max(1), m.l[:, :n_bars].min(1)
        mid = (oh + ol) / 2
        win = (S_IDX >= n_bars) & (S_IDX <= slot_of_close(until))
        if touch:
            up, dn = win & (m.h > oh[:, None]), win & (m.l < ol[:, None])
        else:
            up, dn = win & (m.c > oh[:, None]), win & (m.c < ol[:, None])
        side, slot = _pick(first_true(up), first_true(dn))
        rows = np.where(side != 0)[0]
        s, sd = slot[rows], side[rows]
        if touch:   # a stop order at the level: filled there (or at the open on a gap)
            lvl = np.where(sd > 0, oh[rows], ol[rows])
            o = m.o[rows, s]
            e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
        else:
            e = m.c[rows, s]
        return rows, s, sd, e, mid[rows], touch
    return fn


def level_breakout(name):
    def fn(m, f, lv):
        hi, lo = {"PDH/PDL": (lv.pdh, lv.pdl), "CPR": (lv.TC, lv.BC), "Camarilla H3/L3": (lv.H3, lv.L3),
                  "Camarilla H4/L4": (lv.H4, lv.L4), "Pivot R1/S1": (lv.R1, lv.S1),
                  "20-day high/low": (f.high20, f.low20), "Weekly high/low": (lv.wkh, lv.wkl)}[name]
        o0 = m.o[:, 0]
        inside = (o0 < hi) & (o0 > lo)
        win = (S_IDX >= 2) & (S_IDX <= slot_of_close("14:30"))
        up = np.where(inside, first_true(win & (m.c > hi[:, None])), -1)
        dn = np.where(inside, first_true(win & (m.c < lo[:, None])), -1)
        side, slot = _pick(up, dn)
        rows, s, sd, e, st = _finish(m, side, slot, lambda r, s, sd, e: _swing(m, r, s, sd))
        return rows, s, sd, e, st, False
    return fn


def level_rejection(name):
    """Trades through a resistance (support) level and closes back under (over) it: fade it."""
    def fn(m, f, lv):
        hi, lo = {"PDH/PDL": (lv.pdh, lv.pdl), "Camarilla H3/L3": (lv.H3, lv.L3), "Pivot R1/S1": (lv.R1, lv.S1),
                  "20-day high/low": (f.high20, f.low20)}[name]
        win = (S_IDX >= 2) & (S_IDX <= slot_of_close("14:30"))
        dn = first_true(win & (m.h > hi[:, None]) & (m.c < hi[:, None]) & (m.o < hi[:, None]))
        up = first_true(win & (m.l < lo[:, None]) & (m.c > lo[:, None]) & (m.o > lo[:, None]))
        side, slot = _pick(up, dn)
        rows = np.where(side != 0)[0]
        s, sd = slot[rows], side[rows]
        e = m.c[rows, s]
        st = np.where(sd > 0, m.l[rows, s] * 0.999, m.h[rows, s] * 1.001)
        return rows, s, sd, e, st, False
    return fn


def vwap_cross(n_side):
    """Closes on the far side of VWAP after n_side bars on the other side."""
    def fn(m, f, lv):
        above = m.c > f.vwap
        win = (S_IDX >= max(3, n_side)) & (S_IDX <= slot_of_close("14:30"))
        k = n_side
        below_run = np.ones_like(above)
        above_run = np.ones_like(above)
        for j in range(1, k + 1):
            below_run &= np.roll(~above, j, axis=1)
            above_run &= np.roll(above, j, axis=1)
        up = first_true(win & above & below_run)
        dn = first_true(win & ~above & above_run)
        side, slot = _pick(up, dn)
        rows, s, sd, e, st = _finish(m, side, slot, lambda r, s, sd, e: _swing(m, r, s, sd, 4))
        return rows, s, sd, e, st, False
    return fn


def vwap_band(sig):
    def fn(m, f, lv):
        win = (S_IDX >= slot_of_close("10:00")) & (S_IDX <= slot_of_close("14:30")) & (f.vsd > 0)
        up = first_true(win & (m.c < f.vwap - sig * f.vsd))
        dn = first_true(win & (m.c > f.vwap + sig * f.vsd))
        side, slot = _pick(up, dn)
        rows = np.where(side != 0)[0]
        s, sd = slot[rows], side[rows]
        e = m.c[rows, s]
        vw = f.vwap[rows, s]
        st = e - (vw - e)
        return rows, s, sd, e, st, False
    return fn


def gap_play(lo_pct, hi_pct, follow):
    def fn(m, f, lv):
        o0, c0, h0, l0 = m.o[:, 0], m.c[:, 0], m.h[:, 0], m.l[:, 0]
        gap = 100 * (o0 / f.prev_close - 1)
        big = (np.abs(gap) >= lo_pct) & (np.abs(gap) < hi_pct)
        if follow:   # first bar continues the gap: stop order at its far end
            side = np.where(big & (gap > 0) & (c0 > o0), 1, np.where(big & (gap < 0) & (c0 < o0), -1, 0))
            rows = np.where(side != 0)[0]
            sd = side[rows]
            lvl = np.where(sd > 0, h0[rows], l0[rows])
            hit = (S_IDX[None, :] >= 1) & (S_IDX[None, :] <= slot_of_close("11:00")) & np.where(
                sd[:, None] > 0, m.h[rows] > lvl[:, None], m.l[rows] < lvl[:, None])
            k = first_true(hit)
            ok = k >= 0
            rows, sd, lvl, k = rows[ok], sd[ok], lvl[ok], k[ok]
            o = m.o[rows, k]
            e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
            st = np.where(sd > 0, l0[rows], h0[rows])
            return rows, k, sd, e, st, True
        side = np.where(big & (gap > 0) & (c0 < o0), -1, np.where(big & (gap < 0) & (c0 > o0), 1, 0))
        rows = np.where(side != 0)[0]
        sd = side[rows]
        st = np.where(sd < 0, h0[rows] * 1.001, l0[rows] * 0.999)
        return rows, np.zeros(len(rows), int), sd, c0[rows], st, False
    return fn


def time_momentum(at, pct):
    def fn(m, f, lv):
        s = slot_of_close(at)
        ret = 100 * (m.c[:, s] / m.o[:, 0] - 1)
        vw = f.vwap[:, s]
        side = np.where((ret >= pct) & (m.c[:, s] > vw), 1, np.where((ret <= -pct) & (m.c[:, s] < vw), -1, 0))
        rows = np.where(side != 0)[0]
        sd = side[rows]
        slot = np.full(len(rows), s)
        st = np.where(sd > 0, m.l[rows, :s + 1].min(1), m.h[rows, :s + 1].max(1))
        return rows, slot, sd, m.c[rows, s], st, False
    return fn


def volume_spike(k):
    def fn(m, f, lv):
        v = m.v
        avg = np.cumsum(v, axis=1) / (S_IDX[None, :] + 1)
        prev_avg = np.concatenate([np.full((m.n, 1), np.nan), avg[:, :-1]], axis=1)
        body = m.c - m.o
        rng = np.maximum(m.h - m.l, 1e-9)
        win = (S_IDX >= 3) & (S_IDX <= slot_of_close("14:30"))
        strong = np.abs(body) >= 0.6 * rng
        spike = win & (v >= k * prev_avg) & strong
        up = first_true(spike & (body > 0))
        dn = first_true(spike & (body < 0))
        side, slot = _pick(up, dn)
        rows = np.where(side != 0)[0]
        s, sd = slot[rows], side[rows]
        st = np.where(sd > 0, m.l[rows, s], m.h[rows, s])
        return rows, s, sd, m.c[rows, s], st, False
    return fn


def opening_drive(pct):
    """The first 30 minutes run more than pct% one way and close in the top (bottom) quarter of
    that range: continuation at 09:45."""
    def fn(m, f, lv):
        s = slot_of_close("09:45")
        hi, lo = m.h[:, :s + 1].max(1), m.l[:, :s + 1].min(1)
        mv = 100 * (m.c[:, s] / m.o[:, 0] - 1)
        pos = (m.c[:, s] - lo) / np.maximum(hi - lo, 1e-9)
        side = np.where((mv >= pct) & (pos >= 0.75), 1, np.where((mv <= -pct) & (pos <= 0.25), -1, 0))
        rows = np.where(side != 0)[0]
        sd = side[rows]
        st = np.where(sd > 0, lo[rows], hi[rows])
        return rows, np.full(len(rows), s), sd, m.c[rows, s], st, False
    return fn


SETUPS: Dict[str, Callable] = {}
for nb, lab in ((1, "OR5"), (3, "OR15"), (6, "OR30"), (12, "OR60")):
    for until in ("11:00", "13:00"):
        SETUPS[f"{lab} close-breakout till {until}"] = range_breakout(nb, until)
for nb, lab in ((1, "OR5"), (3, "OR15"), (6, "OR30")):
    SETUPS[f"{lab} stop-order breakout"] = range_breakout(nb, "13:00", touch=True)
for nm in ("PDH/PDL", "CPR", "Camarilla H3/L3", "Camarilla H4/L4", "Pivot R1/S1", "20-day high/low", "Weekly high/low"):
    SETUPS[f"{nm} breakout"] = level_breakout(nm)
for nm in ("PDH/PDL", "Camarilla H3/L3", "Pivot R1/S1", "20-day high/low"):
    SETUPS[f"{nm} rejection (fade)"] = level_rejection(nm)
for n in (3, 6):
    SETUPS[f"VWAP cross after {n} bars"] = vwap_cross(n)
for sg in (1.5, 2.0, 2.5):
    SETUPS[f"VWAP {sg} sigma reversion"] = vwap_band(sg)
for lo_, hi_ in ((0.5, 2.0), (2.0, 5.0)):
    SETUPS[f"Gap {lo_}-{hi_}% follow"] = gap_play(lo_, hi_, True)
for lo_, hi_ in ((1.0, 2.0), (2.0, 5.0)):
    SETUPS[f"Gap {lo_}-{hi_}% fade"] = gap_play(lo_, hi_, False)
for at in ("10:15", "11:00", "13:00"):
    SETUPS[f"Momentum at {at} (>1% & VWAP)"] = time_momentum(at, 1.0)
SETUPS["Momentum at 10:15 (>2% & VWAP)"] = time_momentum("10:15", 2.0)
for k in (3, 5):
    SETUPS[f"Volume spike {k}x"] = volume_spike(k)
SETUPS["Opening drive 1%"] = opening_drive(1.0)
SETUPS["Opening drive 2%"] = opening_drive(2.0)


# ---------------------------------------------------------------- filters (masks over the candidates)
FILTERS = ["none", "trend", "nifty", "rvol", "trend+nifty", "trend+rvol", "nifty+rvol", "trend+nifty+rvol",
           "narrow CPR", "NR7", "VIX high", "VIX low"]


class Ctx:
    def __init__(self, nifty, vix):
        self.N = nifty
        self.pos = {d: i for i, d in enumerate(nifty.dates)}
        vpos = {d: i for i, d in enumerate(vix.dates)}
        self.vix_open = {d: vix.o[vpos[d], 0] for d in nifty.dates if d in vpos}
        tr = [v for d, v in self.vix_open.items() if d <= TRAIN_END]
        self.vix_t = np.percentile(tr, [33.3, 66.7])


def filter_mask(name, m, f, lv, ctx, rows, slots, sides, touch=False):
    """Each filter only uses what is known when the trade is entered. `slots` is the entry bar: a bar-close
    entry knows that bar; a stop order filled inside the bar (touch) knows only the bars before it.
      rvol, top10 rvol: volume of the 09:20 and 09:25 bars, known when the 09:25 bar closes (09:30)
      top10 rvol5: volume of the 09:15 bar, known at 09:20
      nifty: the NIFTY's side of its open at the last closed bar"""
    mask = np.ones(len(rows), bool)
    seen = slots - 1 if touch else slots          # the last bar whose close is known at the entry
    if name == "none":
        return mask
    if name.startswith("gap rank"):
        from research.factory_equity3 import new_filter_mask
        return new_filter_mask(name, m, f, lv, rows, slots, sides)
    parts = name.split("+")
    for p in parts:
        if p == "trend":
            mask &= np.where(sides > 0, f.trend_up[rows], f.trend_down[rows])
        elif p == "nifty":
            mask &= (seen >= 0) & (lv.nifty_dir[rows, np.maximum(seen, 0)] * sides > 0)
        elif p == "rvol":
            mask &= (seen >= 2) & (np.nan_to_num(lv.rvol15[rows]) >= 1.5)
        elif p == "top10 rvol":
            mask &= (seen >= 2) & (lv.rvol_rank[rows] <= 10)
        elif p == "top10 rvol5":
            mask &= (seen >= 0) & (lv.rvol5_rank[rows] <= 10)
        elif p == "top20 ATR%":
            mask &= lv.atr_rank[rows] <= 20
        elif p == "narrow CPR":
            mask &= lv.cpr_w[rows] < 0.15
        elif p == "NR7":
            mask &= lv.nr7[rows]
        elif p in ("VIX high", "VIX low"):
            vx = lv.vix_open[rows]
            mask &= (vx >= ctx.vix_t[1]) if p == "VIX high" else (vx < ctx.vix_t[0])
    return mask


# ---------------------------------------------------------------- exits
@dataclass(frozen=True)
class Exit:
    stop: str        # struct | atr0.1 | atr0.25 | atr0.5 | pct0.5 | pct1.0
    target: float    # R multiple, 0 = none
    time: str        # 13:00 | 15:15
    breakeven: bool  # move the stop to the entry after 1R
    trail: float = 0.0     # trailing stop k x daily ATR behind the best price since the entry (0 = none)
    time_stop: int = 0     # out at the close of bar entry + N if +1R was not reached by then (0 = none)

    @property
    def label(self):
        t = f"{self.target:g}R" if self.target else "no target"
        return (f"stop {self.stop}, {t}, out {self.time}" + (", breakeven at 1R" if self.breakeven else "")
                + (f", trail {self.trail:g} ATR" if self.trail else "")
                + (f", time stop {self.time_stop * 5} min" if self.time_stop else ""))


DEFAULT_EXITS = [Exit("struct", 2.0, "15:15", False), Exit("struct", 0, "15:15", False), Exit("atr0.25", 2.0, "15:15", False)]
EXIT_GRID = [Exit(s, t, x, b) for s, t, x, b in itertools.product(
    ["struct", "atr0.1", "atr0.25", "atr0.5", "pct0.5", "pct1.0"], [1.0, 1.5, 2.0, 3.0, 0], ["13:00", "15:15"], [False, True])]


def resolve(m, f, rows, slots, sides, entry, sstop, touch, ex: Exit, costs):
    """Net P&L per trade with RISK sizing; returns (risk_ok mask, pnl, entry slots)."""
    ok, rows, slots, sides, entry, stop, risk, exit_px, xs, out = resolve_exits(m, f, rows, slots, sides, entry, sstop, ex,
                                                                                touch=touch)
    if len(rows) == 0:
        return ok, np.zeros(0), np.zeros(0, int)
    fill_in = entry * (1 + SLIP * sides)
    fill_out = np.where(out == 1, exit_px, exit_px * (1 - SLIP * sides))
    qty = np.floor(np.minimum(RISK / ((fill_in - stop) * sides).clip(1e-9), MAX_POS_VALUE / fill_in))
    gross = (fill_out - fill_in) * qty * sides
    val_in, val_out = fill_in * qty, fill_out * qty
    ch = _ch(val_in, sides > 0, costs) + _ch(val_out, sides < 0, costs)
    pnl = np.where(qty >= 1, gross - ch, 0.0)
    return ok, pnl, slots


def resolve_exits(m, f, rows, slots, sides, entry, sstop, ex: Exit, min_risk_pct=0.15, max_risk_pct=3.0, atr=None,
                  touch=False, entry_bar="ohlc"):
    """Stop / target / breakeven / time exit, bar by bar. Returns the risk-ok mask and, for the
    kept trades: rows, slots, sides, entry, stop, risk, exit price, exit slot, outcome."""
    atr = f.atr14 if atr is None else atr
    if ex.stop == "struct":
        stop = sstop
    elif ex.stop.startswith("atr"):
        k = float(ex.stop[3:])
        stop = entry - sides * k * atr[rows]
    else:
        k = float(ex.stop[3:]) / 100
        stop = entry * (1 - sides * k)
    risk = (entry - stop) * sides
    rp = 100 * risk / entry
    ok = (risk > 0) & (rp >= min_risk_pct) & (rp <= max_risk_pct) & ~np.isnan(rp)
    rows, slots, sides, entry, stop, risk = rows[ok], slots[ok], sides[ok], entry[ok], stop[ok], risk[ok]
    X = FLAT_SLOT if ex.time == "15:15" else slot_of_close(ex.time)
    n = len(rows)
    if n == 0:
        z = np.zeros(0)
        return ok, rows, slots, sides, entry, stop, risk, z, np.zeros(0, int), np.zeros(0, int)
    H, L, O, C = m.h[rows], m.l[rows], m.o[rows], m.c[rows]
    es = slots[:, None]
    after = (S_IDX[None, :] > es) & (S_IDX[None, :] <= X)
    sd = sides[:, None]
    tgt = entry + sides * ex.target * risk if ex.target else np.full(n, np.nan)
    st = stop.copy()
    if ex.breakeven:
        one_r = entry + sides * risk
        reach = after & np.where(sd > 0, H >= one_r[:, None], L <= one_r[:, None])
        b1 = np.where(reach.any(1), reach.argmax(1), 999)
        st_mat = np.where(S_IDX[None, :] > b1[:, None], entry[:, None], st[:, None])
    else:
        st_mat = np.repeat(st[:, None], data.SLOTS, axis=1)
    if ex.trail:
        # the best price of the bars BEFORE each bar, from the entry bar on, minus k ATR (never loosens the stop)
        live = S_IDX[None, :] >= es
        best = np.where(sd > 0, np.where(live, H, -np.inf), np.where(live, -L, -np.inf))
        run = np.maximum.accumulate(best, axis=1)
        prev = np.c_[np.full((n, 1), -np.inf), run[:, :-1]]
        k = ex.trail * atr[rows][:, None]
        trail = np.where(sd > 0, prev - k, -prev + k)
        st_mat = np.where(sd > 0, np.fmax(st_mat, np.where(np.isfinite(prev), trail, -np.inf)),
                          np.fmin(st_mat, np.where(np.isfinite(prev), trail, np.inf)))
    if ex.time_stop:
        one_r = entry + sides * risk
        reach = after & np.where(sd > 0, H >= one_r[:, None], L <= one_r[:, None])
        b1 = np.where(reach.any(1), reach.argmax(1), 999)
        cut = slots + ex.time_stop
        X_trade = np.where(b1 > cut, np.minimum(cut, X), X)
        after = (S_IDX[None, :] > es) & (S_IDX[None, :] <= X_trade[:, None])
    else:
        X_trade = np.full(n, X)
    stop_hit = after & np.where(sd > 0, L <= st_mat, H >= st_mat)
    if touch:
        # a stop order filled inside the entry bar: the usual bar path (a green bar goes low then high, a red bar
        # high then low) decides whether the stop was reached after the fill; the bar's close beyond the stop always counts
        eb = S_IDX[None, :] == es
        if entry_bar == "worst":      # any touch of the stop in the entry bar counts as stopped
            hit_eb = np.where(sd > 0, L <= st[:, None], H >= st[:, None])
        else:
            hit_eb = np.where(sd > 0, (C <= st[:, None]) | ((O > C) & (L <= st[:, None])),
                              (C >= st[:, None]) | ((O < C) & (H >= st[:, None])))
        stop_hit = stop_hit | (eb & hit_eb)
    tgt_hit = after & ~np.isnan(tgt)[:, None] & np.where(sd > 0, H >= tgt[:, None], L <= tgt[:, None])
    fs = np.where(stop_hit.any(1), stop_hit.argmax(1), 999)
    ft = np.where(tgt_hit.any(1), tgt_hit.argmax(1), 999)
    out = np.where((fs <= ft) & (fs < 999), -1, np.where(ft < 999, 1, 0))
    xs = np.where(out == -1, fs, np.where(out == 1, ft, X_trade))
    r = np.arange(n)
    o_at = O[r, np.minimum(xs, data.SLOTS - 1)]
    stv = st_mat[r, np.minimum(xs, data.SLOTS - 1)]
    stop_px = np.where(sides > 0, np.minimum(stv, o_at), np.maximum(stv, o_at))
    stop_px = np.where(xs == slots, stv, stop_px)                 # inside the entry bar: at the stop
    tgt_px = np.where(sides > 0, np.maximum(tgt, o_at), np.minimum(tgt, o_at))
    exit_px = np.where(out == -1, stop_px, np.where(out == 1, tgt_px, C[r, X_trade]))
    return ok, rows, slots, sides, entry, stop, risk, exit_px, xs, out


def _ch(value, is_buy, costs):
    b = np.minimum(costs.get("brokerage_per_order_inr", 20), value * costs.get("brokerage_pct", 0.03) / 100)
    txn = value * costs.get("exchange_txn_pct", 0.00297) / 100
    sebi = value * costs.get("sebi_pct", 0.0001) / 100
    stt = np.where(is_buy, 0, value * costs.get("stt_sell_pct", 0.025) / 100)
    stamp = np.where(is_buy, value * costs.get("stamp_buy_pct", 0.003) / 100, 0)
    return b + txn + sebi + stt + stamp + 0.18 * (b + txn + sebi)


# ---------------------------------------------------------------- the search
class Universe:
    def __init__(self, symbols, log=print, ctx=None):
        self.items = []
        for s in symbols:
            m = data.load(s)
            if m is None or m.n < 80:
                continue
            f = features(m)
            lv = levels(m, f)
            if ctx is not None:
                # NIFTY's side of its open at every bar, and the India VIX at the open, aligned to this stock
                nd = np.zeros((m.n, data.SLOTS))
                vx = np.full(m.n, np.nan)
                for r, d in enumerate(m.dates):
                    j = ctx.pos.get(d)
                    if j is not None:
                        nd[r] = np.sign(ctx.N.c[j] - ctx.N.o[j, 0])
                    vx[r] = ctx.vix_open.get(d, np.nan)
                lv.nifty_dir, lv.vix_open = nd, vx
            self.items.append((m, f, lv))
        log(f"  loaded {len(self.items)} stocks")
        self.cache = {}

    def setup(self, name):
        if name not in self.cache:
            fn = SETUPS[name]
            self.cache[name] = [fn(m, f, lv) for m, f, lv in self.items]
        return self.cache[name]


def evaluate(uni: Universe, ctx: Ctx, setup: str, filt: str, ex: Exit, master_dates, costs):
    day_idx_all, slot_all, pnl_all = [], [], []
    pos = {d: i for i, d in enumerate(master_dates)}
    for (m, f, lv), cand in zip(uni.items, uni.setup(setup)):
        rows, slots, sides, entry, sstop, touch = cand
        if len(rows) == 0:
            continue
        valid = f.valid[rows]
        rows, slots, sides, entry, sstop = rows[valid], slots[valid], sides[valid], entry[valid], sstop[valid]
        fm = filter_mask(filt, m, f, lv, ctx, rows, slots, sides, touch=touch)
        rows, slots, sides, entry, sstop = rows[fm], slots[fm], sides[fm], entry[fm], sstop[fm]
        if len(rows) == 0:
            continue
        ok, pnl, sl = resolve(m, f, rows, slots, sides, entry, sstop, touch, ex, costs)
        rr = rows[ok]
        di = np.array([pos.get(m.dates[r], -1) for r in rr], dtype=int)
        day_idx_all.append(di)
        slot_all.append(sl)
        pnl_all.append(pnl)
    if not pnl_all:
        return None
    di, sl, pnl = np.concatenate(day_idx_all), np.concatenate(slot_all), np.concatenate(pnl_all)
    keep = di >= 0
    di, sl, pnl = di[keep], sl[keep], pnl[keep]
    order = np.lexsort((sl, di))
    di, sl, pnl = di[order], sl[order], pnl[order]
    # first MAX_TRADES_DAY trades of each day, by entry time
    starts = np.r_[0, np.flatnonzero(np.diff(di)) + 1]
    rank = np.arange(len(di)) - np.repeat(starts, np.diff(np.r_[starts, len(di)]))
    take = rank < MAX_TRADES_DAY
    di, pnl = di[take], pnl[take]
    daily = np.bincount(di, weights=pnl, minlength=len(master_dates))
    trades = np.bincount(di, minlength=len(master_dates))
    return daily, trades, pnl


def stats(daily, trades, pnl_trades, dates, meta):
    train = dates <= TRAIN_END
    month = np.array([str(d)[:7] for d in dates])
    out = dict(meta)
    for name, m in (("train", train), ("test", ~train)):
        q = daily[m]
        months = len(set(month[m]))
        eq = np.cumsum(q)
        dd = float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(q) else 0.0
        out[f"{name}_net"] = round(float(q.sum()), 0)
        out[f"{name}_per_month"] = round(float(q.sum()) / months, 0)
        out[f"{name}_trades"] = int(trades[m].sum())
        out[f"{name}_maxdd"] = round(dd, 0)
        out[f"{name}_pos_months"] = round(100 * float(np.mean([q[month[m] == mm].sum() > 0 for mm in sorted(set(month[m]))])), 0)
    return out


def run_search(symbols, nifty, vix, costs, stage2_top=30, log=print):
    t0 = time.time()
    ctx = Ctx(nifty, vix)
    uni = Universe(symbols, log, ctx)
    dates = nifty.dates
    stage1 = []
    for setup in SETUPS:
        for filt in FILTERS:
            for ex in DEFAULT_EXITS:
                res = evaluate(uni, ctx, setup, filt, ex, dates, costs)
                if res is None:
                    continue
                daily, trades, pt = res
                stage1.append(stats(daily, trades, pt, dates, {"family": "equity", "setup": setup, "filter": filt,
                                                              "exit": ex.label, **_exit_fields(ex)}))
        log(f"  stage 1: {setup} ({len(stage1):,} variants, {time.time() - t0:.0f}s)")
    pairs = {}
    for r in sorted(stage1, key=lambda r: -r["train_per_month"]):
        if r["train_trades"] >= 200 and (r["setup"], r["filter"]) not in pairs:
            pairs[(r["setup"], r["filter"])] = r
        if len(pairs) >= stage2_top:
            break
    stage2 = []
    for (setup, filt) in pairs:
        for ex in EXIT_GRID:
            res = evaluate(uni, ctx, setup, filt, ex, dates, costs)
            if res is None:
                continue
            daily, trades, pt = res
            stage2.append(stats(daily, trades, pt, dates, {"family": "equity", "setup": setup, "filter": filt,
                                                          "exit": ex.label, **_exit_fields(ex)}))
        log(f"  stage 2: {setup} | {filt} ({len(stage2):,} variants, {time.time() - t0:.0f}s)")
    return stage1, stage2


def _exit_fields(ex: Exit):
    return {"exit_stop": ex.stop, "exit_target": ex.target, "exit_time": ex.time, "exit_breakeven": ex.breakeven,
            "exit_trail": ex.trail, "exit_time_stop": ex.time_stop}
