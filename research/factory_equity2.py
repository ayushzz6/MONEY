"""More intraday equity setups for the strategy factory, and the cross-sectional "stocks in play"
filters (ranked across the NIFTY 100 each morning, using only what is known at 09:30).

Each setup returns (rows, slots, sides, entry, structural stop, stop_order_flag) like the ones
in research.factory_equity, so the same filters and exit grid apply.
"""
from typing import Callable, Dict

import numpy as np

from research.factory_equity import S_IDX, _pick, _swing
from research.sim import first_true, slot_of_close


def _ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for j in range(1, x.shape[1]):
        out[:, j] = a * x[:, j] + (1 - a) * out[:, j - 1]
    return out


def _rsi(c, n=14):
    d = np.diff(c, axis=1, prepend=c[:, :1])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = _ema(up, 2 * n - 1), _ema(dn, 2 * n - 1)      # Wilder smoothing = EMA with span 2n-1
    rs = au / np.maximum(ad, 1e-12)
    out = 100 - 100 / (1 + rs)
    out[:, :n] = np.nan
    return out


def _result(m, rows, slots, sides, stop, entry=None, touch=False):
    rows, slots, sides = np.asarray(rows, int), np.asarray(slots, int), np.asarray(sides, int)
    e = m.c[rows, slots] if entry is None else np.asarray(entry, float)
    return rows, slots, sides, e, np.asarray(stop, float), touch


def _from_side(m, side, slot, stop_fn):
    rows = np.where(side != 0)[0]
    s, sd = slot[rows], side[rows]
    return _result(m, rows, s, sd, stop_fn(rows, s, sd))


def orb_retest(m, f, lv):
    """A close above the 15-minute range high, then a later bar dips back to it (low within 0.1%)
    and closes above it again: buy that close. Mirror for shorts. Stop: the retest bar's extreme."""
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    win = (S_IDX >= 3) & (S_IDX <= slot_of_close("13:00"))
    bu, bd = first_true(win & (m.c > oh[:, None])), first_true(win & (m.c < ol[:, None]))
    after_u = S_IDX[None, :] > np.where(bu >= 0, bu, 999)[:, None]
    after_d = S_IDX[None, :] > np.where(bd >= 0, bd, 999)[:, None]
    ru = first_true(win & after_u & (m.l <= oh[:, None] * 1.001) & (m.c > oh[:, None]))
    rd = first_true(win & after_d & (m.h >= ol[:, None] * 0.999) & (m.c < ol[:, None]))
    side, slot = _pick(ru, rd)
    return _from_side(m, side, slot, lambda r, s, sd: np.where(sd > 0, m.l[r, s], m.h[r, s]))


def drive_pullback(m, f, lv):
    """The first 30 minutes run 1%+ (closing in the top quarter of their range), then the first
    bar after 09:45 that touches VWAP and closes green (red for shorts) is the entry."""
    s0 = slot_of_close("09:45")
    hi, lo = m.h[:, :s0 + 1].max(1), m.l[:, :s0 + 1].min(1)
    mv = m.c[:, s0] / m.o[:, 0] - 1
    pos = (m.c[:, s0] - lo) / np.maximum(hi - lo, 1e-9)
    up_day = (mv >= 0.01) & (pos >= 0.75)
    dn_day = (mv <= -0.01) & (pos <= 0.25)
    win = (S_IDX > s0) & (S_IDX <= slot_of_close("13:30"))
    ku = np.where(up_day, first_true(win & (m.l <= f.vwap) & (m.c > m.o) & (m.c > f.vwap)), -1)
    kd = np.where(dn_day, first_true(win & (m.h >= f.vwap) & (m.c < m.o) & (m.c < f.vwap)), -1)
    side, slot = _pick(ku, kd)
    return _from_side(m, side, slot, lambda r, s, sd: np.where(sd > 0, m.l[r, s], m.h[r, s]))


def inside_bar_breakout(m, f, lv):
    """A 5-minute inside bar after 10:00; the next close beyond the mother bar's high (low)."""
    H, L, C = m.h, m.l, m.c
    inside = np.zeros_like(H, bool)
    inside[:, 1:] = (H[:, 1:] < H[:, :-1]) & (L[:, 1:] > L[:, :-1])
    mh = np.full_like(H, np.nan)
    ml = np.full_like(L, np.nan)
    mh[:, 2:] = np.where(inside[:, 1:-1], H[:, :-2], np.nan)
    ml[:, 2:] = np.where(inside[:, 1:-1], L[:, :-2], np.nan)
    win = (S_IDX >= slot_of_close("10:00")) & (S_IDX <= slot_of_close("14:30"))
    ku, kd = first_true(win & (C > mh)), first_true(win & (C < ml))
    side, slot = _pick(ku, kd)
    return _from_side(m, side, slot, lambda r, s, sd: np.where(sd > 0, ml[r, s], mh[r, s]))


def ema_cross(m, f, lv):
    """9 / 21 EMA cross on 5-minute closes after 09:45."""
    e9, e21 = _ema(m.c, 9), _ema(m.c, 21)
    above = e9 > e21
    cu = np.zeros_like(above)
    cu[:, 1:] = above[:, 1:] & ~above[:, :-1]
    cd = np.zeros_like(above)
    cd[:, 1:] = ~above[:, 1:] & above[:, :-1]
    win = (S_IDX >= slot_of_close("09:45")) & (S_IDX <= slot_of_close("14:30"))
    side, slot = _pick(first_true(win & cu), first_true(win & cd))
    return _from_side(m, side, slot, lambda r, s, sd: _swing(m, r, s, sd, 5))


def rsi_reversal(m, f, lv):
    """5-minute RSI(14) back above 30 after being under it (long), back under 70 (short), after 10:00."""
    r = _rsi(m.c)
    up = np.zeros_like(m.c, bool)
    dn = np.zeros_like(m.c, bool)
    up[:, 1:] = (r[:, 1:] > 30) & (r[:, :-1] <= 30)
    dn[:, 1:] = (r[:, 1:] < 70) & (r[:, :-1] >= 70)
    win = (S_IDX >= slot_of_close("10:00")) & (S_IDX <= slot_of_close("14:30"))
    side, slot = _pick(first_true(win & up), first_true(win & dn))
    return _from_side(m, side, slot, lambda r_, s, sd: _swing(m, r_, s, sd, 4))


def bb_squeeze_breakout(m, f, lv):
    """20-bar Bollinger bands on 5-minute closes; when the band width is the narrowest of the day so
    far, the next close outside the band is followed (from 11:00)."""
    C = m.c
    n = 20
    mean = np.full_like(C, np.nan)
    sd = np.full_like(C, np.nan)
    for j in range(n - 1, C.shape[1]):
        w = C[:, j - n + 1:j + 1]
        mean[:, j], sd[:, j] = w.mean(1), w.std(1)
    width = 4 * sd / mean
    run_min = np.fmin.accumulate(np.where(np.isnan(width), np.inf, width), axis=1)
    squeezed = np.zeros_like(C, bool)
    squeezed[:, 1:] = width[:, :-1] <= run_min[:, :-1] * 1.0001
    win = (S_IDX >= slot_of_close("11:00")) & (S_IDX <= slot_of_close("14:30"))
    up = win & squeezed & (C > mean + 2 * sd)
    dn = win & squeezed & (C < mean - 2 * sd)
    side, slot = _pick(first_true(up), first_true(dn))
    return _from_side(m, side, slot, lambda r, s, sdd: np.where(sdd > 0, mean[r, s], mean[r, s]))


def supertrend_flip(m, f, lv, period=10, mult=3.0):
    """Supertrend (10, 3) on 5-minute bars, reset each session; follow the first flip after 10:05.
    Stop: the supertrend line at the flip."""
    H, L, C = m.h, m.l, m.c
    D, S = C.shape
    tr = np.maximum(H - L, np.abs(H - np.c_[C[:, :1], C[:, :-1]]))
    tr = np.maximum(tr, np.abs(L - np.c_[C[:, :1], C[:, :-1]]))
    atr = np.full_like(C, np.nan)
    atr[:, period - 1] = tr[:, :period].mean(1)
    for j in range(period, S):
        atr[:, j] = (atr[:, j - 1] * (period - 1) + tr[:, j]) / period
    mid = (H + L) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    fub, flb = ub.copy(), lb.copy()
    trend = np.ones((D, S), int)
    line = np.full_like(C, np.nan)
    for j in range(period, S):
        fub[:, j] = np.where((ub[:, j] < fub[:, j - 1]) | (C[:, j - 1] > fub[:, j - 1]), ub[:, j], fub[:, j - 1])
        flb[:, j] = np.where((lb[:, j] > flb[:, j - 1]) | (C[:, j - 1] < flb[:, j - 1]), lb[:, j], flb[:, j - 1])
        prev = trend[:, j - 1]
        trend[:, j] = np.where((prev < 0) & (C[:, j] > fub[:, j - 1]), 1, np.where((prev > 0) & (C[:, j] < flb[:, j - 1]), -1, prev))
        line[:, j] = np.where(trend[:, j] > 0, flb[:, j], fub[:, j])
    flip_up = np.zeros_like(C, bool)
    flip_dn = np.zeros_like(C, bool)
    flip_up[:, 1:] = (trend[:, 1:] > 0) & (trend[:, :-1] < 0)
    flip_dn[:, 1:] = (trend[:, 1:] < 0) & (trend[:, :-1] > 0)
    win = (S_IDX >= slot_of_close("10:05")) & (S_IDX <= slot_of_close("14:30"))
    side, slot = _pick(first_true(win & flip_up), first_true(win & flip_dn))
    return _from_side(m, side, slot, lambda r, s, sd: line[r, s])


def late_trend(m, f, lv):
    """At 14:00 the stock is 1.5%+ from its open, on the right side of VWAP and within 0.5% of the
    day's extreme: ride it into the close. Stop: VWAP."""
    s = slot_of_close("14:00")
    mv = m.c[:, s] / m.o[:, 0] - 1
    hod, lod = m.h[:, :s + 1].max(1), m.l[:, :s + 1].min(1)
    c, vw = m.c[:, s], f.vwap[:, s]
    up = (mv >= 0.015) & (c > vw) & (c >= hod * 0.995)
    dn = (mv <= -0.015) & (c < vw) & (c <= lod * 1.005)
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.where(side != 0)[0]
    return _result(m, rows, np.full(len(rows), s), side[rows], vw[rows])


def afternoon_reversal(m, f, lv):
    """Down 3%+ from yesterday's close at 14:30 but 0.5%+ off the day's low (up 3%+, off the high):
    fade into the close. Stop: the day's extreme."""
    s = slot_of_close("14:30")
    ch = m.c[:, s] / f.prev_close - 1
    lod, hod = m.l[:, :s + 1].min(1), m.h[:, :s + 1].max(1)
    c = m.c[:, s]
    up = (ch <= -0.03) & (c >= lod * 1.005)
    dn = (ch >= 0.03) & (c <= hod * 0.995)
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.where(side != 0)[0]
    stop = np.where(side[rows] > 0, lod[rows] * 0.999, hod[rows] * 1.001)
    return _result(m, rows, np.full(len(rows), s), side[rows], stop)


def new_hod_volume(m, f, lv):
    """After 11:00, the first close above the day's high so far (below the low) on 2x the day's
    average bar volume."""
    H, L, C, V = m.h, m.l, m.c, m.v
    hod = np.fmax.accumulate(H, axis=1)
    lod = np.fmin.accumulate(L, axis=1)
    prev_hod = np.c_[np.full((m.n, 1), np.inf), hod[:, :-1]]
    prev_lod = np.c_[np.full((m.n, 1), -np.inf), lod[:, :-1]]
    avg = np.cumsum(V, axis=1) / (S_IDX[None, :] + 1)
    prev_avg = np.c_[np.full((m.n, 1), np.inf), avg[:, :-1]]
    win = (S_IDX >= slot_of_close("11:00")) & (S_IDX <= slot_of_close("14:30"))
    up = win & (C > prev_hod) & (V >= 2 * prev_avg)
    dn = win & (C < prev_lod) & (V >= 2 * prev_avg)
    side, slot = _pick(first_true(up), first_true(dn))
    return _from_side(m, side, slot, lambda r, s, sd: _swing(m, r, s, sd, 3))


def round_number(m, f, lv):
    """The first close beyond a round number above (below) the open: 100s over Rs 1,000, 50s over
    Rs 200, 10s under. Stop: the last 3 bars' extreme."""
    o0 = m.o[:, 0]
    step = np.where(o0 >= 1000, 100.0, np.where(o0 >= 200, 50.0, 10.0))
    above = np.ceil(o0 / step) * step
    below = np.floor(o0 / step) * step
    above = np.where(above - o0 < 0.002 * o0, above + step, above)     # not a level the open already sits on
    below = np.where(o0 - below < 0.002 * o0, below - step, below)
    win = (S_IDX >= 2) & (S_IDX <= slot_of_close("14:00"))
    side, slot = _pick(first_true(win & (m.c > above[:, None])), first_true(win & (m.c < below[:, None])))
    return _from_side(m, side, slot, lambda r, s, sd: _swing(m, r, s, sd, 3))


def cpr_bounce(m, f, lv):
    """The stock opens above the CPR (below it); a later bar dips to the top of the CPR (rallies to its
    bottom) and closes back away from it. Stop: the other side of the CPR."""
    o0 = m.o[:, 0]
    TC, BC = lv.TC, lv.BC
    win = (S_IDX >= 2) & (S_IDX <= slot_of_close("14:00"))
    up = np.where(o0 > TC, first_true(win & (m.l <= TC[:, None] * 1.001) & (m.c > TC[:, None])), -1)
    dn = np.where(o0 < BC, first_true(win & (m.h >= BC[:, None] * 0.999) & (m.c < BC[:, None])), -1)
    side, slot = _pick(up, dn)
    return _from_side(m, side, slot, lambda r, s, sd: np.where(sd > 0, BC[r] * 0.999, TC[r] * 1.001))


def close_location(m, f, lv):
    """Yesterday closed in the top (bottom) 15% of its range and today opens within 0.5% of that
    close: a stop order at the 15-minute range high (low)."""
    pr = f.prev_high - f.prev_low
    loc = (f.prev_close - f.prev_low) / np.maximum(pr, 1e-9)
    o0 = m.o[:, 0]
    flat_open = np.abs(o0 / f.prev_close - 1) <= 0.005
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    win = (S_IDX >= 3) & (S_IDX <= slot_of_close("12:00"))
    ku = np.where(flat_open & (loc >= 0.85), first_true(win & (m.h > oh[:, None])), -1)
    kd = np.where(flat_open & (loc <= 0.15), first_true(win & (m.l < ol[:, None])), -1)
    side, slot = _pick(ku, kd)
    rows = np.where(side != 0)[0]
    s, sd = slot[rows], side[rows]
    lvl = np.where(sd > 0, oh[rows], ol[rows])
    o = m.o[rows, s]
    e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
    return _result(m, rows, s, sd, np.where(sd > 0, ol[rows], oh[rows]), entry=e, touch=True)


def three_bar_pullback(m, f, lv):
    """After 10:30, above VWAP: three lower closes in a row, then a close above the prior bar's high.
    Mirror below VWAP. Stop: the pullback's low."""
    C, H, L = m.c, m.h, m.l
    dn3 = np.zeros_like(C, bool)
    up3 = np.zeros_like(C, bool)
    dn3[:, 3:] = (C[:, 1:-2] < C[:, :-3]) & (C[:, 2:-1] < C[:, 1:-2]) & (C[:, 3:] < C[:, 2:-1])
    up3[:, 3:] = (C[:, 1:-2] > C[:, :-3]) & (C[:, 2:-1] > C[:, 1:-2]) & (C[:, 3:] > C[:, 2:-1])
    prev_dn3 = np.c_[np.zeros((m.n, 1), bool), dn3[:, :-1]]
    prev_up3 = np.c_[np.zeros((m.n, 1), bool), up3[:, :-1]]
    ph = np.c_[H[:, :1], H[:, :-1]]
    pl = np.c_[L[:, :1], L[:, :-1]]
    win = (S_IDX >= slot_of_close("10:30")) & (S_IDX <= slot_of_close("14:30"))
    up = win & prev_dn3 & (C > ph) & (C > f.vwap)
    dn = win & prev_up3 & (C < pl) & (C < f.vwap)
    side, slot = _pick(first_true(up), first_true(dn))
    return _from_side(m, side, slot, lambda r, s, sd: _swing(m, r, s, sd, 4))


def orb_fakeout(m, f, lv):
    """A trade through the 15-minute range high that closes back inside within 2 bars: short, as the
    breakout failed (mirror for the low). Stop: the failed breakout's extreme."""
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    win = (S_IDX >= 3) & (S_IDX <= slot_of_close("12:30"))
    pu = first_true(win & (m.h > oh[:, None]))
    pd = first_true(win & (m.l < ol[:, None]))
    rows_out, slots, sides, stops = [], [], [], []
    for r in range(m.n):
        best = None
        for k, sd, lvl in ((pu[r], -1, oh[r]), (pd[r], 1, ol[r])):
            if k < 0:
                continue
            for j in range(k, min(k + 3, slot_of_close("12:30") + 1)):
                back = m.c[r, j] < lvl if sd < 0 else m.c[r, j] > lvl
                if back:
                    ext = m.h[r, k:j + 1].max() if sd < 0 else m.l[r, k:j + 1].min()
                    if best is None or j < best[0]:
                        best = (j, sd, ext)
                    break
        if best:
            rows_out.append(r); slots.append(best[0]); sides.append(best[1])
            stops.append(best[2] * (1.001 if best[1] < 0 else 0.999))
    return _result(m, rows_out, slots, sides, stops)


EXTRA_SETUPS: Dict[str, Callable] = {
    "ORB-15 breakout retest": orb_retest,
    "Opening drive, first VWAP pullback": drive_pullback,
    "5-min inside bar breakout (after 10:00)": inside_bar_breakout,
    "9/21 EMA cross (5-min)": ema_cross,
    "RSI(14) 5-min reversal 30/70": rsi_reversal,
    "Bollinger squeeze breakout (from 11:00)": bb_squeeze_breakout,
    "Supertrend(10,3) flip": supertrend_flip,
    "Late-day trend at 14:00": late_trend,
    "Afternoon reversal at 14:30 (3% movers)": afternoon_reversal,
    "New high/low of day on 2x volume (after 11:00)": new_hod_volume,
    "Round-number breakout": round_number,
    "CPR bounce": cpr_bounce,
    "Close-location continuation (stop order)": close_location,
    "3-bar pullback with VWAP": three_bar_pullback,
    "ORB-15 fakeout (failed breakout fade)": orb_fakeout,
}

XS_FILTERS = ["top10 rvol", "top10 rvol+trend", "top10 rvol+nifty", "top10 rvol5", "top10 rvol5+trend", "top10 rvol5+nifty",
              "top20 ATR%", "top20 ATR%+nifty"]


def add_cross_sectional_ranks(uni):
    """Rank every stock each morning on its opening relative volume and on yesterday's ATR as a % of price.
    rvol_rank: the 09:20 + 09:25 bars vs their 20-day average, known at 09:30.
    rvol5_rank: the 09:15 bar vs its 14-day average, known at 09:20 (stocks in play, Zarattini et al.).
    atr_rank: known before the open. filter_mask makes sure no trade uses a rank before it is known."""
    from collections import defaultdict
    by_day_rv, by_day_rv5, by_day_atr = defaultdict(list), defaultdict(list), defaultdict(list)
    for k, (m, f, lv) in enumerate(uni.items):
        atrp = f.atr14 / np.maximum(f.prev_close, 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            rv5 = f.v0 / f.v0_avg
        for r, d in enumerate(m.dates):
            rv = lv.rvol15[r]
            if np.isfinite(rv):
                by_day_rv[d].append((rv, k, r))
            if np.isfinite(rv5[r]):
                by_day_rv5[d].append((rv5[r], k, r))
            if np.isfinite(atrp[r]):
                by_day_atr[d].append((atrp[r], k, r))
    for m, f, lv in uni.items:
        lv.rvol_rank = np.full(m.n, 999)
        lv.rvol5_rank = np.full(m.n, 999)
        lv.atr_rank = np.full(m.n, 999)
    for store, attr in ((by_day_rv, "rvol_rank"), (by_day_rv5, "rvol5_rank"), (by_day_atr, "atr_rank")):
        for d, rows in store.items():
            rows.sort(reverse=True)
            for rank, (_, k, r) in enumerate(rows, 1):
                getattr(uni.items[k][2], attr)[r] = rank
