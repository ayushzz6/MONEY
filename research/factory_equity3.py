"""Round 3 of the equity factory: new intraday concepts built on cross-stock information (sectors, gap
ranks, beta, volume against its usual level at that time of day), plus a market-neutral sector-pair engine.

Setups return (rows, slots, sides, entry, structural stop, stop_order_flag) like research.factory_equity,
so the same filters, exits, account simulator and train/test split apply.
"""
from collections import defaultdict
from typing import Callable, Dict

import numpy as np

from research import data
from research.factory_equity import S_IDX, _pick, _swing
from research.sim import first_true, slot_of_close


# ---------------------------------------------------------------- cross-stock features
def add_cross_features(uni, sectors: Dict[str, str], nifty):
    """Adds to each stock's levels: gap ranks, sector move matrix, beta, NIFTY return matrix, volume vs usual."""
    npos = {d: i for i, d in enumerate(nifty.dates)}
    n_ret = nifty.c / nifty.o[:, :1] - 1.0
    n_daily = np.r_[np.nan, nifty.c[1:, -1] / nifty.c[:-1, -1] - 1]
    n_gap = np.r_[np.nan, nifty.o[1:, 0] / nifty.c[:-1, -1] - 1]
    # per-day collections
    gaps = defaultdict(list)
    sector_sum = defaultdict(lambda: np.zeros(data.SLOTS))
    sector_n = defaultdict(int)
    for k, (m, f, lv) in enumerate(uni.items):
        sec = sectors.get(m.symbol, "")
        ret = m.c / m.o[:, :1] - 1.0
        lv.ret = ret
        lv.gap = m.o[:, 0] / f.prev_close - 1.0
        lv.sector = sec
        for r, d in enumerate(m.dates):
            if np.isfinite(lv.gap[r]):
                gaps[d].append((lv.gap[r], k, r))
            if sec:
                sector_sum[(sec, d)] += np.nan_to_num(ret[r])
                sector_n[(sec, d)] += 1
        # NIFTY aligned
        j = np.array([npos.get(d, -1) for d in m.dates])
        lv.n_ret = np.where(j[:, None] >= 0, n_ret[np.maximum(j, 0)], np.nan)
        lv.n_gap = np.where(j >= 0, n_gap[np.maximum(j, 0)], np.nan)
        # beta of daily returns to the NIFTY over the 60 sessions before each day
        o, h, l, c, v = m.daily()
        sr = np.r_[np.nan, c[1:] / c[:-1] - 1]
        nr = np.where(j >= 0, n_daily[np.maximum(j, 0)], np.nan)
        beta = np.full(m.n, np.nan)
        for i in range(61, m.n):
            a, b = sr[i - 60:i], nr[i - 60:i]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 40 and np.var(b[ok]) > 0:
                beta[i] = np.cov(a[ok], b[ok])[0, 1] / np.var(b[ok])
        lv.beta = beta
        # cumulative volume by bar vs its 20-session average at the same bar
        cv = np.cumsum(m.v, axis=1)
        base = np.full_like(cv, np.nan)
        for i in range(20, m.n):
            base[i] = cv[i - 20:i].mean(0)
        lv.rvol_bar = cv / np.where(base > 0, base, np.nan)
        lv.day_range_prev = f.prev_high - f.prev_low
    for d, rows in gaps.items():
        rows.sort()
        for rank, (_, k, r) in enumerate(rows, 1):            # 1 = biggest gap down
            uni.items[k][2].__dict__.setdefault("gap_rank_dn", np.full(uni.items[k][0].n, 999))[r] = rank
        for rank, (_, k, r) in enumerate(reversed(rows), 1):  # 1 = biggest gap up
            uni.items[k][2].__dict__.setdefault("gap_rank_up", np.full(uni.items[k][0].n, 999))[r] = rank
    for m, f, lv in uni.items:
        lv.__dict__.setdefault("gap_rank_dn", np.full(m.n, 999))
        lv.__dict__.setdefault("gap_rank_up", np.full(m.n, 999))
        sm = np.full((m.n, data.SLOTS), np.nan)
        cnt = np.zeros(m.n)
        for r, d in enumerate(m.dates):
            n_ = sector_n.get((lv.sector, d), 0)
            if n_ >= 3:                                        # the sector average without the stock itself
                sm[r] = (sector_sum[(lv.sector, d)] - np.nan_to_num(lv.ret[r])) / (n_ - 1)
                cnt[r] = n_
        lv.sector_ret = sm
        lv.sector_n = cnt


# ---------------------------------------------------------------- setups
def _res(m, rows, slots, sides, stop, entry=None, touch=False):
    rows, slots, sides = np.asarray(rows, int), np.asarray(slots, int), np.asarray(sides, int)
    e = m.c[rows, slots] if entry is None else np.asarray(entry, float)
    return rows, slots, sides, e, np.asarray(stop, float), touch


def gap_reversal(m, f, lv):
    """Overnight-gap reversal at 09:20: short gap-ups, buy gap-downs (0.5%+), stop 1 daily ATR, out 15:15.
    Meant with the 'gap rank' filters: only the day's most extreme gappers on each side."""
    g = lv.gap
    side = np.where(g >= 0.005, -1, np.where(g <= -0.005, 1, 0))
    rows = np.flatnonzero((side != 0) & np.isfinite(f.atr14))
    sd = side[rows]
    e = m.c[rows, 0]
    return _res(m, rows, np.zeros(len(rows)), sd, e - sd * f.atr14[rows])


def sector_catchup(m, f, lv, at="10:00", leader=False):
    s = slot_of_close(at)
    sec = lv.sector_ret[:, s]
    own = lv.ret[:, s]
    strong = np.abs(sec) >= 0.008
    if leader:       # the stock is ahead of its sector, which is moving: momentum
        cond = strong & (np.sign(own) == np.sign(sec)) & (np.abs(own) > np.abs(sec) * 1.5)
    else:            # the stock lags its moving sector: catch-up
        cond = strong & (np.abs(own) < np.abs(sec) * 0.4) & (np.sign(own) != -np.sign(sec))
    side = np.where(cond, np.sign(sec), 0).astype(int)
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    stop = np.where(sd > 0, m.l[rows, :s + 1].min(1), m.h[rows, :s + 1].max(1))
    return _res(m, rows, np.full(len(rows), s), sd, stop)


def idio_gap_fade(m, f, lv):
    g, ng = lv.gap, lv.n_gap
    c0, o0, h0, l0 = m.c[:, 0], m.o[:, 0], m.h[:, 0], m.l[:, 0]
    quiet = np.abs(np.nan_to_num(ng, nan=1)) < 0.003
    up = quiet & (g >= 0.015) & (g <= 0.06) & (c0 < o0)
    dn = quiet & (g <= -0.015) & (g >= -0.06) & (c0 > o0)
    side = np.where(up, -1, np.where(dn, 1, 0))
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    return _res(m, rows, np.zeros(len(rows)), sd, np.where(sd < 0, h0[rows] * 1.001, l0[rows] * 0.999))


def aligned_gap_follow(m, f, lv):
    g, ng = lv.gap, lv.n_gap
    c0, o0, h0, l0 = m.c[:, 0], m.o[:, 0], m.h[:, 0], m.l[:, 0]
    up = (g >= 0.01) & (ng >= 0.004) & (c0 > o0)
    dn = (g <= -0.01) & (ng <= -0.004) & (c0 < o0)
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    lvl = np.where(sd > 0, h0[rows], l0[rows])
    win = (S_IDX[None, :] >= 1) & (S_IDX[None, :] <= slot_of_close("11:00"))
    hit = win & np.where(sd[:, None] > 0, m.h[rows] > lvl[:, None], m.l[rows] < lvl[:, None])
    k = first_true(hit)
    ok = k >= 0
    rows, sd, lvl, k = rows[ok], sd[ok], lvl[ok], k[ok]
    o = m.o[rows, k]
    e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
    return _res(m, rows, k, sd, np.where(sd > 0, l0[rows], h0[rows]), entry=e, touch=True)


def failed_gap(m, f, lv):
    """A 1%+ gap that closes back through yesterday's close (09:25 to 11:00): trade the failure."""
    g, pc = lv.gap, f.prev_close
    win = (S_IDX >= 1) & (S_IDX <= slot_of_close("11:00"))
    dn = np.where(g >= 0.01, first_true(win & (m.c < pc[:, None])), -1)      # gap up failed: short
    up = np.where(g <= -0.01, first_true(win & (m.c > pc[:, None])), -1)     # gap down failed: long
    side, slot = _pick(up, dn)
    rows = np.flatnonzero(side != 0)
    s, sd = slot[rows], side[rows]
    stop = np.array([m.h[r, :k + 1].max() if d < 0 else m.l[r, :k + 1].min() for r, k, d in zip(rows, s, sd)])
    return _res(m, rows, s, sd, stop)


def lunch_breakout(m, f, lv):
    a, b = slot_of_close("12:05"), slot_of_close("13:30")
    rh, rl = m.h[:, a:b + 1].max(1), m.l[:, a:b + 1].min(1)
    avg = m.v[:, :b + 1].mean(1)
    win = (S_IDX > b) & (S_IDX <= slot_of_close("14:45")) & (m.v >= 1.5 * avg[:, None])
    side, slot = _pick(first_true(win & (m.c > rh[:, None])), first_true(win & (m.c < rl[:, None])))
    rows = np.flatnonzero(side != 0)
    return _res(m, rows, slot[rows], side[rows], ((rh + rl) / 2)[rows])


def volatility_breakout(m, f, lv, k=0.5):
    """Larry Williams: stop orders at the open +/- k x yesterday's range, from 09:20 to 13:00. Stop: the open."""
    o0 = m.o[:, 0]
    rng = lv.day_range_prev
    up_l, dn_l = o0 + k * rng, o0 - k * rng
    win = (S_IDX >= 1) & (S_IDX <= slot_of_close("13:00"))
    side, slot = _pick(first_true(win & (m.h > up_l[:, None])), first_true(win & (m.l < dn_l[:, None])))
    rows = np.flatnonzero((side != 0) & np.isfinite(rng))
    s, sd = slot[rows], side[rows]
    lvl = np.where(sd > 0, up_l[rows], dn_l[rows])
    o = m.o[rows, s]
    e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
    return _res(m, rows, s, sd, o0[rows], entry=e, touch=True)


def open_overreaction(m, f, lv):
    """The first 5-minute bar moves 1%+ from yesterday's close on ordinary volume (no news): fade at 09:20."""
    c0, h0, l0 = m.c[:, 0], m.h[:, 0], m.l[:, 0]
    mv = c0 / f.prev_close - 1
    calm = np.nan_to_num(f.v0 / f.v0_avg, nan=9) < 1.0
    side = np.where(calm & (mv >= 0.01), -1, np.where(calm & (mv <= -0.01), 1, 0))
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    return _res(m, rows, np.zeros(len(rows)), sd, np.where(sd < 0, h0[rows] * 1.001, l0[rows] * 0.999))


def quiet_accumulation(m, f, lv):
    """By 11:00 the range is under 0.6 daily ATR but volume runs 1.3x its usual: stop orders at that
    morning range (11:00 to 14:00). Stop: the middle of the range."""
    s = slot_of_close("11:00")
    hi, lo = m.h[:, :s + 1].max(1), m.l[:, :s + 1].min(1)
    tight = (hi - lo) < 0.6 * f.atr14
    heavy = np.nan_to_num(lv.rvol_bar[:, s]) >= 1.3
    ok = tight & heavy
    win = (S_IDX > s) & (S_IDX <= slot_of_close("14:00"))
    side, slot = _pick(np.where(ok, first_true(win & (m.h > hi[:, None])), -1),
                       np.where(ok, first_true(win & (m.l < lo[:, None])), -1))
    rows = np.flatnonzero(side != 0)
    k, sd = slot[rows], side[rows]
    lvl = np.where(sd > 0, hi[rows], lo[rows])
    o = m.o[rows, k]
    e = np.where(sd > 0, np.maximum(lvl, o), np.minimum(lvl, o))
    return _res(m, rows, k, sd, ((hi + lo) / 2)[rows], entry=e, touch=True)


def breakout_day_pullback(m, f, lv):
    """Yesterday closed at a 20-day closing high; today the first green bar after 09:45 that dips to VWAP
    and closes back above it is bought. Stop: that bar's low."""
    o, h, l, c, v = m.daily()
    # yesterday's close was at or above the highest close of the 20 sessions before it
    brk = np.r_[False, [c[i - 1] >= np.nanmax(c[max(0, i - 21):i - 1]) if i > 21 else False for i in range(1, m.n)]]
    win = (S_IDX >= slot_of_close("09:45")) & (S_IDX <= slot_of_close("13:30"))
    k = np.where(brk, first_true(win & (m.l <= f.vwap) & (m.c > f.vwap) & (m.c > m.o)), -1)
    rows = np.flatnonzero(k >= 0)
    return _res(m, rows, k[rows], np.ones(len(rows)), m.l[rows, k[rows]])


def closing_momentum(m, f, lv):
    s = slot_of_close("14:45")
    mv = m.c[:, s] / m.o[:, 0] - 1
    hod, lod = m.h[:, :s + 1].max(1), m.l[:, :s + 1].min(1)
    pos = (m.c[:, s] - lod) / np.maximum(hod - lod, 1e-9)
    heavy = np.nan_to_num(lv.rvol_bar[:, s]) >= 1.2
    up = heavy & (mv >= 0.015) & (m.c[:, s] > f.vwap[:, s]) & (pos >= 0.9)
    dn = heavy & (mv <= -0.015) & (m.c[:, s] < f.vwap[:, s]) & (pos <= 0.1)
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.flatnonzero(side != 0)
    return _res(m, rows, np.full(len(rows), s), side[rows], f.vwap[rows, s])


def open_rs_in_play(m, f, lv):
    """At 09:30, a stock among the day's 10 most active (opening relative volume) that is 0.5% ahead of the
    NIFTY and above VWAP is bought (behind and below VWAP: shorted). Stop: the day's extreme so far."""
    s = slot_of_close("09:30")
    rs = lv.ret[:, s] - lv.n_ret[:, s]
    inplay = lv.rvol_rank <= 10
    up = inplay & (rs >= 0.005) & (m.c[:, s] > f.vwap[:, s])
    dn = inplay & (rs <= -0.005) & (m.c[:, s] < f.vwap[:, s])
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    stop = np.where(sd > 0, m.l[rows, :s + 1].min(1), m.h[rows, :s + 1].max(1))
    return _res(m, rows, np.full(len(rows), s), sd, stop)


def trend_day_beta(m, f, lv):
    """At 11:00 the NIFTY is 0.5%+ from its open: buy (sell) stocks with beta >= 1.2 that are on the same side
    of their VWAP. Stop: 0.5 daily ATR."""
    s = slot_of_close("11:00")
    nr = lv.n_ret[:, s]
    hb = np.nan_to_num(lv.beta) >= 1.2
    up = hb & (nr >= 0.005) & (m.c[:, s] > f.vwap[:, s])
    dn = hb & (nr <= -0.005) & (m.c[:, s] < f.vwap[:, s])
    side = np.where(up, 1, np.where(dn, -1, 0))
    rows = np.flatnonzero((side != 0) & np.isfinite(f.atr14))
    sd = side[rows]
    e = m.c[rows, s]
    return _res(m, rows, np.full(len(rows), s), sd, e - sd * 0.5 * f.atr14[rows])


def coiled_orb(m, f, lv):
    """The 15-minute range sits inside yesterday's range and yesterday was the narrowest of 4 sessions:
    stop orders at the range. Stop: its far side."""
    o, h, l, c, v = m.daily()
    dr = h - l
    nr4 = np.r_[[False] * 4, [dr[i - 1] <= dr[i - 4:i].min() for i in range(4, m.n)]]
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    inside = (oh < f.prev_high) & (ol > f.prev_low)
    ok = nr4 & inside
    win = (S_IDX >= 3) & (S_IDX <= slot_of_close("13:00"))
    side, slot = _pick(np.where(ok, first_true(win & (m.h > oh[:, None])), -1),
                       np.where(ok, first_true(win & (m.l < ol[:, None])), -1))
    rows = np.flatnonzero(side != 0)
    k, sd = slot[rows], side[rows]
    lvl = np.where(sd > 0, oh[rows], ol[rows])
    oo = m.o[rows, k]
    e = np.where(sd > 0, np.maximum(lvl, oo), np.minimum(lvl, oo))
    return _res(m, rows, k, sd, np.where(sd > 0, ol[rows], oh[rows]), entry=e, touch=True)


def or_double_test(m, f, lv):
    """After 10:00 the price comes back to the 15-minute range low (within 0.1%) on below-average volume and
    closes above it: buy (mirror at the high). Stop: just beyond the range."""
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    avg = np.cumsum(m.v, 1) / (S_IDX[None, :] + 1)
    lowvol = m.v < np.c_[np.full((m.n, 1), np.inf), avg[:, :-1]]
    win = (S_IDX >= slot_of_close("10:00")) & (S_IDX <= slot_of_close("14:00"))
    up = first_true(win & lowvol & (m.l <= ol[:, None] * 1.001) & (m.c > ol[:, None]))
    dn = first_true(win & lowvol & (m.h >= oh[:, None] * 0.999) & (m.c < oh[:, None]))
    side, slot = _pick(up, dn)
    rows = np.flatnonzero(side != 0)
    sd = side[rows]
    return _res(m, rows, slot[rows], sd, np.where(sd > 0, ol[rows] * 0.998, oh[rows] * 1.002))


def vwap_persistence(m, f, lv):
    """Every 5-minute close from 09:30 to 11:00 on one side of VWAP: join that side at 11:00. Stop: VWAP."""
    a, s = slot_of_close("09:30"), slot_of_close("11:00")
    above = (m.c[:, a:s + 1] > f.vwap[:, a:s + 1]).all(1)
    below = (m.c[:, a:s + 1] < f.vwap[:, a:s + 1]).all(1)
    side = np.where(above, 1, np.where(below, -1, 0))
    rows = np.flatnonzero(side != 0)
    return _res(m, rows, np.full(len(rows), s), side[rows], f.vwap[rows, s])


NEW_SETUPS: Dict[str, Callable] = {
    "Overnight-gap reversal (09:20)": gap_reversal,
    "Sector catch-up at 10:00": sector_catchup,
    "Sector leader at 10:00": lambda m, f, lv: sector_catchup(m, f, lv, leader=True),
    "Idiosyncratic gap fade (NIFTY flat)": idio_gap_fade,
    "Market-aligned gap follow": aligned_gap_follow,
    "Failed gap (back through prev close)": failed_gap,
    "Lunch-range breakout on volume": lunch_breakout,
    "Volatility breakout (open +/- 0.5 prev range)": volatility_breakout,
    "Opening overreaction fade (normal volume)": open_overreaction,
    "Quiet accumulation breakout (11:00)": quiet_accumulation,
    "Breakout-day first VWAP pullback": breakout_day_pullback,
    "Closing momentum at 14:45": closing_momentum,
    "Opening relative strength, stocks in play": open_rs_in_play,
    "Trend day: high-beta stocks at 11:00": trend_day_beta,
    "Coiled ORB (inside range after NR4)": coiled_orb,
    "Opening-range double test": or_double_test,
    "VWAP persistence 09:30-11:00": vwap_persistence,
}

NEW_FILTERS = ["gap rank 3", "gap rank 5", "gap rank 5+nifty-flat"]


def new_filter_mask(name, m, f, lv, rows, slots, sides):
    """Filters that need the round-3 features. Returns None for names it does not know."""
    if not name.startswith("gap rank"):
        return None
    parts = name.split("+")
    k = int(parts[0].split()[-1])
    mask = np.where(sides > 0, lv.gap_rank_dn[rows] <= k, lv.gap_rank_up[rows] <= k)
    if "nifty-flat" in parts:
        mask &= np.abs(np.nan_to_num(lv.n_gap[rows], nan=1)) < 0.004
    return mask
