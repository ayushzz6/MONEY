"""Nine intraday equity strategies (NSE cash, MIS, NIFTY 100). Rules are fixed in advance
(textbook parameters, not optimized on this data). Each returns candidate trades with the
bracket already resolved bar by bar; research.sim.simulate_equity applies the account."""
import numpy as np

from research.data import SLOTS, DayMatrix
from research.sim import Features, candidates, first_true, slot_of_close, stop_entries

S_IDX = np.arange(SLOTS)


class Market:
    """NIFTY 50 direction at a bar: +1 above its open, -1 below, 0 unknown."""

    def __init__(self, nifty: DayMatrix):
        self.n = nifty
        self.pos = {d: i for i, d in enumerate(nifty.dates)}

    def direction(self, m: DayMatrix, rows, slots):
        out = np.zeros(len(rows), int)
        for i, (r, s) in enumerate(zip(rows, slots)):
            j = self.pos.get(m.dates[r])
            if j is None:
                continue
            c, o = self.n.c[j, s], self.n.o[j, 0]
            out[i] = 1 if c > o else (-1 if c < o else 0)
        return out

    def ret(self, day, slot):
        j = self.pos.get(day)
        if j is None:
            return np.nan
        return self.n.c[j, slot] / self.n.o[j, 0] - 1.0


def _with_market(m, mk, rows, slots, sides):
    d = mk.direction(m, rows, slots)
    return (d * sides) > 0


def _pick_side(up_k, dn_k):
    side = np.where((up_k >= 0) & ((dn_k < 0) | (up_k < dn_k)), 1, np.where(dn_k >= 0, -1, 0))
    slot = np.where(side == 1, up_k, dn_k)
    return side, slot


def _swing_stop(m, rows, slots, sides, n=3):
    out = np.empty(len(rows))
    for i, (r, s, sd) in enumerate(zip(rows, slots, sides)):
        a = max(0, s - n + 1)
        out[i] = m.l[r, a:s + 1].min() if sd > 0 else m.h[r, a:s + 1].max()
    return out


# ---------------------------------------------------------------- E1
def e1_orb15_in_play(m: DayMatrix, f: Features, mk: Market):
    """15-minute opening-range breakout on stocks in play (range volume >= 1.5x its 20-day
    average), first close beyond the range 09:30-13:00, with the NIFTY. Stop: range middle.
    Target 2R. Flat 15:15."""
    oh, ol = m.h[:, :3].max(1), m.l[:, :3].min(1)
    mid = (oh + ol) / 2
    width = 100 * (oh - ol) / mid
    rvol = f.or_vol / f.or_vol_avg
    ok = f.valid & (width >= 0.3) & (width <= 3.0) & (rvol >= 1.5)
    win = (S_IDX >= 3) & (S_IDX <= slot_of_close("13:00"))
    side, slot = _pick_side(first_true(win & (m.c > oh[:, None])), first_true(win & (m.c < ol[:, None])))
    rows = np.where(ok & (side != 0))[0]
    slot, side = slot[rows], side[rows]
    keep = _with_market(m, mk, rows, slot, side)
    rows, slot, side = rows[keep], slot[keep], side[keep]
    entry = m.c[rows, slot]
    stop = mid[rows]
    target = entry + 2 * (entry - stop)
    return candidates(m, rows, slot, side, entry, stop, target, scores=rvol[rows])


# ---------------------------------------------------------------- E2
def e2_orb5_stocks_in_play(m: DayMatrix, f: Features, mk: Market):
    """5-minute ORB on stocks in play (after Zarattini, Barbon and Aziz, 2024): the first bar's
    direction picks the side; a stop order at its high (low); stop 10% of the daily ATR(14);
    no target, flat 15:15. Only the 10 stocks with the highest first-bar relative volume each
    day (ranked at 09:20, across the universe)."""
    o0, c0, h0, l0 = m.o[:, 0], m.c[:, 0], m.h[:, 0], m.l[:, 0]
    rvol = f.v0 / f.v0_avg
    side = np.sign(c0 - o0).astype(int)
    ok = f.valid & (rvol >= 1.0) & (side != 0) & (f.atr14 > 0)
    rows = np.where(ok)[0]
    side = side[rows]
    level = np.where(side > 0, h0[rows], l0[rows])
    k, fill, hit = stop_entries(m, rows, level, side, 1, slot_of_close("12:00"))
    eligible = [(m.dates[r], m.symbol, float(rvol[r])) for r in rows]
    rows, k, fill, side = rows[hit], k[hit], fill[hit], side[hit]
    stop = fill - side * 0.10 * f.atr14[rows]
    target = np.full(len(rows), np.nan)
    c = candidates(m, rows, k, side, fill, stop, target, scores=rvol[rows], include_entry_bar=True)
    return c, eligible


# ---------------------------------------------------------------- E3
def e3_gap_fade(m: DayMatrix, f: Features, mk: Market):
    """Fade an opening gap of 1.5% to 5% when the first 5-minute bar already turns against it:
    enter at 09:20, stop just beyond the first bar's extreme, target the previous close (gap
    filled). Flat 15:15."""
    o0, c0, h0, l0 = m.o[:, 0], m.c[:, 0], m.h[:, 0], m.l[:, 0]
    gap = 100 * (o0 / f.prev_close - 1)
    up = (gap >= 1.5) & (gap <= 5) & (c0 < o0)
    dn = (gap <= -1.5) & (gap >= -5) & (c0 > o0)
    rows = np.where(~np.isnan(f.prev_close) & (up | dn))[0]
    side = np.where(up[rows], -1, 1)
    entry = c0[rows]
    stop = np.where(side < 0, h0[rows] * 1.001, l0[rows] * 0.999)
    target = f.prev_close[rows]
    rr = np.abs(target - entry) / np.abs(entry - stop)
    keep = rr >= 1.0
    rows, side, entry, stop, target = rows[keep], side[keep], entry[keep], stop[keep], target[keep]
    return candidates(m, rows, np.zeros(len(rows), int), side, entry, stop, target, scores=np.abs(gap[rows]))


# ---------------------------------------------------------------- E4
def e4_gap_and_go(m: DayMatrix, f: Features, mk: Market):
    """Gap of 1% to 5% in the direction of the daily trend, first bar strong in the same
    direction (closes in its upper half for a long): stop order at the first bar's high
    (low) until 11:15, stop at its other end, target 2R. Flat 15:15."""
    o0, c0, h0, l0 = m.o[:, 0], m.c[:, 0], m.h[:, 0], m.l[:, 0]
    gap = 100 * (o0 / f.prev_close - 1)
    mid0 = (h0 + l0) / 2
    lo = f.valid & f.trend_up & (gap >= 1) & (gap <= 5) & (c0 > o0) & (c0 >= mid0)
    sh = f.valid & f.trend_down & (gap <= -1) & (gap >= -5) & (c0 < o0) & (c0 <= mid0)
    rows = np.where(lo | sh)[0]
    side = np.where(lo[rows], 1, -1)
    level = np.where(side > 0, h0[rows], l0[rows])
    k, fill, hit = stop_entries(m, rows, level, side, 1, slot_of_close("11:15"))
    rows, k, fill, side = rows[hit], k[hit], fill[hit], side[hit]
    stop = np.where(side > 0, l0[rows], h0[rows])
    target = fill + 2 * (fill - stop)
    return candidates(m, rows, k, side, fill, stop, target, scores=np.abs(gap[rows]), include_entry_bar=True)


# ---------------------------------------------------------------- E5
def e5_vwap_pullback(m: DayMatrix, f: Features, mk: Market):
    """Trend pullback to VWAP: daily uptrend (close > 20-day > 50-day average); after 09:30 a
    close under VWAP, then within two bars a close back above VWAP and the prior bar's high,
    from 10:00 to 14:30, with the NIFTY. Stop: the dip's low. Target 2R. Mirror for shorts."""
    rows_out, slots, sides, stops = [], [], [], []
    until = slot_of_close("14:30")
    nb = slot_of_close("10:00")
    for r in np.where(f.valid & (f.trend_up | f.trend_down))[0]:
        longs = bool(f.trend_up[r])
        c, h, l, vw = m.c[r], m.h[r], m.l[r], f.vwap[r]
        last_dip, ext = -10, None
        for s in range(3, until + 1):
            below = c[s] < vw[s]
            if longs:
                if below:
                    ext = l[s] if s - last_dip > 2 else min(ext, l[s])
                    last_dip = s
                    continue
                if s - last_dip <= 2 and s >= nb and c[s] > h[s - 1]:
                    rows_out.append(r); slots.append(s); sides.append(1); stops.append(min(ext, l[s]))
                    break
            else:
                if not below:
                    ext = h[s] if s - last_dip > 2 else max(ext, h[s])
                    last_dip = s
                    continue
                if s - last_dip <= 2 and s >= nb and c[s] < l[s - 1]:
                    rows_out.append(r); slots.append(s); sides.append(-1); stops.append(max(ext, h[s]))
                    break
    rows, slots, sides, stops = (np.array(x) for x in (rows_out, slots, sides, stops))
    if len(rows) == 0:
        return []
    keep = _with_market(m, mk, rows, slots, sides)
    rows, slots, sides, stops = rows[keep], slots[keep], sides[keep], stops[keep]
    entry = m.c[rows, slots]
    target = entry + 2 * (entry - stops)
    return candidates(m, rows, slots, sides, entry, stops, target)


# ---------------------------------------------------------------- E6
def e6_20day_breakout(m: DayMatrix, f: Features, mk: Market):
    """20-day high breakout: yesterday's close within 3% under the 20-session high, no gap of
    more than 1.5% over it; first close above it 09:30-14:30, with the NIFTY. Stop: low of the
    last 3 bars. Target 2R. Mirror on the 20-session low for shorts."""
    o0 = m.o[:, 0]
    pc = f.prev_close
    near_hi = f.valid & (pc >= f.high20 * 0.97) & (pc < f.high20) & (o0 <= f.high20 * 1.015)
    near_lo = f.valid & (pc <= f.low20 * 1.03) & (pc > f.low20) & (o0 >= f.low20 * 0.985)
    win = (S_IDX >= 2) & (S_IDX <= slot_of_close("14:30"))
    ku = np.where(near_hi, first_true(win & (m.c > f.high20[:, None])), -1)
    kd = np.where(near_lo, first_true(win & (m.c < f.low20[:, None])), -1)
    side, slot = _pick_side(ku, kd)
    rows = np.where(side != 0)[0]
    slot, side = slot[rows], side[rows]
    keep = _with_market(m, mk, rows, slot, side)
    rows, slot, side = rows[keep], slot[keep], side[keep]
    entry = m.c[rows, slot]
    stop = _swing_stop(m, rows, slot, side)
    target = entry + 2 * (entry - stop)
    return candidates(m, rows, slot, side, entry, stop, target)


# ---------------------------------------------------------------- E7
def e7_prev_day_breakout(m: DayMatrix, f: Features, mk: Market):
    """Previous-day high/low breakout: the stock opens inside yesterday's range, then a bar
    closes above yesterday's high (below its low) 09:30-13:00, with the NIFTY. Stop: low of
    the last 3 bars. Target 2R."""
    o0 = m.o[:, 0]
    inside = (o0 < f.prev_high) & (o0 > f.prev_low)
    win = (S_IDX >= 2) & (S_IDX <= slot_of_close("13:00"))
    ku = np.where(inside, first_true(win & (m.c > f.prev_high[:, None])), -1)
    kd = np.where(inside, first_true(win & (m.c < f.prev_low[:, None])), -1)
    side, slot = _pick_side(ku, kd)
    rows = np.where(side != 0)[0]
    slot, side = slot[rows], side[rows]
    keep = _with_market(m, mk, rows, slot, side)
    rows, slot, side = rows[keep], slot[keep], side[keep]
    entry = m.c[rows, slot]
    stop = _swing_stop(m, rows, slot, side)
    target = entry + 2 * (entry - stop)
    return candidates(m, rows, slot, side, entry, stop, target)


# ---------------------------------------------------------------- E8
def e8_vwap_band_reversion(m: DayMatrix, f: Features, mk: Market):
    """Mean reversion to VWAP inside a daily trend: in an uptrend, a close more than 2
    volume-weighted standard deviations under VWAP (10:00-14:30) is bought; target VWAP at
    that moment; stop the same distance below (1:1). Mirror in a downtrend."""
    win = (S_IDX >= slot_of_close("10:00")) & (S_IDX <= slot_of_close("14:30"))
    lo_band = f.vwap - 2 * f.vsd
    hi_band = f.vwap + 2 * f.vsd
    ku = np.where(f.valid & f.trend_up, first_true(win & (f.vsd > 0) & (m.c < lo_band)), -1)
    kd = np.where(f.valid & f.trend_down, first_true(win & (f.vsd > 0) & (m.c > hi_band)), -1)
    side, slot = _pick_side(ku, kd)
    rows = np.where(side != 0)[0]
    slot, side = slot[rows], side[rows]
    entry = m.c[rows, slot]
    target = f.vwap[rows, slot]
    stop = entry - (target - entry)
    return candidates(m, rows, slot, side, entry, stop, target)


# ---------------------------------------------------------------- E9
def e9_relative_strength_1015(m: DayMatrix, f: Features, mk: Market):
    """Relative-strength momentum: at 10:15, if the NIFTY is up more than 0.1% from its open,
    buy the 3 NIFTY 100 stocks that are strongest against it (at least 0.5% ahead and up on
    the day); if it is down, short the 3 weakest. Stop: the day's low (high) so far. No target,
    flat 15:15."""
    s = slot_of_close("10:15")
    ret = m.c[:, s] / m.o[:, 0] - 1
    rows_out, sides, scores = [], [], []
    for r in np.where(f.valid)[0]:
        nr = mk.ret(m.dates[r], s)
        if np.isnan(nr) or abs(nr) <= 0.001:
            continue
        rs = ret[r] - nr
        if nr > 0 and rs > 0.005 and ret[r] > 0:
            rows_out.append(r); sides.append(1); scores.append(rs)
        elif nr < 0 and rs < -0.005 and ret[r] < 0:
            rows_out.append(r); sides.append(-1); scores.append(-rs)
    if not rows_out:
        return []
    rows, sides = np.array(rows_out), np.array(sides)
    slots = np.full(len(rows), s)
    entry = m.c[rows, s]
    stop = np.where(sides > 0, m.l[rows, :s + 1].min(1), m.h[rows, :s + 1].max(1))
    target = np.full(len(rows), np.nan)
    return candidates(m, rows, slots, sides, entry, stop, target, scores=np.array(scores))


EQUITY = [
    ("E1", "ORB-15 stocks in play", e1_orb15_in_play, {}),
    ("E2", "ORB-5 stocks in play (top 10 by volume)", e2_orb5_stocks_in_play, {"rank_top": 10}),
    ("E3", "Opening gap fade", e3_gap_fade, {}),
    ("E4", "Gap and go with the trend", e4_gap_and_go, {}),
    ("E5", "VWAP pullback in trend", e5_vwap_pullback, {}),
    ("E6", "20-day high/low breakout", e6_20day_breakout, {}),
    ("E7", "Previous-day high/low breakout", e7_prev_day_breakout, {}),
    ("E8", "VWAP 2-sigma reversion in trend", e8_vwap_band_reversion, {}),
    ("E9", "Relative strength at 10:15 (top 3)", e9_relative_strength_1015, {"top_n": 3}),
]
