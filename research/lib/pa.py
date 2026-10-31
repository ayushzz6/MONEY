"""Price-action strategy library for round 6 (NIFTY 500, 5-minute bars).

Every rule is decided at a bar's close from that bar and earlier ones; the engine enters at the next
bar's open. Each rule also gives its structural stop. Long rules are written once on (X, s) and mirrored
for shorts (s = -1 swaps highs and lows and flips comparisons).

Sources (paraphrased, adapted to 5-minute NSE bars): Al Brooks, Trading Price Action Trends / Reversals
(2011-12); Bob Volman, Understanding Price Action (2014); Lance Beggs, YTC Price Action Trader (2010);
Thomas Bulkowski, Encyclopedia of Chart Patterns (2005); Linda Raschke and Larry Connors, Street Smarts
(1995); Toby Crabel (1990); the "smart money concepts" vocabulary (break of structure, change of character,
liquidity sweep, fair value gap) as retail traders use it. Plus the 19 price-action and level rules of the
round 1 library.
"""
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

from research.lib.strategies import (LIBRARY as OLD, S, Strat, T0945, T1000, T1200, T1430, col, gt, ge, hi, lo,
                                     mirror, prev, xdn, xup)
from research.sim import slot_of_close

T1030, T1330 = slot_of_close("10:30"), slot_of_close("13:30")


# ---------------------------------------------------------------- helpers
def roll(A, n, how, lag=0):
    """max / min / sum / mean over the n bars ending `lag` bars before each bar (across sessions, like prev)."""
    flat = np.asarray(A, float).reshape(-1)
    x = np.concatenate([np.full(n - 1 + lag, np.nan), flat])
    w = _swv(x, n)[: flat.size]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"max": np.nanmax, "min": np.nanmin, "sum": np.nansum, "mean": np.nanmean}[how](w, axis=1)
    return out.reshape(np.shape(A))


def adverse(A, n, s, lag=0):
    """The extreme against a trade over the window: the lowest for a long, the highest for a short."""
    return roll(A, n, "min" if s > 0 else "max", lag)


def favour(A, n, s, lag=0):
    return roll(A, n, "max" if s > 0 else "min", lag)


def ext(a, b, s):
    """The level further against the trade."""
    return np.minimum(a, b) if s > 0 else np.maximum(a, b)


def dlo(X, s):
    return X.day_lo if s > 0 else X.day_hi


def dhi(X, s):
    return X.day_hi if s > 0 else X.day_lo


def cpos(X, s):
    rng = np.maximum(X.H - X.L, 1e-9)
    return (X.C - X.L) / rng if s > 0 else (X.H - X.C) / rng


def bull(X, s):
    return gt(X.C, X.O, s)


def tail(X, s):
    return (np.minimum(X.O, X.C) - X.L) if s > 0 else (X.H - np.maximum(X.O, X.C))


def cross(a, b, s):
    return xup(a, b) if s > 0 else xdn(a, b)


def pb(A, k=1):
    """prev() for booleans: True only where the earlier bar was True."""
    return prev(np.asarray(A, float), k) == 1


def any_before_today(B, lag=1):
    """Whether B held on an earlier bar of the same session (at least `lag` bars ago)."""
    c = np.logical_or.accumulate(np.asarray(B, bool), axis=1)
    out = np.zeros_like(c)
    out[:, lag:] = c[:, :-lag]
    return out


def ffill_day(A):
    """Forward-fill finite values along each session (NaN before the first)."""
    A = np.asarray(A, float)
    idx = np.where(np.isfinite(A), np.arange(A.shape[1])[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.take_along_axis(A, np.maximum(idx, 0), axis=1)
    return np.where(idx >= 0, out, np.nan)


def prev_level(A):
    """The value a forward-filled level had before its latest change."""
    flat = np.asarray(A, float).reshape(-1)
    before = np.r_[np.nan, flat[:-1]]
    chg = np.isfinite(flat) & np.isfinite(before) & (flat != before)
    v = np.where(chg, before, np.nan)
    idx = np.where(np.isfinite(v), np.arange(flat.size), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    return out.reshape(np.shape(A))


def swing_hi(X, s):
    return X.swing_hi if s > 0 else X.swing_lo


def swing_lo(X, s):
    return X.swing_lo if s > 0 else X.swing_hi


# ---------------------------------------------------------------- trend continuation
@mirror
def ema_pullback(X, s):
    trend = gt(X.ema20, prev(X.ema20, 6), s) & gt(X.ema9, X.ema21, s) & gt(X.C, X.ema50, s)
    touch = ge(X.ema20 * (1 + 0.001 * s), adverse(lo(X, s), 3, s), s)
    sig = trend & touch & gt(X.C, prev(hi(X, s)), s) & (cpos(X, s) >= 0.6) & (S >= 4) & (S <= T1430)
    return sig, adverse(lo(X, s), 5, s)


def retest(level):
    @mirror
    def fn(X, s):
        lvl = level(X, s)
        beyond = gt(X.C, lvl, s)
        broke = any_before_today(beyond, lag=2)
        near = np.abs(lo(X, s) - lvl) <= 0.0015 * X.C
        sig = broke & near & beyond & bull(X, s) & (S >= 5) & (S <= T1430)
        return sig, ext(lo(X, s), lvl * (1 - 0.001 * s), s)
    return fn


@mirror
def bull_flag(X, s):
    base = adverse(lo(X, s), 6, s, lag=4)
    pole = (prev(X.C, 3) - base) * s
    fh = favour(hi(X, s), 3, s, lag=1)
    fl = adverse(lo(X, s), 3, s, lag=1)
    shallow = (fl - base) * s >= 0.5 * pole
    sig = (pole >= 2.5 * X.atr5) & (pole >= 0.006 * X.C) & (np.abs(fh - fl) <= 0.5 * pole) & shallow
    sig &= gt(X.C, fh, s) & (S >= 10) & (S <= T1430)
    return sig, fl


@mirror
def trend_from_open(X, s):
    o0, h0, l0, c0 = X.O[:, 0], X.H[:, 0], X.L[:, 0], X.C[:, 0]
    r0 = np.maximum(h0 - l0, 1e-9)
    if s > 0:
        shape = ((o0 - l0) / r0 <= 0.25) & ((c0 - l0) / r0 >= 0.75)
        held = np.minimum(X.L[:, 1], X.L[:, 2]) > (h0 + l0) / 2
        brk = X.C[:, 2] > h0
        stop = l0
    else:
        shape = ((h0 - o0) / r0 <= 0.25) & ((h0 - c0) / r0 >= 0.75)
        held = np.maximum(X.H[:, 1], X.H[:, 2]) < (h0 + l0) / 2
        brk = X.C[:, 2] < l0
        stop = h0
    sig = np.zeros(X.C.shape, bool)
    sig[:, 2] = shape & held & brk & (r0 >= 0.006 * o0)
    return sig, col(stop) * np.ones(X.C.shape)


@mirror
def momentum_ignition(X, s):
    rng = X.H - X.L
    big = (np.abs(X.C - X.O) >= 0.7 * rng) & (rng >= 2 * roll(rng, 20, "mean", lag=1)) & (np.nan_to_num(X.rv) >= 2)
    sig = big & bull(X, s) & gt(X.C, prev(dhi(X, s)), s) & (S >= T0945) & (S <= T1430)
    return sig, (X.H + X.L) / 2


# ---------------------------------------------------------------- structure (swing highs and lows)
@mirror
def bos(X, s):
    higher_low = gt(swing_lo(X, s), prev_level(swing_lo(X, s)), s)
    sig = higher_low & cross(X.C, swing_hi(X, s), s) & (S >= 6) & (S <= T1430)
    return sig, swing_lo(X, s)


@mirror
def choch(X, s):
    lower_high = gt(prev_level(swing_hi(X, s)), swing_hi(X, s), s)
    sig = lower_high & cross(X.C, swing_hi(X, s), s) & (S >= 6) & (S <= T1430)
    return sig, adverse(lo(X, s), 10, s)


@mirror
def liquidity_sweep(X, s):
    lvl = prev(swing_lo(X, s))
    rng = np.maximum(X.H - X.L, 1e-9)
    sig = gt(lvl, lo(X, s), s) & gt(X.C, lvl, s) & (tail(X, s) >= 0.5 * rng) & (S >= 6) & (S <= T1430)
    return sig, lo(X, s)


@mirror
def fvg_retest(X, s):
    pr_body = np.abs(prev(X.C) - prev(X.O))
    pr_rng = np.maximum(prev(X.H) - prev(X.L), 1e-9)
    create = gt(lo(X, s), prev(hi(X, s), 2), s) & (pr_body >= 0.6 * pr_rng) & (S >= 2)
    near = ffill_day(np.where(create, lo(X, s), np.nan))
    far = ffill_day(np.where(create, prev(hi(X, s), 2), np.nan))
    born = ffill_day(np.where(create, S * np.ones(X.C.shape), np.nan))
    sig = (S > born + 1) & ge(near, lo(X, s), s) & ge(X.C, far, s) & bull(X, s) & (S <= T1430)
    return sig, far * (1 - 0.001 * s)


@mirror
def hod_trap(X, s):
    d1, d2 = prev(dlo(X, s)), prev(dlo(X, s), 2)
    broke = gt(prev(lo(X, s)), d2, -s)
    sig = broke & gt(X.C, d2, s) & bull(X, s) & (S >= T1030) & (S <= T1430)
    return sig, d1


@mirror
def three_push(X, s):
    newx = gt(lo(X, s), prev(dlo(X, s)), -s) & (S >= 1)
    edges = newx & ~pb(newx)
    pushes = np.zeros(X.C.shape)
    pushes[:, 1:] = np.cumsum(edges, axis=1)[:, :-1]
    sig = (pushes >= 3) & bull(X, s) & gt(X.C, prev(hi(X, s)), s) & (S >= 12) & (S <= T1430)
    return sig, dlo(X, s)


@mirror
def double_bottom_low(X, s):
    newx = gt(lo(X, s), prev(dlo(X, s)), -s) | (S == 0)
    last = np.maximum.accumulate(np.where(newx, S * np.ones(X.C.shape), -1), axis=1)
    since = S - np.concatenate([np.full((X.C.shape[0], 1), -1.0), last[:, :-1]], axis=1)
    d1 = prev(dlo(X, s))
    near = np.abs(lo(X, s) - d1) <= 0.001 * X.C
    sig = near & (since >= 6) & bull(X, s) & (cpos(X, s) >= 0.5) & (S >= 8) & (S <= T1430)
    return sig, ext(lo(X, s), d1, s)


# ---------------------------------------------------------------- bar patterns and ranges
@mirror
def inside_ii(X, s):
    ib = (X.H < prev(X.H)) & (X.L > prev(X.L))
    sig = pb(ib) & pb(ib, 2) & gt(X.C, prev(hi(X, s)), s) & gt(X.C, X.vwap, s) & (S >= 4) & (S <= T1430)
    return sig, prev(lo(X, s))


@mirror
def nr7_break(X, s):
    rng = X.H - X.L
    nr = rng < roll(rng, 6, "min", lag=1)
    sig = pb(nr) & gt(X.C, prev(hi(X, s)), s) & (S >= 8) & (S <= T1430)
    return sig, prev(lo(X, s))


@mirror
def box_breakout(X, s):
    bh, bl = favour(hi(X, s), 12, s, lag=1), adverse(lo(X, s), 12, s, lag=1)
    sig = (np.abs(bh - bl) <= 0.005 * X.C) & gt(X.C, bh, s) & (S >= 13) & (S <= T1430)
    return sig, bl


@mirror
def late_range_break(X, s):
    bh, bl = favour(hi(X, s), 12, s, lag=1), adverse(lo(X, s), 12, s, lag=1)
    sig = (np.abs(bh - bl) <= 0.004 * X.C) & gt(X.C, prev(dhi(X, s)), s) & (S >= T1330) & (S <= T1430)
    return sig, bl


@mirror
def climax_reversal(X, s):
    rng = X.H - X.L
    climax = (prev(np.nan_to_num(X.rv)) >= 2.5) & (prev(rng) >= 2.5 * roll(rng, 20, "mean", lag=2))
    climax &= gt(prev(X.C), prev(X.O), -s) & gt(prev(lo(X, s)), prev(dlo(X, s), 2), -s)
    sig = climax & gt(X.C, (prev(X.H) + prev(X.L)) / 2, s) & bull(X, s) & (S >= 3) & (S <= T1430)
    return sig, prev(lo(X, s))


@mirror
def doji_reversal(X, s):
    prng = prev(X.H - X.L)
    doji = (np.abs(prev(X.C) - prev(X.O)) <= 0.1 * prng) & (prng > 0)
    at_ext = gt(prev(lo(X, s)), prev(dlo(X, s), 2), -s)
    sig = doji & at_ext & gt(X.C, prev(hi(X, s)), s) & bull(X, s) & (S >= 3) & (S <= T1430)
    return sig, ext(lo(X, s), prev(lo(X, s)), s)


# ---------------------------------------------------------------- levels
def _step(X):
    p = col(X.f.prev_close)
    return np.select([p < 200, p < 500, p < 2000, p < 10000], [5.0, 10.0, 50.0, 100.0], 500.0)


@mirror
def round_bounce(X, s):
    step = _step(X)
    lvl = np.round(lo(X, s) / step) * step
    near = np.abs(lo(X, s) - lvl) <= 0.0015 * X.C
    sig = near & gt(X.C, lvl, s) & (cpos(X, s) >= 0.6) & bull(X, s) & (S >= 3) & (S <= T1430)
    return sig, ext(lo(X, s), lvl * (1 - 0.002 * s), s)


@mirror
def pin_key(X, s):
    levels = [col(X.f.prev_low), col(X.lv.wk_low), X.vwap, X.or15_lo] if s > 0 else \
        [col(X.f.prev_high), col(X.lv.wk_high), X.vwap, X.or15_hi]
    at = np.logical_or.reduce([np.abs(lo(X, s) - lv_) <= 0.002 * X.C for lv_ in levels])
    body = np.abs(X.C - X.O)
    sig = at & (tail(X, s) >= 2 * body) & (cpos(X, s) >= 0.66) & (S >= 3) & (S <= T1430)
    return sig, lo(X, s)


@mirror
def opening_reversal(X, s):
    o0 = X.O[:, :1]
    drove = ge(o0 * (1 - 0.004 * s), prev(dlo(X, s)), s)
    sig = drove & cross(X.C, o0 * np.ones(X.C.shape), s) & (S >= 3) & (S <= 12)
    return sig, dlo(X, s)


@mirror
def pdc_reclaim(X, s):
    pc = col(X.f.prev_close) * np.ones(X.C.shape)
    other_side = roll(gt(X.C, pc, -s).astype(float), 6, "min", lag=1) == 1
    sig = other_side & gt(X.C, pc, s) & (S >= T1000) & (S <= T1430)
    return sig, adverse(lo(X, s), 6, s)


@mirror
def vwap_two_bar(X, s):
    trend = gt(X.ema20, prev(X.ema20, 6), s)
    tagged = ge(prev(X.vwap) * (1 + 0.0005 * s), prev(lo(X, s)), s)
    sig = trend & tagged & gt(X.C, prev(hi(X, s)), s) & bull(X, s) & gt(X.C, X.vwap, s) & (S >= 6) & (S <= T1430)
    return sig, ext(lo(X, s), prev(lo(X, s)), s)


@mirror
def gap_fill_hold(X, s):
    pc = col(X.f.prev_close)
    gap = (X.O[:, :1] / pc - 1) * s >= 0.004
    touched = ge(pc * (1 + 0.001 * s), lo(X, s), s)
    sig = gap & touched & gt(X.C, pc, s) & bull(X, s) & (cpos(X, s) >= 0.6) & (S >= 1) & (S <= T1200)
    return sig, ext(lo(X, s), pc * (1 - 0.002 * s), s)


NEW = [
    ("pa_ema_pb", "First pullback to the 20 EMA", "Trend continuation", "Brooks", "Trend up, low tags the 20 EMA, bar closes above the prior high", ema_pullback),
    ("pa_retest_or", "Opening-range breakout retest", "Trend continuation", "Volman / Beggs", "After breaking the 15-minute range, a pullback to it holds", retest(lambda X, s: X.or15_hi if s > 0 else X.or15_lo)),
    ("pa_retest_pdh", "Prior-day high retest (role reversal)", "Trend continuation", "Murphy / Beggs", "After breaking yesterday's high, a pullback to it holds", retest(lambda X, s: col(X.f.prev_high) * np.ones(X.C.shape) if s > 0 else col(X.f.prev_low) * np.ones(X.C.shape))),
    ("pa_flag", "Bull / bear flag", "Trend continuation", "Bulkowski / Brooks", "A 2.5-ATR pole, a tight shallow 3-bar flag, close beyond it", bull_flag),
    ("pa_tfo", "Trend from the open", "Trend continuation", "Brooks / Dalton", "First bar opens at its low and closes at its high; 09:25 bar closes above it", trend_from_open),
    ("pa_ignite", "Momentum ignition bar", "Trend continuation", "Volman", "Wide trend bar on 2x volume closing at a new day high", momentum_ignition),
    ("pa_bos", "Break of structure (higher low, then new swing high)", "Market structure", "SMC / Dow", "Higher swing low, then a close above the last swing high", bos),
    ("pa_choch", "Change of character (reversal)", "Market structure", "SMC / Brooks", "Lower swing highs, then a close above the last swing high", choch),
    ("pa_sweep", "Liquidity sweep of a swing low / high", "Market structure", "SMC / Raschke", "Wick below the last swing low, closes back above with a long tail", liquidity_sweep),
    ("pa_fvg", "Fair value gap retest", "Market structure", "SMC", "A 3-bar gap after a strong bar; price returns into it and holds", fvg_retest),
    ("pa_trap", "Failed new day low / high (trap)", "Reversal", "Brooks / Raschke", "A bar makes a new day low, the next closes back above the old low (after 10:30)", hod_trap),
    ("pa_3push", "Three pushes (wedge) reversal", "Reversal", "Brooks", "Three separate new day lows, then a bar closes above the prior high", three_push),
    ("pa_db_low", "Double bottom at the day's low", "Reversal", "Bulkowski / Brooks", "Retest of the day low 6+ bars later that holds", double_bottom_low),
    ("pa_climax", "Climax bar reversal", "Reversal", "Brooks / Wyckoff", "2.5x volume, 2.5x range bar at a new low, then a close above its middle", climax_reversal),
    ("pa_doji", "Doji at the day's extreme, then confirmation", "Reversal", "Nison", "Doji makes a new day low; next bar closes above its high", doji_reversal),
    ("pa_opening_rev", "Opening reversal", "Reversal", "Dalton / Crabel", "Drops 0.4% from the open, then closes back above the open (by 10:15)", opening_reversal),
    ("pa_ii", "Inside-inside (ii) breakout with VWAP", "Bar patterns", "Brooks / Crabel", "Two inside bars, close beyond the second on VWAP's side", inside_ii),
    ("pa_nr7", "NR7 bar breakout", "Bar patterns", "Crabel", "After the narrowest of 7 bars, close beyond its high", nr7_break),
    ("pa_box", "Tight box breakout", "Bar patterns", "Volman", "12 bars inside 0.5%, close beyond the box", box_breakout),
    ("pa_late_box", "Afternoon box to new day high", "Bar patterns", "Brooks", "After 13:30, a 0.4% box breaks to a new day high", late_range_break),
    ("pa_round", "Round-number bounce", "Levels", "Beggs", "Tags a round number and closes back with a strong bar", round_bounce),
    ("pa_pin", "Pin bar at a key level", "Levels", "Nison / Beggs", "Long tail at yesterday's low, last week's low, VWAP or the opening range", pin_key),
    ("pa_pdc", "Prior close reclaim", "Levels", "Beggs", "6 closes below yesterday's close, then a close above it (after 10:00)", pdc_reclaim),
    ("pa_vwap2", "Two-bar reversal at VWAP in trend", "Levels", "Brooks / Shannon", "Trend up, previous bar tags VWAP, this bar closes above its high", vwap_two_bar),
    ("pa_gapfill", "Gap fill and hold", "Levels", "Crabel / Raschke", "Gap of 0.4%+ fills to yesterday's close and holds (by 12:00)", gap_fill_hold),
]
OLD_KEYS = ["engulf", "hammer", "inside_bar", "h2", "three_bar", "outside", "wrb", "engulf15", "inside15", "orb_fail",
            "double_bottom", "morning_star", "tweezer", "turtle_soup", "pdh_break", "fractal_break", "late_high",
            "week_break", "sr_cluster"]


def _lib():
    lib = [Strat(k, n, f, src, rule, fn) for k, n, f, src, rule, fn in NEW]
    old = {s.key: s for s in OLD}
    for k in OLD_KEYS:
        s = old[k]
        lib.append(Strat(s.key, s.name, "Round 1: " + s.family, s.source, s.rule, s.fn))
    return lib


PA_LIBRARY = _lib()
