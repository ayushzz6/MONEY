"""Causal features for the strategy library: every value at a 5-minute bar uses only that bar and earlier ones.

Indicators run on each stock's bars as one continuous series (sessions joined, as on a 5-minute chart) and are
reshaped to sessions x 75 slots. Session values (VWAP, the day's high so far, opening ranges) restart each day.
Daily levels (yesterday's high / low / close, pivots, CPR, Camarilla, last week's range, ATR, moving averages)
come from completed sessions only. research/lib/leak.py checks all of this by cutting the data at random bars.
"""
from types import SimpleNamespace

import numpy as np
from scipy.signal import lfilter

from research.data import SLOTS

SL = np.arange(SLOTS)[None, :]


# ---------------------------------------------------------------- series helpers (1-D, causal)
def ema(x, n):
    a = 2.0 / (n + 1)
    y, _ = lfilter([a], [1, a - 1], x, zi=[x[0] * (1 - a)])
    return y


def wilder(x, n):
    a = 1.0 / n
    y, _ = lfilter([a], [1, a - 1], x, zi=[x[0] * (1 - a)])
    return y


def roll(x, n, fn):
    """fn over the last n values ending at each point (NaN before n values exist)."""
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = fn(w, axis=1)
    return out


def sma(x, n):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def shift(x, k=1, fill=np.nan):
    out = np.empty_like(x, dtype=float)
    out[:k] = fill
    out[k:] = x[:-k]
    return out


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), n), wilder(np.maximum(-d, 0), n)
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def supertrend(h, l, c, n=10, mult=3.0):
    tr = np.maximum(h - l, np.maximum(np.abs(h - shift(c, 1, c[0])), np.abs(l - shift(c, 1, c[0]))))
    atr = wilder(tr, n)
    mid = (h + l) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(len(c))
    line = np.empty(len(c))
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] > 0:
            dirn[i] = -1 if c[i] < fl[i] else 1
        else:
            dirn[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if dirn[i] > 0 else fu[i]
    line[0] = fl[0]
    return dirn, line


def psar(h, l, step=0.02, mx=0.2):
    n = len(h)
    sar = np.empty(n)
    up = True
    af, ep, sar[0] = step, h[0], l[0]
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        sar[i] = s
    return sar


# ---------------------------------------------------------------- per-session helpers (2-D, causal along slots)
def first_bar_from(mask):
    """Index of the first True per row, -1 if none."""
    any_ = mask.any(1)
    return np.where(any_, mask.argmax(1), -1)


def after_slot(arr_d, k):
    """A per-session value that only exists once bar k has closed: NaN on the slots before (and at) k-1."""
    out = np.repeat(arr_d[:, None], SLOTS, 1)
    out[:, :k] = np.nan
    return out


def daily_levels(m, f):
    o, h, l, c, v = m.daily()
    H, L, C = f.prev_high, f.prev_low, f.prev_close
    P = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2 * P - BC
    rng = H - L
    lv = SimpleNamespace(P=P, R1=2 * P - L, S1=2 * P - H, R2=P + rng, S2=P - rng, TC=np.maximum(TC, BC),
                         BC=np.minimum(TC, BC), cpr_w=np.abs(TC - BC) / P,
                         H3=C + 1.1 * rng / 4, L3=C - 1.1 * rng / 4, H4=C + 1.1 * rng / 2, L4=C - 1.1 * rng / 2)
    # last week's high and low
    wk = (m.dates.astype("datetime64[D]").astype(int) + 3) // 7       # week number, weeks starting Monday
    wkh, wkl = np.full(m.n, np.nan), np.full(m.n, np.nan)
    weeks = np.unique(wk)
    hi = {w: h[wk == w].max() for w in weeks}
    lo = {w: l[wk == w].min() for w in weeks}
    for i in range(m.n):
        j = np.searchsorted(weeks, wk[i]) - 1
        if j >= 0:
            wkh[i], wkl[i] = hi[weeks[j]], lo[weeks[j]]
    lv.wk_high, lv.wk_low = wkh, wkl
    rngd = h - l
    lv.nr7 = np.r_[[False] * 7, [rngd[i - 1] <= rngd[i - 7:i].min() for i in range(7, m.n)]]
    lv.nr4 = np.r_[[False] * 4, [rngd[i - 1] <= rngd[i - 4:i].min() for i in range(4, m.n)]]
    lv.inside = np.r_[[False] * 2, (h[1:-1] <= h[:-2]) & (l[1:-1] >= l[:-2])]
    lv.high20d = np.r_[np.nan, roll(h, 20, np.max)[:-1]]
    lv.turn20 = np.r_[np.nan, sma(c * v, 20)[:-1]]
    return lv


def build(m, f, ctx_rows=None):
    """All features of one stock. m: DayMatrix, f: research.sim.features(m). ctx_rows: dict of index name ->
    DayMatrix-like arrays aligned to m's sessions (o, c of the NIFTY and the stock's sector)."""
    D = m.n
    O, H, L, C, V = m.o, m.h, m.l, m.c, m.v
    o, h, l, c, v = (a.reshape(-1) for a in (O, H, L, C, V))
    R = lambda x: x.reshape(D, SLOTS)
    X = SimpleNamespace(O=O, H=H, L=L, C=C, V=V, D=D, f=f)
    # moving averages and trend
    for n in (9, 13, 20, 21, 50, 200, 240):
        setattr(X, f"ema{n}", R(ema(c, n)))
    pc = shift(c, 1, c[0])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = wilder(tr, 14)
    X.atr5 = R(atr)
    X.rsi2, X.rsi14 = R(rsi(c, 2)), R(rsi(c, 14))
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    X.macd, X.macd_sig, X.macd_hist = R(macd), R(sig), R(macd - sig)
    hh14, ll14 = roll(h, 14, np.max), roll(l, 14, np.min)
    k = 100 * (c - ll14) / np.maximum(hh14 - ll14, 1e-9)
    X.stoch_k = R(sma(np.nan_to_num(k, nan=50), 3))
    X.stoch_d = R(sma(np.nan_to_num(sma(np.nan_to_num(k, nan=50), 3), nan=50), 3))
    X.willr = R(-100 * (hh14 - c) / np.maximum(hh14 - ll14, 1e-9))
    tp = (h + l + c) / 3
    tp_sma = sma(tp, 20)
    mad = roll(tp, 20, lambda w, axis: np.mean(np.abs(w - w.mean(axis=axis, keepdims=True)), axis=axis))
    X.cci = R((tp - tp_sma) / np.maximum(0.015 * mad, 1e-9))
    # ADX / DMI
    upm = np.diff(h, prepend=h[0])
    dnm = -np.diff(l, prepend=l[0])
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atr_w = np.maximum(wilder(tr, 14), 1e-9)
    pdi, ndi = 100 * wilder(pdm, 14) / atr_w, 100 * wilder(ndm, 14) / atr_w
    dx = 100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-9)
    X.pdi, X.ndi, X.adx = R(pdi), R(ndi), R(wilder(dx, 14))
    st_dir, st_line = supertrend(h, l, c)
    X.st_dir, X.st_line = R(st_dir), R(st_line)
    X.psar = R(psar(h, l))
    mid = sma(c, 20)
    sd = roll(c, 20, np.std)
    X.bb_mid, X.bb_up, X.bb_lo = R(mid), R(mid + 2 * sd), R(mid - 2 * sd)
    X.kc_up, X.kc_lo = R(ema(c, 20) + 1.5 * atr), R(ema(c, 20) - 1.5 * atr)
    X.don_hi20 = R(shift(roll(h, 20, np.max)))          # the 20 bars before this one
    X.don_lo20 = R(shift(roll(l, 20, np.min)))
    X.don_lo10 = R(shift(roll(l, 10, np.min)))
    X.don_hi10 = R(shift(roll(h, 10, np.max)))
    # channel levels known at the close of each bar (this bar included), for stop orders on the next bar
    X.hi20_incl, X.lo20_incl = R(roll(h, 20, np.max)), R(roll(l, 20, np.min))
    X.hi55_incl, X.lo55_incl = R(roll(h, 55, np.max)), R(roll(l, 55, np.min))
    X.low3, X.high3 = R(roll(l, 3, np.min)), R(roll(h, 3, np.max))
    X.low5, X.high5 = R(roll(l, 5, np.min)), R(roll(h, 5, np.max))
    X.prior_low3, X.prior_high3 = R(shift(roll(l, 3, np.min))), R(shift(roll(h, 3, np.max)))
    obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
    X.obv = R(obv)
    X.obv_hi50, X.obv_lo50 = R(shift(roll(obv, 50, np.max))), R(shift(roll(obv, 50, np.min)))
    X.c_hi50, X.c_lo50 = R(shift(roll(c, 50, np.max))), R(shift(roll(c, 50, np.min)))
    rng = h - l
    X.rng_avg20 = R(shift(sma(rng, 20)))
    tenkan = (roll(h, 9, np.max) + roll(l, 9, np.min)) / 2
    kijun = (roll(h, 26, np.max) + roll(l, 26, np.min)) / 2
    span_a = shift((tenkan + kijun) / 2, 26)
    span_b = shift((roll(h, 52, np.max) + roll(l, 52, np.min)) / 2, 26)
    X.tenkan, X.kijun = R(tenkan), R(kijun)
    X.cloud_top, X.cloud_bot = R(np.fmax(span_a, span_b)), R(np.fmin(span_a, span_b))
    # swing points (5-bar fractals) are only known two bars later
    sh = np.full(len(h), np.nan)
    sl = np.full(len(l), np.nan)
    if len(h) > 5:
        hw = np.lib.stride_tricks.sliding_window_view(h, 5)
        lw = np.lib.stride_tricks.sliding_window_view(l, 5)
        is_h = hw[:, 2] >= hw.max(1)
        is_l = lw[:, 2] <= lw.min(1)
        idx = np.arange(4, len(h))                       # confirmed at the 5th bar of the window
        sh[idx[is_h]] = hw[is_h, 2]
        sl[idx[is_l]] = lw[is_l, 2]
    X.swing_hi = R(_ffill1(sh))
    X.swing_lo = R(_ffill1(sl))
    # session values
    vv = np.where(V > 0, V, 0.0)
    cv = np.cumsum(vv, 1)
    tp2 = (H + L + C) / 3
    vw = np.cumsum(tp2 * vv, 1) / np.maximum(cv, 1e-9)
    var = np.cumsum(vv * tp2 * tp2, 1) / np.maximum(cv, 1e-9) - vw * vw
    X.vwap, X.vsd = vw, np.sqrt(np.maximum(var, 0))
    X.day_hi, X.day_lo = np.maximum.accumulate(H, 1), np.minimum.accumulate(L, 1)
    X.prior_day_hi = np.c_[np.full((D, 1), np.nan), X.day_hi[:, :-1]]
    X.prior_day_lo = np.c_[np.full((D, 1), np.nan), X.day_lo[:, :-1]]
    X.o0 = O[:, 0]
    for k_, name in ((1, "or5"), (3, "or15"), (6, "or30")):
        setattr(X, f"{name}_hi", after_slot(H[:, :k_].max(1), k_))
        setattr(X, f"{name}_lo", after_slot(L[:, :k_].min(1), k_))
    X.ret_open = C / O[:, :1] - 1
    X.ib_hi, X.ib_lo = after_slot(H[:, :12].max(1), 12), after_slot(L[:, :12].min(1), 12)    # first hour, 09:15-10:15
    poc, vah, val = value_area(H, L, C, V)
    X.pv_poc, X.pv_vah, X.pv_val = (np.r_[np.nan, a_[:-1]] for a_ in (poc, vah, val))      # yesterday's profile
    X.sma5 = R(sma(c, 5))
    X.above_vwap_n = np.cumsum(C > vw, 1)          # bars closed above VWAP so far today
    # relative volume against the same slot over the previous 20 sessions
    base = np.full_like(V, np.nan)
    cbase = np.full_like(V, np.nan)
    if D > 21:
        cs = np.cumsum(np.r_[np.zeros((1, SLOTS)), V], 0)
        base[20:] = (cs[20:-1] - cs[:-21]) / 20.0
        cs2 = np.cumsum(np.r_[np.zeros((1, SLOTS)), cv], 0)
        cbase[20:] = (cs2[20:-1] - cs2[:-21]) / 20.0
    X.rv = V / np.where(base > 0, base, np.nan)
    X.crv = cv / np.where(cbase > 0, cbase, np.nan)
    # daily context
    X.lv = daily_levels(m, f)
    X.gap = X.o0 / f.prev_close - 1
    X.atr_d = f.atr14
    X.trend_up, X.trend_dn = f.trend_up, f.trend_down
    X.prev_range = f.prev_high - f.prev_low
    # the NIFTY and the sector, aligned to this stock's sessions
    for key in ("nifty", "sector"):
        a = (ctx_rows or {}).get(key)
        if a is None:
            setattr(X, f"{key}_ret", np.zeros((D, SLOTS)))
            setattr(X, f"{key}_above", np.zeros((D, SLOTS), bool))
            setattr(X, f"{key}_below", np.zeros((D, SLOTS), bool))
            continue
        io, ic, ih, il = a
        twap = np.cumsum((ih + il + ic) / 3, 1) / (SL + 1)
        setattr(X, f"{key}_ret", ic / io[:, :1] - 1)
        setattr(X, f"{key}_above", ic >= twap)
        setattr(X, f"{key}_below", ic <= twap)
    X.sector_rank = (ctx_rows or {}).get("sector_rank", np.full((D, SLOTS), 99))
    X.beta = (ctx_rows or {}).get("beta", np.full(D, np.nan))
    return X


def value_area(H, L, C, V, bins=40, share=0.70):
    """Each session's volume profile from its 5-minute bars (each bar's volume at its typical price): the point of
    control and the value area holding 70% of the volume (Dalton / Steidlmayer)."""
    D = H.shape[0]
    poc, vah, val = np.full(D, np.nan), np.full(D, np.nan), np.full(D, np.nan)
    tp = (H + L + C) / 3
    for d in range(D):
        lo, hi = np.nanmin(L[d]), np.nanmax(H[d])
        if not np.isfinite(lo) or hi <= lo:
            continue
        k = np.clip(((tp[d] - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
        vol = np.bincount(k, weights=np.nan_to_num(V[d]), minlength=bins)
        if vol.sum() <= 0:
            continue
        p = int(vol.argmax())
        a, b, tot = p, p, vol[p]
        while tot < share * vol.sum() and (a > 0 or b < bins - 1):
            left = vol[a - 1] if a > 0 else -1
            right = vol[b + 1] if b < bins - 1 else -1
            if right >= left:
                b += 1
                tot += vol[b]
            else:
                a -= 1
                tot += vol[a]
        w = (hi - lo) / bins
        poc[d], val[d], vah[d] = lo + (p + 0.5) * w, lo + a * w, lo + (b + 1) * w
    return poc, vah, val


def _ffill1(x):
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    first = np.flatnonzero(~np.isnan(x))
    out[:first[0] if len(first) else len(x)] = np.nan
    return out
