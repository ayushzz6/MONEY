"""Entry signals on the NIFTY / BANKNIFTY 5-minute charts, plus option-chain levels from the previous close.

Each signal gives, per session, the first bar (by 13:45) whose close triggers it and the direction
(+1 buy a call, -1 buy a put). Entries happen at the next bar's open. Everything uses bars that have
closed and the chain as it stood at the previous close.
"""
import numpy as np

from core.config.settings import DATA_DIR

LAST = 53                     # the 13:45 bar is the last that can trigger an entry (at 13:50)
OUT = DATA_DIR / "research" / "opt"

NAMES = {
    "orb15": "15-minute opening range breakout",
    "orb30": "30-minute opening range breakout",
    "orb60": "First-hour range breakout",
    "pdhl": "Previous day's high / low breakout",
    "gap_go": "Gap and go (first 15 minutes confirm the gap)",
    "gap_fade": "Gap fade (price crosses back through the open)",
    "ema_cross": "EMA 9 / 21 crossover",
    "supertrend": "Supertrend (10, 3) flip",
    "breadth": "Market breadth push (65% of NIFTY 500 stocks up / 35%)",
    "momentum": "Momentum: index 0.5% from its open after 10:00",
    "wall_break": "Breaks the biggest call-OI / put-OI strike (OI wall)",
    "wall_bounce": "Bounces off the put-OI wall / rejects the call-OI wall",
    "max_pain": "Expiry day: pull towards max pain",
    "late_trend": "Afternoon trend (13:00, with the day's average price)",
    "inside_break": "Inside-day breakout",
    "cpr": "Narrow CPR breakout",
    "rsi_rev": "RSI 14 reversal from 30 / 70",
    "twap_cross": "Crosses the day's average price, after 10:00, with the day's move",
}


def breadth():
    """(D, 75): share of liquid NIFTY 500 stocks above their own open at each bar's close (cached)."""
    path = OUT / "breadth.npy"
    if path.exists():
        return np.load(path)
    from research.lead import intraday as I
    syms, inds, dates, P, idx = I.panel()
    A = I.aux(syms, inds, dates, P, idx)
    elig = A["eligible"][..., None]
    above = np.where(elig, (P["c"] > P["o"][..., :1]).astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        b = np.nanmean(above, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, b)
    return b


def _first(up, dn, start, last=LAST):
    """First slot in [start, last] where up or dn holds: (slot, side); slot -1 when none."""
    D = up.shape[0]
    win = np.zeros(up.shape[1], bool)
    win[start:last + 1] = True
    u_ = up & win
    d_ = dn & win
    any_u, any_d = u_.any(1), d_.any(1)
    fu = np.where(any_u, u_.argmax(1), 999)
    fd = np.where(any_d, d_.argmax(1), 999)
    slot = np.minimum(fu, fd)
    side = np.where(fu <= fd, 1, -1)
    slot = np.where(slot == 999, -1, slot)
    return slot.astype(int), np.where(slot >= 0, side, 0).astype(int)


def _ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    m = np.nan
    for i, v in enumerate(x):
        if np.isfinite(v):
            m = v if not np.isfinite(m) else m + a * (v - m)
        out[i] = m
    return out


def _rsi(x, n=14):
    d = np.diff(x, prepend=x[0])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = _ema(up, 2 * n - 1), _ema(dn, 2 * n - 1)      # Wilder smoothing
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def _supertrend(h, l, c, n=10, mult=3.0):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = _ema(tr, 2 * n - 1)
    mid = (h + l) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    fu, fl = ub.copy(), lb.copy()
    trend = np.ones(len(c), int)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        trend[i] = 1 if c[i] > fu[i - 1] else -1 if c[i] < fl[i - 1] else trend[i - 1]
    return trend


def compute(mk, u, br):
    """{name: (slot (D,), side (D,))} for underlying u."""
    ix = mk.idx[u]
    O, H, L, C = ix["o"], ix["h"], ix["l"], ix["c"]
    D = O.shape[0]
    O0 = O[:, :1]
    pdh = np.concatenate([[np.nan], np.nanmax(H, 1)[:-1]])[:, None]
    pdl = np.concatenate([[np.nan], np.nanmin(L, 1)[:-1]])[:, None]
    pdc = np.concatenate([[np.nan], C[:-1, -1]])[:, None]
    f = mk.f
    prev = lambda k: np.concatenate([[np.nan], f[k][:-1, u]])[:, None]     # chain feature from the previous close
    sig = {}
    for name, n in (("orb15", 3), ("orb30", 6), ("orb60", 12)):
        hi, lo = np.nanmax(H[:, :n], 1, keepdims=True), np.nanmin(L[:, :n], 1, keepdims=True)
        sig[name] = _first(C > hi, C < lo, n)
    inside_prev = (O0 <= pdh) & (O0 >= pdl)
    sig["pdhl"] = _first((C > pdh) & inside_prev, (C < pdl) & inside_prev, 0)
    gap = O0 / pdc - 1
    up = np.zeros_like(C, bool)
    dn = np.zeros_like(C, bool)
    up[:, 2] = (gap[:, 0] >= 0.003) & (C[:, 2] > O[:, 0])
    dn[:, 2] = (gap[:, 0] <= -0.003) & (C[:, 2] < O[:, 0])
    sig["gap_go"] = _first(up, dn, 2)
    sig["gap_fade"] = _first((gap <= -0.004) & (C > O0), (gap >= 0.004) & (C < O0), 2)
    flat_c, flat_h, flat_l = C.ravel(), H.ravel(), L.ravel()
    e9, e21 = _ema(flat_c, 9).reshape(D, 75), _ema(flat_c, 21).reshape(D, 75)
    above = e9 > e21
    was = np.concatenate([above.ravel()[:1], above.ravel()[:-1]]).reshape(D, 75)
    sig["ema_cross"] = _first(above & ~was, ~above & was, 6)
    st = _supertrend(flat_h, flat_l, flat_c).reshape(D, 75)
    stp = np.concatenate([st.ravel()[:1], st.ravel()[:-1]]).reshape(D, 75)
    sig["supertrend"] = _first((st == 1) & (stp == -1), (st == -1) & (stp == 1), 6)
    sig["breadth"] = _first(br >= 0.65, br <= 0.35, 6)
    mv = C / O0 - 1
    sig["momentum"] = _first(mv >= 0.005, mv <= -0.005, 9)
    cw, pw = prev("call_wall"), prev("put_wall")
    between = (O0 < cw) & (O0 > pw)
    sig["wall_break"] = _first((C > cw) & between, (C < pw) & between, 0)
    sig["wall_bounce"] = _first((L <= pw * 1.0015) & (C > O) & (C > pw), (H >= cw * 0.9985) & (C < O) & (C < cw), 3)
    mp = prev("max_pain")
    expiry_today = np.array([mk.expiry_for(d, u) == d if d > 0 else False for d in range(D)])[:, None]
    up = np.zeros_like(C, bool)
    dn = np.zeros_like(C, bool)
    up[:, 9] = (expiry_today[:, 0]) & (C[:, 9] / mp[:, 0] - 1 <= -0.005)
    dn[:, 9] = (expiry_today[:, 0]) & (C[:, 9] / mp[:, 0] - 1 >= 0.005)
    sig["max_pain"] = _first(up, dn, 9)
    twap = np.cumsum((H + L + C) / 3, axis=1) / np.arange(1, 76)
    up = np.zeros_like(C, bool)
    dn = np.zeros_like(C, bool)
    up[:, 44] = (mv[:, 44] >= 0.004) & (C[:, 44] > twap[:, 44])
    dn[:, 44] = (mv[:, 44] <= -0.004) & (C[:, 44] < twap[:, 44])
    sig["late_trend"] = _first(up, dn, 44)
    h2 = np.concatenate([[np.nan, np.nan], np.nanmax(H, 1)[:-2]])[:, None]
    l2 = np.concatenate([[np.nan, np.nan], np.nanmin(L, 1)[:-2]])[:, None]
    inside = (pdh <= h2) & (pdl >= l2)
    sig["inside_break"] = _first((C > pdh) & inside, (C < pdl) & inside, 0)
    P_ = (pdh + pdl + pdc) / 3
    bc = (pdh + pdl) / 2
    tc = 2 * P_ - bc
    top, bot = np.maximum(tc, bc), np.minimum(tc, bc)
    narrow = (top - bot) / P_ < 0.0025
    sig["cpr"] = _first((C > top) & narrow, (C < bot) & narrow, 1)
    r = _rsi(flat_c).reshape(D, 75)
    rp = np.concatenate([r.ravel()[:1], r.ravel()[:-1]]).reshape(D, 75)
    sig["rsi_rev"] = _first((r > 30) & (rp <= 30), (r < 70) & (rp >= 70), 6)
    tp = np.concatenate([twap.ravel()[:1], twap.ravel()[:-1]]).reshape(D, 75)
    cp_ = np.concatenate([C.ravel()[:1], C.ravel()[:-1]]).reshape(D, 75)
    sig["twap_cross"] = _first((C > twap) & (cp_ <= tp) & (mv > 0), (C < twap) & (cp_ >= tp) & (mv < 0), 9)
    for k in sig:                                              # day 0 has no previous close
        s, sd = sig[k]
        s[0], sd[0] = -1, 0
    return sig
