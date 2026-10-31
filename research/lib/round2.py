"""Library round 2: three ways past the cost wall, on the same NIFTY 500 5-minute data.

1. Stocks with big expected moves only (known at 09:20): the 10 or 25 stocks with the most 09:15-09:20 volume
   against their 14-day normal ("stocks in play"), or the 50 with the largest ATR as a % of price (known
   before the open). All 102 book strategies, signal at a bar's close, entry at the next bar's open.
2. Stop-order entries for the 28 breakout strategies: the order rests at the level (known at the previous bar's
   close) and fills when price trades through it (the level, or the bar's open on a gap). Inside the entry bar
   the usual bar path decides a stop touch (green: low then high; red: high then low); a worst-case version
   counts any touch of the stop in that bar as a loss.
3. Agreement: only when 3, 5 or 8 of the 102 strategies (or 3 or 4 different families) give the same stock the
   same direction within the last three bars. Stop: the extreme of the last 5 bars.

Each with the books' confluence filter or none, and three exits: 2R target, trailing stop (break-even at +1R,
then the 3-bar low) or hold to 15:15 behind the stop. Then the day-type routers of research/lib/analyze.py.

    ./research.sh library2
"""
import json
import time

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.factory_equity import _ch
from research.lib import analyze as A
from research.lib import build as B
from research.lib import engine as E
from research.lib import features as F
from research.lib.sectors import sector_index
from research.lib.strategies import (LIBRARY, S as SL, T0930, T0945, T1100, T1200, T1330, T1430, after_range, col, colb,
                                     prev)
from research.sim import FLAT_SLOT
from research.sim import features as sim_features

OUT = DATA_DIR / "research" / "library2"
RES = DATA_DIR / "research" / "results"
UNIVERSES = ["all", "in play 10", "in play 25", "ATR top 50"]
CONTEXTS = ["none", "confluence"]
EXITS = ["target 2R", "trail", "hold"]
AGREE = ["3+ strategies", "5+ strategies", "8+ strategies", "3+ families", "4+ families"]
I = np.arange(data.SLOTS)[None, :]


# ---------------------------------------------------------------- stop-order breakouts: level, condition, stop at bar t
def _orb(name, k, cond=None, stop="mid"):
    def fn(X, s):
        hi_, lo_ = getattr(X, f"{name}_hi"), getattr(X, f"{name}_lo")
        lvl = hi_ if s > 0 else lo_
        st = (hi_ + lo_) / 2 if stop == "mid" else (lo_ if s > 0 else hi_)
        ok = SL >= k - 1
        if cond is not None:
            ok = ok & cond(X, s)
        return lvl, ok, st
    return fn


def _const(level_fn, stop_fn, cond=None, start=0):
    def fn(X, s):
        lvl = np.repeat(col(level_fn(X, s)), data.SLOTS, 1)
        st = stop_fn(X, s)
        ok = (SL >= start) & np.isfinite(lvl)
        if cond is not None:
            ok = ok & cond(X, s)
        return lvl, ok, st * np.ones(X.C.shape)
    return fn


def _dayhigh(cond, start, stop_fn):
    def fn(X, s):
        lvl = X.day_hi if s > 0 else X.day_lo
        return lvl, (SL >= start) & cond(X, s), stop_fn(X, s) * np.ones(X.C.shape)
    return fn


def _gt(a, b, s):
    return a > b if s > 0 else a < b


low5 = lambda X, s: X.low5 if s > 0 else X.high5
STOPS = {}


def _lunch(X, s):
    a, b = T1200 + 1, T1330
    hi_, lo_ = after_range(X.H, a, b, np.max), after_range(X.L, a, b, np.min)
    return (hi_ if s > 0 else lo_), SL >= b, (hi_ + lo_) / 2


def _don(n):
    def fn(X, s):
        lvl = getattr(X, f"hi{n}_incl") if s > 0 else getattr(X, f"lo{n}_incl")
        ok = (SL >= 1) & ((X.adx > 20) if n == 20 else True)
        return lvl, ok, (X.don_lo10 if s > 0 else X.don_hi10)
    return fn


def _keltner(X, s):
    return (X.kc_up if s > 0 else X.kc_lo), SL >= 1, X.ema20


def _acd(X, s):
    a = 0.1 * col(X.atr_d)
    return (X.or15_hi + a if s > 0 else X.or15_lo - a), SL >= 2, (X.or15_hi + X.or15_lo) / 2


def _wvb(X, s):
    lvl = np.repeat(col(X.o0 + s * 0.6 * X.prev_range), data.SLOTS, 1)
    return lvl, SL >= 0, np.repeat(col(X.o0), data.SLOTS, 1)


def _gapgo(X, s):
    g = col(X.gap)
    first = col(np.sign(X.C[:, 0] - X.O[:, 0]))
    ok = (g * s >= 0.01) & (first == s) & (SL >= 0) & (SL <= T1100)
    return (X.or5_hi if s > 0 else X.or5_lo), ok, (X.or5_lo if s > 0 else X.or5_hi)


def _fractal(X, s):
    return (X.swing_hi if s > 0 else X.swing_lo), SL >= 1, (X.swing_lo if s > 0 else X.swing_hi)


BREAKOUTS = {
    "orb5": _orb("or5", 1, stop="far"), "orb15": _orb("or15", 3), "orb30": _orb("or30", 6),
    "orb15_nr7": _orb("or15", 3, cond=lambda X, s: colb(X.lv.nr7)),
    "orb15_inside": _orb("or15", 3, cond=lambda X, s: colb(X.lv.inside)),
    "acd": _acd, "williams_vb": _wvb, "gap_go": _gapgo,
    "pdh_break": _const(lambda X, s: X.f.prev_high if s > 0 else X.f.prev_low, low5,
                        cond=lambda X, s: _gt(col(X.f.prev_high if s > 0 else X.f.prev_low), col(X.o0), s)),
    "pivot_break": _const(lambda X, s: X.lv.R1 if s > 0 else X.lv.S1, lambda X, s: np.repeat(col(X.lv.P), data.SLOTS, 1)),
    "cpr_narrow": _const(lambda X, s: X.lv.TC if s > 0 else X.lv.BC,
                         lambda X, s: np.repeat(col(X.lv.BC if s > 0 else X.lv.TC), data.SLOTS, 1),
                         cond=lambda X, s: colb(X.lv.cpr_w < 0.0025), start=2),
    "cam_break": _const(lambda X, s: X.lv.H4 if s > 0 else X.lv.L4,
                        lambda X, s: np.repeat(col(X.lv.H3 if s > 0 else X.lv.L3), data.SLOTS, 1)),
    "week_break": _const(lambda X, s: X.lv.wk_high if s > 0 else X.lv.wk_low, low5),
    "fractal_break": _fractal,
    "late_high": _dayhigh(lambda X, s: np.ones(X.C.shape, bool), T1200, lambda X, s: X.vwap),
    "donchian": _don(20), "donchian55": _don(55), "keltner": _keltner,
    "in_play": _dayhigh(lambda X, s: (np.nan_to_num(X.crv) >= 2.5) & _gt(X.C, X.vwap, s), T0930, lambda X, s: X.vwap),
    "sector_leader": _dayhigh(lambda X, s: ((X.sector_rank <= 2) if s > 0 else (X.sector_rank >= 13)) & _gt(X.C, X.vwap, s),
                              T0945, lambda X, s: X.vwap),
    "lunch": _lunch,
    "adx_high": _dayhigh(lambda X, s: (X.adx > 25) & (X.adx > prev(X.adx, 3)), T0945, low5),
    "orb_vwap_side": _orb("or15", 3, cond=lambda X, s: _gt(X.C, X.vwap, s)),
    "orb_rsi": _orb("or15", 3, cond=lambda X, s: (X.rsi14 > 55) if s > 0 else (X.rsi14 < 45)),
    "orb5_st": _orb("or5", 1, cond=lambda X, s: X.st_dir == s, stop="far"),
    "orb_cpr": _orb("or15", 3, cond=lambda X, s: colb(X.lv.cpr_w < 0.0035)),
    "gap_vwap": _orb("or15", 3, cond=lambda X, s: (col(X.gap) * s >= 0.005) & _gt(X.C, X.vwap, s)),
    "cam_orb": _orb("or15", 3, cond=lambda X, s: _gt(X.C, col(X.lv.H3 if s > 0 else X.lv.L3), s)),
}


# ---------------------------------------------------------------- variant registry
def _registry():
    rows = []
    names = {s.key: s for s in LIBRARY}
    for s in LIBRARY:
        for u in UNIVERSES[1:]:
            for c in CONTEXTS:
                for e in EXITS:
                    rows.append({"idea": "Big expected moves", "key": s.key, "name": s.name, "family": s.family,
                                 "universe": u, "context": c, "exit": e, "entry": "next open"})
    for k in BREAKOUTS:
        for u in UNIVERSES:
            for c in CONTEXTS:
                for e in EXITS:
                    for eb in ("path", "worst"):
                        rows.append({"idea": "Stop-order entries", "key": k, "name": names[k].name, "family": names[k].family,
                                     "universe": u, "context": c, "exit": e, "entry": f"stop order ({eb})"})
    for a in AGREE:
        for u in UNIVERSES:
            for c in CONTEXTS:
                for e in EXITS:
                    rows.append({"idea": "Agreement", "key": a, "name": f"Agreement: {a}", "family": "Agreement",
                                 "universe": u, "context": c, "exit": e, "entry": "next open"})
    for i, r in enumerate(rows):
        r["v"] = i
    return rows


VARIANTS = _registry()
VID = {(r["idea"], r["key"], r["universe"], r["context"], r["exit"], r["entry"]): r["v"] for r in VARIANTS}
NV2 = len(VARIANTS)


# ---------------------------------------------------------------- exits shared by both entry kinds
def _exits(X, rows, es, sides, entry, stop, exit_kind, eb_hit=None):
    """Bars from the entry bar on. eb_hit: stop already hit inside the entry bar (stop orders)."""
    O, H, L, C = X.O[rows], X.H[rows], X.L[rows], X.C[rows]
    n = len(rows)
    r = np.arange(n)
    risk = (entry - stop) * sides
    sd = sides[:, None]
    first = es + (1 if eb_hit is not None else 0)          # a stop-order fill: checks resume on the next bar
    live = (I >= first[:, None]) & (I <= FLAT_SLOT)
    if exit_kind.startswith("target"):
        tgt = entry + float(exit_kind.split()[1].rstrip("R")) * sides * risk
        st_mat = np.repeat(stop[:, None], data.SLOTS, 1)
        tg_hit = live & np.where(sd > 0, H >= tgt[:, None], L <= tgt[:, None])
    else:
        tgt = np.full(n, np.nan)
        st_mat = np.repeat(stop[:, None], data.SLOTS, 1)
        if exit_kind == "trail":
            one = entry + sides * risk
            reach = (I >= es[:, None]) & (I <= FLAT_SLOT) & np.where(sd > 0, H >= one[:, None], L <= one[:, None])
            b1 = np.where(reach.any(1), reach.argmax(1), 999)
            trail = np.where(sd > 0, np.fmax(entry[:, None], X.prior_low3[rows]), np.fmin(entry[:, None], X.prior_high3[rows]))
            st_mat = np.where(I > b1[:, None], np.where(sd > 0, np.fmax(stop[:, None], trail), np.fmin(stop[:, None], trail)),
                              stop[:, None])
            st_mat = np.where(sd > 0, np.maximum.accumulate(np.where(live, st_mat, -np.inf), 1),
                              -np.maximum.accumulate(np.where(live, -st_mat, -np.inf), 1))
        tg_hit = np.zeros_like(live)
    st_hit = live & np.where(sd > 0, L <= st_mat, H >= st_mat)
    fs = np.where(st_hit.any(1), st_hit.argmax(1), 999)
    ft = np.where(tg_hit.any(1), tg_hit.argmax(1), 999)
    out = np.where((fs <= ft) & (fs < 999), -1, np.where(ft < 999, 1, 0))
    xs = np.where(out == -1, fs, np.where(out == 1, ft, FLAT_SLOT))
    o_at = O[r, np.minimum(xs, data.SLOTS - 1)]
    stv = st_mat[r, np.minimum(xs, data.SLOTS - 1)]
    stop_px = np.where(sides > 0, np.minimum(stv, o_at), np.maximum(stv, o_at))
    stop_px = np.where(xs == es, stv, stop_px)
    tgt_px = np.where(sides > 0, np.maximum(tgt, o_at), np.minimum(tgt, o_at))
    tgt_px = np.where(xs == es, tgt, tgt_px)
    exit_px = np.where(out == -1, stop_px, np.where(out == 1, tgt_px, C[r, FLAT_SLOT]))
    if eb_hit is not None:                       # stopped inside the entry bar
        out = np.where(eb_hit, -1, out)
        xs = np.where(eb_hit, es, xs)
        exit_px = np.where(eb_hit, stop, exit_px)
    return out, xs, exit_px


def _rnet(entry, stop, exit_px, out, sides, costs):
    k = E.SLIP_BPS / 10000.0
    fin = entry * (1 + k * sides)
    fout = np.where(out == 1, exit_px, exit_px * (1 - k * sides))
    rf = (fin - stop) * sides
    qty = np.floor(np.minimum(E.RISK_INR / np.maximum(rf, 1e-9), E.MAX_POS / fin))
    good = (qty >= 1) & (rf > 0)
    gross = (fout - fin) * qty * sides
    ch = _ch(fin * qty, sides > 0, costs) + _ch(fout * qty, sides < 0, costs)
    return np.where(good, (gross - ch) / np.maximum(rf * qty, 1e-9), np.nan)


def _records(v, rows, es, xs, sides, out, entry, stop, exit_px, rnet, score, day_index, stock_id):
    ok = np.isfinite(rnet)
    rec = np.empty(int(ok.sum()), dtype=E.REC)
    rec["v"], rec["day"], rec["stock"] = v, day_index[rows[ok]], stock_id
    rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = es[ok], xs[ok], sides[ok], out[ok]
    rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"], rec["score"] = entry[ok], stop[ok], exit_px[ok], rnet[ok], score[ok]
    return rec


def _risk_ok(entry, stop, sides, es):
    rp = (entry - stop) * sides / entry
    return np.isfinite(rp) & (rp >= E.MIN_RISK) & (rp <= E.MAX_RISK) & (es < FLAT_SLOT)


# ---------------------------------------------------------------- one stock
def process(X, stock_id, day_index, train, costs, uni_masks, ctx_masks):
    agg = np.zeros((NV2, 2, 5))
    parts = []
    W = (I >= 1) & (I <= T1430)

    def add(v, rows, es, sides, entry, stop, out, xs, exit_px):
        rn = _rnet(entry, stop, exit_px, out, sides, costs)
        score = np.nan_to_num(X.rv[rows, np.maximum(es - 1, 0)], nan=0.0)
        rec = _records(v, rows, es, xs, sides, out, entry, stop, exit_px, rn, score, day_index, stock_id)
        if len(rec):
            per = train[rows[np.isfinite(rn)]]
            q = rec["rnet"]
            for pi, pm in ((0, per), (1, ~per)):
                agg[v, pi] += (pm.sum(), (q[pm] > 0).sum(), q[pm].sum(), (q[pm] ** 2).sum(), 0.0)
            parts.append(rec)

    def close_entries(L, Sx, stL, stS, umask, cname):
        cL, cS = ctx_masks[1][cname], ctx_masks[-1][cname]
        aL = L & umask & (True if cL is None else cL)
        aS = Sx & umask & (True if cS is None else cS)
        rows, slots, sides, stop = E.first_signals(aL, aS, stL, stS)
        if len(rows) == 0:
            return None
        es = slots + 1
        entry = X.O[rows, es]
        ok = _risk_ok(entry, stop, sides, es)
        return rows[ok], es[ok], sides[ok], entry[ok], stop[ok]

    # 1. big expected moves + collect every strategy's raw signals for the agreement counts
    nL = np.zeros(X.C.shape)
    nS = np.zeros(X.C.shape)
    famL, famS = {}, {}
    for s in LIBRARY:
        L, Sx, stL, stS = s.fn(X)
        stL = np.asarray(stL, float) * np.ones(X.C.shape)
        stS = np.asarray(stS, float) * np.ones(X.C.shape)
        L = np.asarray(L, bool) & W & np.isfinite(stL)
        Sx = np.asarray(Sx, bool) & W & np.isfinite(stS)
        recentL = L | prev(L.astype(float)).astype(bool) | prev(L.astype(float), 2).astype(bool)
        recentS = Sx | prev(Sx.astype(float)).astype(bool) | prev(Sx.astype(float), 2).astype(bool)
        nL += recentL
        nS += recentS
        famL[s.family] = famL.get(s.family, np.zeros(X.C.shape, bool)) | recentL
        famS[s.family] = famS.get(s.family, np.zeros(X.C.shape, bool)) | recentS
        for u in UNIVERSES[1:]:
            for c in CONTEXTS:
                got = close_entries(L, Sx, stL, stS, uni_masks[u], c)
                if got is None or len(got[0]) == 0:
                    continue
                rows, es, sides, entry, stop = got
                for e in EXITS:
                    out, xs, exit_px = _exits(X, rows, es, sides, entry, stop, e)
                    add(VID[("Big expected moves", s.key, u, c, e, "next open")], rows, es, sides, entry, stop, out, xs, exit_px)
    # 3. agreement
    fL = sum(m.astype(int) for m in famL.values())
    fS = sum(m.astype(int) for m in famS.values())
    rules = {"3+ strategies": (nL >= 3, nS >= 3), "5+ strategies": (nL >= 5, nS >= 5), "8+ strategies": (nL >= 8, nS >= 8),
             "3+ families": (fL >= 3, fS >= 3), "4+ families": (fL >= 4, fS >= 4)}
    for a, (gL, gS) in rules.items():
        gL, gS = gL & W & ~(nS > 0), gS & W & ~(nL > 0)          # one-sided agreement only
        for u in UNIVERSES:
            for c in CONTEXTS:
                got = close_entries(gL, gS, X.low5, X.high5, uni_masks[u], c)
                if got is None or len(got[0]) == 0:
                    continue
                rows, es, sides, entry, stop = got
                for e in EXITS:
                    out, xs, exit_px = _exits(X, rows, es, sides, entry, stop, e)
                    add(VID[("Agreement", a, u, c, e, "next open")], rows, es, sides, entry, stop, out, xs, exit_px)
    # 2. stop orders: level, condition and stop known at bar t-1, order works during bar t
    for k, fn in BREAKOUTS.items():
        lvL, okL, stL = (np.asarray(a_, float) * np.ones(X.C.shape) for a_ in fn(X, 1))
        lvS, okS, stS = (np.asarray(a_, float) * np.ones(X.C.shape) for a_ in fn(X, -1))
        pL, pS = prev(lvL), prev(lvS)
        oL, oS = prev(okL).astype(bool), prev(okS).astype(bool)
        sL, sS = prev(stL), prev(stS)
        win = (I >= 1) & (I <= T1430 + 1)
        trigL = win & oL & np.isfinite(pL) & (X.H > pL) & (prev(X.C) <= pL)      # not already beyond at the last close
        trigS = win & oS & np.isfinite(pS) & (X.L < pS) & (prev(X.C) >= pS)
        for u in UNIVERSES:
            for c in CONTEXTS:
                cL, cS = ctx_masks[1][c], ctx_masks[-1][c]
                aL = trigL & uni_masks[u] & (True if cL is None else prev(cL.astype(float)).astype(bool))
                aS = trigS & uni_masks[u] & (True if cS is None else prev(cS.astype(float)).astype(bool))
                kL, kS = F.first_bar_from(aL), F.first_bar_from(aS)
                side = np.where((kL >= 0) & ((kS < 0) | (kL < kS)), 1, np.where((kS >= 0) & ((kL < 0) | (kS < kL)), -1, 0))
                rows = np.flatnonzero(side != 0)
                if len(rows) == 0:
                    continue
                sides = side[rows]
                es = np.where(sides > 0, kL[rows], kS[rows])
                lvl = np.where(sides > 0, pL[rows, es], pS[rows, es])
                o = X.O[rows, es]
                entry = np.where(sides > 0, np.maximum(lvl, o), np.minimum(lvl, o))
                stop = np.where(sides > 0, sL[rows, es], sS[rows, es])
                ok = _risk_ok(entry, stop, sides, es)
                rows, sides, es, entry, stop = rows[ok], sides[ok], es[ok], entry[ok], stop[ok]
                if len(rows) == 0:
                    continue
                Ob, Hb, Lb, Cb = X.O[rows, es], X.H[rows, es], X.L[rows, es], X.C[rows, es]
                path = np.where(sides > 0, (Cb <= stop) | ((Ob > Cb) & (Lb <= stop)), (Cb >= stop) | ((Ob < Cb) & (Hb >= stop)))
                worst = np.where(sides > 0, Lb <= stop, Hb >= stop)
                for e in EXITS:
                    for eb, hit in (("path", path), ("worst", worst)):
                        out, xs, exit_px = _exits(X, rows, es, sides, entry, stop, e, eb_hit=hit)
                        add(VID[("Stop-order entries", k, u, c, e, f"stop order ({eb})")], rows, es, sides, entry, stop, out, xs, exit_px)
    recs = np.concatenate(parts) if parts else np.empty(0, dtype=E.REC)
    return recs, agg


# ---------------------------------------------------------------- the run
_G = {}


def _prepass(sym):
    m = data.load(sym)
    if m is None or m.n < 260:
        return sym, None
    v0 = m.v[:, 0]
    cs = np.cumsum(np.r_[0.0, v0])
    avg = np.full(m.n, np.nan)
    avg[14:] = (cs[14:-1] - cs[:-15]) / 14.0
    rv5 = v0 / np.where(avg > 0, avg, np.nan)
    dc, dh, dl = m.c[:, -1], m.h.max(1), m.l.min(1)
    pc = np.r_[np.nan, dc[:-1]]
    tr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc)))
    atr = np.r_[np.nan, np.array([np.nanmean(tr[max(0, i - 14):i]) if i >= 14 else np.nan for i in range(1, m.n)])]
    turn = np.r_[np.nan, np.convolve(dc * m.v.sum(1), np.ones(20) / 20, "full")[:m.n - 1]]
    return sym, ([str(d) for d in m.dates], rv5, atr / pc, turn, pc)


def _ranks(pre, master):
    mpos = {str(d): i for i, d in enumerate(master)}
    rv = np.full((len(pre), len(master)), np.nan)
    at = np.full((len(pre), len(master)), np.nan)
    syms = list(pre)
    for k, s in enumerate(syms):
        ds, rv5, atrp, turn, pc = pre[s]
        idx = np.array([mpos[d] for d in ds if d in mpos])
        keep = np.array([d in mpos for d in ds])
        liquid = (np.nan_to_num(turn) >= B.MIN_TURNOVER) & (np.nan_to_num(pc) >= B.MIN_PRICE)
        rv[k, idx] = np.where(liquid[keep], rv5[keep], np.nan)
        at[k, idx] = np.where(liquid[keep], atrp[keep], np.nan)
    out = {}
    rr = np.argsort(np.argsort(-np.nan_to_num(rv, nan=-1), axis=0), axis=0) + 1
    ra = np.argsort(np.argsort(-np.nan_to_num(at, nan=-1), axis=0), axis=0) + 1
    rr = np.where(np.isnan(rv), 9999, rr)
    ra = np.where(np.isnan(at), 9999, ra)
    for k, s in enumerate(syms):
        out[s] = (rr[k], ra[k])
    return out


def _chunk(syms):
    N, master, sectors, ranks, industries, costs, xs = (_G[k] for k in ("N", "master", "sectors", "ranks", "industries", "costs", "xs"))
    mpos = {d: i for i, d in enumerate(master)}
    kept, agg = [], np.zeros((NV2, 2, 5))
    for sid, sym in syms:
        if sym not in xs:
            continue
        m = data.load(sym)
        keep = np.array([d in mpos for d in m.dates])
        m = data.DayMatrix(m.symbol, m.dates[keep], m.o[keep], m.h[keep], m.l[keep], m.c[keep], m.v[keep], m.start[keep])
        f = sim_features(m)
        sec = sector_index(sym, industries.get(sym, ""))
        rows_m = np.array([mpos[d] for d in m.dates])
        ctx = {"nifty": B._rows_for(N, m.dates), "beta": B._beta(m, N)}
        if sec in sectors:
            ctx["sector"] = B._rows_for(sectors[sec], m.dates)
        if sec in ranks:
            ctx["sector_rank"] = ranks[sec][rows_m]
        X = F.build(m, f, ctx)
        liquid = (np.nan_to_num(X.lv.turn20) >= B.MIN_TURNOVER) & (np.nan_to_num(f.prev_close) >= B.MIN_PRICE) & f.valid
        rv_rank, atr_rank = xs[sym]
        rv_rank, atr_rank = rv_rank[rows_m], atr_rank[rows_m]
        L2 = liquid[:, None] & np.ones(X.C.shape, bool)
        uni = {"all": L2, "in play 10": L2 & (rv_rank <= 10)[:, None], "in play 25": L2 & (rv_rank <= 25)[:, None],
               "ATR top 50": L2 & (atr_rank <= 50)[:, None]}
        fm = E.filter_masks(X)
        ctxm = {s: {"none": None, "confluence": fm[s]["confluence"]} for s in (1, -1)}
        recs, a = process(X, sid, rows_m.astype(np.int16), m.dates <= B.TRAIN_END, costs, uni, ctxm)
        agg += a
        kept.append(recs)
        if len(kept) >= 4:
            kept = [E.keep_earliest(np.concatenate(kept))]
    recs = E.keep_earliest(np.concatenate(kept)) if kept else np.empty(0, dtype=E.REC)
    return recs, agg


def build(log=print, workers=8):
    import multiprocessing as mp
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    N, master, sectors, ranks = B.context()
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    with mp.get_context("fork").Pool(workers) as pool:
        pre = {s: r for s, r in pool.map(_prepass, syms, chunksize=8) if r is not None}
    xs = _ranks(pre, master)
    log(f"  round 2: {NV2:,} variants, {len(xs)} stocks ranked ({time.time() - t0:.0f}s)")
    _G.update(N=N, master=master, sectors=sectors, ranks=ranks, industries=industries, costs=Settings().get("costs"), xs=xs)
    ids = [(i, s) for i, s in enumerate(syms) if s in xs]
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    kept = np.empty(0, dtype=E.REC)
    agg = np.zeros((NV2, 2, 5))
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (r, a) in enumerate(pool.imap_unordered(_chunk, chunks), 1):
            kept = E.keep_earliest(np.concatenate([kept, r]))
            agg += a
            if k % 8 == 0 or k == len(chunks):
                log(f"  {k}/{len(chunks)} chunks, {len(kept):,} kept trades ({time.time() - t0:.0f}s)")
    np.savez_compressed(OUT / "trades.npz", recs=kept)
    np.save(OUT / "agg.npy", agg)
    (OUT / "meta.json").write_text(json.dumps({"variants": VARIANTS, "dates": [str(d) for d in master],
                                               "train_end": str(B.TRAIN_END), "stocks": len(ids),
                                               "built_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}))
    log(f"  saved ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- analysis
def analyze(log=print):
    meta = json.loads((OUT / "meta.json").read_text())
    recs = np.load(OUT / "trades.npz")["recs"]
    agg = np.load(OUT / "agg.npy")
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    nd = len(dates)
    train_end = np.datetime64(meta["train_end"])
    train = dates <= train_end
    order = np.argsort(recs["v"], kind="stable")
    rs = recs[order]
    vb = np.searchsorted(rs["v"], np.arange(NV2 + 1))
    rows, daily_by = [], {}
    side = rs["side"].astype(float)
    raw_all = (rs["exit_px"] - rs["entry"]) * side / ((rs["entry"] - rs["stop"]) * side)
    for v, meta_v in enumerate(VARIANTS):
        a = agg[v]
        row = dict(meta_v)
        for pi, pn in ((0, "train"), (1, "test")):
            n, w, sr, sr2 = a[pi, 0], a[pi, 1], a[pi, 2], a[pi, 3]
            mean = sr / n if n else 0.0
            sd = np.sqrt(max(sr2 / n - mean * mean, 1e-12)) if n else 0.0
            row[f"{pn}_trades"], row[f"{pn}_win"] = int(n), (round(100 * w / n, 1) if n else None)
            row[f"{pn}_avgR"], row[f"{pn}_t"] = round(mean, 4), (round(mean / sd * np.sqrt(n), 2) if n else 0.0)
        seg = slice(vb[v], vb[v + 1])
        row["raw_R"] = round(float(np.nanmean(raw_all[seg])), 4) if vb[v + 1] > vb[v] else None
        daily, _ = A.account(rs[seg], nd)
        daily_by[v] = daily
        row.update({f"acct_{k}": val for k, val in A.period_stats(daily, dates, train_end).items()})
        rows.append(row)
        if v % 500 == 0:
            log(f"  accounts {v}/{NV2}")
    # routers over every round-2 variant (the worst-case stop-order versions are stress tests, not candidates)
    cand = np.array([not r["entry"].endswith("(worst)") for r in VARIANTS])
    rr = recs[cand[recs["v"]]]
    br = dict(np.load(DATA_DIR / "research" / "library" / "breadth.npz"))
    Fd = A.day_features(dates, br)
    ot, dtp = A.open_type(Fd), A.day_type(Fd)
    units, ud, us, uc, rec_unit = A.unit_day_table(rr, nd)
    early = A.select_by_type(units, ud, us, uc, ot, train, len(A.OPEN_TYPES), timing=0)
    late = A.select_by_type(units, ud, us, uc, dtp, train, len(A.DAY_TYPES), timing=1)
    day_units = {d: set(early.get(int(ot[d]), [])) | set(late.get(int(dtp[d]), [])) for d in range(nd)}
    router_daily, router_n = A.account(A.routed_recs(rr, rec_unit, day_units), nd)
    label = lambda u: (lambda r: f"{r['name']} · {r['universe']} · {r['context']} · {r['exit']} · {r['entry']}")(VARIANTS[u // 2])
    picks = [{"type": t, "picked": [label(u) for u in late.get(i, [])]} for i, t in enumerate(A.DAY_TYPES)]
    picks_open = [{"type": t, "picked": [label(u) for u in early.get(i, [])]} for i, t in enumerate(A.OPEN_TYPES)]
    # each idea's best variant, chosen on 2021-24 by the account result
    best = {}
    for idea in ("Big expected moves", "Stop-order entries", "Agreement"):
        pool_ = [r for r in rows if r["idea"] == idea and not r["entry"].endswith("(worst)") and r["train_trades"] >= 200]
        top = sorted(pool_, key=lambda r: -r["acct_train"]["per_month"])[:10]
        best[idea] = top
    # the worst-case twin of each stop-order finalist
    twins = {}
    for r in best["Stop-order entries"]:
        w = VID[("Stop-order entries", r["key"], r["universe"], r["context"], r["exit"], "stop order (worst)")]
        twins[r["v"]] = {k: rows[w][k] for k in ("acct_train", "acct_test", "train_avgR", "test_avgR")}
    approaches = [("Day-type router over all three ideas", router_daily, router_n)]
    for idea, top in best.items():
        if top:
            approaches.append((f"Best '{idea}' version (picked on 2021-24)", daily_by[top[0]["v"]], top[0]["acct_train"]))
    appr = [{"name": n, **A.period_stats(dly, dates, train_end), "curve": [round(float(x), 0) for x in np.cumsum(dly)]}
            for n, dly, _ in approaches]
    nonworst = [r for r in rows if not r["entry"].endswith("(worst)")]
    summary = {"built_at": meta["built_at"], "stocks": meta["stocks"], "variants": NV2, "train_end": meta["train_end"],
               "dates": [str(dates[0]), str(dates[-1])], "dates_axis": [str(d) for d in dates],
               "approaches": appr, "best": best, "twins": twins, "picks": picks, "picks_open": picks_open,
               "positive_train": sum(r["train_avgR"] > 0 for r in nonworst if r["train_trades"] >= 100),
               "positive_test": sum(r["test_avgR"] > 0 for r in nonworst if r["test_trades"] >= 50),
               "positive_both": sum(r["train_avgR"] > 0 and r["test_avgR"] > 0 for r in nonworst
                                    if r["train_trades"] >= 100 and r["test_trades"] >= 50),
               "acct_20k_both": sum(r["acct_train"]["per_month"] >= 20000 and r["acct_test"]["per_month"] >= 20000 for r in nonworst),
               "by_idea": {idea: {"variants": sum(1 for r in nonworst if r["idea"] == idea),
                                  "raw_positive": sum(1 for r in nonworst if r["idea"] == idea and (r["raw_R"] or 0) > 0),
                                  "net_positive_both": sum(1 for r in nonworst if r["idea"] == idea and r["train_avgR"] > 0
                                                           and r["test_avgR"] > 0 and r["train_trades"] >= 100 and r["test_trades"] >= 50)}
                           for idea in ("Big expected moves", "Stop-order entries", "Agreement")}}
    (RES / "library2_summary.json").write_text(json.dumps(summary, default=str))
    (RES / "library2_variants.json").write_text(json.dumps(rows, default=str))
    log("  round 2 analysis written")
    return summary, rows
