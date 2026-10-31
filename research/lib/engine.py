"""Signals to trades for the strategy library, one stock at a time.

A signal is decided at a bar's close with only past data; the trade enters at the NEXT bar's open. Exits use the
bars after the entry: the strategy's protective stop, then either a 2R target or a trailing stop (break-even at
+1R, then the low of the 3 bars before), and everything is out at 15:15. Inside one bar, a stop and a target
both touched count as the stop (the cautious reading). One trade per strategy variant per stock per day: the
first signal that passes the variant's filter.

Filters (confluence, the books' "trade with the context"):
  none | volume: the signal bar's volume 1.5x normal for that time of day
  sector: the stock's sectoral index is up (down) on the day and above (below) its intraday average
  market: the same for the NIFTY | htf: the daily trend (close > 20 DMA > 50 DMA for a long)
  room: the next support / resistance in the trade's way is at least 2R away
  confluence: volume + sector + market
"""
import numpy as np

from research.data import SLOTS
from research.factory_equity import _ch
from research.lib import features as F
from research.lib.strategies import LIBRARY, T1430
from research.sim import FLAT_SLOT

FILTERS = ["none", "volume", "sector", "market", "htf", "room", "confluence"]
EXITS = ["target 2R", "trail"]
NV = len(LIBRARY) * len(FILTERS) * len(EXITS)
MIN_RISK, MAX_RISK = 0.002, 0.025
RISK_INR, MAX_POS, SLIP_BPS = 1000.0, 250000.0, 3.0
KEEP_PER_DAY = 5
I = np.arange(SLOTS)[None, :]
REC = [("v", np.int16), ("day", np.int16), ("stock", np.int16), ("slot", np.int8), ("exit_slot", np.int8),
       ("side", np.int8), ("outcome", np.int8), ("entry", np.float32), ("stop", np.float32), ("exit_px", np.float32),
       ("score", np.float32), ("rnet", np.float32)]


def vid(si, fi, ei):
    return (si * len(FILTERS) + fi) * len(EXITS) + ei


def unvid(v):
    ei = v % len(EXITS)
    fi = (v // len(EXITS)) % len(FILTERS)
    si = v // (len(EXITS) * len(FILTERS))
    return si, fi, ei


def filter_masks(X):
    vol = np.nan_to_num(X.rv) >= 1.5
    out = {}
    for s in (1, -1):
        sec = (s * X.sector_ret > 0) & (X.sector_above if s > 0 else X.sector_below)
        mkt = (s * X.nifty_ret > 0) & (X.nifty_above if s > 0 else X.nifty_below)
        htf = np.repeat(np.asarray(X.trend_up if s > 0 else X.trend_dn, bool)[:, None], SLOTS, 1)
        out[s] = {"none": None, "volume": vol, "sector": sec, "market": mkt, "htf": htf, "room": "room",
                  "confluence": vol & sec & mkt}
    return out


def room_levels(X):
    """Nearest resistance above / support below each close (daily levels and the day's range so far)."""
    lv, f = X.lv, X.f
    ups = [f.prev_high, lv.R1, lv.R2, lv.H3, lv.H4, lv.wk_high]
    dns = [f.prev_low, lv.S1, lv.S2, lv.L3, lv.L4, lv.wk_low]
    C = X.C
    res = np.full(C.shape, np.inf)
    sup = np.full(C.shape, -np.inf)
    for a in ups:
        a2 = np.asarray(a, float)[:, None]
        res = np.where((a2 > C) & (a2 < res), a2, res)
    for a in dns:
        a2 = np.asarray(a, float)[:, None]
        sup = np.where((a2 < C) & (a2 > sup), a2, sup)
    res = np.where((X.prior_day_hi > C) & (X.prior_day_hi < res), X.prior_day_hi, res)
    sup = np.where((X.prior_day_lo < C) & (X.prior_day_lo > sup), X.prior_day_lo, sup)
    return res, sup


def first_signals(L, Sx, stL, stS):
    """The first long or short bar per session (both on the same bar: skipped)."""
    kL, kS = F.first_bar_from(L), F.first_bar_from(Sx)
    side = np.where((kL >= 0) & ((kS < 0) | (kL < kS)), 1, np.where((kS >= 0) & ((kL < 0) | (kS < kL)), -1, 0))
    rows = np.flatnonzero(side != 0)
    slots = np.where(side[rows] > 0, kL[rows], kS[rows])
    sd = side[rows]
    stop = np.where(sd > 0, stL[rows, slots], stS[rows, slots])
    return rows, slots, sd, stop


def simulate(X, rows, slots, sides, stop, exit_kind, costs):
    es = slots + 1
    O, H, L, C = X.O[rows], X.H[rows], X.L[rows], X.C[rows]
    n = len(rows)
    r = np.arange(n)
    entry = O[r, es]
    risk = (entry - stop) * sides
    rp = risk / entry
    ok = np.isfinite(rp) & (rp >= MIN_RISK) & (rp <= MAX_RISK) & (es < FLAT_SLOT)
    if not ok.any():
        return None
    rows, es, sides, stop, entry, risk = rows[ok], es[ok], sides[ok], stop[ok], entry[ok], risk[ok]
    O, H, L, C = O[ok], H[ok], L[ok], C[ok]
    n = len(rows)
    r = np.arange(n)
    sd = sides[:, None]
    live = (I >= es[:, None]) & (I <= FLAT_SLOT)
    if exit_kind == "target 2R":
        tgt = entry + 2 * sides * risk
        st_mat = np.repeat(stop[:, None], SLOTS, 1)
        tg_hit = live & np.where(sd > 0, H >= tgt[:, None], L <= tgt[:, None])
    else:
        tgt = np.full(n, np.nan)
        one = entry + sides * risk
        reach = live & np.where(sd > 0, H >= one[:, None], L <= one[:, None])
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
    o_at = O[r, np.minimum(xs, SLOTS - 1)]
    stv = st_mat[r, np.minimum(xs, SLOTS - 1)]
    stop_px = np.where(sides > 0, np.minimum(stv, o_at), np.maximum(stv, o_at))
    stop_px = np.where(xs == es, stv, stop_px)                  # the entry bar: entered at its open, beyond the stop
    tgt_px = np.where(sides > 0, np.maximum(tgt, o_at), np.minimum(tgt, o_at))
    tgt_px = np.where(xs == es, tgt, tgt_px)
    exit_px = np.where(out == -1, stop_px, np.where(out == 1, tgt_px, C[r, FLAT_SLOT]))
    # rupees on a Rs 1,000-risk trade, charges and slippage included
    k = SLIP_BPS / 10000.0
    fin = entry * (1 + k * sides)
    fout = np.where(out == 1, exit_px, exit_px * (1 - k * sides))
    rf = (fin - stop) * sides
    qty = np.floor(np.minimum(RISK_INR / np.maximum(rf, 1e-9), MAX_POS / fin))
    good = (qty >= 1) & (rf > 0)
    gross = (fout - fin) * qty * sides
    ch = _ch(fin * qty, sides > 0, costs) + _ch(fout * qty, sides < 0, costs)
    rnet = np.where(good, (gross - ch) / np.maximum(rf * qty, 1e-9), 0.0)
    return {"rows": rows[good], "es": es[good], "xs": xs[good], "sides": sides[good], "entry": entry[good],
            "stop": stop[good], "exit_px": exit_px[good], "out": out[good], "rnet": rnet[good]}


def process_stock(m, f, X, stock_id, day_index, train_mask, costs, liquid):
    """All variants for one stock. Returns (records structured array, aggregates [NV, 2, 5])."""
    W = (I >= 1) & (I <= T1430) & np.asarray(liquid, bool)[:, None]
    fm = filter_masks(X)
    res, sup = room_levels(X)
    agg = np.zeros((NV, 2, 5))
    parts = []
    for si, strat in enumerate(LIBRARY):
        try:
            L, Sx, stL, stS = strat.fn(X)
        except Exception as e:                    # a broken rule must not stop the run: it trades nothing
            raise RuntimeError(f"{strat.key}: {e}") from e
        stL = np.asarray(stL, float) * np.ones(X.C.shape)
        stS = np.asarray(stS, float) * np.ones(X.C.shape)
        L = np.asarray(L, bool) & W & np.isfinite(stL)
        Sx = np.asarray(Sx, bool) & W & np.isfinite(stS)
        if not (L.any() or Sx.any()):
            continue
        for fi, fname in enumerate(FILTERS):
            mL, mS = fm[1][fname], fm[-1][fname]
            if fname == "room":
                aL = L & ((res - X.C) >= 2 * (X.C - stL))
                aS = Sx & ((X.C - sup) >= 2 * (stS - X.C))
            elif mL is None:
                aL, aS = L, Sx
            else:
                aL, aS = L & mL, Sx & mS
            rows, slots, sides, stop = first_signals(aL, aS, stL, stS)
            if len(rows) == 0:
                continue
            for ei, ex in enumerate(EXITS):
                t = simulate(X, rows, slots, sides, stop, ex, costs)
                if t is None or len(t["rows"]) == 0:
                    continue
                v = vid(si, fi, ei)
                per = train_mask[t["rows"]]
                for pi, pm in ((0, per), (1, ~per)):
                    q = t["rnet"][pm]
                    agg[v, pi] += (len(q), (q > 0).sum(), q.sum(), (q * q).sum(), 0.0)
                rec = np.empty(len(t["rows"]), dtype=REC)
                rec["v"], rec["day"], rec["stock"] = v, day_index[t["rows"]], stock_id
                rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = t["es"], t["xs"], t["sides"], t["out"]
                rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"] = t["entry"], t["stop"], t["exit_px"], t["rnet"]
                rec["score"] = np.nan_to_num(X.rv[t["rows"], t["es"] - 1], nan=0.0)
                parts.append(rec)
    recs = np.concatenate(parts) if parts else np.empty(0, dtype=REC)
    return recs, agg


def keep_earliest(recs, k=KEEP_PER_DAY):
    """Per (variant, day): the k earliest trades (higher relative volume first on the same bar)."""
    if len(recs) == 0:
        return recs
    order = np.lexsort((-recs["score"], recs["slot"], recs["day"], recs["v"]))
    r = recs[order]
    key = r["v"].astype(np.int64) * 100000 + r["day"]
    start = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    rank = np.arange(len(r)) - first[gid]
    return r[rank < k]
