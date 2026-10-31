"""Strong support / resistance breakouts on the NIFTY 500 (round 7).

Every evening, for every liquid stock (daily bars up to that close only):
- swing highs and lows (a day whose high / low is the extreme of the 5 days either side, known 5 days later)
  over the last 250 sessions are grouped when they sit within max(0.5 ATR, 0.5%) of each other; a group's
  "touches" is how many swing points it holds (highs and lows both count: old resistance becomes support);
  a 52-week high / low and a round number near a group each add a touch
- a strong level has 3+ touches; the watchlist is every stock whose close is within 1.5 ATR under its nearest
  strong resistance (or over its nearest strong support), with the last 5 closes all still on the near side

Next day, on 5-minute bars, the trade is the first close beyond the level by 0.1 ATR (from 09:30 to 14:00,
the day not having gapped through it), entered at the next bar's open. Each breakout is recorded with the
facts known at that bar: its volume against the same time of day over 20 sessions, the day's volume so far,
the NIFTY's move and breadth, the sector's move, the stock's daily trend and 20-day strength against the
NIFTY, VWAP, room to the next strong level, how stretched the day already is, and whether the stock was coiled.
Stops: 0.25 ATR back inside the level, or the breakout bar's far end. Exits: 2R, 3R, a trail, or 15:15.
"""
import json
import time
from itertools import product

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.lib import analyze as A
from research.lib import build as B
from research.lib import engine as E
from research.lib import features as F
from research.lib import round2 as R2
from research.lib.sectors import sector_index
from research.sim import features as sim_features
from research.sim import slot_of_close

OUT = DATA_DIR / "research" / "srbreak"
RES = DATA_DIR / "research" / "results"
LEARN_START = np.datetime64("2022-10-25")
TEST_START = np.datetime64("2024-10-24")
T0930, T1400 = slot_of_close("09:30"), slot_of_close("14:00")
STOPS = ["0.25 ATR back inside the level", "the breakout bar's far end"]
EXITS = ["target 2R", "target 3R", "trail", "hold"]
EXIT_LABELS = ["target 2R", "target 3R", "trail (stop to cost at 1R, then under the last 3 bars)", "hold to 15:15"]
FACTORS = ["market agrees", "sector agrees", "daily trend agrees", "stronger than the NIFTY (20 days)", "VWAP side agrees",
           "room to the next level 2R+", "not stretched", "coiled before the break"]
VOLS = [1.5, 2.0, 3.0]
STRENGTHS = [3, 4]
COLS = ["day", "stock", "slot", "side", "touches", "is52", "rv", "crv", "nifty", "breadth", "sector", "trend", "rs20", "vwap",
        "room0", "room1", "ret_open", "gap", "coil", "level", "atr"]


# ---------------------------------------------------------------- levels from daily bars
def _round_step(p):
    return 5.0 if p < 200 else 10.0 if p < 500 else 50.0 if p < 2000 else 100.0 if p < 10000 else 500.0


def daily_levels(dh, dl, dc, atr, k=5, look=250, min_hist=120):
    """Per session d (using days up to d-1): nearest strong resistance / support, touches, 52-week flag,
    the next strong level beyond each, and whether the last 5 closes stayed on the near side."""
    D = len(dc)
    ph = np.zeros(D, bool)
    pl = np.zeros(D, bool)
    for i in range(k, D - k):
        ph[i] = dh[i] >= dh[i - k:i + k + 1].max()
        pl[i] = dl[i] <= dl[i - k:i + k + 1].min()
    out = {n: np.full(D, np.nan) for n in ("res", "res_t", "res_52", "res_next", "sup", "sup_t", "sup_52", "sup_next")}
    for d in range(min_hist, D):
        lo_i, hi_i = max(0, d - look), d - 1 - k                 # a swing point is known k days after it
        if hi_i <= lo_i or not np.isfinite(atr[d]):
            continue
        ih = np.flatnonzero(ph[lo_i:hi_i + 1]) + lo_i
        il = np.flatnonzero(pl[lo_i:hi_i + 1]) + lo_i
        pts = np.r_[dh[ih], dl[il]]
        pc = dc[d - 1]
        if pts.size < 3 or not np.isfinite(pc):
            continue
        tol = max(0.5 * atr[d], 0.005 * pc)
        ps = np.sort(pts)
        cut = np.flatnonzero(np.diff(ps) > tol) + 1
        groups = np.split(ps, cut)
        lv = np.array([g.mean() for g in groups])
        tc = np.array([len(g) for g in groups], float)
        h52, l52 = dh[max(0, d - 252):d].max(), dl[max(0, d - 252):d].min()
        is52 = (np.abs(lv - h52) <= tol) | (np.abs(lv - l52) <= tol)
        step = _round_step(pc)
        rnd = np.abs(lv - np.round(lv / step) * step) <= tol / 2
        tc = tc + 1.0 * is52 + 1.0 * rnd
        strong = tc >= 3
        last5 = dc[max(0, d - 5):d]
        for side, name in ((1, "res"), (-1, "sup")):
            cand = strong & ((lv > pc) if side > 0 else (lv < pc))
            if not cand.any():
                continue
            j = np.flatnonzero(cand)[np.argmin(np.abs(lv[cand] - pc))]
            L = lv[j]
            if abs(L - pc) > 1.5 * atr[d] or not ((last5 < L).all() if side > 0 else (last5 > L).all()):
                continue
            out[name][d], out[name + "_t"][d], out[name + "_52"][d] = L, tc[j], float(is52[j])
            beyond = strong & ((lv > L + tol) if side > 0 else (lv < L - tol))
            if beyond.any():
                out[name + "_next"][d] = lv[beyond][np.argmin(np.abs(lv[beyond] - L))]
    return out


# ---------------------------------------------------------------- one stock
def process(X, f, stock_id, day_index, costs, liquid, ctx):
    D = X.C.shape[0]
    dh, dl, dc = np.nanmax(X.H, 1), np.nanmin(X.L, 1), X.C[:, -1]
    atr = np.asarray(f.atr14, float)
    lv = daily_levels(dh, dl, dc, atr)
    nifty_c = ctx["nifty_close"]
    rs20 = np.full(D, np.nan)
    rs20[21:] = dc[20:-1] / dc[:-21] - nifty_c[day_index[21:] - 1] / nifty_c[day_index[21:] - 21]
    coil = np.full(D, np.nan)
    for d in range(6, D):
        coil[d] = (dh[d - 5:d].max() - dl[d - 5:d].min()) / max(atr[d], 1e-9)
    I = np.arange(data.SLOTS)[None, :]
    res, sup = [], []
    recs = []
    br = ctx["breadth"][day_index]
    for side, name in ((1, "res"), (-1, "sup")):
        L = lv[name]
        buf = 0.1 * atr
        trig = (L + buf * side)[:, None]
        beyond = (X.C - trig) * side > 0
        first = beyond & ~np.concatenate([np.zeros((D, 1), bool), np.logical_or.accumulate(beyond, axis=1)[:, :-1]], axis=1)
        ok_day = np.isfinite(L) & np.asarray(liquid, bool) & ((X.O[:, 0] - trig[:, 0]) * side < 0)
        sig = first & ok_day[:, None] & (I >= T0930) & (I <= T1400)
        rows, slots = np.nonzero(sig)
        if rows.size == 0:
            continue
        es = slots + 1
        entry = X.O[rows, es]
        sides = np.full(rows.size, side)
        stop0 = L[rows] - side * 0.25 * atr[rows]
        stop1 = X.L[rows, slots] if side > 0 else X.H[rows, slots]
        rec = {"day": day_index[rows], "stock": np.full(rows.size, stock_id), "slot": es, "side": sides,
               "touches": lv[name + "_t"][rows], "is52": lv[name + "_52"][rows], "rv": X.rv[rows, slots], "crv": X.crv[rows, slots],
               "nifty": X.nifty_ret[rows, slots] * side, "breadth": (br[rows, slots] - 0.5) * side,
               "sector": X.sector_ret[rows, slots] * side,
               "trend": (np.asarray(X.trend_up, float)[rows] - np.asarray(X.trend_dn, float)[rows]) * side,
               "rs20": rs20[rows] * side, "vwap": (X.C[rows, slots] - X.vwap[rows, slots]) * side,
               "ret_open": X.ret_open[rows, slots] * side, "gap": np.asarray(X.gap, float)[rows] * side, "coil": coil[rows],
               "level": L[rows], "atr": atr[rows]}
        nxt = lv[name + "_next"][rows]
        for si, stop in enumerate((stop0, stop1)):
            risk = (entry - stop) * side
            rec[f"room{si}"] = np.where(np.isfinite(nxt), (nxt - entry) * side / np.maximum(risk, 1e-9), 99.0)
            ok = R2._risk_ok(entry, stop, sides, es)
            for ei, ex in enumerate(EXITS):
                rn = np.full(rows.size, np.nan)
                xs = np.zeros(rows.size, int)
                xp = np.full(rows.size, np.nan)
                oc = np.zeros(rows.size, int)
                if ok.any():
                    out, x_, px = R2._exits(X, rows[ok], es[ok], sides[ok], entry[ok], stop[ok], ex)
                    rn[ok] = R2._rnet(entry[ok], stop[ok], px, out, sides[ok], costs)
                    xs[ok], xp[ok], oc[ok] = x_, px, out
                rec[f"r{si}{ei}"], rec[f"x{si}{ei}"], rec[f"p{si}{ei}"], rec[f"o{si}{ei}"] = rn, xs, xp, oc
            rec[f"stop{si}"] = stop
        rec["entry"] = entry
        recs.append(rec)
    if not recs:
        return None
    return {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}


_G = {}


def _chunk(syms):
    N, master, sectors, industries, costs, breadth, nifty_close = (_G[k] for k in ("N", "master", "sectors", "industries", "costs",
                                                                                "breadth", "nifty_close"))
    mpos = {d: i for i, d in enumerate(master)}
    parts = []
    for sid, sym in syms:
        m = data.load(sym)
        if m is None or m.n < 300:
            continue
        kp = np.array([d in mpos for d in m.dates])
        m = data.DayMatrix(m.symbol, m.dates[kp], m.o[kp], m.h[kp], m.l[kp], m.c[kp], m.v[kp], m.start[kp])
        f = sim_features(m)
        sec = sector_index(sym, industries.get(sym, ""))
        rows_m = np.array([mpos[d] for d in m.dates])
        ctx = {"nifty": B._rows_for(N, m.dates), "beta": B._beta(m, N)}
        if sec in sectors:
            ctx["sector"] = B._rows_for(sectors[sec], m.dates)
        X = F.build(m, f, ctx)
        liquid = (np.nan_to_num(X.lv.turn20) >= B.MIN_TURNOVER) & (np.nan_to_num(f.prev_close) >= B.MIN_PRICE) & f.valid
        r = process(X, f, sid, rows_m, costs, liquid, {"breadth": breadth, "nifty_close": nifty_close})
        if r is not None:
            parts.append(r)
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build(log=print, workers=8):
    import multiprocessing as mp
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    N, master, sectors, _ = B.context()
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    breadth = np.load(DATA_DIR / "research" / "opt" / "breadth.npy")
    if breadth.shape[0] != len(master):
        raise RuntimeError("breadth is not aligned to the master calendar")
    nifty_close = N.c[:, -1]
    _G.update(N=N, master=master, sectors=sectors, industries=industries, costs=Settings().get("costs"), breadth=breadth,
              nifty_close=nifty_close)
    ids = list(enumerate(syms))
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    log(f"  strong-level breakouts: {len(ids)} stocks, {len(master)} sessions")
    parts = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_chunk, chunks), 1):
            if r is not None:
                parts.append(r)
            if k % 10 == 0 or k == len(chunks):
                log(f"  {k}/{len(chunks)} chunks, {sum(len(p['day']) for p in parts):,} breakouts ({time.time() - t0:.0f}s)")
    T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(OUT / "breakouts.npz", **T)
    (OUT / "meta.json").write_text(json.dumps({"dates": [str(d) for d in master], "symbols": syms,
                                               "built_at": time.strftime("%Y-%m-%d %H:%M")}))
    log(f"  saved {len(T['day']):,} breakouts ({time.time() - t0:.0f}s)")
    return T


# ---------------------------------------------------------------- analysis
def factor_masks(T, si):
    return [T["nifty"] > 0, T["sector"] > 0, T["trend"] > 0, T["rs20"] > 0, T["vwap"] > 0,
            T[f"room{si}"] >= 2, (np.abs(T["ret_open"]) < 0.025) & (np.abs(T["gap"]) < 0.02), T["coil"] <= 3.0]


def _recs(T, mask, si, ei):
    r = T[f"r{si}{ei}"]
    m = mask & np.isfinite(r)
    rec = np.empty(int(m.sum()), dtype=E.REC)
    rec["v"], rec["day"], rec["stock"] = 0, T["day"][m], T["stock"][m]
    rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = T["slot"][m], T[f"x{si}{ei}"][m], T["side"][m], T[f"o{si}{ei}"][m]
    rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"] = T["entry"][m], T[f"stop{si}"][m], T[f"p{si}{ei}"][m], r[m]
    rec["score"] = np.nan_to_num(T["rv"][m])
    return rec


def _account(T, mask, si, ei, dates):
    daily, n = A.account(_recs(T, mask, si, ei), len(dates))
    month = np.array([str(d)[:7] for d in dates])
    out = {"trades": int(n)}
    for name, m in (("learn", (dates >= LEARN_START) & (dates < TEST_START)), ("test", dates >= TEST_START)):
        mm = sorted(set(month[m]))
        sums = np.array([daily[m][month[m] == x].sum() for x in mm])
        cum = np.cumsum(daily[m])
        out[name] = {"per_month": float(sums.mean()), "months": len(mm), "losing_months": int((sums < 0).sum()),
                     "max_dd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()),
                     "best_month": float(sums.max()), "worst_month": float(sums.min()),
                     "monthly": {a: round(float(b)) for a, b in zip(mm, sums)}}
    sel = dates >= LEARN_START
    out["curve"] = np.round(np.cumsum(daily[sel])).astype(int).tolist()
    return out


def _stats(r, m):
    x = r[m & np.isfinite(r)]
    return {"n": int(x.size), "avg_r": float(x.mean()) if x.size else None, "win": float((x > 0).mean() * 100) if x.size else None,
            "total_r": float(x.sum())}


def analyze(log=print):
    t0 = time.time()
    z = np.load(OUT / "breakouts.npz")
    T = {k: z[k] for k in z.files}
    meta = json.loads((OUT / "meta.json").read_text())
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    day = dates[T["day"]]
    learn = (day >= LEARN_START) & (day < TEST_START)
    test = day >= TEST_START
    log(f"  {int(learn.sum()):,} breakouts in Oct 2022 - Oct 2024, {int(test.sum()):,} in the last 2 years")
    # the strategy exactly as described: strong level, good volume, every confirming factor, 2R target
    fm0 = factor_masks(T, 0)
    described = (T["touches"] >= 3) & (T["rv"] >= 2.0) & (T["crv"] >= 1.2) & np.logical_and.reduce(fm0[:7])
    acct_described = _account(T, described, 0, 0, dates)
    desc_stats = {p: _stats(T["r00"], described & m) for p, m in (("learn", learn), ("test", test))}
    log(f"  as described: last 2 years Rs {acct_described['test']['per_month']:,.0f}/month on {desc_stats['test']['n']} breakouts")
    # one factor at a time, on top of the base (strong level + 1.5x volume), 2R target, level stop
    base = (T["touches"] >= 3) & (T["rv"] >= 1.5)
    one = [{"factor": "none (base: strong level, 1.5x volume)", "learn": _stats(T["r00"], base & learn), "test": _stats(T["r00"], base & test)}]
    for name, fmask in zip(FACTORS, fm0):
        one.append({"factor": name, "learn": _stats(T["r00"], base & fmask & learn), "test": _stats(T["r00"], base & fmask & test)})
    for thr in (2.0, 3.0):
        one.append({"factor": f"volume {thr:g}x instead of 1.5x", "learn": _stats(T["r00"], (T["touches"] >= 3) & (T["rv"] >= thr) & learn),
                    "test": _stats(T["r00"], (T["touches"] >= 3) & (T["rv"] >= thr) & test)})
    one.append({"factor": "level with 4+ touches", "learn": _stats(T["r00"], base & (T["touches"] >= 4) & learn),
                "test": _stats(T["r00"], base & (T["touches"] >= 4) & test)})
    one.append({"factor": "52-week high / low level", "learn": _stats(T["r00"], base & (T["is52"] > 0) & learn),
                "test": _stats(T["r00"], base & (T["is52"] > 0) & test)})
    # grid: every factor combination x volume x level strength x stop x exit; chosen on the 2 years before
    grid = []
    for si in (0, 1):
        fm = factor_masks(T, si)
        for combo in product((0, 1), repeat=len(FACTORS)):
            fmask = np.ones(len(T["day"]), bool)
            for on, mm in zip(combo, fm):
                if on:
                    fmask &= mm
            for vt, st in product(VOLS, STRENGTHS):
                m = fmask & (T["rv"] >= vt) & (T["touches"] >= st)
                for ei in range(len(EXITS)):
                    r = T[f"r{si}{ei}"]
                    L_, Te = _stats(r, m & learn), _stats(r, m & test)
                    grid.append({"stop": si, "exit": ei, "vol": vt, "strength": st, "factors": combo, "learn": L_, "test": Te})
    log(f"  {len(grid):,} variants scored ({time.time() - t0:.0f}s)")
    ok = [g for g in grid if g["learn"]["n"] >= 150]
    ok.sort(key=lambda g: -g["learn"]["total_r"])
    top = ok[:15]
    for g in top:
        si, ei = g["stop"], g["exit"]
        fm = factor_masks(T, si)
        m = (T["rv"] >= g["vol"]) & (T["touches"] >= g["strength"])
        for on, mm in zip(g["factors"], fm):
            if on:
                m &= mm
        g["account"] = _account(T, m, si, ei, dates)
    pos_test = sum(1 for g in ok if (g["test"]["avg_r"] or -1) > 0)
    # long vs short and example trades for the described version
    sides = {nm: {p: _stats(T["r00"], described & (T["side"] == s) & m) for p, m in (("learn", learn), ("test", test))}
             for s, nm in ((1, "breakouts (buy)"), (-1, "breakdowns (sell)"))}
    ex = np.flatnonzero(described & test & np.isfinite(T["r00"]))
    ex = ex[np.argsort(T["day"][ex])][-12:]
    syms = meta["symbols"]
    examples = [{"date": str(dates[T["day"][i]]), "stock": syms[int(T["stock"][i])], "side": "buy" if T["side"][i] > 0 else "sell",
                 "level": round(float(T["level"][i]), 2), "touches": int(T["touches"][i]), "volume_x": round(float(T["rv"][i]), 1),
                 "entry": round(float(T["entry"][i]), 2), "stop": round(float(T["stop0"][i]), 2), "exit": round(float(T["p00"][i]), 2),
                 "r": round(float(T["r00"][i]), 2)} for i in ex]
    per_day = np.bincount(T["day"][learn | test], minlength=len(dates))
    doc = {"built_at": meta["built_at"], "dates": [str(LEARN_START), str(dates[-1])], "test_start": str(TEST_START),
           "breakouts": {"learn": int(learn.sum()), "test": int(test.sum())},
           "per_day": {"median": float(np.median(per_day[(dates >= LEARN_START)])), "max": int(per_day.max())},
           "described": {"stats": desc_stats, "account": acct_described, "sides": sides, "examples": examples},
           "one_factor": one, "variants": len(grid), "variants_ok": len(ok), "variants_positive_test": pos_test,
           "top": [{**g, "factors": [FACTORS[i] for i, on in enumerate(g["factors"]) if on],
                    "stop_label": STOPS[g["stop"]], "exit_label": EXIT_LABELS[g["exit"]]} for g in top],
           "dates_axis": [str(d) for d in dates[dates >= LEARN_START]]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "srbreak_summary.json").write_text(json.dumps(doc, indent=1, default=float))
    log(f"  saved results/srbreak_summary.json ({time.time() - t0:.0f}s)")
    return doc


# ---------------------------------------------------------------- swing version (daily bars, long only)
SWING_EXITS = ["target 2R (10 days max)", "target 3R (15 days max)", "trail under the 5-day low (20 days max)"]
SWING_POS = 20000.0          # Rs 1 lakh split over at most 5 positions, no leverage for delivery
SWING_MAX_POS = 5


def swing_cost(value, slip_bps=5.0):
    """Delivery round trip on `value` rupees: brokerage min(Rs 20, 0.1%) a side, STT 0.1% both sides, stamp 0.015%,
    NSE 0.00297% a side, SEBI, a ~Rs 16 depository charge on the sell, GST 18%, plus slippage."""
    brk = np.minimum(20.0, value * 0.001)
    exch = value * 0.0000297
    sebi = value * 0.000001
    dp = 15.93
    gst = 0.18 * (2 * brk + 2 * exch + 2 * sebi + dp)
    return 2 * brk + 2 * exch + 2 * sebi + dp + gst + value * 0.002 + value * 0.00015 + 2 * value * slip_bps / 1e4


def _swing_stock(args):
    sym, sid, mrow, nifty_up, sector_up_all, industries = args
    m = data.load(sym)
    if m is None or m.n < 300:
        return None
    rows = np.array([mrow.get(d, -1) for d in m.dates])
    keep = rows >= 0
    if keep.sum() < 300:
        return None
    o, h, l, c, v = m.o[keep][:, 0], np.nanmax(m.h[keep], 1), np.nanmin(m.l[keep], 1), m.c[keep][:, -1], np.nansum(m.v[keep], 1)
    rows = rows[keep]
    f = sim_features(data.DayMatrix(m.symbol, m.dates[keep], m.o[keep], m.h[keep], m.l[keep], m.c[keep], m.v[keep], m.start[keep]))
    atr = np.asarray(f.atr14, float)
    D = len(c)
    lv = daily_levels(h, l, c, atr)
    sma20 = np.array([c[max(0, i - 19):i + 1].mean() for i in range(D)])
    sma50 = np.array([c[max(0, i - 49):i + 1].mean() for i in range(D)])
    vavg = np.array([v[max(0, i - 20):i].mean() if i >= 20 else np.nan for i in range(D)])
    turn = np.array([(v[max(0, i - 20):i] * c[max(0, i - 20):i]).mean() if i >= 20 else np.nan for i in range(D)])
    sec = sector_index(sym, industries.get(sym, ""))
    sector_up = sector_up_all.get(sec)
    out = []
    for d in range(130, D - 1):
        L = lv["res"][d]
        if not np.isfinite(L) or not (turn[d] >= B.MIN_TURNOVER) or c[d - 1] < B.MIN_PRICE:
            continue
        if not (c[d] > L + 0.1 * atr[d]) or o[d] > L + 0.1 * atr[d]:
            continue                                                 # needs a close beyond, not a gap over
        e = o[d + 1]
        stop = L - 0.5 * atr[d]
        risk = e - stop
        if not (risk > 0.003 * e and risk < 0.08 * e):
            continue
        if np.floor(SWING_POS / e) < 1:
            continue                                                 # one share costs more than a position
        rs = c[d] / c[max(0, d - 20)] - 1
        r = {"day": rows[d + 1], "stock": sid, "entry": e, "stop": stop, "vol_x": v[d] / vavg[d] if vavg[d] > 0 else np.nan,
             "market": float(nifty_up[rows[d]]), "sector": float(sector_up[rows[d]]) if sector_up is not None else np.nan,
             "trend": float(c[d] > sma20[d] > sma50[d]), "rs": rs, "touches": lv["res_t"][d], "is52": lv["res_52"][d],
             "room": (lv["res_next"][d] - e) / risk if np.isfinite(lv["res_next"][d]) else 99.0,
             "stretch": c[d] / c[d - 1] - 1}
        for k, (tgt_r, hold) in enumerate(((2.0, 10), (3.0, 15), (None, 20))):
            tgt = e + tgt_r * risk if tgt_r else np.inf
            trail = stop
            xp, xd = None, None
            for j in range(d + 1, min(D, d + 1 + hold)):
                if tgt_r is None and j > d + 1:
                    trail = max(trail, l[max(d + 1, j - 5):j].min())
                lvl = trail
                if l[j] <= lvl:
                    xp, xd = min(lvl, o[j]) if j > d + 1 else lvl, j
                    break
                if h[j] >= tgt:
                    xp, xd = max(tgt, o[j]) if j > d + 1 else tgt, j
                    break
            if xp is None:
                xd = min(D - 1, d + hold)
                xp = c[xd]
            qty = np.floor(SWING_POS / e)
            pnl = (xp - e) * qty - swing_cost(e * qty)
            r[f"pnl{k}"], r[f"gross{k}"], r[f"exit_day{k}"] = pnl, (xp - e) * qty, rows[xd]
            r[f"r{k}"] = pnl / max(risk * qty, 1e-9)
        out.append(r)
    return out


def build_swing(log=print, workers=8):
    import multiprocessing as mp
    t0 = time.time()
    N, master, sectors, _ = B.context()
    mrow = {d: i for i, d in enumerate(master)}
    nc = N.c[:, -1]
    nifty_up = np.array([nc[i] > nc[max(0, i - 19):i + 1].mean() for i in range(len(nc))])
    sector_up = {}
    for s, M_ in sectors.items():
        sc = np.full(len(master), np.nan)
        pos = np.array([mrow.get(d, -1) for d in M_.dates])
        sc[pos[pos >= 0]] = M_.c[pos >= 0, -1]
        sector_up[s] = np.array([sc[i] > np.nanmean(sc[max(0, i - 19):i + 1]) for i in range(len(sc))])
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    jobs = [(s, i, mrow, nifty_up, sector_up, industries) for i, s in enumerate(syms)]
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for res in pool.imap_unordered(_swing_stock, jobs, chunksize=4):
            if res:
                rows.extend(res)
    T = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(OUT / "swing.npz", **T)
    log(f"  swing: {len(rows):,} daily-close breakouts of strong resistance ({time.time() - t0:.0f}s)")
    return T


def _swing_account(T, mask, k, dates):
    """Rs 1 lakh, at most 5 positions of Rs 20,000, first come first served (ties: higher volume)."""
    idx = np.flatnonzero(mask & np.isfinite(T[f"pnl{k}"]))
    idx = idx[np.lexsort((-np.nan_to_num(T["vol_x"][idx]), T["day"][idx]))]
    daily = np.zeros(len(dates))
    open_ = []
    taken = 0
    for i in idx:
        d = int(T["day"][i])
        open_ = [x for x in open_ if x > d]
        if len(open_) >= SWING_MAX_POS:
            continue
        open_.append(int(T[f"exit_day{k}"][i]))
        daily[int(T[f"exit_day{k}"][i])] += float(T[f"pnl{k}"][i])
        taken += 1
    month = np.array([str(x)[:7] for x in dates])
    out = {"trades": taken}
    for name, m in (("learn", (dates >= LEARN_START) & (dates < TEST_START)), ("test", dates >= TEST_START)):
        mm = sorted(set(month[m]))
        sums = np.array([daily[m][month[m] == x].sum() for x in mm])
        cum = np.cumsum(daily[m])
        out[name] = {"per_month": float(sums.mean()), "losing_months": int((sums < 0).sum()), "months": len(mm),
                     "max_dd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())}
    return out


def analyze_swing(log=print):
    z = np.load(OUT / "swing.npz")
    T = {k: z[k] for k in z.files}
    dates = np.array(json.loads((OUT / "meta.json").read_text())["dates"], dtype="datetime64[D]")
    day = dates[T["day"].astype(int)]
    learn, test = (day >= LEARN_START) & (day < TEST_START), day >= TEST_START
    conf = (T["vol_x"] >= 2.0) & (T["market"] > 0) & (T["sector"] > 0) & (T["trend"] > 0) & (T["rs"] > 0) & (T["room"] >= 2) & (T["stretch"] < 0.05)
    base = T["vol_x"] >= 1.5
    out = {"breakouts": {"learn": int(learn.sum()), "test": int(test.sum())}, "versions": []}
    for name, m in (("as described (2x volume + market, sector, trend, strength, room, not stretched)", conf),
                    ("base: strong level + 1.5x volume", base)):
        for k, ex in enumerate(SWING_EXITS):
            row = {"version": name, "exit": ex}
            for per, pm in (("learn", learn), ("test", test)):
                mm = m & pm & np.isfinite(T[f"r{k}"])
                g = T[f"gross{k}"][mm] / np.maximum((T["entry"][mm] - T["stop"][mm]) * np.floor(SWING_POS / T["entry"][mm]), 1e-9)
                row[per] = {"n": int(mm.sum()), "avg_r": float(T[f"r{k}"][mm].mean()) if mm.any() else None,
                            "gross_r": float(g.mean()) if mm.any() else None, "win": float((T[f"pnl{k}"][mm] > 0).mean() * 100) if mm.any() else None}
            row["account"] = _swing_account(T, m, k, dates)
            out["versions"].append(row)
            log(f"  swing {name[:24]} / {ex[:12]}: last 2 years Rs {row['account']['test']['per_month']:,.0f}/month "
                f"({row['test']['n']} breakouts, {row['test']['gross_r'] or 0:+.2f}R before costs, {row['test']['avg_r'] or 0:+.2f}R after)")
    (RES / "srbreak_swing.json").write_text(json.dumps(out, indent=1, default=float))
    return out
