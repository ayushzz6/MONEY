"""Library round 3: every strategy at 1:1 (the target as far as the stop) and a machine-learning router that
aims at 75% winning trades.

Trades: all 123 book strategies, signal at a bar's close, entry at the next bar's open, target 1R, stop 1R
(a bar touching both counts as the stop), out at 15:15; one trade per strategy per stock per day; the 15
earliest a day per strategy are kept. For each trade, 31 things known at the signal are saved: time of day,
the strategy, volume against normal, distance from VWAP and the 20 EMA, ADX / RSI / DMI, the stock's, the
NIFTY's and the sector's move since the open, the sector's rank, beta, the stop in ATR and in %, room to the
next support / resistance, the signal bar's shape, the day's range so far, the gap, the daily trend, the
stock's volatility and opening-volume ranks, breadth (the share of stocks above their open) as it stands at the
signal bar, the VIX, the open type (known at 09:15) and the day type (known at 09:30: blank for earlier signals).

Router (meta-labelling, Lopez de Prado): a gradient-boosted classifier estimates each trade's chance of a net
win. It learns on Oct 2021 - Dec 2023, its confidence cut-off is set on 2024 (the lowest cut-off that gives 75%
winners there), and it is judged once on Jan 2025 - Oct 2026. Account: the first 5 trades a day above the
cut-off, Rs 1 lakh, NSE charges and 3 bps slippage a side.

    ./research.sh library3
"""
import json
import time

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.lib import analyze as A
from research.lib import build as B
from research.lib import engine as E
from research.lib import features as F
from research.lib import round2 as R2
from research.lib.sectors import sector_index
from research.lib.strategies import LIBRARY, T1430
from research.sim import features as sim_features

OUT = DATA_DIR / "research" / "library3"
RES = DATA_DIR / "research" / "results"
KEEP = 15
TRAIN_END = np.datetime64("2023-12-31")
CAL_END = np.datetime64("2024-12-31")
TARGET_WIN = 0.75
MIN_CAL_TRADES = 150
FEATS = ["slot", "long", "strategy", "family", "rel_volume", "cum_rel_volume", "vwap_dist_atr", "ema20_dist_atr", "adx",
         "rsi14_dir", "dmi_dir", "stock_move", "nifty_move", "sector_move", "sector_rank_dir", "beta", "risk_pct",
         "stop_atr", "room_R", "bar_close_pos", "bar_range_atr", "day_range_atr", "gap_dir", "daily_trend_dir",
         "atr_rank", "open_volume_rank", "breadth_dir", "vix_pct", "day_type", "open_type", "vwap_side_frac"]
CATEGORICAL = ["long", "strategy", "family", "day_type", "open_type"]
I = np.arange(data.SLOTS)[None, :]
FAMILIES = sorted({s.family for s in LIBRARY})


# ---------------------------------------------------------------- per-trade features (all known at the signal bar)
def trade_features(X, si, rows, slots, sides, entry, stop, res, sup, rv_rank, atr_rank, dayf, day_index):
    r, k = rows, slots
    sd = sides.astype(float)
    C, O, H, L = X.C[r, k], X.O[r, k], X.H[r, k], X.L[r, k]
    atr5 = np.maximum(X.atr5[r, k], 1e-9)
    risk = np.maximum((entry - stop) * sd, 1e-9)
    rank = X.sector_rank[r, k].astype(float)
    rank = np.where(rank > 50, np.nan, rank)
    d = day_index[r]
    rng = np.maximum(H - L, 1e-9)
    cols = [
        k, (sd > 0).astype(float), np.full(len(r), si, float), np.full(len(r), FAMILIES.index(LIBRARY[si].family), float),
        X.rv[r, k], X.crv[r, k],
        (C - X.vwap[r, k]) / atr5 * sd, (C - X.ema20[r, k]) / atr5 * sd, X.adx[r, k],
        np.where(sd > 0, X.rsi14[r, k], 100 - X.rsi14[r, k]), (X.pdi[r, k] - X.ndi[r, k]) * sd,
        X.ret_open[r, k] * sd, X.nifty_ret[r, k] * sd, X.sector_ret[r, k] * sd, np.where(sd > 0, rank, 16 - rank),
        np.asarray(X.beta, float)[r], risk / entry, risk / atr5,
        np.clip(np.where(sd > 0, res[r, k] - entry, entry - sup[r, k]) / risk, 0, 20),
        np.where(sd > 0, (C - L) / rng, (H - C) / rng), (H - L) / atr5,
        (X.day_hi[r, k] - X.day_lo[r, k]) / np.maximum(np.asarray(X.atr_d, float)[r], 1e-9),
        np.asarray(X.gap, float)[r] * sd, (np.asarray(X.trend_up, float)[r] - np.asarray(X.trend_dn, float)[r]) * sd,
        np.where(atr_rank[r] > 2000, np.nan, atr_rank[r]).astype(float), np.where(rv_rank[r] > 2000, np.nan, rv_rank[r]).astype(float),
        np.where(sd > 0, dayf["breadth_slot"][d, k], 1 - dayf["breadth_slot"][d, k]), dayf["vix_pct"][d],
        np.where(k >= 2, dayf["day_type"][d], np.nan), dayf["open_type"][d],
        np.where(sd > 0, X.above_vwap_n[r, k], k + 1 - X.above_vwap_n[r, k]) / (k + 1),
    ]
    return np.column_stack(cols).astype(np.float32)


def process(X, stock_id, day_index, costs, liquid, rv_rank, atr_rank, dayf):
    W = (I <= T1430) & np.asarray(liquid, bool)[:, None]
    res, sup = E.room_levels(X)
    recs, feats = [], []
    for si, s in enumerate(LIBRARY):
        L, Sx, stL, stS = s.fn(X)
        stL = np.asarray(stL, float) * np.ones(X.C.shape)
        stS = np.asarray(stS, float) * np.ones(X.C.shape)
        L = np.asarray(L, bool) & W & np.isfinite(stL)
        Sx = np.asarray(Sx, bool) & W & np.isfinite(stS)
        rows, slots, sides, stop = E.first_signals(L, Sx, stL, stS)
        if len(rows) == 0:
            continue
        es = slots + 1
        entry = X.O[rows, es]
        ok = R2._risk_ok(entry, stop, sides, es)
        rows, slots, es, sides, entry, stop = (a[ok] for a in (rows, slots, es, sides, entry, stop))
        if len(rows) == 0:
            continue
        out, xs, exit_px = R2._exits(X, rows, es, sides, entry, stop, "target 1R")
        rn = R2._rnet(entry, stop, exit_px, out, sides, costs)
        good = np.isfinite(rn)
        if not good.any():
            continue
        rec = np.empty(int(good.sum()), dtype=E.REC)
        rec["v"], rec["day"], rec["stock"] = si, day_index[rows[good]], stock_id
        rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = es[good], xs[good], sides[good], out[good]
        rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"] = entry[good], stop[good], exit_px[good], rn[good]
        rec["score"] = np.nan_to_num(X.rv[rows[good], slots[good]], nan=0.0)
        recs.append(rec)
        feats.append(trade_features(X, si, rows, slots, sides, entry, stop, res, sup, rv_rank, atr_rank, dayf, day_index)[good])
    if not recs:
        return np.empty(0, dtype=E.REC), np.empty((0, len(FEATS)), np.float32)
    return np.concatenate(recs), np.concatenate(feats)


def keep(recs, feats, k=KEEP):
    if len(recs) == 0:
        return recs, feats
    order = np.lexsort((-recs["score"], recs["slot"], recs["day"], recs["v"]))
    r, f = recs[order], feats[order]
    key = r["v"].astype(np.int64) * 100000 + r["day"]
    start = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    rank = np.arange(len(r)) - first[gid]
    return r[rank < k], f[rank < k]


def _breadth_pass(sym):
    m = data.load(sym)
    if m is None or m.n < 260:
        return None
    return [str(d) for d in m.dates], (m.c > m.o[:, :1])


_G = {}


def _chunk(syms):
    N, master, sectors, ranks, industries, costs, xs, dayf = (_G[k] for k in ("N", "master", "sectors", "ranks", "industries",
                                                                           "costs", "xs", "dayf"))
    mpos = {d: i for i, d in enumerate(master)}
    recs, feats = [], []
    for sid, sym in syms:
        if sym not in xs:
            continue
        m = data.load(sym)
        if m is None or m.n < 260:
            continue
        kp = np.array([d in mpos for d in m.dates])
        m = data.DayMatrix(m.symbol, m.dates[kp], m.o[kp], m.h[kp], m.l[kp], m.c[kp], m.v[kp], m.start[kp])
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
        r, ft = process(X, sid, rows_m.astype(np.int16), costs, liquid, rv_rank[rows_m], atr_rank[rows_m], dayf)
        recs.append(r)
        feats.append(ft)
    if not recs:
        return np.empty(0, dtype=E.REC), np.empty((0, len(FEATS)), np.float32)
    return keep(np.concatenate(recs), np.concatenate(feats))


def build(log=print, workers=8):
    import multiprocessing as mp
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    N, master, sectors, ranks = B.context()
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    with mp.get_context("fork").Pool(workers) as pool:
        pre = {s: r for s, r in pool.map(R2._prepass, syms, chunksize=8) if r is not None}
    xs = R2._ranks(pre, master)
    br = dict(np.load(DATA_DIR / "research" / "library" / "breadth.npz"))
    Fd = A.day_features(master, br)
    mpos = {str(d): i for i, d in enumerate(master)}
    up = np.zeros((len(master), data.SLOTS))
    cnt = np.zeros(len(master))
    with mp.get_context("fork").Pool(workers) as pool:
        for got in pool.imap_unordered(_breadth_pass, syms, chunksize=8):
            if got is None:
                continue
            ds, above = got
            idx = np.array([mpos.get(d, -1) for d in ds])
            ok = idx >= 0
            up[idx[ok]] += above[ok]
            cnt[idx[ok]] += 1
    breadth_slot = up / np.maximum(cnt, 1)[:, None]
    dayf = {"breadth_slot": breadth_slot, "vix_pct": Fd["vix_pct"],
            "day_type": A.day_type(Fd).astype(float), "open_type": A.open_type(Fd).astype(float)}
    _G.update(N=N, master=master, sectors=sectors, ranks=ranks, industries=industries, costs=Settings().get("costs"), xs=xs, dayf=dayf)
    ids = [(i, s) for i, s in enumerate(syms) if s in xs]
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    log(f"  round 3: {len(LIBRARY)} strategies at 1:1, {len(ids)} stocks, {len(FEATS)} features a trade")
    recs, feats = np.empty(0, dtype=E.REC), np.empty((0, len(FEATS)), np.float32)
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (r, ft) in enumerate(pool.imap_unordered(_chunk, chunks), 1):
            recs, feats = keep(np.concatenate([recs, r]), np.concatenate([feats, ft]))
            if k % 8 == 0 or k == len(chunks):
                log(f"  {k}/{len(chunks)} chunks, {len(recs):,} trades ({time.time() - t0:.0f}s)")
    np.savez_compressed(OUT / "trades.npz", recs=recs, feats=feats)
    (OUT / "meta.json").write_text(json.dumps({"dates": [str(d) for d in master], "stocks": len(ids), "feats": FEATS,
                                               "strategies": [{"key": s.key, "name": s.name, "family": s.family, "source": s.source,
                                                               "rule": s.rule} for s in LIBRARY],
                                               "built_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}))
    log(f"  saved {len(recs):,} trades ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- the router
def _stats(recs, m):
    q = recs["rnet"][m]
    return {"trades": int(m.sum()), "win": round(100 * float((q > 0).mean()), 1) if m.any() else None,
            "avgR": round(float(q.mean()), 4) if m.any() else None}


def _curve(p, y, pct_list):
    """Win rate when the top x% of trades by confidence are taken."""
    order = np.argsort(-p)
    ys = y[order]
    out = []
    for pct in pct_list:
        n = max(1, int(len(ys) * pct / 100))
        out.append({"top_pct": pct, "trades": n, "win": round(100 * float(ys[:n].mean()), 1), "min_p": round(float(p[order][n - 1]), 3)})
    return out


def analyze(log=print):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    meta = json.loads((OUT / "meta.json").read_text())
    z = np.load(OUT / "trades.npz")
    recs, feats = z["recs"], z["feats"]
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    nd = len(dates)
    d = dates[recs["day"]]
    y = recs["rnet"] > 0
    tr, cal, te = d <= TRAIN_END, (d > TRAIN_END) & (d <= CAL_END), d > CAL_END
    log(f"  {len(recs):,} trades: {tr.sum():,} to learn on, {cal.sum():,} to set the cut-off, {te.sum():,} to judge")
    # 1. every strategy at 1:1, no router
    strategies = meta["strategies"]
    per = []
    for si, s in enumerate(strategies):
        m = recs["v"] == si
        per.append({"key": s["key"], "name": s["name"], "family": s["family"], "source": s["source"],
                    "early": _stats(recs, m & ~te), "late": _stats(recs, m & te)})
    # 2. the router
    cat = [FEATS.index(k) for k in CATEGORICAL]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
                                         l2_regularization=1.0, categorical_features=cat, random_state=1,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    t0 = time.time()
    clf.fit(feats[tr], y[tr])
    p = clf.predict_proba(feats)[:, 1].astype(np.float32)
    log(f"  router trained: {clf.n_iter_} rounds ({time.time() - t0:.0f}s)")
    # the cut-off: the lowest confidence that gives 75% winners on 2024
    pc, yc = p[cal], y[cal]
    order = np.argsort(-pc)
    cum = np.cumsum(yc[order]) / np.arange(1, len(order) + 1)
    n_ok = np.flatnonzero((cum >= TARGET_WIN) & (np.arange(1, len(order) + 1) >= MIN_CAL_TRADES))
    reached = len(n_ok) > 0
    if reached:
        n_take = int(n_ok.max()) + 1                       # the largest set of 2024 trades that still wins 75%
        thr = float(pc[order][n_take - 1])
    else:
        best = int(np.argmax(np.where(np.arange(1, len(order) + 1) >= MIN_CAL_TRADES, cum, -1)))
        n_take, thr = best + 1, float(pc[order][best])
    cal_win = float(cum[n_take - 1])
    sel_te = te & (p >= thr)
    router_test = _stats(recs, sel_te)
    router_cal = {"trades": n_take, "win": round(100 * cal_win, 1), "threshold": round(thr, 3), "reached_75": reached}
    # thresholds table: 2024 (used to choose) against 2025-26 (unseen)
    thr_rows = []
    for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        thr_rows.append({"threshold": t, "cal": _stats(recs, cal & (p >= t)), "test": _stats(recs, te & (p >= t))})
    coverage = {"cal": _curve(pc, yc, [0.5, 1, 2, 5, 10, 20, 50, 100]), "test": _curve(p[te], y[te], [0.5, 1, 2, 5, 10, 20, 50, 100])}
    # 3. the account on 2025-26: router at the cut-off, no router, the best single strategy picked on 2021-24
    rr = recs.copy()
    rr["score"] = p
    acct = {}
    for name, m in (("Router at the 2024 cut-off", sel_te | (~te & (p >= thr))), ("Every trade, no router", np.ones(len(rr), bool))):
        daily, n = A.account(rr[m], nd)
        acct[name] = dict(A.period_stats(daily, dates, CAL_END), trades=n, curve=[round(float(x), 0) for x in np.cumsum(daily)])
    early_ok = [x for x in per if x["early"]["trades"] >= 500]
    best1 = max(early_ok, key=lambda x: x["early"]["avgR"]) if early_ok else None
    if best1:
        si = [s["key"] for s in strategies].index(best1["key"])
        daily, n = A.account(rr[rr["v"] == si], nd)
        acct[f"Best single strategy on 2021-24: {best1['name']}"] = dict(A.period_stats(daily, dates, CAL_END), trades=n,
                                                                        curve=[round(float(x), 0) for x in np.cumsum(daily)])
    # the router's trades, up to 10 a day instead of 5 (its picks cluster on the days whose context it likes)
    A.MAX_TRADES, A.MAX_POS = 10, 10
    try:
        daily10, n10 = A.account(rr[sel_te | (~te & (p >= thr))], nd)
    finally:
        A.MAX_TRADES, A.MAX_POS = 5, 5
    acct["Router at the 2024 cut-off, up to 10 trades a day"] = dict(A.period_stats(daily10, dates, CAL_END), trades=n10,
                                                                     curve=[round(float(x), 0) for x in np.cumsum(daily10)])
    # month by month, router, 2025-26
    daily_r, _ = A.account(rr[sel_te], nd)
    month = np.array([str(x)[:7] for x in dates])
    months = {k: round(float(daily_r[month == k].sum()), 0) for k in sorted(set(month[dates > CAL_END]))}
    # how the flagged trades spread over the days of 2025-26
    fd = np.bincount(recs["day"][sel_te], minlength=nd)[dates > CAL_END]
    spread = {"days": int(len(fd)), "days_with_trades": int((fd > 0).sum()), "mean_per_active_day": round(float(fd[fd > 0].mean()), 1) if (fd > 0).any() else 0,
              "median_per_active_day": float(np.median(fd[fd > 0])) if (fd > 0).any() else 0, "max_in_a_day": int(fd.max()),
              "share_of_trades_on_top_20_days": round(float(np.sort(fd)[::-1][:20].sum() / max(fd.sum(), 1)), 2)}
    # what the router leans on (permutation importance on 40,000 trades of 2024)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(cal), size=min(40000, int(cal.sum())), replace=False)
    pi = permutation_importance(clf, feats[idx], y[idx], n_repeats=3, random_state=0, scoring="roc_auc")
    importance = sorted([{"feature": f_, "gain": round(float(g), 4)} for f_, g in zip(FEATS, pi.importances_mean)],
                        key=lambda x: -x["gain"])
    # which strategies the router actually trades, 2025-26
    picked = {}
    for si in np.unique(recs["v"][sel_te]):
        m = sel_te & (recs["v"] == si)
        picked[strategies[si]["name"]] = _stats(recs, m)
    picked = dict(sorted(picked.items(), key=lambda kv: -kv[1]["trades"])[:15])
    from sklearn.metrics import roc_auc_score
    summary = {"built_at": meta["built_at"], "stocks": meta["stocks"], "strategies": len(strategies), "trades": int(len(recs)),
               "dates": [str(dates[0]), str(dates[-1])], "dates_axis": [str(x) for x in dates],
               "splits": {"learn": [str(dates[0]), str(TRAIN_END)], "cutoff": ["2024-01-01", str(CAL_END)], "judge": ["2025-01-01", str(dates[-1])]},
               "base_win": {"early": _stats(recs, ~te), "test": _stats(recs, te)},
               "auc": {"cal": round(float(roc_auc_score(yc, pc)), 3), "test": round(float(roc_auc_score(y[te], p[te])), 3)},
               "router_cal": router_cal, "router_test": router_test, "thresholds": thr_rows, "coverage": coverage,
               "accounts": acct, "months": months, "spread": spread, "importance": importance, "picked": picked,
               "per_strategy": sorted(per, key=lambda x: -(x["late"]["win"] or 0)), "features": FEATS}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "library3_summary.json").write_text(json.dumps(summary, default=str))
    np.save(OUT / "p.npy", p)
    log(f"  router: 2024 cut-off {thr:.3f} -> {router_cal['win']}% winners on {n_take} trades; "
        f"2025-26: {router_test['win']}% on {router_test['trades']} trades")
    return summary
