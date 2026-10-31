"""Round 6: price action on the NIFTY 500, and routers that pick setups by market type.

Market type of a session, all known by 09:15:
- NIFTY trend at yesterday's close: up (close > 20-day average > 50-day), down (the reverse), sideways
- volatility: India VIX at the open against its past year (calm below the median, high above)
- the open: gap up (+0.3% or more), gap down (-0.3% or less), flat
= 18 market types. Every price-action setup (44 rules x long / short x 3 exits) is scored in every type.

Routers, all chosen without seeing 2025-26:
- regime table, checked: setups that earned in a type on 2021-23 AND again in that type in 2024
- regime table, pooled: setups that earned in a type on 2021-24 with a t-stat of 2+
- machine learning: a gradient-boosted model of each trade's R from the market type, the setup and 25
  facts known at the signal; its cut-off is set on 2024
Baselines: every setup, trading only with the NIFTY trend, and the best single setup on 2021-24.
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
from research.lib import round3 as R3
from research.lib.pa import PA_LIBRARY
from research.lib.sectors import sector_index
from research.sim import features as sim_features
from research.sim import slot_of_close

OUT = DATA_DIR / "research" / "library6"
RES = DATA_DIR / "research" / "results"
EXITS = ["target 1R", "target 2R", "trail"]
EXIT_LABELS = ["target 1R", "target 2R", "trail (stop to cost at 1R, then under the last 3 bars)"]
KEEP = 12
LEARN_END = np.datetime64("2023-12-31")
CHECK_END = np.datetime64("2024-12-31")
T1430 = slot_of_close("14:30")
I = np.arange(data.SLOTS)[None, :]
TRENDS = ["NIFTY uptrend", "NIFTY sideways", "NIFTY downtrend"]
VOLS = ["calm VIX", "high VIX"]
OPENS = ["gap up", "flat open", "gap down"]
FAMILIES = sorted({s.family for s in PA_LIBRARY})
NS, NE = len(PA_LIBRARY), len(EXITS)
NU = NS * NE * 2
FEATS = ["slot", "long", "strategy", "exit", "family", "trend_with", "vix_pct", "gap_with", "open3", "day_type",
         "breadth_with", "nifty_with", "sector_with", "stock_trend_with", "rel_volume", "cum_rel_volume", "atr_rank",
         "open_volume_rank", "risk_pct", "stop_atr", "room_R", "bar_close_pos", "bar_range_atr", "vwap_dist_atr",
         "ema20_dist_atr", "adx", "beta", "day_range_atr"]
CATEGORICAL = ["long", "strategy", "exit", "family", "open3", "day_type"]


def regime_name(r):
    return f"{TRENDS[r // 6]}, {VOLS[(r // 3) % 2]}, {OPENS[r % 3]}"


def market_regimes(master, Fd):
    """(D,) trend (0 up, 1 sideways, 2 down), vol (0 calm, 1 high), open (0 up, 1 flat, 2 down), regime 0-17."""
    N = data.load("NIFTY")
    pos = {d: i for i, d in enumerate(N.dates)}
    dc = N.c[:, -1]
    trend = np.ones(len(master), int)
    for k, d in enumerate(master):
        i = pos.get(d)
        if i is None or i < 51:
            continue
        pc, s20, s50 = dc[i - 1], dc[i - 20:i].mean(), dc[i - 50:i].mean()
        trend[k] = 0 if (pc > s20 > s50) else 2 if (pc < s20 < s50) else 1
    vol = (np.nan_to_num(Fd["vix_pct"], nan=0.5) >= 0.5).astype(int)
    g = np.nan_to_num(Fd["gap"])
    opn = np.where(g >= 0.003, 0, np.where(g <= -0.003, 2, 1))
    return trend, vol, opn, trend * 6 + vol * 3 + opn


def trade_features(X, si, ei, rows, slots, sides, entry, stop, res, sup, rv_rank, atr_rank, dayf, day_index):
    r, k = rows, slots
    sd = sides.astype(float)
    C, H, L = X.C[r, k], X.H[r, k], X.L[r, k]
    atr5 = np.maximum(X.atr5[r, k], 1e-9)
    risk = np.maximum((entry - stop) * sd, 1e-9)
    d = day_index[r]
    rng = np.maximum(H - L, 1e-9)
    tr = dayf["trend"][d]
    cols = [
        k, (sd > 0).astype(float), np.full(len(r), si, float), np.full(len(r), ei, float),
        np.full(len(r), FAMILIES.index(PA_LIBRARY[si].family), float),
        np.where(tr == 1, 0.0, np.where(tr == 0, 1.0, -1.0)) * sd, dayf["vix_pct"][d], dayf["gap"][d] * sd, dayf["open3"][d],
        np.where(k >= 2, dayf["day_type"][d], np.nan),
        np.where(sd > 0, dayf["breadth_slot"][d, k], 1 - dayf["breadth_slot"][d, k]),
        X.nifty_ret[r, k] * sd, X.sector_ret[r, k] * sd,
        (np.asarray(X.trend_up, float)[r] - np.asarray(X.trend_dn, float)[r]) * sd,
        X.rv[r, k], X.crv[r, k],
        np.where(atr_rank[r] > 2000, np.nan, atr_rank[r]).astype(float), np.where(rv_rank[r] > 2000, np.nan, rv_rank[r]).astype(float),
        risk / entry, risk / atr5, np.clip(np.where(sd > 0, res[r, k] - entry, entry - sup[r, k]) / risk, 0, 20),
        np.where(sd > 0, (C - L) / rng, (H - C) / rng), (H - L) / atr5,
        (C - X.vwap[r, k]) / atr5 * sd, (C - X.ema20[r, k]) / atr5 * sd, X.adx[r, k], np.asarray(X.beta, float)[r],
        (X.day_hi[r, k] - X.day_lo[r, k]) / np.maximum(np.asarray(X.atr_d, float)[r], 1e-9),
    ]
    return np.column_stack(cols).astype(np.float32)


def process(X, stock_id, day_index, costs, liquid, rv_rank, atr_rank, dayf):
    W = (I <= T1430) & np.asarray(liquid, bool)[:, None]
    res, sup = E.room_levels(X)
    recs, feats = [], []
    for si, s in enumerate(PA_LIBRARY):
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
        ft = trade_features(X, si, 0, rows, slots, sides, entry, stop, res, sup, rv_rank, atr_rank, dayf, day_index)
        score = np.nan_to_num(X.rv[rows, slots], nan=0.0)
        for ei, ex in enumerate(EXITS):
            out, xs, exit_px = R2._exits(X, rows, es, sides, entry, stop, ex)
            rn = R2._rnet(entry, stop, exit_px, out, sides, costs)
            good = np.isfinite(rn)
            if not good.any():
                continue
            rec = np.empty(int(good.sum()), dtype=E.REC)
            rec["v"], rec["day"], rec["stock"] = si * NE + ei, day_index[rows[good]], stock_id
            rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = es[good], xs[good], sides[good], out[good]
            rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"], rec["score"] = entry[good], stop[good], exit_px[good], rn[good], score[good]
            f2 = ft[good].copy()
            f2[:, FEATS.index("exit")] = ei
            recs.append(rec)
            feats.append(f2)
    if not recs:
        return np.empty(0, dtype=E.REC), np.empty((0, len(FEATS)), np.float32)
    return np.concatenate(recs), np.concatenate(feats)


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
    return R3.keep(np.concatenate(recs), np.concatenate(feats), k=KEEP)


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
        for got in pool.imap_unordered(R3._breadth_pass, syms, chunksize=8):
            if got is None:
                continue
            ds, above = got
            idx = np.array([mpos.get(d, -1) for d in ds])
            ok = idx >= 0
            up[idx[ok]] += above[ok]
            cnt[idx[ok]] += 1
    trend, vol, opn, regime = market_regimes(master, Fd)
    dayf = {"breadth_slot": up / np.maximum(cnt, 1)[:, None], "vix_pct": Fd["vix_pct"], "gap": Fd["gap"],
            "day_type": A.day_type(Fd).astype(float), "open3": opn.astype(float), "trend": trend}
    _G.update(N=N, master=master, sectors=sectors, ranks=ranks, industries=industries, costs=Settings().get("costs"), xs=xs, dayf=dayf)
    ids = [(i, s) for i, s in enumerate(syms) if s in xs]
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    log(f"  round 6: {NS} price-action rules x 2 sides x {NE} exits, {len(ids)} stocks, 18 market types")
    recs, feats = np.empty(0, dtype=E.REC), np.empty((0, len(FEATS)), np.float32)
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (r, ft) in enumerate(pool.imap_unordered(_chunk, chunks), 1):
            recs, feats = R3.keep(np.concatenate([recs, r]), np.concatenate([feats, ft]), k=KEEP)
            if k % 8 == 0 or k == len(chunks):
                log(f"  {k}/{len(chunks)} chunks, {len(recs):,} trades ({time.time() - t0:.0f}s)")
    np.savez_compressed(OUT / "trades.npz", recs=recs, feats=feats, trend=trend, vol=vol, opn=opn, regime=regime)
    (OUT / "meta.json").write_text(json.dumps({"dates": [str(d) for d in master], "stocks": len(ids), "feats": FEATS,
                                               "strategies": [{"key": s.key, "name": s.name, "family": s.family, "source": s.source,
                                                               "rule": s.rule} for s in PA_LIBRARY],
                                               "built_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}))
    log(f"  saved {len(recs):,} trades ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- analysis
def _unit(recs):
    return recs["v"].astype(np.int64) * 2 + (recs["side"] > 0)


def unit_name(u):
    v, long = divmod(int(u), 2)
    si, ei = divmod(v, NE)
    return f"{PA_LIBRARY[si].name} ({'long' if long else 'short'}, {EXIT_LABELS[ei]})"


def _cell_stats(key, r, mask, n_keys):
    n = np.bincount(key[mask], minlength=n_keys).astype(float)
    s = np.bincount(key[mask], weights=r[mask], minlength=n_keys)
    q = np.bincount(key[mask], weights=r[mask] ** 2, minlength=n_keys)
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - mean ** 2, 1e-12))
    return n, mean, mean / sd * np.sqrt(np.maximum(n, 1))


def _acct(recs, mask, n_days, dates):
    daily, ntr = A.account(recs[mask], n_days)
    out = {"trades": int(ntr)}
    month = np.array([str(d)[:7] for d in dates])
    for name, m in (("learn", dates <= LEARN_END), ("check", (dates > LEARN_END) & (dates <= CHECK_END)), ("judge", dates > CHECK_END)):
        mm = sorted(set(month[m]))
        sums = np.array([daily[m][month[m] == x].sum() for x in mm])
        cum = np.cumsum(daily[m])
        out[name] = {"per_month": float(sums.mean()) if len(sums) else 0.0, "months": len(mm),
                     "losing_months": int((sums < 0).sum()), "max_dd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())}
    out["curve"] = np.round(np.cumsum(daily)).astype(int).tolist()
    return out


def analyze(log=print):
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    z = np.load(OUT / "trades.npz")
    recs, feats, regime_day = z["recs"], z["feats"], z["regime"]
    meta = json.loads((OUT / "meta.json").read_text())
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    n_days = len(dates)
    day = dates[recs["day"]]
    learn, check, judge = day <= LEARN_END, (day > LEARN_END) & (day <= CHECK_END), day > CHECK_END
    r = recs["rnet"].astype(float)
    unit = _unit(recs)
    reg = regime_day[recs["day"]]
    key = reg * NU + unit
    K = 18 * NU
    log(f"  {len(recs):,} trades; {NU} setups (rule x side x exit) x 18 market types")
    nl, ml, tl = _cell_stats(key, r, learn, K)
    nc, mc, tc = _cell_stats(key, r, check, K)
    nj, mj, tj = _cell_stats(key, r, judge, K)
    nlc, mlc, tlc = _cell_stats(key, r, learn | check, K)
    sel_a = (nl >= 80) & (ml >= 0.05) & (tl >= 1.5) & (nc >= 20) & (mc > 0)
    sel_b = (nlc >= 100) & (mlc >= 0.05) & (tlc >= 2.0)
    # baselines
    tr_day = z["trend"][recs["day"]]
    with_trend = ((recs["side"] > 0) & (tr_day == 0)) | ((recs["side"] < 0) & (tr_day == 2))
    nu_l, mu_l, _ = _cell_stats(unit, r, learn | check, NU)
    best_unit = int(np.nanargmax(np.where(nu_l >= 300, mu_l, -np.inf)))
    # machine learning router
    Xf = feats.astype(np.float32)
    cat = [FEATS.index(c) for c in CATEGORICAL]
    Xr = np.column_stack([Xf, reg.astype(np.float32)])
    cat_mask = np.zeros(Xr.shape[1], bool)
    cat_mask[cat] = True
    cat_mask[-1] = True
    rng = np.random.default_rng(1)
    tr_idx = np.flatnonzero(learn)
    tr_idx = rng.choice(tr_idx, size=min(2_500_000, tr_idx.size), replace=False)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                          l2_regularization=1.0, categorical_features=cat_mask, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=1)
    model.fit(Xr[tr_idx], np.clip(r[tr_idx], -1.5, 3.0))
    pred = np.empty(len(r), np.float32)
    for i in range(0, len(r), 2_000_000):
        pred[i:i + 2_000_000] = model.predict(Xr[i:i + 2_000_000])
    best_q, best_val, q_table = None, -1e18, []
    for q in (0.5, 1, 2, 5, 10, 20):
        thr = float(np.quantile(pred[check], 1 - q / 100))
        a = _acct(recs, check & (pred >= thr), n_days, dates)
        q_table.append({"top_pct": q, "threshold": thr, "check_per_month": a["check"]["per_month"]})
        if a["check"]["per_month"] > best_val:
            best_q, best_val, best_thr = q, a["check"]["per_month"], thr
    log(f"  ML router: cut-off top {best_q}% of 2024 predictions ({time.time() - t0:.0f}s)")
    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else None
    ml_ic = {"learn": corr(pred[learn], r[learn]), "check": corr(pred[check], r[check]), "judge": corr(pred[judge], r[judge])}
    approaches = {
        "Every price-action setup, no router": np.ones(len(r), bool),
        "Only with the NIFTY trend (long in uptrends, short in downtrends)": with_trend,
        f"Best single setup on 2021-24: {unit_name(best_unit)}": unit == best_unit,
        "Router: regime table, earned on 2021-23 and again in 2024": sel_a[key],
        "Router: regime table, chosen on 2021-24 (t-stat 2+)": sel_b[key],
        f"Router: machine learning, top {best_q}% (cut-off from 2024)": pred >= best_thr,
    }
    accounts = {}
    for name, m in approaches.items():
        accounts[name] = _acct(recs, m, n_days, dates)
        jm = m & judge
        accounts[name]["judge_trades_avail"] = int(jm.sum())
        accounts[name]["judge_avg_r"] = float(r[jm].mean()) if jm.any() else None
        accounts[name]["judge_win"] = float((r[jm] > 0).mean() * 100) if jm.any() else None
        log(f"  {name[:70]}: 2025-26 Rs {accounts[name]['judge']['per_month']:,.0f}/month")
    # which setup works best in which market (chosen on 2021-23, then 2024 and 2025-26 shown)
    table = []
    for g in range(18):
        sl = slice(g * NU, (g + 1) * NU)
        cand = np.where(nl[sl] >= 80, ml[sl], -np.inf)
        u = int(np.argmax(cand))
        hj = np.where(nj[sl] >= 30, mj[sl], -np.inf)
        uh = int(np.argmax(hj))
        days_ = int((regime_day == g).sum())
        has = bool(np.isfinite(cand[u]))
        row = {"regime": regime_name(g), "days": days_, "days_judge": int(((regime_day == g) & (dates > CHECK_END)).sum()),
               "best": unit_name(u) if has else None,
               "learn": {"n": int(nl[sl][u]) if has else 0, "avg_r": float(ml[sl][u]) if has else None},
               "check": {"n": int(nc[sl][u]) if has else 0, "avg_r": float(mc[sl][u]) if has and nc[sl][u] else None},
               "judge": {"n": int(nj[sl][u]) if has else 0, "avg_r": float(mj[sl][u]) if has and nj[sl][u] else None},
               "selected_a": int(sel_a[sl].sum()), "selected_b": int(sel_b[sl].sum()),
               "hindsight_best": unit_name(uh) if np.isfinite(hj[uh]) else None,
               "hindsight_avg_r": float(mj[sl][uh]) if np.isfinite(hj[uh]) else None}
        table.append(row)
    # family x market trend, 2025-26 (descriptive)
    fam = np.array([FAMILIES.index(PA_LIBRARY[int(v) // NE].family) for v in range(NS * NE)])[recs["v"]]
    fam_tbl = {}
    for fi, fname in enumerate(FAMILIES):
        fam_tbl[fname] = {}
        for t, tname in enumerate(TRENDS):
            for side, sname in ((1, "long"), (-1, "short")):
                m = judge & (fam == fi) & (tr_day == t) & (recs["side"] == side)
                ml_ = learn & (fam == fi) & (tr_day == t) & (recs["side"] == side)
                fam_tbl[fname][f"{tname}, {sname}"] = {"judge_r": float(r[m].mean()) if m.sum() > 50 else None,
                                                      "learn_r": float(r[ml_].mean()) if ml_.sum() > 50 else None,
                                                      "n_judge": int(m.sum())}
    per_setup = []
    for u in range(NU):
        m = unit == u
        if not m.any():
            continue
        per_setup.append({"setup": unit_name(u), "learn": {"n": int((m & learn).sum()), "avg_r": float(r[m & learn].mean()) if (m & learn).any() else None},
                          "judge": {"n": int((m & judge).sum()), "avg_r": float(r[m & judge].mean()) if (m & judge).any() else None,
                                    "win": float((r[m & judge] > 0).mean() * 100) if (m & judge).any() else None}})
    per_setup.sort(key=lambda x: -(x["judge"]["avg_r"] if x["judge"]["avg_r"] is not None else -9))
    positive = {p: int(sum(1 for x in range(NU) if (unit == x).any() and ((unit == x) & m).sum() >= 300 and r[(unit == x) & m].mean() > 0))
                for p, m in (("learn", learn), ("check", check), ("judge", judge))}
    regime_days = {regime_name(g): {"learn": int(((regime_day == g) & (dates <= LEARN_END)).sum()),
                                    "check": int(((regime_day == g) & (dates > LEARN_END) & (dates <= CHECK_END)).sum()),
                                    "judge": int(((regime_day == g) & (dates > CHECK_END)).sum())} for g in range(18)}
    doc = {"built_at": meta["built_at"], "trades": int(len(recs)), "stocks": meta["stocks"], "rules": NS, "setups": NU,
           "dates": [str(dates[0]), str(dates[-1])], "dates_axis": [str(d) for d in dates], "check_end": str(CHECK_END),
           "learn_end": str(LEARN_END), "base": {p: {"n": int(m.sum()), "avg_r": float(r[m].mean()), "win": float((r[m] > 0).mean() * 100)}
                                                for p, m in (("learn", learn), ("check", check), ("judge", judge))},
           "cells_selected": {"checked": int(sel_a.sum()), "pooled": int(sel_b.sum())}, "setups_positive": positive, "ml": {"ic": ml_ic, "q_table": q_table, "top_pct": best_q},
           "accounts": accounts, "regime_table": table, "family_trend": fam_tbl, "per_setup": per_setup, "regime_days": regime_days,
           "strategies": meta["strategies"]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "library6_summary.json").write_text(json.dumps(doc, indent=1, default=float))
    log(f"  saved results/library6_summary.json ({time.time() - t0:.0f}s)")
    return doc
