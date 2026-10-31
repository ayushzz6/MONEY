"""Library results, the day-type router and the comparison, from research/lib/build.py's saved trades.

Account: Rs 1 lakh, Rs 5 lakh intraday (MIS) exposure, Rs 1,000 risk a trade (positions up to Rs 2.5 lakh), at
most 5 open and 5 a day, one trade per stock a day, trading stops for the day at a Rs 3,000 loss. Every trade
already carries NSE charges and 3 bps slippage a side (research/lib/engine.py).

The router never looks ahead: signals before 09:30 are routed on what is known at 09:15 (the NIFTY's gap, the
India VIX, the NIFTY's trend, yesterday's breadth); signals from 09:30 on what is known at 09:30 as well (how the
NIFTY opened, breadth at 09:30, the size of its first 15 minutes). Which strategies trade on which day type is
learned on Oct 2021 - Dec 2024 and judged on Jan 2025 - Oct 2026.
"""
import json
from collections import defaultdict

import numpy as np

from core.config.settings import DATA_DIR
from research import data
from research.lib import engine as E

LIB = DATA_DIR / "research" / "library"
RES = DATA_DIR / "research" / "results"
CAPITAL, LEVERAGE, MAX_POS, MAX_TRADES, DAILY_STOP, RISK = 100000.0, 5.0, 5, 5, 3000.0, 1000.0
K_PER_TYPE, MIN_N, MIN_T = 5, 60, 2.0


# ---------------------------------------------------------------- loading
def load():
    meta = json.loads((LIB / "meta.json").read_text())
    recs = np.load(LIB / "trades.npz")["recs"]
    agg = np.load(LIB / "agg.npy")
    br = dict(np.load(LIB / "breadth.npz"))
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    return meta, recs, agg, br, dates


# ---------------------------------------------------------------- the account
def account(recs, n_days):
    """Daily P&L of a set of candidate trades under the account's limits."""
    daily = np.zeros(n_days)
    ntr = 0
    if len(recs) == 0:
        return daily, ntr
    order = np.lexsort((-recs["score"], recs["slot"], recs["day"]))
    r = recs[order]
    k = E.SLIP_BPS / 10000.0
    side = r["side"].astype(float)
    fin = r["entry"] * (1 + k * side)
    rf = (fin - r["stop"]) * side
    qfull = np.floor(np.minimum(RISK / np.maximum(rf, 1e-9), E.MAX_POS / fin))
    days = r["day"]
    bounds = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        open_, closed, used, n = [], 0.0, set(), 0
        for i in range(a, b):
            if r["stock"][i] in used:
                continue
            slot = r["slot"][i]
            still = []
            for (xs, val, net) in open_:
                if xs <= slot:
                    closed += net
                else:
                    still.append((xs, val, net))
            open_ = still
            if len(open_) >= MAX_POS or n >= MAX_TRADES or closed <= -DAILY_STOP:
                continue
            room = CAPITAL * LEVERAGE - sum(v for _, v, _ in open_)
            q = min(qfull[i], np.floor(room / fin[i]))
            if q < 1:
                continue
            net = float(r["rnet"][i] * rf[i] * q)
            open_.append((r["exit_slot"][i], q * fin[i], net))
            used.add(r["stock"][i])
            n += 1
        daily[days[a]] += closed + sum(net for _, _, net in open_)
        ntr += n
    return daily, ntr


def period_stats(daily, dates, train_end):
    month = np.array([str(d)[:7] for d in dates])
    out = {}
    for name, msk in (("train", dates <= train_end), ("test", dates > train_end)):
        q = daily[msk]
        mm = sorted(set(month[msk]))
        sums = np.array([q[month[msk] == x].sum() for x in mm])
        eq = np.cumsum(q)
        out[name] = {"per_month": round(float(q.sum()) / max(1, len(mm)), 0),
                     "maxdd": round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0) if len(q) else 0,
                     "months_20k": int((sums >= 20000).sum()), "months_loss": int((sums < 0).sum()), "months": len(mm)}
    return out


# ---------------------------------------------------------------- day features (known at 09:15 / 09:30) and types
def day_features(dates, br):
    N, V = data.load("NIFTY"), data.load("INDIAVIX")
    pos = {d: i for i, d in enumerate(N.dates)}
    vpos = {d: i for i, d in enumerate(V.dates)}
    n = len(dates)
    feat = defaultdict(lambda: np.full(n, np.nan))
    o, h, l, c = N.o, N.h, N.l, N.c
    dc = c[:, -1]
    dh, dl = h.max(1), l.min(1)
    tr = np.maximum(dh - dl, np.abs(dh - np.r_[np.nan, dc[:-1]]))
    atr = np.r_[np.nan, np.array([np.nanmean(tr[max(0, i - 14):i]) if i >= 14 else np.nan for i in range(1, len(dc))])]
    vc = V.c[:, -1]
    for k, d in enumerate(dates):
        i = pos.get(d)
        if i is None or i < 21:
            continue
        pc = dc[i - 1]
        feat["gap"][k] = o[i, 0] / pc - 1
        feat["trend"][k] = pc / np.mean(dc[i - 20:i]) - 1
        feat["ret5"][k] = pc / dc[i - 5] - 1
        feat["r930"][k] = c[i, 2] / o[i, 0] - 1
        feat["or15_atr"][k] = (h[i, :3].max() - l[i, :3].min()) / atr[i]
        feat["prev_range_atr"][k] = (dh[i - 1] - dl[i - 1]) / atr[i]
        j = vpos.get(d)
        if j is not None and j > 250:
            feat["vix_pct"][k] = np.mean(vc[j - 250:j] < V.o[j, 0])
            feat["vix_chg"][k] = V.o[j, 0] / vc[j - 1] - 1
        # hindsight only (for the descriptive table, never for routing)
        body, rng = dc[i] - o[i, 0], max(dh[i] - dl[i], 1e-9)
        feat["h_body_atr"][k] = body / atr[i]
        feat["h_range_atr"][k] = rng / atr[i]
        feat["h_close_pos"][k] = (dc[i] - dl[i]) / rng
    cnt = np.maximum(br["cnt"], 1)
    feat["breadth930"] = br["up930"] / cnt
    feat["breadth_prev"] = np.r_[np.nan, (br["upclose"] / cnt)[:-1]]
    feat["weekday"] = np.array([d.astype(object).weekday() for d in dates], float)
    return dict(feat)


OPEN_TYPES = ["Gap up, calm", "Gap up, volatile", "Flat open, calm", "Flat open, volatile", "Gap down, calm", "Gap down, volatile"]
DAY_TYPES = ["Gap up and drive", "Gap down and drive", "Gap faded", "Drive up", "Drive down", "Rotational, volatile",
             "Rotational, quiet", "Rotational"]
HINDSIGHT = ["Trend up", "Trend down", "Reversal", "Range", "Normal"]


def open_type(F):
    g = np.nan_to_num(F["gap"])
    hv = np.nan_to_num(F["vix_pct"]) >= 0.6
    gi = np.where(g >= 0.004, 0, np.where(g <= -0.004, 2, 1))
    return gi * 2 + hv.astype(int)


def day_type(F):
    g, r, b = np.nan_to_num(F["gap"]), np.nan_to_num(F["r930"]), np.nan_to_num(F["breadth930"], nan=0.5)
    w, hv = np.nan_to_num(F["or15_atr"]), np.nan_to_num(F["vix_pct"]) >= 0.6
    up = (r > 0.0015) & (b > 0.6)
    dn = (r < -0.0015) & (b < 0.4)
    t = np.full(len(g), 7)
    t = np.where(~up & ~dn & (w < 0.2) & ~hv, 6, t)
    t = np.where(~up & ~dn & ((w >= 0.35) | hv), 5, t)
    t = np.where(dn, 4, t)
    t = np.where(up, 3, t)
    t = np.where((np.abs(g) >= 0.004) & (np.sign(r) == -np.sign(g)) & (np.abs(r) > 0.001), 2, t)
    t = np.where((g <= -0.004) & dn, 1, t)
    t = np.where((g >= 0.004) & up, 0, t)
    return t


def hindsight_type(F):
    body, rng, pos = np.nan_to_num(F["h_body_atr"]), np.nan_to_num(F["h_range_atr"]), np.nan_to_num(F["h_close_pos"], nan=0.5)
    r = np.nan_to_num(F["r930"])
    t = np.full(len(body), 4)
    t = np.where((rng < 0.7) & (np.abs(body) < 0.2), 3, t)
    t = np.where((np.abs(r) >= 0.0015) & (np.sign(body) == -np.sign(r)) & (np.abs(body) >= 0.3), 2, t)
    t = np.where((body <= -0.5) & (pos <= 0.25), 1, t)
    t = np.where((body >= 0.5) & (pos >= 0.75), 0, t)
    return t


# ---------------------------------------------------------------- routers
def unit_day_table(recs, n_days):
    """Per (unit, day): sum and count of R. A unit is a variant x timing (0: signal before 09:30, 1: after)."""
    timing = (recs["slot"] >= 3).astype(np.int64)       # entry at 09:30 or later = signal at 09:30 or later
    unit = recs["v"].astype(np.int64) * 2 + timing
    key = unit * n_days + recs["day"]
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=recs["rnet"])
    c = np.bincount(inv)
    return u // n_days, u % n_days, s, c, unit


def select_by_type(units, days, s, c, types, train, n_types, k=K_PER_TYPE, min_n=MIN_N, min_t=MIN_T, timing=None):
    """For each type: the k units with the best average R per trade over its train days (enough trades, t >= min_t)."""
    picks = {}
    keep = train[days]
    if timing is not None:
        keep &= (units % 2) == timing
    uu, dd, ss, cc = units[keep], days[keep], s[keep], c[keep]
    tt = types[dd]
    for t in range(n_types):
        m = tt == t
        if not m.any():
            picks[t] = []
            continue
        U, inv = np.unique(uu[m], return_inverse=True)
        n = np.bincount(inv, weights=cc[m])
        sm = np.bincount(inv, weights=ss[m])
        mean = sm / np.maximum(n, 1)
        # day-level dispersion for the t-statistic (days, not trades, are the independent draws)
        dm = ss[m] / np.maximum(cc[m], 1)
        nd = np.bincount(inv)
        var = np.bincount(inv, weights=dm * dm) / np.maximum(nd, 1) - (np.bincount(inv, weights=dm) / np.maximum(nd, 1)) ** 2
        tstat = (np.bincount(inv, weights=dm) / np.maximum(nd, 1)) / np.sqrt(np.maximum(var, 1e-12) / np.maximum(nd, 1))
        ok = (n >= min_n) & (tstat >= min_t) & (mean > 0)
        order = np.argsort(-mean[ok])[:k]
        picks[t] = [int(x) for x in U[ok][order]]
    return picks


def routed_recs(recs, rec_unit, day_units):
    """Candidates of the units chosen for each day. day_units: day -> set of units."""
    keys = np.array([d * 1_000_000 + u for d, us in day_units.items() for u in us], np.int64)
    if len(keys) == 0:
        return recs[:0]
    rk = recs["day"].astype(np.int64) * 1_000_000 + rec_unit
    return recs[np.isin(rk, keys)]


def run(log=print):
    meta, recs, agg, br, dates = load()
    nd = len(dates)
    train_end = np.datetime64(meta["train_end"])
    train = dates <= train_end
    strategies = meta["strategies"]
    F = day_features(dates, br)
    ot, dtp, ht = open_type(F), day_type(F), hindsight_type(F)
    log(f"  {len(recs):,} trades, {E.NV:,} variants, {nd} sessions")
    # 1. every variant: per-trade statistics (all trades) and the account (5 earliest a day)
    order = np.argsort(recs["v"], kind="stable")
    rs = recs[order]
    vb = np.searchsorted(rs["v"], np.arange(E.NV + 1))
    variants = []
    daily_by_v = {}
    for v in range(E.NV):
        si, fi, ei = E.unvid(v)
        a = agg[v]
        row = {"v": v, "strategy": strategies[si]["key"], "name": strategies[si]["name"], "family": strategies[si]["family"],
               "source": strategies[si]["source"], "filter": E.FILTERS[fi], "exit": E.EXITS[ei]}
        for pi, pn in ((0, "train"), (1, "test")):
            n, w, sr, sr2 = a[pi, 0], a[pi, 1], a[pi, 2], a[pi, 3]
            mean = sr / n if n else 0.0
            sd = np.sqrt(max(sr2 / n - mean * mean, 1e-12)) if n else 0.0
            row[f"{pn}_trades"] = int(n)
            row[f"{pn}_win"] = round(100 * w / n, 1) if n else None
            row[f"{pn}_avgR"] = round(mean, 4)
            row[f"{pn}_t"] = round(mean / sd * np.sqrt(n), 2) if n else 0.0
        daily, ntr = account(rs[vb[v]:vb[v + 1]], nd)
        daily_by_v[v] = daily
        row.update({f"acct_{k}": val for k, val in period_stats(daily, dates, train_end).items()})
        variants.append(row)
        if v % 200 == 0:
            log(f"  accounts: {v}/{E.NV}")
    # 2. routers
    units, udays, us, uc, rec_unit = unit_day_table(recs, nd)
    early_picks = select_by_type(units, udays, us, uc, ot, train, len(OPEN_TYPES), timing=0)
    late_picks = select_by_type(units, udays, us, uc, dtp, train, len(DAY_TYPES), timing=1)
    day_units = {d: set(early_picks.get(int(ot[d]), [])) | set(late_picks.get(int(dtp[d]), [])) for d in range(nd)}
    rule_daily, rule_n = account(routed_recs(recs, rec_unit, day_units), nd)
    # data-driven day types: k-means on the 09:30 features, fitted on the train days
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    cols = ["gap", "r930", "or15_atr", "breadth930", "vix_pct", "trend", "ret5", "breadth_prev"]
    Xd = np.column_stack([np.nan_to_num(F[c_]) for c_ in cols])
    sc = StandardScaler().fit(Xd[train])
    km = KMeans(n_clusters=8, n_init=10, random_state=1).fit(sc.transform(Xd[train]))
    cl = km.predict(sc.transform(Xd))
    cl_picks = select_by_type(units, udays, us, uc, cl, train, 8, timing=1)
    early_only = {d: set(early_picks.get(int(ot[d]), [])) for d in range(nd)}
    day_units_km = {d: early_only[d] | set(cl_picks.get(int(cl[d]), [])) for d in range(nd)}
    km_daily, km_n = account(routed_recs(recs, rec_unit, day_units_km), nd)
    # recent-performance router: each day, the 5 units with the best average R over the previous 60 sessions
    tab = defaultdict(lambda: [0.0, 0])
    by_day = defaultdict(list)
    for uu_, dd_, s_, c_ in zip(units, udays, us, uc):
        by_day[int(dd_)].append((int(uu_), s_, c_))
    window = []
    rp_units = {}
    for d in range(nd):
        ranked = sorted(((v_[0] / v_[1], u_) for u_, v_ in tab.items() if v_[1] >= 30), reverse=True)
        rp_units[d] = {u_ for m_, u_ in ranked[:5] if m_ > 0}
        window.append(by_day.get(d, []))
        for u_, s_, c_ in by_day.get(d, []):
            tab[u_][0] += s_
            tab[u_][1] += c_
        if len(window) > 60:
            for u_, s_, c_ in window.pop(0):
                tab[u_][0] -= s_
                tab[u_][1] -= c_
    rp_daily, rp_n = account(routed_recs(recs, rec_unit, rp_units), nd)
    # baselines
    tr_rank = sorted(variants, key=lambda r: -r["acct_train"]["per_month"])
    best1 = tr_rank[0]
    top10 = sorted([r for r in variants if r["train_trades"] >= 300], key=lambda r: -r["train_t"])[:10]
    top10_daily, top10_n = account(recs[np.isin(recs["v"], [r["v"] for r in top10])], nd)
    approaches = [
        ("Day-type router (rules)", rule_daily, rule_n, "Day types from the gap, VIX, how the NIFTY opened and breadth at 09:30"),
        ("Day-type router (k-means)", km_daily, km_n, "8 day clusters learned from the same 09:30 features"),
        ("Recent-performance router", rp_daily, rp_n, "Each day, the 5 strategies with the best last-60-session record"),
        ("Best single strategy (picked on 2021-24)", daily_by_v[best1["v"]], None, f"{best1['name']} · {best1['filter']} · {best1['exit']}"),
        ("Top 10 by 2021-24 t-statistic, every day", top10_daily, top10_n, "No routing"),
    ]
    appr = [{"name": n, "desc": d, "trades": t, **period_stats(dly, dates, train_end),
             "curve": [round(float(x), 0) for x in np.cumsum(dly)]} for n, dly, t, d in approaches]
    # 3. day types: what the router picked, and how families do on each (train / test)
    def unit_label(u):
        v, timing = u // 2, u % 2
        si, fi, ei = E.unvid(v)
        return f"{strategies[si]['name']} · {E.FILTERS[fi]} · {E.EXITS[ei]}" + (" (before 09:30)" if timing == 0 else "")
    type_rows = []
    for t, name in enumerate(DAY_TYPES):
        m = dtp == t
        type_rows.append({"type": name, "train_days": int((m & train).sum()), "test_days": int((m & ~train).sum()),
                          "picked": [unit_label(u) for u in late_picks.get(t, [])],
                          "router_train": round(float(rule_daily[m & train].mean()), 0) if (m & train).any() else 0,
                          "router_test": round(float(rule_daily[m & ~train].mean()), 0) if (m & ~train).any() else 0})
    open_rows = [{"type": n, "train_days": int(((ot == t) & train).sum()), "test_days": int(((ot == t) & ~train).sum()),
                  "picked": [unit_label(u) for u in early_picks.get(t, [])]} for t, n in enumerate(OPEN_TYPES)]
    # family x day type, base variants (no filter, 2R target), average R per trade
    fam_names = sorted({s["family"] for s in strategies})
    base = recs[(recs["v"] % (len(E.FILTERS) * len(E.EXITS))) == 0]
    fam_of = np.array([fam_names.index(strategies[E.unvid(int(v))[0]]["family"]) for v in range(E.NV)])
    fam = fam_of[base["v"]]
    heat = {}
    for label, types, names in (("router", dtp, DAY_TYPES), ("hindsight", ht, HINDSIGHT)):
        for pn, pm in (("train", train), ("test", ~train)):
            grid = []
            for fi_, fn in enumerate(fam_names):
                row = []
                for t in range(len(names)):
                    m = (fam == fi_) & (types[base["day"]] == t) & pm[base["day"]]
                    row.append(round(float(base["rnet"][m].mean()), 3) if m.sum() >= 50 else None)
                grid.append(row)
            heat[f"{label}_{pn}"] = {"rows": fam_names, "cols": names, "grid": grid}
    # router day-type prediction vs what the day turned out to be
    conf = [[int(((dtp == a) & (ht == b)).sum()) for b in range(len(HINDSIGHT))] for a in range(len(DAY_TYPES))]
    # filters: average change in R per trade versus no filter (test), across strategies
    fil = []
    for fi, fname in enumerate(E.FILTERS):
        d_tr, d_te = [], []
        for si in range(len(strategies)):
            b0, b1 = variants[E.vid(si, 0, 0)], variants[E.vid(si, fi, 0)]
            if b1["train_trades"] >= 100 and b1["test_trades"] >= 50:
                d_tr.append(b1["train_avgR"] - b0["train_avgR"])
                d_te.append(b1["test_avgR"] - b0["test_avgR"])
        fil.append({"filter": fname, "strategies": len(d_tr), "train_dR": round(float(np.mean(d_tr)), 4) if d_tr else None,
                    "test_dR": round(float(np.mean(d_te)), 4) if d_te else None})
    summary = {"built_at": meta["built_at"], "stocks": meta["stocks_done"], "strategies": len(strategies),
               "variants": E.NV, "sessions": nd, "dates": [str(dates[0]), str(dates[-1])], "train_end": meta["train_end"],
               "approaches": appr, "day_types": type_rows, "open_types": open_rows, "heat": heat,
               "confusion": {"rows": DAY_TYPES, "cols": HINDSIGHT, "grid": conf}, "filters": fil,
               "positive": {"train": sum(r["train_avgR"] > 0 for r in variants if r["train_trades"] >= 100),
                            "test": sum(r["test_avgR"] > 0 for r in variants if r["test_trades"] >= 50),
                            "both": sum(r["train_avgR"] > 0 and r["test_avgR"] > 0 for r in variants
                                        if r["train_trades"] >= 100 and r["test_trades"] >= 50)},
               "acct_20k_both": sum(r["acct_train"]["per_month"] >= 20000 and r["acct_test"]["per_month"] >= 20000 for r in variants),
               "dates_axis": [str(d) for d in dates]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "library_summary.json").write_text(json.dumps(summary, default=str))
    (RES / "library_variants.json").write_text(json.dumps(variants, default=str))
    log("  router and summary written")
    return summary, variants
