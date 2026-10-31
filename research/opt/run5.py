"""Round 5 driver: validate the option simulator, run the 1:2 option-buying grid, choose on 2021-24,
judge on 2025-26, build the account and the ruin odds, and save results/opt5_summary.json."""
import json
import time

import numpy as np

from core.config.settings import DATA_DIR
from research.opt import features as FT
from research.opt import grid as G
from research.opt import signals as SG
from research.opt import sim as SIM

RES = DATA_DIR / "research" / "results"
TOP = 25
DAILY_STOP = 6000.0
MAX_OPEN = 2


def choose_iv_mode(val):
    """The intraday IV rule with the smallest close + high + low error on 2021-24 (never on 2025-26)."""
    best, score = None, 1e9
    for mode, res in val["modes"].items():
        r = res["2021-24"]
        s = r["close"]["median_abs_pct"] + r["high"]["median_abs_pct"] + r["low"]["median_abs_pct"]
        if s < score:
            best, score = mode, s
    return best


def account(mk, picks, meta, pnl, xslot, ev, flags):
    """Several chosen variants in one Rs 1 lakh account: at most 2 trades open at once, no new trade
    after the day's realised loss reaches Rs 6,000, one trade per variant a day."""
    trades = []
    pe = meta[:, 0].astype(int)
    for v in picks:
        k = v["exit"]
        rows = np.flatnonzero((np.array([ev[i][0] for i in pe]) == v["signal"]) & (meta[:, 3] == v["u"]) &
                              (meta[:, 1] == v["contract"]) & flags[pe, v["filter"]] & np.isfinite(pnl[:, k]))
        for r in rows:
            trades.append((int(meta[r, 2]), int(meta[r, 7]) + 1, int(xslot[r, k]), float(pnl[r, k])))
    trades.sort()
    D = len(mk.dates)
    daily = np.zeros(D)
    taken = 0
    by_day = {}
    for t in trades:
        by_day.setdefault(t[0], []).append(t)
    for d, ts in by_day.items():
        open_ = []
        realised = 0.0
        for (dd, s_in, s_out, p) in sorted(ts, key=lambda x: x[1]):
            done = [o for o in open_ if o[0] <= s_in]
            realised += sum(o[1] for o in done)
            open_ = [o for o in open_ if o[0] > s_in]
            if realised <= -DAILY_STOP or len(open_) >= MAX_OPEN:
                continue
            open_.append((s_out, p))
            taken += 1
        realised += sum(o[1] for o in open_)
        daily[d] = realised
    return daily, taken


def stats(mk, daily):
    dates = mk.dates
    out = {}
    for per, m in (("learn", dates <= G.LEARN_END), ("judge", dates > G.LEARN_END)):
        x = daily[m]
        months = dates[m].astype("datetime64[M]")
        uniq, inv = np.unique(months, return_inverse=True)
        ms = np.bincount(inv, weights=x)
        cum = np.cumsum(x)
        dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
        out[per] = {"per_month": float(ms.mean()), "months": int(uniq.size), "losing_months": int((ms < 0).sum()),
                    "worst_month": float(ms.min()), "best_month": float(ms.max()), "max_dd": dd,
                    "months_at_20k": int((ms >= 20000).sum()), "monthly": {str(a): round(float(b)) for a, b in zip(uniq, ms)}}
    return out


def ruin(daily_judge, sims=20000, days=250, seed=3):
    """Bootstrap a year of 2025-26 trading days: odds of a Rs 50k drawdown and of losing Rs 1 lakh."""
    rng = np.random.default_rng(seed)
    x = daily_judge
    paths = rng.choice(x, size=(sims, days), replace=True)
    cum = np.cumsum(paths, axis=1)
    dd = (np.maximum.accumulate(np.maximum(cum, 0), axis=1) - cum).max(axis=1)
    low = cum.min(axis=1)
    return {"p_drawdown_50k": float((dd >= 50000).mean()), "p_lose_1L": float((low <= -100000).mean()),
            "median_year": float(np.median(cum[:, -1])), "p_year_negative": float((cum[:, -1] < 0).mean())}


def edge(mk, ev, flags, mode):
    """Edge per unit of risk with no account-size limit (every signal traded with at least 1 lot)."""
    saved = (G.MAX_RISK_1LOT, G.CAPITAL)
    G.MAX_RISK_1LOT, G.CAPITAL = 1e9, 1e9
    try:
        meta, pnl, rr, outcome, _ = G.simulate(mk, ev, mode, log=lambda *_: None)
        V = G.variants(mk, ev, flags, meta, pnl, rr, outcome)
    finally:
        G.MAX_RISK_1LOT, G.CAPITAL = saved
    ok = [v for v in V if v["learn"]["trades"] >= 60 and v["judge"]["trades"] >= 30]
    both = [v for v in ok if v["learn"]["avg_r"] > 0 and v["judge"]["avg_r"] > 0]
    cons = sorted([v for v in ok if all(v["learn_years"][y] > 0 for y in (2022, 2023, 2024))], key=lambda v: -v["learn"]["avg_r"])
    top50 = sorted(ok, key=lambda v: -v["learn"]["avg_r"])[:50]
    doc = {"ok": len(ok), "both_positive": len(both), "consistent": len(cons),
           "median_learn_r": float(np.median([v["learn"]["avg_r"] for v in ok])),
           "median_judge_r": float(np.median([v["judge"]["avg_r"] for v in ok])),
           "top50_judge_mean_r": float(np.mean([v["judge"]["avg_r"] for v in top50])),
           "top50_judge_positive": int(sum(v["judge"]["avg_r"] > 0 for v in top50)),
           "top": [{**v, **G.describe(v)} for v in cons[:15]]}
    (RES / "opt5_edge.json").write_text(json.dumps(doc, indent=1, default=float))
    return doc


def run(log=print):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    FT.build(log=log)
    mk = SIM.Market()
    val = SIM.validate(mk, log=log)
    mode = choose_iv_mode(val)
    log(f"  intraday IV rule chosen on 2021-24: {mode}")
    br = SG.breadth()
    ev = G.events(mk, br, log)
    flags = G.filter_flags(mk, br, ev)
    meta, pnl, rr, outcome, xslot = G.simulate(mk, ev, mode, log=log)
    V = G.variants(mk, ev, flags, meta, pnl, rr, outcome)
    log(f"  {len(V):,} variants scored ({time.time() - t0:.0f}s)")
    cands = [v for v in V if v["learn"]["trades"] >= 60 and all(v["learn_years"][y] > 0 for y in (2022, 2023, 2024))]
    cands.sort(key=lambda v: -v["learn"]["per_month"])
    top = cands[:TOP]
    hind = sorted([v for v in V if v["judge"]["trades"] >= 30], key=lambda v: -v["judge"]["per_month"])[:10]
    counts = {"variants": len(V), "consistent_on_2021_24": len(cands),
              "learn_20k": sum(v["learn"]["per_month"] >= 20000 for v in V),
              "judge_20k": sum(v["judge"]["per_month"] >= 20000 for v in V),
              "both_20k": sum(v["learn"]["per_month"] >= 20000 and v["judge"]["per_month"] >= 20000 for v in V),
              "judge_positive_of_top": sum(v["judge"]["per_month"] > 0 for v in top),
              "judge_positive_all": sum(v["judge"]["per_month"] > 0 for v in V)}
    # the account: the best 5 distinct signal / index pairs from the 2021-24 ranking
    seen, picks = set(), []
    for v in cands:
        key = (v["signal"], v["u"])
        if key in seen:
            continue
        seen.add(key)
        picks.append(v)
        if len(picks) == 5:
            break
    daily, taken = account(mk, picks, meta, pnl, xslot, ev, flags)
    acct = stats(mk, daily)
    acct["trades"] = taken
    acct["ruin"] = ruin(daily[mk.dates > G.LEARN_END]) if (mk.dates > G.LEARN_END).any() else None
    single = []
    for v in picks:
        d1, _ = account(mk, [v], meta, pnl, xslot, ev, flags)
        single.append({**G.describe(v), "account": stats(mk, d1)})
    # stress: the same picks with double slippage and with the other IV rules
    stress = {}
    sub = [e for e in ev if (e[0], e[1]) in seen]
    for label, kw in (("double slippage", {"iv_mode": mode, "slip_mult": 2.0}),
                      *[(f"IV rule: {m}", {"iv_mode": m}) for m in SIM.IV_MODES if m != mode]):
        idx_map = {i: j for j, i in enumerate([k for k, e in enumerate(ev) if (e[0], e[1]) in seen])}
        m2, p2, r2, o2, x2 = G.simulate(mk, sub, log=lambda *_: None, **kw)
        f2 = flags[sorted(idx_map)]
        d2, _ = account(mk, picks, m2, p2, x2, sub, f2)
        stress[label] = stats(mk, d2)
        log(f"  stress {label}: 2025-26 Rs {stress[label]['judge']['per_month']:,.0f}/month")
    # how often one trade fits the Rs 3,000-per-trade cap, before and after SEBI's Nov 2024 lot increase
    day = mk.dates[meta[:, 2].astype(int)]
    sizing = []
    for u, name in enumerate(("NIFTY", "BANKNIFTY")):
        for c, (dl, md) in enumerate(G.CONTRACTS):
            if md != "nearest expiry":
                continue
            row = {"index": name, "delta": dl}
            for per, m in (("2021-24", day <= G.LEARN_END), ("2025-26", day > G.LEARN_END)):
                sel = (meta[:, 3] == u) & (meta[:, 1] == c) & m
                row[per] = {"fits_pct": round(float(np.isfinite(pnl[sel, 2]).mean() * 100), 1) if sel.any() else None,
                            "lot": float(np.nanmedian(meta[sel, 8])) if sel.any() else None}
            sizing.append(row)
    lots = {name: float(mk.smiles[max(k for k in mk.smiles if k[1] == u)]["lot"]) for u, name in enumerate(("NIFTY", "BANKNIFTY"))}
    doc = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "iv_mode": mode, "validation": val, "counts": counts,
           "top": [{**v, **G.describe(v)} for v in top], "hindsight": [{**v, **G.describe(v)} for v in hind],
           "sizing": sizing,
           "picks": [G.describe(v) for v in picks], "account": acct, "single": single, "stress": stress,
           "lots_now": lots, "risk_per_trade": G.RISK, "events": len(ev), "paths": int(len(meta)),
           "signals": SG.NAMES, "filters": G.FILTERS, "exits": [G.exit_label(e) for e in G.EXITS],
           "contracts": [f"delta {a:.2f}, {b}" for a, b in G.CONTRACTS]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "opt5_summary.json").write_text(json.dumps(doc, indent=1, default=float))
    np.savez(DATA_DIR / "research" / "opt" / "grid5.npz", meta=meta, pnl=pnl, rr=rr, outcome=outcome, xslot=xslot)
    e = edge(mk, ev, flags, mode)
    log(f"  edge with no size limit: median {e['median_learn_r']:+.3f}R / {e['median_judge_r']:+.3f}R, "
        f"{e['both_positive']} of {e['ok']} variants positive in both periods")
    log(f"  saved results/opt5_summary.json ({time.time() - t0:.0f}s)")
    return doc
