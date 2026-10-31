"""Round 4, part 2: positioning data published after the close, as a lead for the next day.

NSE publishes these every evening; they are known before the next session opens:
- delivery: how much of today's volume was taken into demat (accumulation vs intraday churn)
- F&O: futures open interest build-up / unwinding, futures premium, options put / call open interest
- participants: FII, DII, Pro and Client long / short contracts in index futures and options
- US: the S&P 500's last full session before India opens (and how the NIFTY's gap compares)

Trades: the next session from 09:20 to 15:15 (intraday, Rs 1 lakh, intraday charges), and a 5-day
swing from the next day's 09:20 to the 5th day's close (delivery charges). Direction and nothing else
is chosen on 2021-23; 2024 checks it; 2025-26 judges it.
"""
import glob
import json
import time

import numpy as np

from core.config.settings import DATA_DIR
from research.lead import intraday as I

DAILY = DATA_DIR / "research" / "lead" / "daily"
RES = DATA_DIR / "research" / "results"
K = 3
PARTICIPANTS = ["Client", "DII", "FII", "Pro"]
P_COLS = ["fut_idx_long", "fut_idx_short", "fut_stk_long", "fut_stk_short", "opt_idx_call_long", "opt_idx_put_long",
          "opt_idx_call_short", "opt_idx_put_short", "opt_stk_call_long", "opt_stk_put_long", "opt_stk_call_short",
          "opt_stk_put_short"]


def swing_cost_bps(notional=I.NOTIONAL, slip_bps=5.0):
    """Delivery round trip: STT 0.1% each side, stamp 0.015% on the buy, exchange, SEBI, GST, a Rs 20
    brokerage cap each side and a ~Rs 16 depository charge on the sell, plus slippage."""
    brk = min(20.0, notional * 0.001)
    exch = notional * 0.0000297
    sebi = notional * 0.000001
    dp = 13.5
    buy = brk + exch + sebi + notional * 0.001 + notional * 0.00015 + 0.18 * (brk + exch + sebi)
    sell = brk + exch + sebi + notional * 0.001 + dp + 0.18 * (brk + exch + sebi + dp)
    return (buy + sell) / notional * 1e4 + 2 * slip_bps


def load_positioning(syms, dates):
    """(S, D) arrays from the daily NSE files, NaN where a file or a stock is missing."""
    S, D = len(syms), len(dates)
    sid = {s: i for i, s in enumerate(syms)}
    pos = {str(d): j for j, d in enumerate(dates)}
    f = {k: np.full((S, D), np.nan, np.float32) for k in
         ("close", "prev", "vol", "dqty", "dper", "fut_oi", "fut_chg", "fut_close", "ce_oi", "pe_oi", "ce_chg", "pe_chg")}
    idx_fo = np.full((D, 7), np.nan)
    part = np.full((D, 4, 12), np.nan)
    for path in sorted(glob.glob(str(DAILY / "*.json"))):
        doc = json.loads(open(path).read())
        j = pos.get(doc["date"])
        if j is None:
            continue
        for s, row in (doc.get("delivery") or {}).items():
            i = sid.get(s)
            if i is not None:
                for k, v in zip(("close", "prev", "vol", "dqty", "dper"), row):
                    f[k][i, j] = np.nan if v is None else v
        fo = doc.get("fo") or {}
        for s, row in fo.items():
            i = sid.get(s)
            if i is not None:
                for k, v in zip(("fut_oi", "fut_chg", "fut_close", "ce_oi", "pe_oi", "ce_chg", "pe_chg"), row):
                    f[k][i, j] = np.nan if v is None else v
        if "NIFTY" in fo:
            idx_fo[j] = [np.nan if v is None else v for v in fo["NIFTY"]]
        for p, row in (doc.get("participants") or {}).items():
            if p in PARTICIPANTS:
                part[j, PARTICIPANTS.index(p)] = [np.nan if v is None else v for v in row]
    return f, idx_fo, part


def _z(x, n=20):
    """(x - previous-n mean) / previous-n std along days."""
    m = I.prior_mean(x, n)
    sd = np.sqrt(np.maximum(I.prior_mean(x * x, n) - m * m, 0))
    return (x - m) / np.where(sd > 0, sd, np.nan)


def stock_signals(f, P):
    """(S, D) signals known after day d's close."""
    ret = f["close"] / f["prev"] - 1
    up = np.sign(ret)
    sig = {}
    sig["Delivery % vs its 20-day normal, x today's direction"] = _z(f["dper"]) * up
    sig["Delivered quantity vs its 20-day normal, x direction"] = (f["dqty"] / I.prior_mean(f["dqty"], 20) - 1) * up
    sig["Delivery % vs its 20-day normal (no direction)"] = _z(f["dper"])
    base = f["fut_oi"] - f["fut_chg"]
    oi_chg = f["fut_chg"] / np.where(base > 0, base, np.nan)
    sig["Futures OI change x today's direction (build-up)"] = oi_chg * up
    sig["Today's move where futures OI rose (fresh positions)"] = np.where(oi_chg > 0, ret, 0.0) + np.where(np.isfinite(oi_chg), 0, np.nan)
    sig["Today's move where futures OI fell (covering / unwinding)"] = np.where(oi_chg < 0, ret, 0.0) + np.where(np.isfinite(oi_chg), 0, np.nan)
    sig["Futures OI vs its 20-day normal, x direction"] = _z(f["fut_oi"]) * up
    basis = f["fut_close"] / f["close"] - 1
    with np.errstate(all="ignore"):
        sig["Futures premium vs other stocks"] = basis - np.nanmedian(basis, axis=0, keepdims=True)
    pcr = f["pe_oi"] / np.where(f["ce_oi"] > 0, f["ce_oi"], np.nan)
    sig["Put / call OI ratio vs its 20-day normal"] = _z(pcr)
    sig["Put / call OI ratio, change today"] = pcr - np.concatenate([np.full((pcr.shape[0], 1), np.nan), pcr[:, :-1]], 1)
    tot = f["ce_oi"] + f["pe_oi"]
    sig["New put OI minus new call OI (share of all options OI)"] = (f["pe_chg"] - f["ce_chg"]) / np.where(tot > 0, tot, np.nan)
    sig["Today's move (baseline)"] = ret
    C = P["c"]
    vol_today = np.nansum(P["v"], axis=2)
    sig["Today's close in the day's range (baseline)"] = ((C[:, :, -1] - np.nanmin(P["l"], 2)) /
                                                       np.maximum(np.nanmax(P["h"], 2) - np.nanmin(P["l"], 2), 1e-9))
    sig["Today's volume vs its 20-day normal, x direction (baseline)"] = (vol_today / I.prior_mean(vol_today, 20) - 1) * up
    return {k: v.astype(np.float32) for k, v in sig.items()}


def market_signals(idx_fo, part, dates, us):
    """(D,) signals known after day d's close (the US one: known before day d+1 opens)."""
    sig = {}
    for i, p in enumerate(PARTICIPANTS):
        L_, S_ = part[:, i, 0], part[:, i, 1]
        ratio = L_ / (L_ + S_)
        sig[f"{p}: share long in index futures"] = ratio
        sig[f"{p}: change in share long, index futures"] = np.concatenate([[np.nan], np.diff(ratio)])
        calls = part[:, i, 4] - part[:, i, 6]
        puts = part[:, i, 5] - part[:, i, 7]
        sig[f"{p}: net index calls minus net index puts"] = (calls - puts) / np.nanmean(np.abs(calls - puts))
    pcr = idx_fo[:, 4] / idx_fo[:, 3]
    sig["NIFTY options put / call OI ratio"] = pcr
    sig["NIFTY options put / call OI ratio, change"] = np.concatenate([[np.nan], np.diff(pcr)])
    oi = idx_fo[:, 0]
    sig["NIFTY futures OI change"] = idx_fo[:, 1] / (oi - idx_fo[:, 1])
    sig["S&P 500, last session before the open"] = us
    return {k: np.asarray(v, np.float64) for k, v in sig.items()}


def us_returns(dates, log=print):
    """S&P 500 return of the last US session that closed before each Indian session (cached)."""
    path = DATA_DIR / "research" / "lead" / "spx.json"
    if not path.exists():
        from market.candles import yahoo
        bars = yahoo.daily("^GSPC", "10y")
        path.write_text(json.dumps([[b.date, b.c] for b in bars]))
    rows = json.loads(path.read_text())
    ud = np.array([np.datetime64(r[0]) for r in rows])
    uc = np.array([r[1] for r in rows])
    ur = np.concatenate([[np.nan], uc[1:] / uc[:-1] - 1])
    k = np.searchsorted(ud, np.asarray(dates), side="left") - 1     # last US date strictly before
    return np.where(k >= 0, ur[np.maximum(k, 0)], np.nan)


def targets(P, A):
    """Next-session intraday (09:20 open -> 15:15) and 5-day swing (next 09:20 -> 5th close), aligned to
    the signal day d; plus each stock's eligibility on the trade day."""
    O, C = P["o"], P["c"]
    S, D, _ = O.shape
    t1 = np.full((S, D), np.nan, np.float32)
    t5 = np.full((S, D), np.nan, np.float32)
    t1[:, :-1] = C[:, 1:, I.FLAT] / O[:, 1:, 1] - 1
    t5[:, :-5] = C[:, 5:, -1] / O[:, 1:-4, 1] - 1
    el = np.zeros((S, D), bool)
    el[:, :-1] = A["eligible"][:, 1:]
    return {"next day, 09:20 to 15:15": t1, "5-day swing": t5}, el


def _period_masks(dates):
    d = np.asarray(dates)
    trade = np.concatenate([d[1:], [d[-1] + np.timedelta64(1, "D")]])   # the trade happens the next session
    return I.periods(trade)


def score_stock(sig, tg, el, dates, costs):
    per = _period_masks(dates)
    out = []
    for name, x in sig.items():
        x = np.where(el, x, np.nan)
        res = {}
        for h, y in tg.items():
            y = np.where(el, y, np.nan)
            ic = I._ic(x, y)
            g = I._picks(x, y)
            res[h] = {}
            for p, m in per.items():
                gg = g[:, m].ravel()
                gg = gg[np.isfinite(gg)] * 1e4
                icp = ic[m]
                icp = icp[np.isfinite(icp)]
                res[h][p] = {"ic": float(icp.mean()) if icp.size else np.nan,
                             "ic_t": float(icp.mean() / (icp.std() + 1e-12) * np.sqrt(icp.size)) if icp.size else np.nan,
                             "gross": float(gg.mean()) if gg.size else np.nan, "g": gg}
        for h in tg:
            learn = res[h]["2021-23 (choose)"]
            sign = 1.0 if (learn["gross"] if np.isfinite(learn["gross"]) else 0) >= 0 else -1.0
            row = {"signal": name, "trade": h, "use": "follow" if sign > 0 else "fade", "periods": {}}
            for p, v in res[h].items():
                gg = sign * v["g"]
                row["periods"][p] = {"ic": round(sign * v["ic"], 4), "ic_t": round(sign * v["ic_t"], 1),
                                     "gross_bps": round(float(gg.mean()), 1) if gg.size else None,
                                     "net_bps": round(float(gg.mean() - costs[h]), 1) if gg.size else None,
                                     "win_pct": round(float((gg > costs[h]).mean() * 100), 1) if gg.size else None,
                                     "trades": int(gg.size)}
            out.append(row)
    out.sort(key=lambda r: -(r["periods"]["2025-26 (judge)"]["net_bps"] or -1e9))
    return out


def score_market(msig, idx, dates, costs):
    """Spearman with the NIFTY's next-session move; NIFTYBEES trades on the most extreme 20% of days
    (thresholds from 2021-23)."""
    from scipy.stats import spearmanr
    per = _period_masks(dates)
    n, b = idx["NIFTY"], idx["NIFTYBEES"]
    D = len(dates)
    ny = {"next day, 09:20 to 15:15": np.concatenate([n["c"][1:, I.FLAT] / n["o"][1:, 1] - 1, [np.nan]])}
    by = {"next day, 09:20 to 15:15": np.concatenate([b["c"][1:, I.FLAT] / b["o"][1:, 1] - 1, [np.nan]])}
    gap = np.concatenate([n["o"][1:, 0] / n["c"][:-1, -1] - 1, [np.nan]])
    out = []
    for name, x in list(msig.items()) + [("NIFTY gap minus the S&P 500's move (known at 09:15)",
                                          gap - msig["S&P 500, last session before the open"])]:
        for h in ny:
            learn = per["2021-23 (choose)"]
            ok = np.isfinite(x) & np.isfinite(ny[h])
            if (ok & learn).sum() < 200:
                continue
            ic_l = spearmanr(x[ok & learn], ny[h][ok & learn]).correlation
            sign = 1.0 if ic_l >= 0 else -1.0
            xs = sign * x
            lx = xs[learn & np.isfinite(xs)]
            hi, lo = np.quantile(lx, 0.8), np.quantile(lx, 0.2)
            row = {"signal": name, "trade": h, "use": "follow" if sign > 0 else "fade", "periods": {}}
            for p, m in per.items():
                okp = ok & m
                icp = spearmanr(xs[okp], ny[h][okp]).correlation if okp.sum() > 50 else np.nan
                t = m & np.isfinite(xs) & np.isfinite(by[h])
                g = np.concatenate([by[h][t & (xs >= hi)], -by[h][t & (xs <= lo)]]) * 1e4
                row["periods"][p] = {"ic": round(float(icp), 4), "days": int(okp.sum()),
                                     "gross_bps": round(float(g.mean()), 1) if g.size else None,
                                     "net_bps": round(float(g.mean() - costs[h]), 1) if g.size else None,
                                     "win_pct": round(float((g > costs[h]).mean() * 100), 1) if g.size else None,
                                     "trades": int(g.size)}
            out.append(row)
    out.sort(key=lambda r: -(r["periods"]["2025-26 (judge)"]["net_bps"] or -1e9))
    return out


def combined(sig, tg, el, dates, costs, log=print, seed=1):
    """All stock-level positioning signals together, one gradient-boosted model per trade type."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    per = _period_masks(dates)
    S, D = el.shape
    X3 = np.stack(list(sig.values()), axis=-1)
    out = []
    for h, y in tg.items():
        ok = el & np.isfinite(y)
        with np.errstate(all="ignore"):
            y_rel = y - np.nanmean(np.where(ok, y, np.nan), axis=0, keepdims=True)
        learn = ok & per["2021-23 (choose)"][None, :]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
                                              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                              n_iter_no_change=20, random_state=seed)
        model.fit(X3[learn], np.clip(y_rel[learn], -0.15, 0.15))
        pred = np.full((S, D), np.nan, np.float32)
        pred[ok] = model.predict(X3[ok])
        yy = np.where(ok, y, np.nan)
        ic = I._ic(pred, yy)
        g = I._picks(pred, yy)
        res = {}
        for p, m in per.items():
            gg = g[:, m].ravel()
            gg = gg[np.isfinite(gg)] * 1e4
            icp = ic[m]
            res[p] = {"ic": round(float(np.nanmean(icp)), 4), "gross_bps": round(float(gg.mean()), 1),
                      "net_bps": round(float(gg.mean() - costs[h]), 1), "win_pct": round(float((gg > costs[h]).mean() * 100), 1),
                      "trades": int(gg.size)}
        log(f"  combined positioning model ({h}): " + "; ".join(f"{p} IC {v['ic']} net {v['net_bps']} bps" for p, v in res.items()))
        out.append({"trade": h, "periods": res, "iterations": int(model.n_iter_)})
    return out


def run(log=print):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    syms, inds, dates, P, idx = I.panel(log)
    A = I.aux(syms, inds, dates, P, idx)
    f, idx_fo, part = load_positioning(syms, dates)
    have = {"delivery": int(np.isfinite(f["dper"]).any(0).sum()), "fo": int(np.isfinite(f["fut_oi"]).any(0).sum()),
            "participants": int(np.isfinite(part[:, 2, 0]).sum()), "fo_stocks": int(np.isfinite(f["fut_oi"]).any(1).sum())}
    log(f"  positioning data: {have}")
    # signals at index d trade on session d+1, so the US session that matters is the last one
    # before d+1 opens (it closes around 01:30-02:30 IST that morning)
    us = us_returns(dates, log)
    us = np.concatenate([us[1:], [np.nan]])
    sig = stock_signals(f, P)
    msig = market_signals(idx_fo, part, dates, us)
    tg, el = targets(P, A)
    costs = {"next day, 09:20 to 15:15": I.cost_bps(), "5-day swing": swing_cost_bps()}
    log(f"  costs per trade: intraday {costs['next day, 09:20 to 15:15']:.1f} bps, 5-day swing {costs['5-day swing']:.1f} bps")
    rows = score_stock(sig, tg, el, dates, costs)
    mrows = score_market(msig, idx, dates, costs)
    comb = combined(sig, tg, el, dates, costs, log)
    doc = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "data": have, "costs_bps": {k: round(v, 1) for k, v in costs.items()},
           "k": K, "stock": rows, "market": mrows, "combined": comb}
    (RES / "lead_daily.json").write_text(json.dumps(doc, indent=1, default=float))
    log(f"  saved results/lead_daily.json ({time.time() - t0:.0f}s)")
    return doc
