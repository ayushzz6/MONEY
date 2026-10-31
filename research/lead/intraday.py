"""Round 4, part 1: intraday leading signals on 5-minute bars (NIFTY 500, Oct 2021 - Oct 2026).

Question: does anything known at the close of a bar predict the move from the NEXT bar's open over
the next 5 / 15 / 30 / 60 minutes (or to 15:15) by more than a trade costs?

Signals, all computed from bars that have closed:
- own price: last 1 / 3 / 6 / 12 bars, since the open, the gap
- lead-lag: sector index, NIFTY x beta, and the industry's three largest stocks, minus the stock's own
  move over the same bars (does the laggard catch up?)
- order flow proxies: close location x volume (accumulation / distribution), the part of it the price
  has not shown yet, volume surges with their direction
- market-wide (judged on the NIFTY and traded through NIFTYBEES): change in breadth, India VIX,
  Bank NIFTY and small caps against the NIFTY, the NIFTY's own last bars

Each signal's direction (follow or fade) and its best horizon are fixed on 2021-23, checked on 2024,
and judged once on 2025-26. Costs: Indian intraday charges on Rs 1 lakh a trade plus slippage.
"""
import json
import time

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.lib.sectors import SECTOR_TOKENS, sector_index

OUT = DATA_DIR / "research" / "lead"
RES = DATA_DIR / "research" / "results"
EVAL = np.arange(2, 63, 3)                   # signals at the close of 09:30, 09:45, ... 14:30
FLAT = 71                                    # the 15:10 bar closes at 15:15
HORIZONS = {"5 min": 1, "15 min": 3, "30 min": 6, "1 hour": 12, "to 15:15": None}
LEARN_END = np.datetime64("2023-12-31")
CUT_END = np.datetime64("2024-12-31")
K = 3                                        # long the 3 strongest, short the 3 weakest, each time
NOTIONAL = 100_000.0
MIN_TURNOVER = 1e8                           # Rs 10 crore a day, previous 20 days
MARKET = ["NIFTY", "BANKNIFTY", "INDIAVIX", "NIFTYBEES"]


# ------------------------------------------------------------------ data
def panel(log=print):
    """(S, D, 75) float32 o/h/l/c/v for the NIFTY 500 stocks and (D, 75) for the indices, cached."""
    path = OUT / "panel.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        P = {k: z[k] for k in "ohlcv"}
        idx = {n: {k: z[f"{n}__{k}"] for k in "ohlcv"} for n in z["idx_names"]}
        return list(z["syms"]), list(z["inds"]), z["dates"], P, idx
    t0 = time.time()
    uni = data.nifty500()
    dates = data.load("NIFTY").dates
    D = len(dates)
    syms, inds, P = [], [], {k: [] for k in "ohlcv"}
    for s, ind in sorted(uni.items()):
        m = data.load(s)
        if m is None or m.n < 250:
            continue
        pos = np.searchsorted(dates, m.dates)
        ok = (pos < D) & (dates[np.minimum(pos, D - 1)] == m.dates)
        for k in "ohlcv":
            a = np.full((D, 75), np.nan, np.float32)
            a[pos[ok]] = getattr(m, k)[ok]
            P[k].append(a)
        syms.append(s)
        inds.append(ind)
    P = {k: np.stack(v) for k, v in P.items()}
    idx = {}
    for n in MARKET + list(SECTOR_TOKENS):
        m = data.load(n)
        if m is None:
            continue
        pos = np.searchsorted(dates, m.dates)
        ok = (pos < D) & (dates[np.minimum(pos, D - 1)] == m.dates)
        idx[n] = {}
        for k in "ohlcv":
            a = np.full((D, 75), np.nan, np.float32)
            a[pos[ok]] = getattr(m, k)[ok]
            idx[n][k] = a
    OUT.mkdir(parents=True, exist_ok=True)
    flat = {f"{n}__{k}": v for n, d in idx.items() for k, v in d.items()}
    np.savez(path, syms=np.array(syms), inds=np.array(inds), dates=dates, idx_names=np.array(list(idx)),
             **P, **flat)
    log(f"  panel: {len(syms)} stocks x {D} sessions ({time.time() - t0:.0f}s)")
    return syms, inds, dates, P, idx


def prior_mean(x, n, axis=1, min_frac=0.5):
    """Mean of the previous n values along `axis` (today excluded), ignoring NaN."""
    x = np.moveaxis(x, axis, 0)
    v = np.nan_to_num(x, nan=0.0).astype(np.float64)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate([np.zeros_like(v[:1]), np.cumsum(v, 0)])
    cc = np.concatenate([np.zeros_like(c[:1]), np.cumsum(c, 0)])
    i = np.arange(x.shape[0])
    lo = np.maximum(i - n, 0)
    s, k = cv[i] - cv[lo], cc[i] - cc[lo]
    out = np.where(k >= n * min_frac, s / np.maximum(k, 1), np.nan)
    return np.moveaxis(out.astype(np.float32), 0, axis)


def cost_bps(notional=NOTIONAL, slip_bps=None):
    """Round-trip charges + slippage on one intraday trade, in basis points of the notional."""
    c = Settings().get("costs")
    slip = float(Settings().get("slippage_bps", 5) if slip_bps is None else slip_bps)
    brk = min(c["brokerage_per_order_inr"], notional * c["brokerage_pct"] / 100)
    exch = notional * c["exchange_txn_pct"] / 100
    sebi = notional * c["sebi_pct"] / 100
    gst = (brk + exch + sebi) * c["gst_pct"] / 100
    buy = brk + exch + sebi + gst + notional * c["stamp_buy_pct"] / 100
    sell = brk + exch + sebi + gst + notional * c["stt_sell_pct"] / 100
    return (buy + sell) / notional * 1e4 + 2 * slip


# ------------------------------------------------------------------ signals
def _retk(C, O, k):
    """Return over the last k bars at each EVAL slot (from the day's open when fewer bars exist)."""
    b = EVAL - k
    base = np.where(b >= 0, C[..., np.maximum(b, 0)], O[..., :1])
    return C[..., EVAL] / base - 1


def _windowed_mean(x, k):
    """Mean of x over the k bars ending at each EVAL slot (fewer at the start of the day)."""
    cs = np.nancumsum(x, axis=-1)
    hi = cs[..., EVAL]
    lo_i = EVAL - k
    lo = np.where(lo_i >= 0, cs[..., np.maximum(lo_i, 0)], 0.0)
    n = np.minimum(EVAL + 1, k)
    return (hi - lo) / n


def _resid(y, x):
    """y minus its cross-sectional least-squares fit on x, at every (day, slot)."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(0)
    xm = np.where(ok, x, 0).sum(0) / np.maximum(n, 1)
    ym = np.where(ok, y, 0).sum(0) / np.maximum(n, 1)
    dx, dy = np.where(ok, x - xm, 0), np.where(ok, y - ym, 0)
    b = (dx * dy).sum(0) / np.maximum((dx * dx).sum(0), 1e-12)
    return np.where(ok, y - ym - b * (x - xm), np.nan)


def aux(syms, inds, dates, P, idx):
    """Things known before the session: liquidity, average volume per slot, beta, sector, leaders."""
    O, C, V = P["o"], P["c"], P["v"]
    close = C[:, :, -1]
    turnover = np.nansum(V * C, axis=2)
    turnover[np.isnan(O[:, :, 0])] = np.nan
    avg_to = prior_mean(turnover, 20)
    prev_close = np.concatenate([np.full((len(syms), 1), np.nan, np.float32), close[:, :-1]], axis=1)
    eligible = (avg_to >= MIN_TURNOVER) & (prev_close >= 50) & np.isfinite(O[:, :, 0])
    avg_v = prior_mean(V, 20)                                     # (S, D, 75)
    nc = idx["NIFTY"]["c"][:, -1]
    rm = nc[1:] / nc[:-1] - 1
    rs = close[:, 1:] / close[:, :-1] - 1
    rm = np.concatenate([[np.nan], rm]).astype(np.float32)
    rs = np.concatenate([np.full((len(syms), 1), np.nan), rs], axis=1).astype(np.float32)
    rm2 = np.broadcast_to(rm, rs.shape)
    exy = prior_mean(rs * rm2, 60)
    ex, ey = prior_mean(rs, 60), prior_mean(rm2.copy(), 60)
    ey2 = prior_mean(rm2 * rm2, 60)
    beta = np.clip(np.nan_to_num((exy - ex * ey) / np.maximum(ey2 - ey * ey, 1e-9), nan=1.0), 0.0, 3.0)
    secs = sorted({sector_index(s, i) for s, i in zip(syms, inds)} & set(idx))
    sec_id = np.array([secs.index(sector_index(s, i)) if sector_index(s, i) in secs else -1
                       for s, i in zip(syms, inds)])
    # the industry's three largest stocks by previous-20-day turnover (leaders); others follow?
    leader = np.zeros_like(eligible)
    groups = {}
    for j, g in enumerate(inds):
        groups.setdefault(g, []).append(j)
    for g, mem in groups.items():
        if len(mem) < 6:
            continue
        mem = np.array(mem)
        t = np.where(eligible[mem], avg_to[mem], -np.inf)
        top = np.argsort(-t, axis=0)[:3]                          # (3, D)
        for r in range(3):
            leader[mem[top[r]], np.arange(len(dates))] = True
        leader[mem] &= eligible[mem]
    return {"eligible": eligible, "avg_v": avg_v, "beta": beta, "secs": secs, "sec_id": sec_id,
            "leader": leader, "groups": groups}


def stock_signals(P, idx, A):
    """Cross-sectional signals, each (S, D, len(EVAL)), known at the close of the EVAL bar."""
    O, H, L, C, V = (P[k] for k in "ohlcv")
    sig = {}
    r = {k: _retk(C, O, k) for k in (1, 3, 6, 12)}
    for k in (1, 3, 6, 12):
        sig[f"Own move, last {k} bar{'s' if k > 1 else ''}"] = r[k]
    sig["Own move since the open"] = C[..., EVAL] / O[..., :1] - 1
    prev_close = np.concatenate([np.full(C[:, :1, -1:].shape, np.nan, np.float32), C[:, :-1, -1:]], axis=1)
    sig["Opening gap"] = np.broadcast_to(O[..., :1] / prev_close - 1, r[1].shape).copy()
    # lead-lag
    sc = np.stack([idx[s]["c"] for s in A["secs"]])               # (n_sec, D, 75)
    so = np.stack([idx[s]["o"] for s in A["secs"]])
    nC, nO = idx["NIFTY"]["c"], idx["NIFTY"]["o"]
    beta = A["beta"][..., None]
    for k in (1, 3, 6):
        srk = _retk(sc, so, k)[np.maximum(A["sec_id"], 0)]
        srk[A["sec_id"] < 0] = np.nan
        sig[f"Sector index minus stock, last {k} bar{'s' if k > 1 else ''}"] = srk - r[k]
        sig[f"NIFTY x beta minus stock, last {k} bar{'s' if k > 1 else ''}"] = beta * _retk(nC, nO, k)[None] - r[k]
        lead = np.full_like(r[k], np.nan)
        for g, mem in A["groups"].items():
            mem = np.array(mem)
            L_ = A["leader"][mem][..., None]
            lr = np.where(L_, r[k][mem], np.nan)
            with np.errstate(all="ignore"):
                m = np.nanmean(lr, axis=0) if L_.any() else np.full(r[k].shape[1:], np.nan)
            lead[mem] = np.where(L_, np.nan, m[None] - r[k][mem])
        sig[f"Industry leaders minus stock, last {k} bar{'s' if k > 1 else ''}"] = lead
    # order flow proxies
    rng = H - L
    clv = np.where(rng > 0, (2 * C - H - L) / np.where(rng > 0, rng, 1), 0.0).astype(np.float32)
    rel_v = V / np.where(A["avg_v"] > 0, A["avg_v"], np.nan)
    flow = clv * rel_v
    for k in (1, 3, 6, 12):
        sig[f"Order flow (close location x volume), last {k} bar{'s' if k > 1 else ''}"] = _windowed_mean(flow, k)
    for k in (3, 6):
        f = _windowed_mean(flow, k)
        S_, D_, E_ = f.shape
        sig[f"Order flow the price has not shown yet, last {k} bars"] = _resid(
            f.reshape(S_, -1), r[k].reshape(S_, -1)).reshape(S_, D_, E_)
    sig["Volume surge x bar direction"] = rel_v[..., EVAL] * np.sign(C[..., EVAL] - O[..., EVAL])
    for k in (3, 6):
        sig[f"Relative volume x direction, last {k} bars"] = _windowed_mean(rel_v, k) * np.sign(r[k])
    return {n: s.astype(np.float32) for n, s in sig.items()}


def market_signals(P, idx, A):
    """Market-wide signals, each (D, len(EVAL))."""
    O, C = P["o"], P["c"]
    elig = A["eligible"][..., None]
    above = np.where(elig, (C > O[..., :1]).astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        breadth = np.nanmean(above, axis=0)                       # (D, 75) share of stocks above their open
    n = idx["NIFTY"]
    sig = {"Breadth (share of stocks above their open)": breadth[:, EVAL] - 0.5}
    for k in (1, 3, 6):
        b = EVAL - k
        base = np.where(b >= 0, breadth[:, np.maximum(b, 0)], 0.5)
        sig[f"Breadth change, last {k} bar{'s' if k > 1 else ''}"] = breadth[:, EVAL] - base
        sig[f"India VIX change, last {k} bar{'s' if k > 1 else ''}"] = _retk(idx["INDIAVIX"]["c"], idx["INDIAVIX"]["o"], k)
        sig[f"NIFTY own move, last {k} bar{'s' if k > 1 else ''}"] = _retk(n["c"], n["o"], k)
    for k in (3, 6):
        nr = _retk(n["c"], n["o"], k)
        sig[f"Bank NIFTY minus NIFTY, last {k} bars"] = _retk(idx["BANKNIFTY"]["c"], idx["BANKNIFTY"]["o"], k) - nr
        sig[f"Small caps minus NIFTY, last {k} bars"] = _retk(idx["SEC_SMLCAP100"]["c"], idx["SEC_SMLCAP100"]["o"], k) - nr
    return {k: v.astype(np.float32) for k, v in sig.items()}


def forward(C, O):
    """Return from the next bar's open to the close h bars on (capped at 15:15), per horizon."""
    out = {}
    nxt = O[..., EVAL + 1]
    for name, h in HORIZONS.items():
        end = np.full_like(EVAL, FLAT) if h is None else np.minimum(EVAL + h, FLAT)
        out[name] = C[..., end] / nxt - 1
    return out


# ------------------------------------------------------------------ scoring
def _xs_rank(a):
    """Cross-sectional ranks in [0, 1] along axis 0; NaN stays NaN."""
    ok = np.isfinite(a)
    order = np.where(ok, a, np.inf).argsort(axis=0, kind="stable")
    rk = np.empty(a.shape, np.float32)
    np.put_along_axis(rk, order, np.arange(a.shape[0], dtype=np.float32)[:, None], axis=0)
    n = ok.sum(0)
    rk = rk / np.maximum(n - 1, 1)
    rk[~ok] = np.nan
    return rk


def _ic(x, y):
    """Spearman correlation per column (cross-section)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
    rx, ry = _xs_rank(x), _xs_rank(y)
    with np.errstate(all="ignore"):
        rx = rx - np.nanmean(rx, axis=0)
        ry = ry - np.nanmean(ry, axis=0)
        return np.nansum(rx * ry, 0) / np.sqrt(np.nansum(rx * rx, 0) * np.nansum(ry * ry, 0))


def _picks(x, y, k=K):
    """Forward returns of the k highest and k lowest x per column (shorts sign-flipped): (2k, N)."""
    ok = np.isfinite(x) & np.isfinite(y)
    hi = np.where(ok, x, -np.inf).argsort(axis=0)[::-1][:k]
    lo = np.where(ok, x, np.inf).argsort(axis=0)[:k]
    n = ok.sum(0)
    g_hi = np.take_along_axis(y, hi, 0)
    g_lo = -np.take_along_axis(y, lo, 0)
    g = np.concatenate([g_hi, g_lo])
    g[:, n < 4 * k] = np.nan
    return g


def periods(dates):
    d = np.asarray(dates)
    return {"2021-23 (choose)": d <= LEARN_END, "2024 (check)": (d > LEARN_END) & (d <= CUT_END),
            "2025-26 (judge)": d > CUT_END}


def score_cross(sig, fwd, elig, dates, cost):
    """For each signal and horizon: rank IC and the k-long / k-short trades, by period."""
    per = periods(dates)
    S, D, E = elig.shape
    out = {}
    fwd_f = {h: np.where(elig, f, np.nan).reshape(S, -1) for h, f in fwd.items()}
    for name, x in sig.items():
        xf = np.where(elig, x, np.nan).reshape(S, -1)
        out[name] = {}
        for h, y in fwd_f.items():
            ic = _ic(xf, y).reshape(D, E)
            g = _picks(xf, y).reshape(2 * K, D, E)
            row = {}
            for p, m in per.items():
                icd = np.nanmean(ic[m], axis=1)
                icd = icd[np.isfinite(icd)]
                gg = g[:, m].ravel()
                gg = gg[np.isfinite(gg)]
                row[p] = {"ic": float(icd.mean()), "ic_t": float(icd.mean() / (icd.std() + 1e-12) * np.sqrt(len(icd))),
                          "gross_bps": float(gg.mean() * 1e4), "trades": int(gg.size)}
            out[name][h] = row
    return finalize(out, cost)


def finalize(out, cost):
    """Fix each signal's direction and horizon on 2021-23; report all periods with that choice."""
    res = []
    for name, by_h in out.items():
        best_h = max(by_h, key=lambda h: abs(by_h[h]["2021-23 (choose)"]["gross_bps"]))
        sign = 1.0 if by_h[best_h]["2021-23 (choose)"]["gross_bps"] >= 0 else -1.0
        row = {"signal": name, "use": "follow" if sign > 0 else "fade", "horizon": best_h, "periods": {}}
        for p, v in by_h[best_h].items():
            g = sign * v["gross_bps"]
            row["periods"][p] = {"ic": round(sign * v["ic"], 4), "ic_t": round(sign * v["ic_t"], 1),
                                 "gross_bps": round(g, 2), "net_bps": round(g - cost, 2), "trades": v["trades"]}
        row["all_horizons"] = {h: {p: {"ic": round(sign * v[p]["ic"], 4), "gross_bps": round(sign * v[p]["gross_bps"], 2)}
                                   for p in v} for h, v in by_h.items()}
        res.append(row)
    res.sort(key=lambda r: -r["periods"]["2025-26 (judge)"]["gross_bps"])
    return res


def win_rates(sig, fwd, elig, rows, cost):
    """Share of trades that made money after costs, for each signal at its chosen horizon and direction."""
    S = elig.shape[0]
    for r in rows:
        x = np.where(elig, sig[r["signal"]], np.nan).reshape(S, -1)
        y = np.where(elig, fwd[r["horizon"]], np.nan).reshape(S, -1)
        s = 1.0 if r["use"] == "follow" else -1.0
        g = _picks(s * x, y).reshape(2 * K, *elig.shape[1:])
        for p, m in r["_per"].items():
            gg = g[:, m].ravel()
            gg = gg[np.isfinite(gg)]
            r["periods"][p]["win_pct"] = round(float((gg * 1e4 > cost).mean() * 100), 1) if gg.size else None
    return rows


def score_market(msig, idx, dates, cost):
    """Market-wide signals: Spearman with the NIFTY's forward move, and NIFTYBEES trades when the
    signal is in its most extreme 10% (thresholds from 2021-23)."""
    per = periods(dates)
    n, b = idx["NIFTY"], idx["NIFTYBEES"]
    fn = forward(n["c"], n["o"])
    fb = forward(b["c"], b["o"])
    res = []
    for name, x in msig.items():
        best = None
        for h in HORIZONS:
            learn = per["2021-23 (choose)"]
            ok = np.isfinite(x) & np.isfinite(fn[h])
            xl, yl = x[learn & True][ok[learn]], fn[h][learn][ok[learn]]
            if xl.size < 500:
                continue
            ic = _spearman(xl, yl)
            if best is None or abs(ic) > abs(best[1]):
                best = (h, ic)
        if best is None:
            continue
        h, ic = best
        sign = 1.0 if ic >= 0 else -1.0
        xs = sign * x
        lx = xs[per["2021-23 (choose)"]]
        lx = lx[np.isfinite(lx)]
        hi_t, lo_t = np.quantile(lx, 0.9), np.quantile(lx, 0.1)
        row = {"signal": name, "use": "follow" if sign > 0 else "fade", "horizon": h, "periods": {}}
        for p, m in per.items():
            ok = np.isfinite(xs[m]) & np.isfinite(fn[h][m])
            ic_p = _spearman(xs[m][ok], fn[h][m][ok]) if ok.sum() > 100 else np.nan
            xb, yb = xs[m], fb[h][m]
            t = np.isfinite(xb) & np.isfinite(yb)
            g = np.concatenate([yb[t & (xb >= hi_t)], -yb[t & (xb <= lo_t)]]) * 1e4
            row["periods"][p] = {"ic": round(float(ic_p), 4), "gross_bps": round(float(g.mean()), 2) if g.size else None,
                                 "net_bps": round(float(g.mean() - cost), 2) if g.size else None,
                                 "win_pct": round(float((g > cost).mean() * 100), 1) if g.size else None,
                                 "trades": int(g.size)}
        res.append(row)
    res.sort(key=lambda r: -(r["periods"]["2025-26 (judge)"]["gross_bps"] or -1e9))
    return res


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


# ------------------------------------------------------------------ combined model
def combined(sig, msig, fwd, elig, dates, cost, horizons=("15 min", "30 min", "to 15:15"), log=print, seed=1):
    """All signals together in one gradient-boosted model per horizon, trained on 2021-23 to predict each
    stock's move relative to the others; judged on 2024 and 2025-26 with the same k-long / k-short trades."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    S, D, E = elig.shape
    per = periods(dates)
    slot = np.broadcast_to(EVAL[None, None, :].astype(np.float32), elig.shape)
    cols = list(sig.values()) + [np.broadcast_to(v[None], elig.shape) for v in msig.values()] + [slot]
    ok = elig.copy()
    for h in horizons:
        ok &= np.isfinite(fwd[h])
    rows_all = np.flatnonzero(ok.ravel())
    X = np.empty((rows_all.size, len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = np.ascontiguousarray(c).ravel()[rows_all]
    day_of = np.broadcast_to(np.arange(D)[None, :, None], elig.shape).ravel()[rows_all]
    learn = per["2021-23 (choose)"][day_of]
    rng = np.random.default_rng(seed)
    tr = rng.choice(np.flatnonzero(learn), size=min(2_500_000, int(learn.sum())), replace=False)
    out = []
    for h in horizons:
        y = fwd[h]
        with np.errstate(all="ignore"):
            y_rel = (y - np.nanmean(np.where(elig, y, np.nan), axis=0, keepdims=True)).ravel()[rows_all]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=2000, l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
        model.fit(X[tr], np.clip(y_rel[tr], -0.05, 0.05))
        pred = np.full(elig.size, np.nan, np.float32)
        pred[rows_all] = model.predict(X)
        yf, pf = np.where(ok, y, np.nan).reshape(S, -1), pred.reshape(S, -1)
        ic = _ic(pf, yf).reshape(D, E)
        g = _picks(pf, yf).reshape(2 * K, D, E)
        res = {}
        for p, m in per.items():
            icd = np.nanmean(ic[m], axis=1)
            gg = g[:, m].ravel()
            gg = gg[np.isfinite(gg)] * 1e4
            res[p] = {"ic": round(float(np.nanmean(icd)), 4), "gross_bps": round(float(gg.mean()), 2),
                      "net_bps": round(float(gg.mean() - cost), 2), "win_pct": round(float((gg > cost).mean() * 100), 1),
                      "trades": int(gg.size)}
        log(f"  combined model ({h}): " + "; ".join(f"{p} IC {v['ic']} gross {v['gross_bps']} bps" for p, v in res.items()))
        out.append({"horizon": h, "features": len(cols), "trained_rows": int(tr.size), "iterations": int(model.n_iter_),
                    "periods": res})
    return out


def leak_test(P, idx, A, samples=3, seed=7, log=print):
    """Blank every bar after a random signal time (that day's later bars and all later days) and check
    that the signals at that time do not change."""
    sig, msig, _ = compute(P, idx, A)
    rng = np.random.default_rng(seed)
    D = P["c"].shape[1]
    bad = []
    for _ in range(samples):
        d, e = int(rng.integers(300, D - 5)), int(rng.integers(0, len(EVAL)))
        t = EVAL[e]
        P2 = {k: v.copy() for k, v in P.items()}
        idx2 = {n: {k: v.copy() for k, v in dd.items()} for n, dd in idx.items()}
        for arr in list(P2.values()):
            arr[:, d, t + 1:] = np.nan
            arr[:, d + 1:] = np.nan
        for dd in idx2.values():
            for arr in dd.values():
                arr[d, t + 1:] = np.nan
                arr[d + 1:] = np.nan
        s2, m2, _ = compute(P2, idx2, A)
        for n in sig:
            a, b = sig[n][:, d, e], s2[n][:, d, e]
            if not np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-7):
                bad.append(n)
        for n in msig:
            if not np.allclose(msig[n][d, e], m2[n][d, e], equal_nan=True, rtol=1e-5, atol=1e-7):
                bad.append(n)
        log(f"  leak test: day {d}, signal time slot {t}: {len(set(bad))} signals changed so far")
    return sorted(set(bad))


# ------------------------------------------------------------------ run
def compute(P, idx, A):
    sig = stock_signals(P, idx, A)
    msig = market_signals(P, idx, A)
    fwd = forward(P["c"], P["o"])
    return sig, msig, fwd


def run(log=print, leak=True):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    syms, inds, dates, P, idx = panel(log)
    log(f"  {len(syms)} stocks, {len(dates)} sessions, {len(EVAL)} signal times a day")
    A = aux(syms, inds, dates, P, idx)
    if leak:
        bad = leak_test(P, idx, A, log=log)
        if bad:
            raise RuntimeError(f"lookahead in: {bad}")
    sig, msig, fwd = compute(P, idx, A)
    elig = np.broadcast_to(A["eligible"][..., None], fwd["5 min"].shape)
    cost = cost_bps()
    cost3 = cost_bps(slip_bps=3)
    log(f"  cost per trade: {cost:.1f} bps (Rs {cost * NOTIONAL / 1e4:,.0f} on Rs 1 lakh); {cost3:.1f} bps at 3 bps slippage")
    rows = score_cross(sig, fwd, elig, dates, cost)
    per = periods(dates)
    for r in rows:
        r["_per"] = {p: m for p, m in per.items()}
    win_rates(sig, fwd, elig, rows, cost)
    for r in rows:
        r.pop("_per", None)
    log(f"  scored {len(rows)} stock signals ({time.time() - t0:.0f}s)")
    mrows = score_market(msig, idx, dates, cost)
    log(f"  scored {len(mrows)} market signals ({time.time() - t0:.0f}s)")
    comb = combined(sig, msig, fwd, elig, dates, cost, log=log)
    doc = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "stocks": len(syms), "sessions": len(dates),
           "signal_times": [f"{9 + (15 + 5 * (t + 1)) // 60:02d}:{(15 + 5 * (t + 1)) % 60:02d}" for t in EVAL],
           "cost_bps": round(cost, 2), "cost_bps_3": round(cost3, 2), "k": K, "stock": rows, "market": mrows,
           "combined": comb}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lead_intraday.json").write_text(json.dumps(doc, indent=1))
    log(f"  saved results/lead_intraday.json ({time.time() - t0:.0f}s)")
    return doc
