"""Round 5 grid: buying NIFTY / BANKNIFTY options with a 1:2 risk : reward, on real option-chain data.

Variant = signal x underlying x option-chain filter x contract (delta, expiry) x exit.
- entry at the open of the bar after the signal, the strike whose delta is nearest the target,
  priced from the previous close's real IV smile (research.opt.sim)
- exits: premium stop x% below the fill and target 2x% above (1:2), optionally moving the stop to
  break-even once the option is up x%; or an index-level stop with the premium target set at twice
  the premium risk; whatever is left closes at 15:15
- each trade risks about Rs 2,000 (whole lots; 1 lot allowed up to Rs 3,000; premium paid <= Rs 1 lakh),
  NSE charges on every order and slippage of max(Rs 0.5, 0.5% of premium) a side, double on stops
- chosen on Oct 2021 - Dec 2024, judged once on Jan 2025 - Oct 2026
"""
import json
import time
from itertools import product

import numpy as np

from core.config.settings import DATA_DIR
from research.opt import model as M
from research.opt import signals as SG
from research.opt import sim as SIM

RES = DATA_DIR / "research" / "results"
OUT = DATA_DIR / "research" / "opt"
DELTAS = (0.35, 0.5, 0.65)
EXPIRY_MODES = ("nearest expiry", "skip expiry day")
CONTRACTS = list(product(DELTAS, EXPIRY_MODES))
STOPS = (0.15, 0.20, 0.25, 0.30, 0.40)
INDEX_STOPS = (0.0025, 0.004)
EXITS = [("premium", x, False) for x in STOPS] + [("premium", x, True) for x in STOPS] + [("index", p, False) for p in INDEX_STOPS]
FLAT = 71
RISK = 2000.0
MAX_RISK_1LOT = 3000.0
CAPITAL = 100000.0
LEARN_END = np.datetime64("2024-12-31")
GST = 0.18


def exit_label(ex):
    kind, x, be = ex
    if kind == "premium":
        return f"stop -{int(x * 100)}% / target +{int(2 * x * 100)}% of premium" + (", stop to cost at +1R" if be else "")
    return f"index stop {x * 100:.2f}% / target 2x the premium risk"


def charges(value, side):
    """option_order_charges, vectorised (research.costs)."""
    value = np.asarray(value, float)
    txn = value * 0.03503 / 100
    sebi = value * 0.0001 / 100
    stt = value * 0.1 / 100 if side == "sell" else 0.0
    stamp = value * 0.003 / 100 if side == "buy" else 0.0
    return 20.0 + txn + sebi + stt + stamp + GST * (20.0 + txn + sebi)


def slip(p):
    return np.maximum(0.5, 0.005 * np.asarray(p, float))


# ------------------------------------------------------------------ events and filters
FILTERS = ["none", "IV rank below 35%", "IV rank above 65%", "IV below realized volatility", "expiry day only",
           "not expiry day", "put/call OI ratio agrees", "put/call OI ratio disagrees", "0.5%+ room to the OI wall",
           "breadth agrees", "on max pain's side", "VIX up since yesterday"]


def events(mk, br, log=print):
    ev = []
    for u in (0, 1):
        sig = SG.compute(mk, u, br)
        for name, (slot, side) in sig.items():
            for d in np.flatnonzero((slot >= 0) & (slot + 1 <= 70)):
                ev.append((name, u, int(d), int(slot[d]), int(side[d])))
    log(f"  {len(ev):,} signal events ({len(SG.NAMES)} signals x 2 indices)")
    return ev


def filter_flags(mk, br, ev):
    f = mk.f
    n = len(ev)
    flags = np.zeros((n, len(FILTERS)), bool)
    for i, (name, u, d, slot, side) in enumerate(ev):
        p = d - 1
        S = mk.idx[u]["o"][d, slot + 1]
        e = mk.expiry_for(d, u)
        ivr, ivrv, pcr = f["iv_rank"][p, u], f["iv_minus_rv"][p, u], f["pcr_front"][p, u]
        cw, pw, mp = f["call_wall"][p, u], f["put_wall"][p, u], f["max_pain"][p, u]
        b = br[d, slot]
        room = (cw / S - 1) if side > 0 else (1 - pw / S)
        flags[i] = [True, ivr < 0.35, ivr > 0.65, ivrv < 0, e == d, e is not None and e != d,
                    (side > 0 and pcr >= 1.0) or (side < 0 and pcr < 1.0), (side > 0 and pcr < 1.0) or (side < 0 and pcr >= 1.0),
                    room >= 0.005, (side > 0 and b > 0.55) or (side < 0 and b < 0.45),
                    (side > 0 and S < mp) or (side < 0 and S > mp), mk.vix_ratio_c[d, slot] > 1.0]
    return flags


# ------------------------------------------------------------------ paths and exits
def _first(mask):
    any_ = mask.any(1)
    return np.where(any_, mask.argmax(1), 999)


def simulate(mk, ev, iv_mode="vix", slip_mult=1.0, log=print):
    """Every event x contract -> option path -> every exit. Returns per-path metadata and (paths, exits) results."""
    t0 = time.time()
    meta = []                                                   # (event, contract, d, u, e, K, cp, slot, lot)
    for i, (name, u, d, slot, side) in enumerate(ev):
        S = mk.idx[u]["o"][d, slot + 1]
        if not np.isfinite(S):
            continue
        for c, (dl, mode) in enumerate(CONTRACTS):
            e = mk.expiry_for(d, u, skip_expiry_day=(mode == "skip expiry day"))
            if e is None or e >= len(mk.dates) + 40:
                continue
            K = mk.pick_strike(d, u, e, S, side, dl, slot)
            if K is None:
                continue
            meta.append((i, c, d, u, e, K, side, slot, mk.smiles[(d - 1, u, e)]["lot"]))
    meta = np.array(meta, dtype=float)
    log(f"  {len(meta):,} option paths to price ({time.time() - t0:.0f}s)")
    n, X = len(meta), len(EXITS)
    pnl = np.full((n, X), np.nan, np.float32)
    rr = np.full((n, X), np.nan, np.float32)
    outcome = np.zeros((n, X), np.int8)
    xslot = np.zeros((n, X), np.int8)
    for u in (0, 1):
        rows = np.flatnonzero(meta[:, 3] == u)
        for s0 in range(0, rows.size, 15000):
            part = rows[s0:s0 + 15000]
            ev_i, c, d, _, e, K, cp, slot, lot = (meta[part, j] for j in range(9))
            d, e, cp, slot = d.astype(int), e.astype(int), cp.astype(int), slot.astype(int)
            p = mk.paths(d, u, e, K, cp, iv_mode)
            Po, Ph, Pl, Pc = p["o"], p["h"], p["l"], p["c"]
            j0 = slot + 1
            bars = np.arange(75)[None, :]
            live = (bars >= j0[:, None]) & (bars <= FLAT)
            entry = Po[np.arange(len(part)), j0]
            fill_in = entry + slip(entry) * slip_mult
            ok = np.isfinite(entry) & (entry >= 2.0)
            Sx = mk.idx[u]
            So = Sx["o"][d, :][np.arange(len(part)), j0]
            for k, (kind, x, be) in enumerate(EXITS):
                if kind == "premium":
                    stop = fill_in * (1 - x)
                    tgt = fill_in * (1 + 2 * x)
                    risk_pu = fill_in - stop
                    hit_s = live & (Pl <= stop[:, None])
                    if be:
                        t1 = live & (Ph >= (fill_in * (1 + x))[:, None])
                        jb = _first(t1)
                        after = bars > jb[:, None]
                        hit_s = (hit_s & ~after) | (live & after & (Pl <= fill_in[:, None]))
                        stop_lvl = lambda j: np.where(j > jb, fill_in, stop)
                    else:
                        stop_lvl = lambda j: stop
                    fs = _first(hit_s)
                    hit_t = live & (Ph >= tgt[:, None])
                    ft = _first(hit_t)
                    js = np.minimum(fs, 74)
                    sl_px = np.minimum(stop_lvl(fs), Po[np.arange(len(part)), js])
                else:
                    S_stop = So * (1 - x * cp)
                    fr = p["fr"][:, 0]
                    prem_at_stop0 = M.black(S_stop * fr, K, p["T_c"][np.arange(len(part)), j0], p["sig_c"][np.arange(len(part)), j0], cp)
                    risk_pu = fill_in - prem_at_stop0
                    tgt = fill_in + 2 * risk_pu
                    ilow = np.where(cp[:, None] > 0, Sx["l"][d], -Sx["h"][d])
                    lvl = np.where(cp > 0, S_stop, -S_stop)[:, None]
                    hit_s = live & (ilow <= lvl)
                    fs = _first(hit_s)
                    hit_t = live & (Ph >= tgt[:, None])
                    ft = _first(hit_t)
                    js = np.minimum(fs, 74)
                    prem_stop_bar = M.black(S_stop * fr, K, p["T_c"][np.arange(len(part)), js], p["sig_c"][np.arange(len(part)), js], cp)
                    sl_px = np.minimum(prem_stop_bar, Po[np.arange(len(part)), js])
                    ok_k = risk_pu > 0.05 * fill_in
                stop_first = fs <= ft                               # same bar: the stop counts
                ex_s = (fs < 999) & stop_first
                ex_t = (ft < 999) & ~ex_s
                time_px = Pc[:, FLAT]
                out_px = np.where(ex_s, sl_px - 2 * slip(sl_px) * slip_mult,
                                  np.where(ex_t, tgt - slip(tgt) * slip_mult, time_px - slip(time_px) * slip_mult))
                out_px = np.maximum(out_px, 0.05)
                units = np.floor(RISK / np.maximum(risk_pu * lot, 1e-9))
                units = np.where((units == 0) & (risk_pu * lot <= MAX_RISK_1LOT), 1, units)
                units = np.minimum(units, np.floor(CAPITAL / np.maximum(fill_in * lot, 1e-9)))
                qty = units * lot
                valid = ok & (units >= 1) & (risk_pu > 0)
                if kind == "index":
                    valid &= ok_k
                money = (out_px - fill_in) * qty - charges(fill_in * qty, "buy") - charges(out_px * qty, "sell")
                pnl[part, k] = np.where(valid, money, np.nan)
                rr[part, k] = np.where(valid, money / np.maximum(risk_pu * qty, 1e-9), np.nan)
                outcome[part, k] = np.where(ex_s, -1, np.where(ex_t, 1, 0))
                xslot[part, k] = np.where(ex_s, js, np.where(ex_t, np.minimum(ft, 74), FLAT))
            log(f"  priced {min(s0 + 15000, rows.size):,}/{rows.size:,} {('NIFTY', 'BANKNIFTY')[u]} paths ({time.time() - t0:.0f}s)")
    return meta, pnl, rr, outcome, xslot


# ------------------------------------------------------------------ variants
def _months(d0, d1):
    return (d1 - d0).astype(int) / 30.44


def variants(mk, ev, flags, meta, pnl, rr, outcome):
    dates = mk.dates
    learn_m = _months(dates[1], LEARN_END)
    judge_m = _months(LEARN_END, dates[-1])
    ev_name = np.array([e[0] for e in ev])
    ev_u = np.array([e[1] for e in ev])
    pe = meta[:, 0].astype(int)
    pc = meta[:, 1].astype(int)
    pd_ = meta[:, 2].astype(int)
    day = dates[pd_]
    learn = day <= LEARN_END
    year = day.astype("datetime64[Y]").astype(int) + 1970
    month = day.astype("datetime64[M]")
    out = []
    for name in SG.NAMES:
        for u in (0, 1):
            for c in range(len(CONTRACTS)):
                base = np.flatnonzero((ev_name[pe] == name) & (ev_u[pe] == u) & (pc == c))
                if base.size == 0:
                    continue
                for fi, fname in enumerate(FILTERS):
                    rows = base[flags[pe[base], fi]]
                    if rows.size < 20:
                        continue
                    for k, ex in enumerate(EXITS):
                        p = pnl[rows, k]
                        good = np.isfinite(p)
                        r = rows[good]
                        p = p[good]
                        if r.size < 20:
                            continue
                        lm = learn[r]
                        res = {"signal": name, "u": u, "contract": c, "filter": fi, "exit": k}
                        for per, m, mo in (("learn", lm, learn_m), ("judge", ~lm, judge_m)):
                            pp = p[m]
                            cum = np.cumsum(pp)
                            res[per] = {"trades": int(m.sum()), "win": float((pp > 0).mean() * 100) if m.any() else None,
                                        "avg_r": float(np.nanmean(rr[r[m], k])) if m.any() else None,
                                        "per_month": float(pp.sum() / mo), "max_dd": float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max()) if m.any() else 0.0,
                                        "targets": float((outcome[r[m], k] == 1).mean() * 100) if m.any() else None}
                        yy = year[r]
                        res["learn_years"] = {int(y): float(p[(yy == y)].sum()) for y in (2022, 2023, 2024)}
                        mm = month[r][~lm]
                        if mm.size:
                            uniq, inv = np.unique(mm, return_inverse=True)
                            ms = np.bincount(inv, weights=p[~lm])
                            res["judge_losing_months"] = int((ms < 0).sum())
                            res["judge_months"] = int(uniq.size)
                        out.append(res)
    return out


def describe(v):
    dl, mode = CONTRACTS[v["contract"]]
    return {"strategy": f"{('NIFTY', 'BANKNIFTY')[v['u']]}: {SG.NAMES[v['signal']]}",
            "filter": FILTERS[v["filter"]], "contract": f"delta {dl:.2f}, {mode}", "exit": exit_label(EXITS[v["exit"]])}
