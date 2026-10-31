"""XAUUSD round 2, the legal way from India: MCX gold (Gold Petal, 1 gram a lot) on 5-minute bars,
Jun 2021 - Oct 2026, on a Rs 1 lakh account.

New against round 1 (research/futures/xau.py):
- stop-order entries resting at a level (the MCX opening range, the Asian range before London, the London
  first hour, the COMEX open, the New York cash open, yesterday's high / low, the open +/- half of
  yesterday's range). Every level is known before the order is placed. A fill inside the entry bar follows
  the usual bar path (green bar: low then high; red: high then low); a worst-case check counts any touch of
  the stop in that bar as a loss;
- trailing stops (k x daily ATR behind the best price of the bars before) and time stops (out after an hour
  unless +1R);
- risk budgets of 1%, 2.5% and 5% of the account a trade, capped by what Rs 1 lakh of margin carries
  (Gold Petal margin 9.34% of notional: about 75 grams at Rs 14,200 a gram);
- positional systems held overnight (Donchian, moving averages, time-series momentum), long only or both ways.

Chosen on Jun 2021 - Dec 2024, judged on Jan 2025 - Oct 2026. Gold rose about 100% in the judged period, so
anything long gold looks good there: long-only and both-ways results are kept apart.
Costs: MCX charges on every order, 1 bp (at least a tick) slippage a side; stress at 3 bp.
"""
import itertools
import json
import time
from dataclasses import dataclass

import numpy as np

from core.config.settings import DATA_DIR
from research.futures import factory as fx
from research.futures import trend as tr
from research.futures import xau as x1
from research.futures.data import SPECS, load

OUT = DATA_DIR / "research" / "results"
SP = SPECS["GOLDPETAL"]
CAPITAL = 100000.0
TRAIN_END = np.datetime64("2024-12-31")
RISKS = [1000.0, 2500.0, 5000.0]
SIDES = ("both", "long only", "short only")


@dataclass(frozen=True)
class Exit2:
    stop: str              # struct | atr0.25 | atr0.5 | atr1.0
    target: float          # R multiple, 0 = none
    trail: float           # k x daily ATR behind the best price, 0 = none
    time_stop: int         # bars: out at the close of entry + N unless +1R was reached, 0 = none

    @property
    def label(self):
        return (f"stop {self.stop}, {f'{self.target:g}R' if self.target else 'no target'}"
                + (f", trail {self.trail:g} ATR" if self.trail else "")
                + (f", out after {self.time_stop * 5} min unless +1R" if self.time_stop else ""))


EXITS = [Exit2(st, tg, trl, ts) for st, tg, trl, ts in itertools.product(
    ["struct", "atr0.25", "atr0.5", "atr1.0"], [0, 2.0, 3.0], [0, 0.5, 1.0], [0, 12])]


# ---------------------------------------------------------------- stop-order setups
def _ffill(a):
    out = a.copy()
    for j in range(1, a.shape[1]):
        m = np.isnan(out[:, j])
        out[m, j] = out[m, j - 1]
    return out


def _stop_orders(s, up, dn, start, end, struct_up, struct_dn):
    """Per session: buy stop at up[i], sell stop at dn[i], live from bar start[i] to end[i]; the first one
    touched wins. Fill: the level, or the bar's open if it opened beyond it."""
    I = np.arange(s.c.shape[1])[None, :]
    win = (I >= start[:, None]) & (I <= end[:, None])
    u = fx.first_true(win & (s.h > up[:, None]))
    d = fx.first_true(win & (s.l < dn[:, None]))
    side, slot = fx._pick(u, d)
    rows = np.flatnonzero((side != 0) & np.isfinite(up) & np.isfinite(dn))
    sl, sd = slot[rows], side[rows]
    lvl = np.where(sd > 0, up[rows], dn[rows])
    o = s.o[rows, sl]
    e = np.where(sd > 0, np.fmax(lvl, o), np.fmin(lvl, o))
    st = np.where(sd > 0, struct_up[rows], struct_dn[rows])
    return rows, sl, sd, e, st, True


def _range_levels(s, shifts, r_from, r_to, until):
    n = s.n
    up, dn = np.full(n, np.nan), np.full(n, np.nan)
    start, end = np.full(n, 10 ** 6), np.full(n, -1)
    for sh in np.unique(shifts):
        rows = np.flatnonzero(shifts == sh)
        a, b, e = x1._slot(s, r_from, sh) + 1, x1._slot(s, r_to, sh), x1._slot(s, until, sh)
        if b < a or e <= b:
            continue
        with np.errstate(all="ignore"):
            up[rows] = np.nanmax(s.h[rows, a:b + 1], 1)
            dn[rows] = np.nanmin(s.l[rows, a:b + 1], 1)
        start[rows], end[rows] = b + 1, e
    return up, dn, start, end


def stop_order_setups():
    def ranged(r_from, r_to, until, which):
        def fn(s, f, sp):
            sh = x1._shifts(s, which) if which else np.zeros(s.n, int)
            up, dn, start, end = _range_levels(s, sh, r_from, r_to, until)
            mid = (up + dn) / 2
            return _stop_orders(s, up, dn, start, end, mid, mid)
        return fn

    def prev_day(s, f, sp):
        o0 = s.o[:, 0]
        start, end = np.zeros(s.n, int), np.full(s.n, s.slot_of_close("21:00"))
        return _stop_orders(s, f.prev_high, f.prev_low, start, end, o0, o0)

    def vol_breakout(s, f, sp):
        o0 = s.o[:, 0]
        rng = f.prev_high - f.prev_low
        start, end = np.ones(s.n, int), np.full(s.n, s.slot_of_close("21:00"))
        return _stop_orders(s, o0 + 0.5 * rng, o0 - 0.5 * rng, start, end, o0, o0)

    return {
        "Stop orders: MCX opening 15-min range (to 13:00)": ranged("09:00", "09:15", "13:00", None),
        "Stop orders: Asian range 09:00-London open (to 16:30)": ranged("09:00", "12:30", "16:30", "uk"),
        "Stop orders: London first hour (to 17:30)": ranged("12:30", "13:30", "17:30", "uk"),
        "Stop orders: COMEX open 30 min (to 21:30)": ranged("17:50", "18:20", "21:30", "us"),
        "Stop orders: New York cash open 30 min (to 22:30)": ranged("19:00", "19:30", "22:30", "us"),
        "Stop orders: yesterday's high / low (to 21:00)": prev_day,
        "Stop orders: open +/- half of yesterday's range (to 21:00)": vol_breakout,
    }


# ---------------------------------------------------------------- one variant on the Rs 1 lakh account
def run_variant(s, f, cand, filt, ex: Exit2, risk_budget, slip_bp=1.0, entry_bar="ohlc", Cf=None):
    rows, slots, sides, entry, sstop, touch = cand
    if len(rows) == 0:
        return None
    keep = np.isfinite(f.atr[rows]) & fx.filt_mask(filt, f, rows, sides)
    rows, slots, sides, entry, sstop = rows[keep], slots[keep], sides[keep], entry[keep], sstop[keep]
    if len(rows) == 0:
        return None
    atr = f.atr[rows]
    stop = sstop if ex.stop == "struct" else entry - sides * float(ex.stop[3:]) * atr
    risk = (entry - stop) * sides
    ok = np.isfinite(risk) & (risk > 0) & (risk / entry < 0.05) & (risk / entry > 0.0005)
    rows, slots, sides, entry, stop, risk, atr = rows[ok], slots[ok], sides[ok], entry[ok], stop[ok], risk[ok], atr[ok]
    n = len(rows)
    if n == 0:
        return None
    X = s.slot_of_close(SP.flat_by)
    H, L, O = s.h[rows], s.l[rows], s.o[rows]
    C = (Cf if Cf is not None else _ffill(s.c))[rows]
    I = np.arange(s.c.shape[1])[None, :]
    es = slots[:, None]
    sd = sides[:, None]
    after = (I > es) & (I <= X)
    tgt = entry + sides * ex.target * risk if ex.target else np.full(n, np.nan)
    st_mat = np.repeat(stop[:, None], s.c.shape[1], 1)
    if ex.trail:
        live = I >= es
        best = np.where(sd > 0, np.where(live, np.nan_to_num(H, nan=-np.inf), -np.inf),
                        np.where(live, np.nan_to_num(-L, nan=-np.inf), -np.inf))
        run = np.maximum.accumulate(best, axis=1)
        prev = np.c_[np.full((n, 1), -np.inf), run[:, :-1]]
        k = ex.trail * atr[:, None]
        with np.errstate(invalid="ignore"):
            trail = np.where(sd > 0, prev - k, -prev + k)
            st_mat = np.where(sd > 0, np.fmax(st_mat, np.where(np.isfinite(prev), trail, -np.inf)),
                              np.fmin(st_mat, np.where(np.isfinite(prev), trail, np.inf)))
    X_trade = np.full(n, X)
    if ex.time_stop:
        one = entry + sides * risk
        reach = after & np.where(sd > 0, H >= one[:, None], L <= one[:, None])
        b1 = np.where(reach.any(1), reach.argmax(1), 999)
        cut = slots + ex.time_stop
        X_trade = np.where(b1 > cut, np.minimum(cut, X), X)
        after = (I > es) & (I <= X_trade[:, None])
    stop_hit = after & np.where(sd > 0, L <= st_mat, H >= st_mat)
    if touch:
        eb = I == es
        if entry_bar == "worst":
            hit_eb = np.where(sd > 0, L <= stop[:, None], H >= stop[:, None])
        else:
            hit_eb = np.where(sd > 0, (C <= stop[:, None]) | ((O > C) & (L <= stop[:, None])),
                              (C >= stop[:, None]) | ((O < C) & (H >= stop[:, None])))
        stop_hit = stop_hit | (eb & hit_eb)
    tgt_hit = after & ~np.isnan(tgt)[:, None] & np.where(sd > 0, H >= tgt[:, None], L <= tgt[:, None])
    fs = np.where(stop_hit.any(1), stop_hit.argmax(1), 999)
    ft = np.where(tgt_hit.any(1), tgt_hit.argmax(1), 999)
    out = np.where((fs <= ft) & (fs < 999), -1, np.where(ft < 999, 1, 0))
    r = np.arange(n)
    xs = np.minimum(np.where(out == -1, fs, np.where(out == 1, ft, X_trade)), s.c.shape[1] - 1)
    o_at = np.where(np.isnan(O[r, xs]), C[r, xs], O[r, xs])
    stv = st_mat[r, xs]
    stop_px = np.where(sides > 0, np.fmin(stv, o_at), np.fmax(stv, o_at))
    stop_px = np.where(xs == slots, stv, stop_px)
    tgt_px = np.where(sides > 0, np.fmax(tgt, o_at), np.fmin(tgt, o_at))
    px = np.where(out == -1, stop_px, np.where(out == 1, tgt_px, C[r, X_trade]))
    slip = np.maximum(SP.tick, slip_bp * 1e-4 * entry)
    fin = entry + sides * slip
    fout = np.where(out == 1, px, px - sides * slip)
    lots = np.floor(risk_budget / ((risk + slip) * SP.multiplier))
    lots = np.minimum(lots, np.floor(CAPITAL / (entry * SP.multiplier * SP.margin_pct)))
    gross = (fout - fin) * sides * lots * SP.multiplier
    ch = np.array([fx.charges(a * q, d > 0, SP) + fx.charges(b * q, d < 0, SP) for a, b, q, d in zip(fin, fout, lots, sides)])
    pnl = np.where(lots >= 1, gross - ch, 0.0)
    daily = np.zeros(s.n)
    np.add.at(daily, rows, pnl)
    return daily, int((lots >= 1).sum()), lots


def stats(daily, ntr, dates, meta):
    out = fx.stats(daily, ntr, dates, meta)
    month = np.array([str(d)[:7] for d in dates])
    for name, m in (("train", dates <= TRAIN_END), ("test", dates > TRAIN_END)):
        mm = sorted(set(month[m]))
        sums = [daily[m][month[m] == k].sum() for k in mm]
        out[f"{name}_months_20k"] = int(sum(v >= 20000 for v in sums))
        out[f"{name}_months_loss"] = int(sum(v < 0 for v in sums))
        out[f"{name}_months"] = len(mm)
    return out


# ---------------------------------------------------------------- the search (forked workers share the data)
_G = {}


def _job(args):
    sname, side_mode, filt, ex, rb = args
    s, f, Cf = _G["s"], _G["f"], _G["Cf"]
    c = _G["cands"][(sname, side_mode)]
    res = run_variant(s, f, c, filt, ex, rb, Cf=Cf)
    if res is None:
        return None
    daily, ntr, lots = res
    return stats(daily, ntr, s.dates, {"setup": sname, "sides": side_mode, "filter": filt, "exit": ex.label,
                                       "stop": ex.stop, "target": ex.target, "trail": ex.trail, "time_stop": ex.time_stop,
                                       "risk": rb, "avg_grams": round(float(lots[lots >= 1].mean()), 1) if (lots >= 1).any() else 0,
                                       "stop_orders": bool(c[5])})


def intraday_search(log=print, workers=8):
    import multiprocessing as mp
    s = load("GOLDPETAL")
    f = fx.features(s)
    setups = dict(fx.setups_for(SP))
    setups.update(x1.gold_setups())
    setups.update(stop_order_setups())
    cands = {}
    for name, fn in setups.items():
        c = fn(s, f, SP)
        for side_mode in SIDES:
            if side_mode == "both" or not len(c[0]):
                cands[(name, side_mode)] = c
            else:
                k = c[2] == (1 if side_mode == "long only" else -1)
                cands[(name, side_mode)] = tuple(v[k] if isinstance(v, np.ndarray) else v for v in c)
    _G.update(s=s, f=f, Cf=_ffill(s.c), cands=cands)
    jobs = [(sn, sm, fl, ex, rb) for (sn, sm) in cands for fl in fx.FILTERS for ex in EXITS for rb in RISKS]
    log(f"  gold intraday: {len(setups)} setups, {len(jobs):,} variants")
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=64) if r is not None]
    log(f"  done: {len(rows):,} variants ({time.time() - t0:.0f}s)")
    return rows, s, f, setups


# ---------------------------------------------------------------- positional (held overnight)
SYSTEMS = [("donchian", 20, 10), ("donchian", 55, 20), ("donchian", 100, 50), ("ma", 10, 50), ("ma", 20, 100),
           ("ma", 50, 200), ("tsmom", 60, 5), ("tsmom", 120, 5), ("tsmom", 250, 5), ("tsmom", 120, 21), ("tsmom", 250, 21)]


def positional_search(log=print):
    rows = []
    orig = tr.signals
    try:
        for system, risk, side_mode in itertools.product(SYSTEMS, [2000.0, 4000.0, 6000.0], ("both", "long only")):
            if side_mode == "long only":
                tr.signals = lambda c, h, l, sy, _o=orig: np.maximum(_o(c, h, l, sy), 0)
            else:
                tr.signals = orig
            r, dates, pnl = tr.backtest("GOLDPETAL", system, risk, str(TRAIN_END))
            month = np.array([str(d)[:7] for d in dates])
            for name, m in (("train", dates <= TRAIN_END), ("test", dates > TRAIN_END)):
                mm = sorted(set(month[m]))
                sums = [pnl[m][month[m] == k].sum() for k in mm]
                r[f"{name}_months_loss"] = int(sum(v < 0 for v in sums))
                r[f"{name}_months"] = len(mm)
            r["sides"] = side_mode
            rows.append(r)
    finally:
        tr.signals = orig
    log(f"  gold positional: {len(rows)} variants")
    return rows


def best(rows, n=10, min_trades=100, max_dd=40000):
    ok = [r for r in rows if r["trades"] >= min_trades and r["train_maxdd"] <= max_dd and r["train_per_month"] > 0]
    ok.sort(key=lambda r: -r["train_per_month"])
    out, seen = [], set()
    for r in ok:
        k = (r["setup"], r["sides"], r["filter"])
        if k not in seen:
            seen.add(k)
            out.append(r)
        if len(out) >= n:
            break
    return out


def stress(s, f, setups, r):
    ex = Exit2(r["stop"], r["target"], r["trail"], r["time_stop"])
    c = setups[r["setup"]](s, f, SP)
    if r["sides"] != "both" and len(c[0]):
        k = c[2] == (1 if r["sides"] == "long only" else -1)
        c = tuple(v[k] if isinstance(v, np.ndarray) else v for v in c)
    out = {}
    for label, bp, eb in (("1 bp slippage", 1.0, "ohlc"), ("3 bp slippage", 3.0, "ohlc"), ("worst case in the entry bar", 1.0, "worst")):
        res = run_variant(s, f, c, r["filter"], ex, r["risk"], slip_bp=bp, entry_bar=eb)
        st = stats(res[0], res[1], s.dates, {})
        out[label] = {k: st[k] for k in ("train_per_month", "test_per_month", "train_maxdd", "test_maxdd")}
    return out


def run(log=print):
    t0 = time.time()
    rows, s, f, setups = intraday_search(log)
    pos = positional_search(log)
    top = best(rows, n=10)
    for r in top[:5]:
        r["stress"] = stress(s, f, setups, r)
    doc = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "variants": len(rows), "top": top,
           "by_risk": {str(rb): best([r for r in rows if r["risk"] == rb], n=3) for rb in RISKS},
           "long_only_top": best([r for r in rows if r["sides"] == "long only"], n=3),
           "short_only_top": best([r for r in rows if r["sides"] == "short only"], n=3),
           "stop_order_top": best([r for r in rows if r.get("stop_orders")], n=5),
           "both_20k": sum(1 for r in rows if r["train_per_month"] >= 20000 and r["test_per_month"] >= 20000),
           "best_test_hindsight": max(r["test_per_month"] for r in rows if r["trades"] >= 100),
           "positional": sorted(pos, key=lambda r: -r["train_per_month"]),
           "seconds": round(time.time() - t0)}
    (OUT / "grid_xau2.json").write_text(json.dumps({"summary": doc, "rows": rows}, default=str))
    return doc
