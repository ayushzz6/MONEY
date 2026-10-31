"""Check a 5-minute stop-order strategy on real 1-minute bars (Angel One), trade by trade.

The 5-minute ORB on stocks in play uses a stop of 10% of the daily ATR, smaller than a typical
5-minute bar, so whether the stop was hit right after the entry cannot be read from 5-minute
bars. This re-runs every candidate trade of a period on 1-minute bars.

    ./research.sh validate1m 2025-01-01 2026-10-23
"""
import datetime as dt
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data

HIST1 = DATA_DIR / "research" / "hist1m"
logger = logging.getLogger("validate_1m")


def _api():
    from brokers.angelone.auth import AngelOneAuth
    import logzero
    logzero.loglevel(logging.CRITICAL)
    a = AngelOneAuth(*Settings().angel_credentials())
    a.login()
    return a.smart_api


def download_1m(pairs, universe, log=print):
    """pairs: {symbol: set(date)}. One request per symbol per 30-day window that holds a needed day."""
    api = _api()
    HIST1.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sym, days in pairs.items():
        days = sorted(days)
        i = 0
        while i < len(days):
            start = days[i]
            end = start + dt.timedelta(days=29)
            jobs.append((sym, start, end))
            while i < len(days) and days[i] <= end:
                i += 1
    log(f"  1-minute download: {len(jobs)} requests for {len(pairs)} stocks")
    t0, last = time.time(), 0.0
    for n, (sym, start, end) in enumerate(jobs, 1):
        out = HIST1 / f"{sym}_{start:%Y%m%d}.npz"
        if out.exists():
            continue
        tok = universe.token(sym)
        rows = None
        for attempt in range(5):
            wait = 0.45 - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            try:
                r = api.getCandleData({"exchange": "NSE", "symboltoken": tok, "interval": "ONE_MINUTE",
                                       "fromdate": f"{start} 09:15", "todate": f"{end} 15:30"})
            except Exception as e:
                r = {"status": False, "message": str(e)}
            if r and r.get("status"):
                rows = r.get("data") or []
                break
            time.sleep(2 + 3 * attempt)
        if rows is None:
            log(f"  {sym} {start}: failed")
            continue
        if rows:
            t = np.array([int(dt.datetime.fromisoformat(x[0]).timestamp()) for x in rows], np.int64)
            a = np.array([x[1:6] for x in rows], float)
            np.savez_compressed(out, t=t, o=a[:, 0], h=a[:, 1], l=a[:, 2], c=a[:, 3], v=a[:, 4])
        if n % 100 == 0:
            log(f"  {n}/{len(jobs)} requests ({time.time() - t0:.0f}s)")


def load_1m_day(sym, day):
    """1-minute bars of one session as arrays indexed by minute from 09:15 (0..374), or None."""
    files = sorted(HIST1.glob(f"{sym}_*.npz"))
    for p in files:
        start = dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()
        if start <= day <= start + dt.timedelta(days=29):
            z = np.load(p)
            ist = z["t"] + 19800
            dday = (ist // 86400).astype("datetime64[D]")
            sel = dday == np.datetime64(day)
            if not sel.any():
                return None
            minute = ((ist[sel] % 86400) - (9 * 3600 + 15 * 60)) // 60
            M = np.full((5, 375), np.nan)
            ok = (minute >= 0) & (minute < 375)
            for k, key in enumerate(("o", "h", "l", "c", "v")):
                M[k, minute[ok]] = z[key][sel][ok]
            return M
    return None


# ---------------------------------------------------------------- the 5-minute ORB on stocks in play, minute by minute
ENTRY_LAST_MIN = 224          # the last minute whose bar closes by 13:00
FLAT_MIN = 359                # the minute that closes at 15:15


def orb5_candidates_1m(pairs, uni, path="ohlc", trail=0.0, time_stop=0, entry="stop", stop_atr=0.1):
    """Candidate trades rebuilt on 1-minute bars. path: 'ohlc' (green minute low-then-high, red high-then-low)
    or 'worst' (any touch of the stop in the entry minute is a loss). trail: trailing stop k x daily ATR behind
    the best price of the minutes before (never loosens). time_stop: N five-minute bars - out at the close of
    5-minute bar entry + N (minute 5 * (entry // 5 + N) + 4) if +1R was not reached by then, as the 5-minute
    version does. entry: 'stop' (stop order at the level), 'close' (market order when a minute CLOSES beyond the
    level, filled at that close) or 'stop_sl_after' (stop order at the level, stop-loss placed only once the
    entry minute has closed: out at that close if it is beyond the stop). 'close' and 'stop_sl_after' need no
    guess about the order of prices inside a minute."""
    items = {m.symbol: (m, f, lv) for m, f, lv in uni.items}
    out, missing = [], 0
    for sym, days in pairs.items():
        if sym not in items:
            continue
        m, f, lv = items[sym]
        pos = {d.astype(object): i for i, d in enumerate(m.dates)}
        for day in sorted(days):
            r = pos.get(day)
            if r is None or not f.valid[r] or not np.isfinite(f.atr14[r]):
                continue
            M = load_1m_day(sym, day)
            if M is None or np.isnan(M[3, :5]).any():
                missing += 1
                continue
            O, H, L, C = M[0], M[1], M[2], M[3]
            o0, c0 = O[0], C[4]
            side = 1 if c0 > o0 else (-1 if c0 < o0 else 0)
            if side == 0:
                continue
            oh, ol = np.nanmax(H[:5]), np.nanmin(L[:5])
            level = oh if side > 0 else ol
            j = None
            for k in range(5, ENTRY_LAST_MIN + 1):
                if np.isnan(H[k]):
                    continue
                if entry == "close":
                    if (side > 0 and C[k] > level) or (side < 0 and C[k] < level):
                        j = k
                        break
                elif (side > 0 and H[k] > level) or (side < 0 and L[k] < level):
                    j = k
                    break
            if j is None:
                continue
            if entry == "close":
                fill = C[j]
            else:
                fill = max(level, O[j]) if side > 0 else min(level, O[j])
            stop = fill - side * stop_atr * f.atr14[r]
            rp = 100 * abs(fill - stop) / fill
            if rp < 0.15 or rp > 3.0:
                continue
            # inside the entry minute
            if entry == "close":
                hit = False                               # filled at the minute's close: the stop starts next minute
            elif entry == "stop_sl_after":
                hit = (C[j] <= stop) if side > 0 else (C[j] >= stop)
            elif path == "worst":
                hit = (L[j] <= stop) if side > 0 else (H[j] >= stop)
            else:
                hit = ((C[j] <= stop) | ((O[j] > C[j]) & (L[j] <= stop))) if side > 0 else \
                      ((C[j] >= stop) | ((O[j] < C[j]) & (H[j] >= stop)))
            exit_min, exit_px, outcome = None, None, 0
            last = FLAT_MIN
            if time_stop:
                one_r = fill + side * abs(fill - stop)
                cut = min(5 * (j // 5 + time_stop) + 4, FLAT_MIN)
                seg = slice(j + 1, cut + 1)
                reached = np.nanmax(H[seg]) >= one_r if side > 0 else np.nanmin(L[seg]) <= one_r
                if not reached:
                    last = cut
            if hit:
                # stop_sl_after: out at the entry minute's close
                exit_min, exit_px, outcome = j, (C[j] if entry == "stop_sl_after" else stop), -1
            else:
                best = C[j] if entry == "close" else (H[j] if side > 0 else L[j])
                for k in range(j + 1, last + 1):
                    if np.isnan(L[k]):
                        continue
                    st_k = stop
                    if trail:
                        st_k = max(stop, best - trail * f.atr14[r]) if side > 0 else min(stop, best + trail * f.atr14[r])
                    if (side > 0 and L[k] <= st_k) or (side < 0 and H[k] >= st_k):
                        exit_min = k
                        exit_px = min(st_k, O[k]) if side > 0 else max(st_k, O[k])
                        outcome = -1
                        break
                    best = max(best, H[k]) if side > 0 else min(best, L[k])
            if exit_min is None:
                k = last
                while k > j and np.isnan(C[k]):
                    k -= 1
                exit_min, exit_px = k, C[k]
            out.append({"day": np.datetime64(day), "symbol": sym, "slot": j, "side": side, "entry": float(fill),
                        "stop": float(stop), "target": None, "exit_slot": int(exit_min), "exit_px": float(exit_px),
                        "outcome": outcome, "score": float(-j)})
    return out, missing


def compare(start="2025-01-01", end="2026-10-23", log=print):
    from research.metrics import daily_series
    from research.search_equity import _setup_universe, exact_check
    from research.sim import EquityRules, simulate_equity
    N, V, syms, ctx, uni = _setup_universe(lambda m: None)
    dates = N.dates
    sel = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    months = len({str(d)[:7] for d in dates[sel]})
    costs = Settings().get("costs")
    pairs = {k: {dt.date.fromisoformat(x) for x in v} for k, v in
             json.loads((DATA_DIR / "research" / "results" / "orb5_pairs_test.json").read_text()).items()}
    rules = lambda bps: EquityRules(capital=100000.0, risk_per_trade=1000.0, leverage=5.0, max_position_value=250000.0,
                                    max_positions=5, max_trades_per_day=5, daily_loss_stop=3000.0, slippage_bps=bps,
                                    min_risk_pct=0.15, max_risk_pct=3.0)
    base = {"setup": "OR5 stop-order breakout", "filter": "top10 rvol", "exit_stop": "atr0.1", "exit_target": 0,
            "exit_time": "15:15", "exit_breakeven": False}
    d5, t5 = exact_check(uni, ctx, base, dates, costs)
    res = {"period": [start, end], "months": months,
           "5-minute bars (bar-path rule)": round(float(d5[sel].sum()) / months, 0)}
    for path in ("ohlc", "worst"):
        cands, missing = orb5_candidates_1m(pairs, uni, path)
        for bps in (3.0, 5.0):
            tr = simulate_equity(cands, rules(bps), costs)
            d1 = daily_series(dates, tr)
            q = d1[sel]
            eq = np.cumsum(q)
            res[f"1-minute bars, {'path rule' if path == 'ohlc' else 'worst case'} in the entry minute, {bps:g} bps"] = {
                "per_month": round(float(q.sum()) / months, 0), "trades": len(tr),
                "win_rate": round(100 * float(np.mean([t['net'] > 0 for t in tr])), 1) if tr else None,
                "max_drawdown": round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0),
                "by_month": {k: round(v, 0) for k, v in sorted(_by_month(dates[sel], q).items())}}
        res[f"missing 1-minute days ({path})"] = missing
    (DATA_DIR / "research" / "results" / "orb5_1m_check.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _by_month(dates, q):
    out = defaultdict(float)
    for d, v in zip(dates, q):
        out[str(d)[:7]] += v
    return out


def per_trade_check(log=print):
    """For 2025-26 trades of the 5-minute version that have 1-minute data: the result of the SAME trade
    (same stock, day, side, entry level and stop) on 1-minute bars, in R (multiples of the planned risk)."""
    from research.factory_equity import filter_mask, resolve_exits, Exit
    from research.search_equity import _setup_universe
    N, V, syms, ctx, uni = _setup_universe(lambda m: None)
    ex = Exit("atr0.1", 0, "15:15", False)
    have = {p.stem.split("_")[0] for p in HIST1.glob("*.npz")}
    r5, r1, r1w, n_missing = [], [], [], 0
    for (m, f, lv), cand in zip(uni.items, uni.setup("OR5 stop-order breakout")):
        if m.symbol not in have:
            continue
        rows, slots, sides, entry, sstop, touch = cand
        keep = f.valid[rows]
        rows, slots, sides, entry, sstop = rows[keep], slots[keep], sides[keep], entry[keep], sstop[keep]
        fm = filter_mask("top10 rvol", m, f, lv, ctx, rows, slots, sides, touch=touch)
        rows, slots, sides, entry, sstop = rows[fm], slots[fm], sides[fm], entry[fm], sstop[fm]
        test = m.dates[rows] >= np.datetime64("2025-01-01")
        rows, slots, sides, entry, sstop = rows[test], slots[test], sides[test], entry[test], sstop[test]
        if len(rows) == 0:
            continue
        ok, rr, ss, sd, en, st, rk, xpx, xs, out = resolve_exits(m, f, rows, slots, sides, entry, sstop, ex, touch=True)
        for i in range(len(rr)):
            day = m.dates[rr[i]].astype(object)
            M = load_1m_day(m.symbol, day)
            if M is None:
                n_missing += 1
                continue
            risk = abs(en[i] - st[i])
            res5 = (xpx[i] - en[i]) * sd[i] / risk
            got = _one_trade_1m(M, sd[i], st[i] if False else None, en[i], f.atr14[rr[i]])
            if got is None:
                continue
            res1, res1w = got
            r5.append(res5); r1.append(res1); r1w.append(res1w)
    r5, r1, r1w = np.array(r5), np.array(r1), np.array(r1w)
    res = {"trades_compared": int(len(r5)), "missing_1m_days": n_missing,
           "avg_R_5min": round(float(r5.mean()), 3) if len(r5) else None,
           "avg_R_1min_path_rule": round(float(r1.mean()), 3) if len(r1) else None,
           "avg_R_1min_worst_case": round(float(r1w.mean()), 3) if len(r1w) else None,
           "stopped_5min_pct": round(100 * float((r5 <= -0.99).mean()), 1) if len(r5) else None,
           "stopped_1min_pct": round(100 * float((r1 <= -0.99).mean()), 1) if len(r1) else None}
    log(res)
    (DATA_DIR / "research" / "results" / "orb5_1m_pertrade.json").write_text(json.dumps(res, indent=1))
    return res


def _one_trade_1m(M, side, _unused, entry_5m, atr):
    """Rebuild one trade on 1-minute bars from scratch (the 1-minute range, trigger and fill), in R before costs.
    Returns (R with the path rule in the entry minute, R with the worst case), or None if it never triggers."""
    O, H, L, C = M[0], M[1], M[2], M[3]
    if np.isnan(M[3, :5]).any():
        return None
    oh, ol = np.nanmax(H[:5]), np.nanmin(L[:5])
    level = oh if side > 0 else ol
    j = next((k for k in range(5, ENTRY_LAST_MIN + 1) if not np.isnan(H[k]) and
              ((side > 0 and H[k] > level) or (side < 0 and L[k] < level))), None)
    if j is None:
        return None
    fill = max(level, O[j]) if side > 0 else min(level, O[j])
    stop = fill - side * 0.1 * atr
    risk = abs(fill - stop)
    outs = []
    for worst in (False, True):
        if worst:
            hit = (L[j] <= stop) if side > 0 else (H[j] >= stop)
        else:
            hit = ((C[j] <= stop) or ((O[j] > C[j]) and (L[j] <= stop))) if side > 0 else \
                  ((C[j] >= stop) or ((O[j] < C[j]) and (H[j] >= stop)))
        px = stop if hit else None
        if px is None:
            for k in range(j + 1, FLAT_MIN + 1):
                if np.isnan(L[k]):
                    continue
                if (side > 0 and L[k] <= stop) or (side < 0 and H[k] >= stop):
                    px = min(stop, O[k]) if side > 0 else max(stop, O[k])
                    break
        if px is None:
            k = FLAT_MIN
            while k > j and np.isnan(C[k]):
                k -= 1
            px = C[k]
        outs.append((px - fill) * side / risk)
    return outs[0], outs[1]


# ---------------------------------------------------------------- any universe (round 3: NIFTY 200)
def covered_days(sym):
    """Dates already covered by downloaded 1-minute files for a symbol (from the file names)."""
    out = set()
    for p in HIST1.glob(f"{sym}_*.npz"):
        start = dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()
        out.update(start + dt.timedelta(days=i) for i in range(30))
    return out


def leader_pairs(uni, ctx, setup="OR5 stop-order breakout", filt="top10 rvol", start="2025-01-01"):
    from research.factory_equity import filter_mask
    pairs = {}
    for (m, f, lv), cand in zip(uni.items, uni.setup(setup)):
        rows, slots, sides, entry, sstop, touch = cand
        if len(rows) == 0:
            continue
        keep = f.valid[rows]
        rows, slots, sides = rows[keep], slots[keep], sides[keep]
        fm = filter_mask(filt, m, f, lv, ctx, rows, slots, sides, touch=touch)
        for r in rows[fm]:
            d = m.dates[r].astype(object)
            if d >= dt.date.fromisoformat(start):
                pairs.setdefault(m.symbol, set()).add(d)
    return pairs


def validate_universe(uni, ctx, dates, costs, universe_label, log=print, download=True, trail=0.0, time_stop=0,
                      filt="top10 rvol", out_name=None):
    """Download what is missing, then run the full account on 1-minute bars next to the 5-minute version."""
    from market.instruments.universe import Universe as U
    from research.metrics import daily_series
    from research.search_equity import exact_check
    from research.sim import EquityRules, simulate_equity
    pairs = leader_pairs(uni, ctx, filt=filt)
    missing = {s: {d for d in ds if d not in covered_days(s)} for s, ds in pairs.items()}
    missing = {s: ds for s, ds in missing.items() if ds}
    log(f"  {sum(len(v) for v in pairs.values())} candidate stock-days in 2025-26; "
        f"{sum(len(v) for v in missing.values())} not covered by 1-minute files yet")
    if download and missing:
        class Tok:
            def __init__(self):
                self.u = U().load()
                import json as _j
                from core.config.settings import CACHE_DIR
                self.master = _j.loads((CACHE_DIR / "angel_nse_eq.json").read_text())

            def token(self, s):
                return self.u.token(s) or self.master.get(s, {}).get("token", "")
        download_1m(missing, Tok(), log=log)
    sel = (dates >= np.datetime64("2025-01-01"))
    months = len({str(d)[:7] for d in dates[sel]})
    base = {"setup": "OR5 stop-order breakout", "filter": filt, "exit_stop": "atr0.1", "exit_target": 0,
            "exit_time": "15:15", "exit_breakeven": False, "exit_trail": trail, "exit_time_stop": time_stop}
    d5, t5 = exact_check(uni, ctx, base, dates, costs)
    rules = lambda bps: EquityRules(capital=100000.0, risk_per_trade=1000.0, leverage=5.0, max_position_value=250000.0,
                                    max_positions=5, max_trades_per_day=5, daily_loss_stop=3000.0, slippage_bps=bps,
                                    min_risk_pct=0.15, max_risk_pct=3.0)
    res = {"universe": universe_label, "exit": {"stop": "0.1 ATR", "trail_atr": trail, "time_stop_min": 5 * time_stop},
           "period": ["2025-01-01", str(dates[-1])], "months": months,
           "5-minute bars (bar-path rule)": round(float(d5[sel].sum()) / months, 0)}
    labels = {"ohlc": "path rule", "worst": "worst case", "after": "stop-loss placed after the entry minute closes"}
    for path in ("ohlc", "worst", "after"):
        if path == "after":
            cands, miss = orb5_candidates_1m(pairs, uni, "ohlc", trail=trail, time_stop=time_stop, entry="stop_sl_after")
        else:
            cands, miss = orb5_candidates_1m(pairs, uni, path, trail=trail, time_stop=time_stop)
        for bps in (3.0, 5.0, 8.0):
            tr = simulate_equity(cands, rules(bps), costs)
            q = daily_series(dates, tr)[sel]
            eq = np.cumsum(q)
            res[f"1-minute bars, {labels[path]} in the entry minute, {bps:g} bps"] = {
                "per_month": round(float(q.sum()) / months, 0), "trades": len(tr),
                "win_rate": round(100 * float(np.mean([t['net'] > 0 for t in tr])), 1) if tr else None,
                "max_drawdown": round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0),
                "by_month": {k: round(v, 0) for k, v in sorted(_by_month(dates[sel], q).items())}}
        res[f"missing 1-minute days ({path})"] = miss
    tag = f"_trail{trail:g}_ts{time_stop}" if (trail or time_stop) else ""
    name = out_name or f"orb5_1m_check_n200{tag}.json"
    res["filter"] = filt
    (DATA_DIR / "research" / "results" / name).write_text(json.dumps(res, indent=1, default=str))
    return res
