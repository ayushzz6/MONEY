"""Run the 15-strategy, 5-year NSE backtest and write the results and the HTML report.

    ./research.sh download     # 5 years of 5-minute bars from Angel One (about an hour, once)
    ./research.sh run          # all 15 strategies -> data/research/results/
    ./research.sh run --only O1,I2
    ./research.sh equity       # equity cash only: intraday + swing factories, then the report
    ./research.sh search       # the strategy factories: thousands of variants, chosen on 2021-24, judged on 2025-26
"""
import argparse
import json
import logging
import sys
import time

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.metrics import daily_series, summarize
from research.sim import EquityRules, features, simulate_equity
from research.strategies_equity import EQUITY, Market
from research.strategies_fno import FNO, FnoContext

OUT = DATA_DIR / "research" / "results"
CAPITAL = 100000.0
TARGET = 2000.0


def _clean(t):
    return {k: (str(v) if isinstance(v, np.datetime64) else (float(v) if isinstance(v, np.floating) else
            (int(v) if isinstance(v, np.integer) else v))) for k, v in t.items()}


def run(only=None, log=print):
    t0 = time.time()
    N, B, V = data.load("NIFTY"), data.load("BANKNIFTY"), data.load("INDIAVIX")
    if N is None or B is None or V is None:
        raise SystemExit("index data missing: run ./research.sh download first")
    days = N.dates
    log(f"Sessions: {len(days)} ({days[0]} to {days[-1]})")
    results, all_trades = [], {}
    rules = EquityRules(capital=CAPITAL)
    costs = Settings().get("costs", {})
    want = set(only) if only else None

    eq = [s for s in EQUITY if not want or s[0] in want]
    if eq:
        mk = Market(N)
        syms = [s for s in data.nifty100() if (data.HIST_DIR / f"{s}.npz").exists()]
        log(f"Equity universe: {len(syms)} NIFTY 100 stocks with data")
        cands = {code: [] for code, *_ in eq}
        eligible = {code: [] for code, *_ in eq}
        for i, s in enumerate(syms, 1):
            m = data.load(s)
            if m is None or m.n < 60:
                continue
            f = features(m)
            for code, name, fn, opts in eq:
                res = fn(m, f, mk)
                if isinstance(res, tuple):
                    c, el = res
                    eligible[code] += el
                else:
                    c = res
                cands[code] += c
            if i % 20 == 0:
                log(f"  {i}/{len(syms)} stocks scanned ({time.time() - t0:.0f}s)")
        for code, name, fn, opts in eq:
            c = cands[code]
            if opts.get("rank_top"):
                by_day = {}
                for d, sym, sc in eligible[code]:
                    by_day.setdefault(d, []).append((sc, sym))
                allowed = {(d, sym) for d, rows in by_day.items() for sc, sym in sorted(rows, reverse=True)[:opts["rank_top"]]}
                c = [x for x in c if (x["day"], x["symbol"]) in allowed]
            r = EquityRules(capital=CAPITAL, top_n_per_day=opts.get("top_n"))
            trades = simulate_equity(c, r, costs)
            pnl = daily_series(days, trades)
            sens = {}
            for bps in (0.0, 5.0):
                r2 = EquityRules(capital=CAPITAL, top_n_per_day=opts.get("top_n"), slippage_bps=bps)
                sens[f"slip_{bps:g}bps"] = round(float(daily_series(days, simulate_equity(c, r2, costs)).sum()), 0)
            gross_r = [t["gross"] / t["risk_inr"] for t in trades if t["risk_inr"]]
            res = summarize(code, name, days, pnl, trades, CAPITAL, TARGET,
                            {"segment": "Equity cash (MIS)", "doc": fn.__doc__.strip(), "signals": len(c),
                             "slippage_sensitivity": sens,
                             "avg_R_before_costs": round(float(np.mean(gross_r)), 3) if gross_r else None,
                             "sizing": f"risk Rs {r.risk_per_trade:,.0f} a trade, up to {r.max_positions} open, "
                                       f"MIS up to {r.leverage:g}x, daily stop Rs {r.daily_loss_stop:,.0f}"})
            results.append(res)
            all_trades[code] = trades
            log(f"{code} {name}: {len(trades)} trades, net Rs {res['net_pnl']:,.0f} (no slippage: "
                f"{sens['slip_0bps']:,.0f}, 5 bps: {sens['slip_5bps']:,.0f})")

    fno = [s for s in FNO if not want or s[0] in want]
    if fno:
        ctx = FnoContext(N, B, V)
        for code, name, fn in fno:
            res = fn(ctx)
            trades, mtm = (res if isinstance(res, tuple) else (res, None))
            pnl = daily_series(days, trades, mtm)
            sens = {}
            for scale in (0.8, 1.0):
                c2 = FnoContext(N, B, V, iv_scale=scale)
                r2 = fn(c2)
                t2, m2 = (r2 if isinstance(r2, tuple) else (r2, None))
                sens[f"iv_{scale:g}x_vix"] = round(float(daily_series(days, t2, m2).sum()), 0)
            extra = {"segment": "Index options (modeled premiums, IV 0.9x India VIX)", "doc": fn.__doc__.strip(),
                     "sizing": "1 lot (NIFTY 65 / BANKNIFTY 30)", "iv_sensitivity": sens}
            if code == "O2":
                extra["unverified"] = True
                extra["doc"] += (" Caveat: same-day-expiry option prices could not be checked against real ones; the "
                                 "model is least reliable here.")
            if trades and "fut_net" in trades[0]:
                extra["futures_equivalent_net"] = round(sum(t["fut_net"] for t in trades), 0)
            s = summarize(code, name, days, pnl, trades, CAPITAL, TARGET, extra)
            results.append(s)
            all_trades[code] = trades
            log(f"{code} {name}: {len(trades)} trades, net Rs {s['net_pnl']:,.0f} (IV 0.8x VIX: {sens['iv_0.8x_vix']:,.0f}, "
                f"1.0x VIX: {sens['iv_1x_vix']:,.0f})")

    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "capital": CAPITAL, "target": TARGET,
           "from": str(days[0]), "to": str(days[-1]), "sessions": len(days), "seconds": round(time.time() - t0, 0),
           "strategies": results}
    (OUT / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    for code, trades in all_trades.items():
        (OUT / f"trades_{code}.json").write_text(json.dumps([_clean(t) for t in trades], default=str))
    from research.report import write_report
    path = write_report(doc, OUT / "report.html")
    log(f"Report: {path}")
    return doc


def print_table(doc):
    print(f"\n{'':4}{'strategy':<44}{'trades':>7}{'win%':>6}{'net Rs':>11}{'/month':>9}{'/day':>7}"
          f"{'maxDD':>9}{'PF':>6}{'mo>=2k':>7}")
    for s in doc["strategies"]:
        pf = f"{s['profit_factor']:.2f}" if s.get("profit_factor") else "-"
        wr = f"{s['win_rate_pct']:.0f}" if s.get("win_rate_pct") is not None else "-"
        print(f"{s['code']:<4}{s['name'][:43]:<44}{s['trades']:>7}{wr:>6}{s['net_pnl']:>11,.0f}{s['avg_per_month']:>9,.0f}"
              f"{s['avg_per_session']:>7,.0f}{s['max_drawdown']:>9,.0f}{pf:>6}{s['months_at_target_pct']:>6.0f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "download500", "run", "search", "equity", "equity3", "validate3", "library", "library2", "library3", "lead", "options", "library6", "srbreak", "obtest", "obstrats"])
    ap.add_argument("--only", help="comma-separated codes, e.g. E1,O1")
    ap.add_argument("--report", action="store_true", help="search: rebuild the report from the saved grids")
    ap.add_argument("--universe", default="nifty200", choices=["nifty200", "nifty500"], help="equity3: stock universe")
    ap.add_argument("--intraday-only", action="store_true", help="equity3: keep swing / pairs / ETF from the saved grid")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if a.cmd == "download":
        import datetime as dt
        from market.instruments.universe import Universe
        syms = list(data.INDEX_TOKENS) + data.nifty100()
        data.download_all(syms, Universe().load(), dt.date.today() - dt.timedelta(days=5 * 365 + 2), dt.date.today())
        return 0
    if a.cmd == "download500":
        import datetime as dt
        import json as _json
        from core.config.settings import CACHE_DIR
        from market.instruments.universe import Universe

        class Tokens:
            def __init__(self):
                self.u = Universe().load()
                self.master = _json.loads((CACHE_DIR / "angel_nse_eq.json").read_text())

            def token(self, s):
                return self.u.token(s) or self.master.get(s, {}).get("token", "")
        syms = [s for s in data.nifty500() if not (data.HIST_DIR / f"{s}.npz").exists()]
        print(f"{len(syms)} NIFTY 500 stocks to download", flush=True)
        data.download_all(syms, Tokens(), dt.date(2021, 10, 25), dt.date(2026, 10, 23), log=lambda m: print(m, flush=True))
        return 0
    if a.cmd == "obtest":
        from research.lib import obtest, report_ob
        log = lambda m: print(m, flush=True)
        if not a.report:
            obtest.build(log=log)
        log(f"Report: {report_ob.write(obtest.analyze(log=log))}")
        return 0
    if a.cmd == "obstrats":
        from research.lib import obstrats
        log = lambda m: print(m, flush=True)
        if not a.report:
            obstrats.build(log=log)
        obstrats.analyze(log=log)
        return 0
    if a.cmd == "srbreak":
        from research.lib import report_sr, srbreak
        log = lambda m: print(m, flush=True)
        if not a.report:
            srbreak.build(log=log)
            srbreak.build_swing(log=log)
        summary = srbreak.analyze(log=log)
        swing = srbreak.analyze_swing(log=log)
        log(f"Report: {report_sr.write(summary, swing)}")
        return 0
    if a.cmd == "library6":
        from research.lib import leak, report6, round6
        from research.lib.pa import PA_LIBRARY
        log = lambda m: print(m, flush=True)
        if not a.report:
            lk = leak.run(samples=24, seed=11, log=log, library=PA_LIBRARY)
            if lk["failing"]:
                raise SystemExit(f"lookahead in: {sorted(lk['failing'])}")
            round6.build(log=log)
        summary = round6.analyze(log=log)
        log(f"Report: {report6.write(summary)}")
        return 0
    if a.cmd == "options":
        from research.opt import chain, run5
        from research.opt import report as opt_report
        log = lambda m: print(m, flush=True)
        if not a.report:
            chain.download(log=log)
            run5.run(log=log)
        log(f"Report: {opt_report.write()}")
        return 0
    if a.cmd == "lead":
        from research import data as _data
        from research.lead import daily, intraday, nse_daily
        from research.lead import report as lead_report
        log = lambda m: print(m, flush=True)
        if not a.report:
            nse_daily.download([d.astype(object) for d in _data.load("NIFTY").dates], log=log)
            intraday.run(log=log)
            daily.run(log=log)
        log(f"Report: {lead_report.write()}")
        return 0
    if a.cmd == "library3":
        from research.lib import report, round3
        log = lambda m: print(m, flush=True)
        if not a.report:
            round3.build(log=log)
        summary = round3.analyze(log=log)
        log(f"Report: {report.write_round3(summary)}")
        return 0
    if a.cmd == "library2":
        from research.lib import report, round2
        log = lambda m: print(m, flush=True)
        if not a.report:
            round2.build(log=log)
        summary, _ = round2.analyze(log=log)
        log(f"Report: {report.write_round2(summary)}")
        return 0
    if a.cmd == "library":
        import json as _json
        from research.lib import analyze, build, leak, report
        log = lambda m: print(m, flush=True)
        if not a.report:
            build.build(log=log)
        lk = leak.run(samples=20, log=log)
        (s3_out := DATA_DIR / "research" / "results").mkdir(parents=True, exist_ok=True)
        (s3_out / "library_leak.json").write_text(_json.dumps(lk))
        summary, variants = analyze.run(log=log)
        log(f"Report: {report.write(summary, variants, lk)}")
        return 0
    if a.cmd == "equity3":
        import json as _json
        from research import search_equity3 as s3
        p = s3.OUT / f"equity3_grid{s3.INDEXES[a.universe][2]}.json"
        doc = _json.loads(p.read_text()) if a.report and p.exists() else s3.run(log=lambda m: print(m, flush=True),
                                                                                  index=a.universe,
                                                                                  intraday_only=a.intraday_only)
        s3.summary(doc)
        s3.write_report(doc)
        return 0
    if a.cmd == "validate3":
        import json as _json
        from research import search_equity3 as s3
        from research.validate_1m import validate_universe
        log = lambda m: print(m, flush=True)
        if a.universe == "nifty500":                      # the corrected NIFTY 500 leader: OR5 + top10 rvol5 + NIFTY direction
            N, V, syms, sectors, ctx, uni = s3.universe(log, index="nifty500")
            costs = Settings().get("costs")
            res = validate_universe(uni, ctx, N.dates, costs, "NIFTY 500", log=log, filt="top10 rvol5+nifty",
                                    out_name="orb5_1m_check_n500_rvol5_nifty.json")
            log(_json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "by_month"})
                             for k, v in res.items()}, indent=1, default=str))
            return 0
        N, V, syms, sectors, ctx, uni = s3.universe(log)
        costs = Settings().get("costs")
        for trail, ts in ((1.0, 6), (0.0, 0)):            # the round 3 leader, then the plain version
            log(f"1-minute check: trail {trail:g} ATR, time stop {5 * ts} min")
            res = validate_universe(uni, ctx, N.dates, costs, "NIFTY 200", log=log, trail=trail, time_stop=ts)
            log(_json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "by_month"})
                             for k, v in res.items()}, indent=1, default=str))
        return 0
    if a.cmd == "equity":
        from research.search_equity import main as eq_main
        eq_main(rebuild_only=a.report, log=lambda m: print(m, flush=True))
        return 0
    if a.cmd == "search":
        from research.search import main as search_main
        search_main(rebuild_only=a.report, log=lambda m: print(m, flush=True))
        return 0
    doc = run([x.strip().upper() for x in a.only.split(",")] if a.only else None)
    print_table(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
