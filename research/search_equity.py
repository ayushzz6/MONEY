"""Equity cash strategy search: intraday (MIS) and swing (CNC) on the NIFTY 100, 5 years.

    ./research.sh equity           # the full search (about 35 minutes), then the report
    ./research.sh equity --report  # rebuild the report from the saved grid

Everything is chosen on Oct 2021 - Dec 2024 and judged on Jan 2025 - Oct 2026.
"""
import json
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import research.factory_equity as fe
from core.config.settings import DATA_DIR, Settings
from research import data
from research.factory_equity import DEFAULT_EXITS, EXIT_GRID, FILTERS, Ctx, Exit, Universe, evaluate, stats
from research.factory_equity2 import EXTRA_SETUPS, XS_FILTERS, add_cross_sectional_ranks
from research.factory_swing import run_swing_search, simulate
from research.report import CSS, E, cls, compact, heatmap, inr, kv
from research.search import _equity_svg, _scale_for_target, _scatter

OUT = DATA_DIR / "research" / "results"
TRAIN_END = np.datetime64("2024-12-31")
TARGET_MONTH, TARGET_DAY, CAPITAL = 20000.0, 2000.0, 100000.0
ALL_FILTERS = FILTERS + XS_FILTERS


def _setup_universe(log):
    fe.SETUPS.update(EXTRA_SETUPS)
    N, V = data.load("NIFTY"), data.load("INDIAVIX")
    syms = [s for s in data.nifty100() if (data.HIST_DIR / f"{s}.npz").exists()]
    ctx = Ctx(N, V)
    uni = Universe(syms, log, ctx)
    add_cross_sectional_ranks(uni)
    return N, V, syms, ctx, uni


def _intraday(N, ctx, uni, costs, log, stage2_top=40):
    t0 = time.time()
    dates = N.dates
    s1 = []
    for setup in fe.SETUPS:
        for filt in ALL_FILTERS:
            for ex in DEFAULT_EXITS:
                res = evaluate(uni, ctx, setup, filt, ex, dates, costs)
                if res is None:
                    continue
                s1.append(stats(res[0], res[1], res[2], dates, _meta(setup, filt, ex)))
        log(f"  intraday stage 1: {setup} ({len(s1):,} variants, {time.time() - t0:.0f}s)")
    pairs = []
    seen = set()
    for r in sorted(s1, key=lambda r: -r["train_per_month"]):
        k = (r["setup"], r["filter"])
        if r["train_trades"] >= 200 and k not in seen:
            seen.add(k)
            pairs.append(k)
        if len(pairs) >= stage2_top:
            break
    s2 = []
    for setup, filt in pairs:
        for ex in EXIT_GRID:
            res = evaluate(uni, ctx, setup, filt, ex, dates, costs)
            if res is not None:
                s2.append(stats(res[0], res[1], res[2], dates, _meta(setup, filt, ex)))
        log(f"  intraday stage 2: {setup} | {filt} ({len(s2):,} variants, {time.time() - t0:.0f}s)")
    return s1, s2


def _meta(setup, filt, ex: Exit):
    return {"family": "intraday", "setup": setup, "filter": filt, "exit": ex.label, "exit_stop": ex.stop,
            "exit_target": ex.target, "exit_time": ex.time, "exit_breakeven": ex.breakeven,
            "new": setup in EXTRA_SETUPS}


def run(log=print):
    t0 = time.time()
    N, V, syms, ctx, uni = _setup_universe(log)
    costs = Settings().get("costs")
    s1, s2 = _intraday(N, ctx, uni, costs, log)
    sw, _ = run_swing_search(syms, N.dates, log)
    doc = {"stocks": len(syms), "intraday": s1 + s2, "swing": sw, "run_at": time.strftime("%Y-%m-%d %H:%M"),
           "seconds": round(time.time() - t0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "equity_grid.json").write_text(json.dumps(doc, default=str))
    return doc


# ---------------------------------------------------------------- finalists
def exact_check(uni, ctx, r, dates, costs, bps=3.0, entry_bar="ohlc"):
    """The finalist through the account simulator: positions open at the same time share the Rs 5 L MIS
    limit, at most 5 open and 5 a day, and a Rs 3,000 daily loss stop."""
    from research.factory_equity import filter_mask, resolve_exits
    from research.metrics import daily_series
    from research.sim import EquityRules, simulate_equity
    ex = Exit(r["exit_stop"], r["exit_target"], r["exit_time"], r["exit_breakeven"],
              trail=r.get("exit_trail", 0.0) or 0.0, time_stop=int(r.get("exit_time_stop", 0) or 0))
    cands = []
    for (m, f, lv), cand in zip(uni.items, uni.setup(r["setup"])):
        rows, slots, sides, entry, sstop, touch = cand
        if len(rows) == 0:
            continue
        keep = f.valid[rows]
        rows, slots, sides, entry, sstop = rows[keep], slots[keep], sides[keep], entry[keep], sstop[keep]
        fm = filter_mask(r["filter"], m, f, lv, ctx, rows, slots, sides, touch=touch)
        rows, slots, sides, entry, sstop = rows[fm], slots[fm], sides[fm], entry[fm], sstop[fm]
        if len(rows) == 0:
            continue
        ok, rr, ss, sd, en, st, rk, xpx, xs, out = resolve_exits(m, f, rows, slots, sides, entry, sstop, ex, touch=touch,
                                                                  entry_bar=entry_bar)
        for i in range(len(rr)):
            cands.append({"day": m.dates[rr[i]], "symbol": m.symbol, "slot": int(ss[i]), "side": int(sd[i]),
                          "entry": float(en[i]), "stop": float(st[i]), "target": None, "exit_slot": int(xs[i]),
                          "exit_px": float(xpx[i]), "outcome": int(out[i]), "score": float(-ss[i])})
    rules = EquityRules(capital=CAPITAL, risk_per_trade=1000.0, leverage=5.0, max_position_value=250000.0,
                        max_positions=5, max_trades_per_day=5, daily_loss_stop=3000.0, slippage_bps=bps,
                        min_risk_pct=0.15, max_risk_pct=3.0)
    trades = simulate_equity(cands, rules, costs)
    return daily_series(dates, trades), trades


def _period(daily, dates):
    out = {}
    month = np.array([str(d)[:7] for d in dates])
    for name, m in (("train", dates <= TRAIN_END), ("test", dates > TRAIN_END)):
        q = daily[m]
        eq = np.cumsum(q)
        out[name] = {"per_month": round(float(q.sum()) / len(set(month[m])), 0),
                     "maxdd": round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0),
                     "days_2k": round(100 * float((q >= TARGET_DAY).mean()), 1)}
    return out


def _finalists(rows, min_trades, max_dd, n):
    ok = sorted([r for r in rows if r["train_trades"] >= min_trades and r["train_maxdd"] <= max_dd
                 and r["train_per_month"] > 0], key=lambda r: -r["train_per_month"])
    out, seen = [], set()
    for r in ok:
        k = (r["setup"], r.get("filter"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= n:
            break
    return out


def _best_per_setup(rows):
    best = {}
    for r in rows:
        if r["train_trades"] < (200 if r["family"] == "intraday" else 60):
            continue
        k = r["setup"]
        if k not in best or r["train_per_month"] > best[k]["train_per_month"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: -r["train_per_month"])


def _describe(r):
    if r["family"] == "intraday":
        return f"{r['setup']} · filter: {r['filter']} · {r['exit']} · risk ₹1,000 a trade, first 5 trades a day"
    ex = {"time": f"exit after {r['hold_days']} sessions", "close>SMA5": f"exit on a close above the 5-day average (max {r['hold_days']} sessions)",
          "close<10-day low": f"exit on a close under the 10-day low (max {r['hold_days']} sessions)"}[r["exit_rule"]]
    st = f"{r['stop_atr']:g} x ATR stop" if r["stop_atr"] else "no stop"
    return f"{r['setup']} · {ex} · {st} · {r['slots']} positions of ₹{CAPITAL / r['slots']:,.0f}"


def stress(uni, ctx, r, dates, costs):
    """The leading intraday finalist under harsher assumptions."""
    out = []
    for label, bps, eb in (("Base: 3 bps slippage, bar-path rule in the entry bar", 3.0, "ohlc"),
                           ("5 bps slippage a side", 5.0, "ohlc"), ("8 bps slippage a side", 8.0, "ohlc"),
                           ("Worst case: any touch of the stop in the entry bar is a loss", 3.0, "worst")):
        d, t = exact_check(uni, ctx, r, dates, costs, bps=bps, entry_bar=eb)
        p = _period(d, dates)
        yr = defaultdict(float)
        for dd, v in zip(dates, d):
            yr[str(dd)[:4]] += v
        out.append({"label": label, "train": p["train"], "test": p["test"], "trades": len(t),
                    "years": {k: round(v) for k, v in sorted(yr.items())}})
    return out


def _validation_html(doc):
    v = doc.get("validation")
    if not v:
        return ""
    rows = "".join(f"<tr><td>{E(x['label'])}</td><td class='r {cls(x['train']['per_month'])}'>{inr(x['train']['per_month'], True)}</td>"
                   f"<td class='r {cls(x['test']['per_month'])}'>{inr(x['test']['per_month'], True)}</td><td class='r'>{inr(x['test']['maxdd'])}</td>"
                   f"<td>{' '.join(f'{y}: {compact(n)}' for y, n in x['years'].items())}</td></tr>" for x in v["stress"])
    one = v.get("one_minute") or {}
    one_rows = "".join(f"<tr><td>{E(k)}</td><td class='r {cls(x['per_month'])}'>{inr(x['per_month'], True)}</td>"
                       f"<td class='r'>{x['trades']:,}</td><td class='r'>{x['win_rate']}%</td><td class='r'>{inr(x['max_drawdown'])}</td></tr>"
                       for k, x in one.items() if isinstance(x, dict) and "per_month" in x)
    one_html = (f"<h4>Re-run on real 1-minute bars, {E(one['period'][0])} to {E(one['period'][1])}</h4>"
                f"<p>The same rule rebuilt from Angel One 1-minute bars, so the stop is checked minute by minute. On 5-minute bars this "
                f"period averaged <b>{inr(one.get('5-minute bars (bar-path rule)'), True)}</b> a month.</p>"
                f"<table><thead><tr><th>1-minute version</th><th class='r'>A month</th><th class='r'>Trades</th><th class='r'>Win</th>"
                f"<th class='r'>Max DD</th></tr></thead><tbody>{one_rows}</tbody></table>") if one_rows else         "<p class='note'>1-minute re-run not available.</p>"
    pt = v.get("per_trade")
    pt_html = ""
    if pt and pt.get("trades_compared"):
        pt_html = (f"<h4>The same trades on 1-minute bars ({pt['trades_compared']:,} trades of 2025-26)</h4>" + kv([
            ("Average result on 5-minute bars", f"{pt['avg_R_5min']:+.2f} R"),
            ("On 1-minute bars, path rule in the entry minute", f"{pt['avg_R_1min_path_rule']:+.2f} R"),
            ("On 1-minute bars, worst case in the entry minute", f"{pt['avg_R_1min_worst_case']:+.2f} R"),
            ("Stopped out: 5-minute / 1-minute", f"{pt['stopped_5min_pct']}% / {pt['stopped_1min_pct']}%")]) +
            "<p class='note'>R = profit or loss in multiples of the planned risk, before charges and slippage (which cost about 0.3 R a trade "
            "here: positions average ₹1.6 L against about ₹500 of risk).</p>")
    return f"""<h2>Stress test of the leading intraday strategy</h2><div class="card">
<p class="rules">{E(v['name'])}</p><table><thead><tr><th>Assumption</th><th class='r'>2021-24 /mo</th><th class='r'>2025-26 /mo</th>
<th class='r'>2025-26 max DD</th><th>By year</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Its stop (10% of the daily ATR) is smaller than a typical 5-minute bar, so on 5-minute data you cannot see whether the
stop was hit in the minutes right after the entry: the "worst case" row assumes it always was. The 1-minute re-run settles it.</p>
{pt_html}{one_html}</div>"""


def write_report(doc, path):
    N = data.load("NIFTY")
    dates = N.dates
    fams = {"Intraday (MIS)": doc["intraday"], "Swing (delivery, 1-20 sessions)": doc["swing"]}
    uni = ctx = stocks = None
    finals = []
    for fname, rows in fams.items():
        mt = 200 if fname.startswith("Intraday") else 60
        for r in _finalists(rows, mt, 35000, 3):
            if r["family"] == "intraday":
                if uni is None:
                    _, _, syms, ctx, uni = _setup_universe(lambda m: None)
                daily, ex_trades = exact_check(uni, ctx, r, dates, Settings().get("costs"))
                r = dict(r, exact=_period(daily, dates), exact_trades=len(ex_trades))
            else:
                if stocks is None:
                    from research.factory_swing import Stock
                    stocks = [Stock(m) for m in (data.load(s) for s in data.nifty100()) if m is not None and m.n > 300]
                daily, _ = simulate(stocks, dates, r["setup"], r["hold_days"], r["stop_atr"], r["slots"], r["exit_rule"])
            finals.append((fname, r, daily, _scale_for_target(daily, dates)))
    # summary
    summ, scat = [], []
    for fname, rows in fams.items():
        tr = np.array([r["train_per_month"] for r in rows])
        te = np.array([r["test_per_month"] for r in rows])
        rho = spearmanr(tr, te).correlation
        summ.append(f"<tr><td>{E(fname)}</td><td class='r'>{len(rows):,}</td><td class='r'>{(tr > 0).mean():.0%}</td>"
                    f"<td class='r'>{(te > 0).mean():.0%}</td><td class='r'>{rho:.2f}</td><td class='r'>{inr(tr.max(), True)}</td>"
                    f"<td class='r'>{inr(te.max(), True)}</td><td class='r'>{int(((tr >= TARGET_MONTH) & (te >= TARGET_MONTH)).sum())}</td></tr>")
        scat.append(f"<div class='card' style='margin:0'><h3>{E(fname)}: every variant, 2021-24 vs 2025-26</h3>{_scatter(rows)}"
                    f"<p class='note'>Dashed lines: ₹20,000 a month. Rank correlation between the periods {rho:.2f}.</p></div>")
    setup_rows = "".join(
        f"<tr><td>{E(r['setup'])}{' <span class=chip>new</span>' if r['setup'] in EXTRA_SETUPS else ''}</td><td>{E(r['family'])}</td>"
        f"<td>{E(r.get('filter') or '–')}</td><td class='r {cls(r['train_per_month'])}'>{inr(r['train_per_month'], True)}</td>"
        f"<td class='r {cls(r['test_per_month'])}'>{inr(r['test_per_month'], True)}</td><td class='r'>{r['train_trades']:,} / {r['test_trades']:,}</td>"
        f"<td class='r'>{inr(r['test_maxdd'])}</td></tr>" for r in _best_per_setup(doc["intraday"] + doc["swing"]))
    fin_html = []
    for fname, r, daily, sc in finals:
        months = defaultdict(float)
        for d, v in zip(dates, daily):
            months[str(d)[:7]] += v
        test = dates > TRAIN_END
        exact = ""
        if r.get("exact"):
            x = r["exact"]
            exact = ("<h4>Exact account check (used for the chart below)</h4>" + kv([
                ("2021-24 a month", f"<span class='{cls(x['train']['per_month'])}'>{inr(x['train']['per_month'], True)}</span>"),
                ("2025-26 a month", f"<b class='{cls(x['test']['per_month'])}'>{inr(x['test']['per_month'], True)}</b>"),
                ("Max drawdown 2021-24 / 2025-26", f"{inr(x['train']['maxdd'])} / {inr(x['test']['maxdd'])}"),
                ("Sessions ≥ ₹2,000, 2021-24 / 2025-26", f"{x['train']['days_2k']}% / {x['test']['days_2k']}%"),
                ("Trades", f"{r['exact_trades']:,}")]) +
                "<p class='note'>Positions open at the same time share the ₹5 L MIS limit; at most 5 open and 5 a day; no new "
                "trades after ₹3,000 lost in a day. The screening numbers above used a simpler model (each position up to ₹1 L, "
                "first 5 trades a day).</p>")
        need = (f"<p>To have averaged ₹20,000 a month in 2025-26, every position would have had to be <b>{sc['multiple']}×</b> bigger "
                f"(risk ₹{1000 * sc['multiple']:,.0f} a trade for intraday); the worst 5-year drawdown at that size: "
                f"<b>{inr(sc['max_drawdown_at_that_size'])}</b> on a ₹1 L account.</p>" if sc else
                "<p>It lost money in 2025-26: no position size reaches the target.</p>")
        fin_html.append(f"""<section class="card"><h3>{E(fname)}</h3><p class="rules">{E(_describe(r))}</p>
          <div class="grid3">{kv([("2021-24 a month", f"<span class='{cls(r['train_per_month'])}'>{inr(r['train_per_month'], True)}</span>"),
                                 ("2025-26 a month (unseen)", f"<b class='{cls(r['test_per_month'])}'>{inr(r['test_per_month'], True)}</b>"),
                                 ("Sessions ≥ ₹2,000 in 2025-26", f"{100 * float((daily[test] >= TARGET_DAY).mean()):.1f}%")])}
          {kv([("Trades 2021-24 / 2025-26", f"{r['train_trades']:,} / {r['test_trades']:,}"),
               ("Max drawdown 2021-24 / 2025-26", f"{inr(r['train_maxdd'])} / {inr(r['test_maxdd'])}"),
               ("Win rate 2025-26", f"{r.get('test_win') if r.get('test_win') is not None else '–'}%" if r['family'] == 'swing' else f"{r.get('test_pos_months', '–')}% positive months")])}</div>
          {exact}{need}{_equity_svg(daily, dates)}{heatmap({k: round(v) for k, v in months.items()}, TARGET_MONTH)}</section>""")
    total = len(doc["intraday"]) + len(doc["swing"])
    best_final = max((r["test_per_month"] for _, r, _, _ in finals), default=0)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Cash Strategy Search</title><style>{CSS}</style></head><body><main>
<h1>{total:,} equity cash strategies on the NIFTY 100, 5 years</h1>
<p class="lede">Intraday (MIS) and swing (delivery) · {doc['stocks']} stocks · 5-minute bars Oct 2021 – Oct 2026 · chosen on 2021-24, judged on
2025-26 · after brokerage, STT, exchange, GST, stamp duty, DP charges and slippage · run {E(doc['run_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>Target: ₹20,000 a month (₹2,000 a day) on ₹1 lakh, which is about 240% (500%) a year.</li>
<li><b>{sum(1 for rows in fams.values() for r in rows if r['train_per_month'] >= TARGET_MONTH and r['test_per_month'] >= TARGET_MONTH)}</b>
 of {total:,} variants averaged ₹20,000 a month in both periods at the tested sizes.</li>
<li>The best finalist picked on 2021-24 averaged <b>{inr(best_final, True)}</b> a month in 2025-26.</li>
<li>Each finalist below shows how much bigger its positions would have to be to reach ₹20,000 a month, and the drawdown that size carried.</li>
</ul></div>
<h2>Families</h2><div class="card scroll"><table><thead><tr><th>Family</th><th class="r">Variants</th><th class="r">2021-24 &gt; 0</th>
<th class="r">2025-26 &gt; 0</th><th class="r">Rank corr.</th><th class="r">Best 2021-24 /mo</th><th class="r">Best 2025-26 /mo</th>
<th class="r">≥ ₹20k both</th></tr></thead><tbody>{''.join(summ)}</tbody></table>
<p class="note">"Best 2025-26" is picked with hindsight from every variant: it shows what luck looks like.</p></div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px">{''.join(scat)}</div>
{_validation_html(doc)}
<h2>Every setup: its best version on 2021-24, and what that version did in 2025-26</h2>
<div class="card scroll"><table><thead><tr><th>Setup</th><th>Type</th><th>Best filter</th><th class="r">2021-24 /mo</th>
<th class="r">2025-26 /mo</th><th class="r">Trades</th><th class="r">2025-26 max DD</th></tr></thead><tbody>{setup_rows}</tbody></table></div>
<h2>Finalists (best on 2021-24 with ≥ 200 trades intraday / 60 swing and a drawdown under ₹35,000)</h2>
{''.join(fin_html)}
<h2>How it was tested</h2><div class="card"><ul>
<li>Intraday: 53 setups (opening ranges 5-60 min, breakouts, retests and fakeouts; previous-day high/low, CPR, pivots, Camarilla,
 20-day and weekly levels, round numbers; VWAP crosses, pullbacks and sigma bands; gaps; time-of-day momentum, late-day trend and
 afternoon reversal; volume spikes and new highs on volume; inside bars, EMA 9/21, RSI 30/70, Bollinger squeeze, Supertrend(10,3))
 x 17 filters (trend, NIFTY direction, opening relative volume, top-10 stocks in play, top-20 by ATR, narrow CPR, NR7, VIX regime and
 combinations) x exits (structural / ATR / % stops, 1-3R or no target, out 13:00 or 15:15, breakeven after 1R). Risk ₹1,000 a trade,
 at most ₹2.5 L a position (MIS), the first 5 trades a day, 3 bps slippage a side, NSE intraday charges.</li>
<li>Swing: 8 setups (RSI(2) pullback, Donchian breakout, near 250-day high, NR7, gap-down reversal, 3 down closes, volume breakout,
 weekly relative-strength rotation) x holding 3-20 sessions x stop none / 2 / 3 ATR x 5 or 10 positions x exit rules. Long only
 (cash shares cannot be shorted overnight), entries at the next open, delivery charges (0.1% STT each side) and DP charges.</li>
<li>Universe: today's NIFTY 100 (survivorship bias in favour of the strategies). Angel One 5-minute bars, split-adjusted.</li>
</ul></div><p class="note">Research on past data, not a forecast and not investment advice.</p></main></body></html>"""
    path.write_text(body, encoding="utf-8")
    return path, finals


def add_validation(doc, log=print):
    """Stress-test the best intraday finalist and attach any 1-minute checks on disk."""
    N, V, syms, ctx, uni = _setup_universe(lambda m: None)
    tops = _finalists(doc["intraday"], 200, 35000, 1)
    if not tops:
        return doc
    top = tops[0]
    log(f"  stress-testing {top['setup']} | {top['filter']}")
    doc["validation"] = {"name": _describe(top), "stress": stress(uni, ctx, top, N.dates, Settings().get("costs"))}
    for key, fname in (("one_minute", "orb5_1m_check.json"), ("per_trade", "orb5_1m_pertrade.json")):
        f = OUT / fname
        if f.exists():
            doc["validation"][key] = json.loads(f.read_text())
    return doc


def main(rebuild_only=False, log=print):
    p = OUT / "equity_grid.json"
    doc = json.loads(p.read_text()) if rebuild_only and p.exists() else run(log)
    doc = add_validation(doc, log)
    path, finals = write_report(doc, OUT / "equity_report.html")
    log(f"Report: {path}")
    for fname, r, daily, sc in finals:
        log(f"  {fname}: {r['setup']} | {r.get('filter', '')} | train {r['train_per_month']:,.0f}/mo test {r['test_per_month']:,.0f}/mo "
            f"(scale x{sc['multiple'] if sc else '-'})")
    return doc, finals
