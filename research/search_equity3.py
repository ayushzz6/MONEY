"""Equity cash, round 3: new strategy concepts on the NIFTY 200, plus pairs, ETFs and new swing setups.

    ./research.sh equity3           # the full search, then data/research/results/equity3_report.html
    ./research.sh equity3 --report  # rebuild the report from the saved grid
"""
import itertools
import json
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import research.factory_equity as fe
from core.config.settings import DATA_DIR, Settings
from research import data
from research.etf import run_etf
from research.factory_equity import DEFAULT_EXITS, EXIT_GRID, FILTERS, Ctx, Exit, Universe, evaluate, stats
from research.factory_equity2 import EXTRA_SETUPS, XS_FILTERS, add_cross_sectional_ranks
from research.factory_equity3 import NEW_FILTERS, NEW_SETUPS, add_cross_features
from research.factory_swing import run_swing_search
from research.pairs import run_pairs
from research.report import CSS, E, cls, compact, heatmap, inr, kv
from research.search import _equity_svg, _scale_for_target, _scatter

OUT = DATA_DIR / "research" / "results"
TRAIN_END = np.datetime64("2024-12-31")
OLD_FILTERS = ["none", "top10 rvol", "top10 rvol+nifty", "top10 rvol5", "top10 rvol5+nifty", "trend+nifty+rvol"]
ALL_FILTERS = FILTERS + XS_FILTERS + NEW_FILTERS
EXTRA_EXITS = [Exit(st, 0, "15:15", False, trail=tr, time_stop=ts) for st, tr, ts in itertools.product(
    ["struct", "atr0.1", "atr0.25", "atr0.5", "pct0.5", "pct1.0"], [0.5, 1.0], [0, 6])]
STAGE2_EXITS = EXIT_GRID + EXTRA_EXITS


INDEXES = {"nifty200": ("NIFTY 200", data.nifty200, ""), "nifty500": ("NIFTY 500", data.nifty500, "_n500")}


def universe(log, index="nifty200"):
    fe.SETUPS.update(EXTRA_SETUPS)
    fe.SETUPS.update(NEW_SETUPS)
    sectors = INDEXES[index][1]()
    N, V = data.load("NIFTY"), data.load("INDIAVIX")
    syms = [s for s in sectors if (data.HIST_DIR / f"{s}.npz").exists()]
    ctx = Ctx(N, V)
    uni = Universe(syms, log, ctx)
    add_cross_sectional_ranks(uni)
    add_cross_features(uni, sectors, N)
    return N, V, syms, sectors, ctx, uni


def _meta(setup, filt, ex):
    return {"family": "intraday", "setup": setup, "filter": filt, "exit": ex.label, "exit_stop": ex.stop,
            "exit_target": ex.target, "exit_time": ex.time, "exit_breakeven": ex.breakeven, "exit_trail": ex.trail,
            "exit_time_stop": ex.time_stop, "new": setup in NEW_SETUPS}


_G = {}


def _job(args):
    setup, filt, ex = args
    res = evaluate(_G["uni"], _G["ctx"], setup, filt, ex, _G["dates"], _G["costs"])
    return stats(res[0], res[1], res[2], _G["dates"], _meta(setup, filt, ex)) if res is not None else None


def _parallel(jobs, workers=8):
    """Evaluate (setup, filter, exit) jobs in forked workers that share the loaded universe."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return [r for r in pool.map(_job, jobs, chunksize=4) if r is not None]


def run(log=print, stage2_top=25, index="nifty200", intraday_only=False):
    """intraday_only: re-run the intraday stages and keep swing / pairs / ETF from the saved grid."""
    t0 = time.time()
    N, V, syms, sectors, ctx, uni = universe(log, index)
    costs = Settings().get("costs")
    dates = N.dates
    _G.update(uni=uni, ctx=ctx, dates=dates, costs=costs)
    jobs = [(setup, filt, ex) for setup in fe.SETUPS
            for filt in (ALL_FILTERS if setup in NEW_SETUPS else OLD_FILTERS)
            for ex in DEFAULT_EXITS + [Exit("struct", 0, "15:15", False, trail=1.0)]]
    log(f"  stage 1: {len(jobs):,} jobs on {len(uni.items)} stocks")
    workers = 6 if len(uni.items) > 300 else 8          # ~10 MB a stock in every worker's view of the universe
    s1 = _parallel(jobs, workers)
    log(f"  stage 1 done: {len(s1):,} variants ({time.time() - t0:.0f}s)")
    pairs, seen = [], set()
    for r in sorted(s1, key=lambda r: -r["train_per_month"]):
        k = (r["setup"], r["filter"])
        if r["train_trades"] >= 200 and k not in seen:
            seen.add(k)
            pairs.append(k)
        if len(pairs) >= stage2_top:
            break
    jobs = [(setup, filt, ex) for setup, filt in pairs for ex in STAGE2_EXITS]
    log(f"  stage 2: {len(jobs):,} jobs ({len(pairs)} setup/filter pairs x {len(STAGE2_EXITS)} exits)")
    s2 = _parallel(jobs, workers)
    log(f"  stage 2 done: {len(s2):,} variants ({time.time() - t0:.0f}s)")
    prev_path = OUT / f"equity3_grid{INDEXES[index][2]}.json"
    if intraday_only and prev_path.exists():
        prev = json.loads(prev_path.read_text())
        sw, pr, et, pair_list = prev["swing"], prev["pairs"], prev["etf"], prev.get("pair_list") or []
        log("  swing, pairs and ETF kept from the previous run")
    else:
        sw, _ = run_swing_search(syms, dates, log)
        mats = {m.symbol: m for m, f, lv in uni.items}
        pr, _, _, pair_list = run_pairs(mats, sectors, costs, log)
        et, _ = run_etf(log)
    doc = {"stocks": len(syms), "index": index, "intraday": s1 + s2, "swing": sw, "pairs": pr, "etf": et,
           "pair_list": pair_list, "run_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"equity3_grid{INDEXES[index][2]}.json").write_text(json.dumps(doc, default=str))
    log(f"  done in {time.time() - t0:.0f}s")
    return doc


def best_rows(rows, min_trades, max_dd=35000, n=5):
    ok = sorted([r for r in rows if r.get("train_trades", r.get("trades", 0)) >= min_trades and r["train_maxdd"] <= max_dd
                 and r["train_per_month"] > 0], key=lambda r: -r["train_per_month"])
    out, seen = [], set()
    for r in ok:
        k = (r["setup"], r.get("filter"))
        if k not in seen:
            seen.add(k)
            out.append(r)
        if len(out) >= n:
            break
    return out


def summary(doc, log=print):
    for fam in ("intraday", "swing", "pairs", "etf"):
        rows = doc.get(fam) or []
        if not rows:
            continue
        tr = np.array([r["train_per_month"] for r in rows])
        te = np.array([r["test_per_month"] for r in rows])
        rho = spearmanr(tr, te).correlation if len(rows) > 2 else float("nan")
        log(f"\n{fam}: {len(rows):,} variants; 2021-24 > 0: {(tr > 0).mean():.0%}; 2025-26 > 0: {(te > 0).mean():.0%}; "
            f"rank corr {rho:.2f}; best 2021-24 {tr.max():,.0f}/mo; best 2025-26 {te.max():,.0f}/mo; "
            f">= 20k in both: {int(((tr >= 20000) & (te >= 20000)).sum())}")
        mt = {"intraday": 200, "swing": 60, "pairs": 100, "etf": 50}[fam]
        for r in best_rows(rows, mt, n=8):
            nt = r.get("train_trades", r.get("trades"))
            log(f"   {r['setup'][:52]:53} {str(r.get('filter', ''))[:20]:21} {str(r.get('exit', r.get('exit_rule', '')))[:44]:45}"
                f" tr {r['train_per_month']:>7,.0f} DD {r['train_maxdd']:>7,.0f} | te {r['test_per_month']:>7,.0f} DD {r['test_maxdd']:>7,.0f} n {nt}")


# ---------------------------------------------------------------- report
def _new_table(rows):
    best = {}
    for r in rows:
        if r["family"] != "intraday" or not r.get("new") or r["train_trades"] < 200:
            continue
        if r["setup"] not in best or r["train_per_month"] > best[r["setup"]]["train_per_month"]:
            best[r["setup"]] = r
    return "".join(
        f"<tr><td>{E(r['setup'])}</td><td>{E(r['filter'])}</td><td class='small'>{E(r['exit'])}</td>"
        f"<td class='r {cls(r['train_per_month'])}'>{inr(r['train_per_month'], True)}</td>"
        f"<td class='r {cls(r['test_per_month'])}'>{inr(r['test_per_month'], True)}</td>"
        f"<td class='r'>{r['train_trades']:,} / {r['test_trades']:,}</td><td class='r'>{inr(r['test_maxdd'])}</td></tr>"
        for r in sorted(best.values(), key=lambda r: -r["train_per_month"]))


def _simple_table(rows):
    return "".join(
        f"<tr><td>{E(r['setup'])}</td><td class='r {cls(r['train_per_month'])}'>{inr(r['train_per_month'], True)}</td>"
        f"<td class='r {cls(r['test_per_month'])}'>{inr(r['test_per_month'], True)}</td><td class='r'>{inr(r['train_maxdd'])} / {inr(r['test_maxdd'])}</td>"
        f"<td class='r'>{r.get('trades', r.get('train_trades', 0)):,}</td></tr>"
        for r in sorted(rows, key=lambda r: -r["train_per_month"])[:12])


def _one_minute_html():
    """The leader's 2025-26 trades replayed on real 1-minute bars (research/validate_1m.py), if that has run."""
    chk, unb = OUT / "orb5_1m_check_n200_trail1_ts6.json", OUT / "orb5_1m_unambiguous_n200.json"
    if not chk.exists():
        return ""
    c = json.loads(chk.read_text())
    rows = [("5-minute bars (bar-path rule), 3 bps", c["5-minute bars (bar-path rule)"], None)]
    for k, v in c.items():
        if k.startswith("1-minute"):
            rows.append((k[len("1-minute bars, "):], v["per_month"], v["max_drawdown"]))
    t1 = "".join(f"<tr><td>{E(a)}</td><td class='r {cls(b)}'>{inr(b, True)}</td><td class='r'>{inr(d) if d is not None else '-'}</td></tr>"
                 for a, b, d in rows)
    t2 = ""
    if unb.exists():
        names = {"stop": "Stop order at the level, stop-loss live at once (needs the path rule)",
                 "close": "Market order when a minute closes beyond the level",
                 "stop_sl_after": "Stop order at the level, stop-loss placed when the entry minute closes"}
        t2 = ("<h3>Entry rules that need no guess inside a minute</h3><table><thead><tr><th>Entry</th><th class='r'>Stop</th>"
              "<th>Exit</th><th class='r'>3 bps</th><th class='r'>5 bps</th><th class='r'>8 bps</th><th class='r'>DD at 5 bps</th>"
              "<th class='r'>Months ≥ ₹20k (3 bps)</th></tr></thead><tbody>"
              + "".join(f"<tr><td>{E(names[r['entry']])}</td><td class='r'>{r['stop_atr']:g} ATR</td>"
                        f"<td>{'trail 1 ATR, 30-min time stop' if r['trail'] else 'out 15:15'}</td>"
                        + "".join(f"<td class='r {cls(r[b]['per_month'])}'>{inr(r[b]['per_month'], True)}</td>" for b in ("3bps", "5bps", "8bps"))
                        + f"<td class='r'>{inr(r['5bps']['maxdd'])}</td><td class='r'>{r['3bps']['months_20k']} / {r['3bps']['months']}</td></tr>"
                        for r in json.loads(unb.read_text()))
              + "</tbody></table><p class='note'>The 0.2 and 0.3 ATR stops were tried on 2025-26 1-minute data only (1-minute bars "
                "for 2021-24 were not downloaded), so their rows are not out-of-sample.</p>")
    return f"""<h2>The leader on real 1-minute bars, 2025-26</h2><div class="card scroll">
<p>Every 2025-26 trade of the leader rebuilt from Angel One 1-minute bars: opening range, trigger, fill, trailing stop and time stop.
{c.get('missing 1-minute days (ohlc)', 0)} stock-days had gaps in the 1-minute data and are left out.</p>
<p>The stop is 0.1 × daily ATR, often smaller than a single minute's range. In about a third of the trades the stop price was touched
inside the entry minute itself. One-minute bars cannot show whether that happened before or after the fill, so the path rule and the
worst case give opposite answers.</p>
<table><thead><tr><th>Assumption</th><th class="r">2025-26 /mo</th><th class="r">Max DD</th></tr></thead><tbody>{t1}</tbody></table>{t2}
<p><b>Verdict:</b> on 1-minute bars the leader makes about ₹16-18k a month at 3 bps slippage, about ₹10-12k at 5 bps, and about ₹2-6k at
8 bps. It does not reach ₹20,000 a month on ₹1 lakh under realistic execution. Entering on a minute close instead of a stop order loses
most of the edge, so the profit sits in the first seconds after the breakout. Only live tick-level paper trading can show what fills
are actually achievable.</p></div>"""


def exact_leaderboard(doc, uni, ctx, dates, costs, n=15, n_stress=3, log=print):
    """The n best setup/filter pairs on 2021-24 (each with its best 2021-24 exit) through the exact Rs 1 lakh account
    (Rs 5 L MIS, 5 positions, 5 trades a day, Rs 3,000 daily stop); the n_stress best of those on 2021-24 again at
    5 and 8 bps slippage and with the worst case in the entry bar. Every choice uses 2021-24 only."""
    from research.search_equity import exact_check, stress, _period
    board = []
    for r in best_rows(doc["intraday"], 200, max_dd=60000, n=n):
        daily, tr = exact_check(uni, ctx, r, dates, costs)
        months = defaultdict(float)
        for d, v in zip(dates, daily):
            months[str(d)[:7]] += v
        tm = [v for k, v in months.items() if k <= "2024-12"]
        sm = [v for k, v in months.items() if k > "2024-12"]
        board.append({**{k: r[k] for k in ("setup", "filter", "exit", "exit_stop", "exit_target", "exit_time",
                                           "exit_breakeven", "exit_trail", "exit_time_stop")},
                      "screen_train": r["train_per_month"], "screen_test": r["test_per_month"],
                      "exact": _period(daily, dates), "trades": len(tr),
                      "months_20k": [int(sum(v >= 20000 for v in tm)), len(tm), int(sum(v >= 20000 for v in sm)), len(sm)],
                      "months_loss": [int(sum(v < 0 for v in tm)), int(sum(v < 0 for v in sm))]})
        log(f"   exact: {r['setup'][:40]:40} {r['filter'][:18]:18} tr {board[-1]['exact']['train']['per_month']:>8,.0f}"
            f" te {board[-1]['exact']['test']['per_month']:>8,.0f}")
    board.sort(key=lambda b: -b["exact"]["train"]["per_month"])
    for b in board[:n_stress]:
        b["stress"] = stress(uni, ctx, b, dates, costs)
    return board


def _board_html(board):
    if not board:
        return ""
    w = board[0]
    te, tr = w["exact"]["test"]["per_month"], w["exact"]["train"]["per_month"]
    st = {x["label"][:6]: x for x in w.get("stress", [])}
    s5 = st.get("5 bps ")
    s8 = st.get("8 bps ")
    swc = st.get("Worst ")
    yes = te >= 20000 and tr >= 20000
    verdict = (f"<b class='{cls(te)}'>{'Yes' if yes else 'No'}</b>, on 5-minute bars at 3 bps slippage. The strategy picked on 2021-24 alone "
               f"made <b>{inr(tr, True)}</b> a month in 2021-24 and <b>{inr(te, True)}</b> a month in 2025-26, which it never saw.")
    extra = ""
    if s5 and s8 and swc:
        extra = (f"<li>At 5 bps slippage: {inr(s5['train']['per_month'], True)} / {inr(s5['test']['per_month'], True)} a month. At 8 bps: "
                 f"{inr(s8['train']['per_month'], True)} / {inr(s8['test']['per_month'], True)}.</li>"
                 f"<li>If every touch of the stop inside the entry bar is counted as a loss (the worst reading of a 5-minute bar): "
                 f"{inr(swc['train']['per_month'], True)} / {inr(swc['test']['per_month'], True)} a month.</li>")
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{E(b['setup'])}<div class='small'>{E(b['filter'])} · {E(b['exit'])}</div></td>"
        f"<td class='r {cls(b['exact']['train']['per_month'])}'>{inr(b['exact']['train']['per_month'], True)}</td>"
        f"<td class='r {cls(b['exact']['test']['per_month'])}'><b>{inr(b['exact']['test']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(b['exact']['train']['maxdd'])} / {inr(b['exact']['test']['maxdd'])}</td>"
        f"<td class='r'>{b['months_20k'][0]}/{b['months_20k'][1]} · {b['months_20k'][2]}/{b['months_20k'][3]}</td>"
        f"<td class='r'>{b['months_loss'][0]} · {b['months_loss'][1]}</td><td class='r'>{b['trades']:,}</td></tr>"
        for i, b in enumerate(board))
    stress_rows = "".join(
        f"<tr><td>{E(b['setup'][:44])}<div class='small'>{E(b['filter'])}</div></td>"
        + "".join(f"<td class='r {cls(x['test']['per_month'])}'>{inr(x['train']['per_month'], True)} / {inr(x['test']['per_month'], True)}</td>"
                  for x in b["stress"]) + "</tr>"
        for b in board if b.get("stress"))
    return f"""<div class="card"><h2 style="margin-top:0">Is ₹20,000 a month on ₹1 lakh possible?</h2><p>{verdict}</p>
<p><b>Strategy:</b> {E(w['setup'])} · filter: {E(w['filter'])} · {E(w['exit'])}</p><ul class="findings">
<li>Worst drawdown: {inr(w['exact']['train']['maxdd'])} (2021-24) and {inr(w['exact']['test']['maxdd'])} (2025-26). Months of ₹20k or more:
{w['months_20k'][0]} of {w['months_20k'][1]} and {w['months_20k'][2]} of {w['months_20k'][3]}. Losing months: {w['months_loss'][0]} and {w['months_loss'][1]}.</li>{extra}
</ul></div>
<h2>Exact ₹1 lakh account: the 15 best setup/filter pairs on 2021-24</h2><div class="card scroll"><table><thead><tr><th>#</th><th>Strategy</th>
<th class="r">2021-24 /mo</th><th class="r">2025-26 /mo</th><th class="r">Max DD</th><th class="r">Months ≥ ₹20k</th><th class="r">Losing months</th>
<th class="r">Trades</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Ranked by 2021-24 only. Account: ₹5 lakh MIS exposure on ₹1 lakh, ₹1,000 risk a trade, at most 5 positions and 5 trades a day,
trading stops for the day at a ₹3,000 loss, 3 bps slippage a side, all NSE charges.</p>
<h3>Harsher assumptions (2021-24 / 2025-26 a month)</h3><table><thead><tr><th>Strategy</th><th class="r">Base 3 bps</th><th class="r">5 bps</th>
<th class="r">8 bps</th><th class="r">Worst case in entry bar</th></tr></thead><tbody>{stress_rows}</tbody></table></div>"""


def write_report(doc, path=None, log=print):
    from research.search_equity import exact_check, stress, _period
    from research.factory_swing import Stock, simulate
    index = doc.get("index", "nifty200")
    path = path or OUT / f"equity3_report{INDEXES[index][2]}.html"
    N, V, syms, sectors, ctx, uni = universe(lambda m: None, index)
    dates = N.dates
    costs = Settings().get("costs")
    bp = OUT / f"equity3_exact{INDEXES[index][2]}.json"
    if bp.exists() and json.loads(bp.read_text()).get("run_at") == doc["run_at"]:
        board = json.loads(bp.read_text())["board"]
    else:
        board = exact_leaderboard(doc, uni, ctx, dates, costs, log=log)
        bp.write_text(json.dumps({"run_at": doc["run_at"], "board": board}, default=str))
    board_html = _board_html(board)
    finals = []
    for r in best_rows(doc["intraday"], 200, n=3):
        daily, tr = exact_check(uni, ctx, r, dates, costs)
        finals.append(("Intraday (MIS)", r, daily, _period(daily, dates), len(tr)))
    sw = best_rows(doc["swing"], 60, n=1)
    if sw:
        stocks = [Stock(m) for m, f, lv in uni.items if m.n > 300]
        r = sw[0]
        daily, _ = simulate(stocks, dates, r["setup"], r["hold_days"], r["stop_atr"], r["slots"], r["exit_rule"])
        finals.append(("Swing (delivery)", r, daily, _period(daily, dates), None))
    stress_rows = stress(uni, ctx, finals[0][1], dates, costs) if finals and finals[0][0].startswith("Intraday") else []
    fams = {k: doc.get(k) or [] for k in ("intraday", "swing", "pairs", "etf")}
    allr = [r for v in fams.values() for r in v]
    both = sum(1 for r in allr if r["train_per_month"] >= 20000 and r["test_per_month"] >= 20000)
    fin_html = []
    for fam, r, daily, p, ntr in finals:
        months = defaultdict(float)
        for d, v in zip(dates, daily):
            months[str(d)[:7]] += v
        sc = _scale_for_target(daily, dates)
        desc = (f"{r['setup']} · filter: {r.get('filter')} · {r.get('exit')}" if fam.startswith("Intraday") else
                f"{r['setup']} · hold up to {r['hold_days']} sessions ({r['exit_rule']}) · stop {r['stop_atr']} ATR · {r['slots']} positions")
        fin_html.append(f"""<section class="card"><h3>{E(fam)}</h3><p class="rules">{E(desc)}</p>{kv([
            ("2021-24 a month (exact account)", f"<span class='{cls(p['train']['per_month'])}'>{inr(p['train']['per_month'], True)}</span>"),
            ("2025-26 a month (unseen)", f"<b class='{cls(p['test']['per_month'])}'>{inr(p['test']['per_month'], True)}</b>"),
            ("Max drawdown 2021-24 / 2025-26", f"{inr(p['train']['maxdd'])} / {inr(p['test']['maxdd'])}"),
            ("Sessions >= ₹2,000, 2021-24 / 2025-26", f"{p['train']['days_2k']}% / {p['test']['days_2k']}%")])}
            <p>{'To average ₹20,000 a month in 2025-26 every position would have had to be <b>' + str(sc['multiple']) + '×</b> bigger (worst 5-year drawdown at that size: <b>' + inr(sc['max_drawdown_at_that_size']) + '</b>).' if sc else 'It lost money in 2025-26.'}</p>
            {_equity_svg(daily, dates)}{heatmap({k: round(v) for k, v in months.items()}, 20000)}</section>""")
    st_html = ""
    if stress_rows:
        st_html = ("<h2>Stress test of the top intraday finalist</h2><div class='card'><table><thead><tr><th>Assumption</th>"
                   "<th class='r'>2021-24 /mo</th><th class='r'>2025-26 /mo</th><th class='r'>2025-26 max DD</th></tr></thead><tbody>"
                   + "".join(f"<tr><td>{E(x['label'])}</td><td class='r {cls(x['train']['per_month'])}'>{inr(x['train']['per_month'], True)}</td>"
                             f"<td class='r {cls(x['test']['per_month'])}'>{inr(x['test']['per_month'], True)}</td><td class='r'>{inr(x['test']['maxdd'])}</td></tr>"
                             for x in stress_rows) + "</tbody></table></div>")
    v1_html = _one_minute_html() if index == "nifty200" else ""
    fam_rows = "".join(
        f"<tr><td>{E(k)}</td><td class='r'>{len(v):,}</td><td class='r'>{np.mean([r['train_per_month'] > 0 for r in v]):.0%}</td>"
        f"<td class='r'>{np.mean([r['test_per_month'] > 0 for r in v]):.0%}</td>"
        f"<td class='r'>{inr(max(r['train_per_month'] for r in v), True)}</td><td class='r'>{inr(max(r['test_per_month'] for r in v), True)}</td></tr>"
        for k, v in fams.items() if v)
    pairs_txt = ", ".join(f"{a}/{b}" for a, b, *_ in (doc.get("pair_list") or [])[:12])
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Strategies Round 3</title><style>{CSS}</style></head><body><main>
<h1>Round 3: new equity cash strategies, {len(allr):,} variants</h1>
<p class="lede">{doc['stocks']} {INDEXES[index][0]} stocks + NIFTYBEES / BANKBEES · 5-minute bars Oct 2021 – Oct 2026 · chosen on 2021-24, judged on 2025-26 ·
after all charges and slippage · run {E(doc['run_at'])}</p>
{board_html}
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>{both}</b> of {len(allr):,} variants averaged ₹20,000 a month on ₹1 lakh in both periods.</li>
<li>Best finalist picked on 2021-24: <b>{inr(max((p['test']['per_month'] for _, _, _, p, _ in finals), default=0), True)}</b> a month in 2025-26 (exact account).</li>
</ul></div>
<h2>Families</h2><div class="card scroll"><table><thead><tr><th>Family</th><th class="r">Variants</th><th class="r">2021-24 &gt; 0</th><th class="r">2025-26 &gt; 0</th>
<th class="r">Best 2021-24 /mo</th><th class="r">Best 2025-26 /mo</th></tr></thead><tbody>{fam_rows}</tbody></table>
<p class="note">"Best 2025-26" is picked with hindsight from every variant.</p></div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px">
<div class="card" style="margin:0"><h3>Intraday, 2021-24 vs 2025-26</h3>{_scatter(fams['intraday'])}</div>
<div class="card" style="margin:0"><h3>Swing, 2021-24 vs 2025-26</h3>{_scatter(fams['swing'])}</div></div>
<h2>The 17 new intraday concepts: each one's best version on 2021-24, and its 2025-26 result</h2>
<div class="card scroll"><table><thead><tr><th>Concept</th><th>Best filter</th><th>Best exit</th><th class="r">2021-24 /mo</th><th class="r">2025-26 /mo</th>
<th class="r">Trades</th><th class="r">2025-26 DD</th></tr></thead><tbody>{_new_table(fams['intraday'])}</tbody></table></div>
<h2>Market-neutral sector pairs</h2><div class="card scroll"><p class="note">Pairs (picked on 2021-24 correlation): {E(pairs_txt)}</p>
<table><thead><tr><th>Rule</th><th class="r">2021-24 /mo</th><th class="r">2025-26 /mo</th><th class="r">DD</th><th class="r">Trades</th></tr></thead><tbody>{_simple_table(fams['pairs'])}</tbody></table></div>
<h2>Index ETFs (cash segment)</h2><div class="card scroll"><table><thead><tr><th>Rule</th><th class="r">2021-24 /mo</th><th class="r">2025-26 /mo</th>
<th class="r">DD</th><th class="r">Trades</th></tr></thead><tbody>{_simple_table(fams['etf'])}</tbody></table>
<p class="note"><b>Ignore the "sell 09:15 open" rows: they come from bad data.</b> The ETFs' first 5-minute bar often opens on stray prints far from
fair value. NIFTYBEES shows overnight "moves" of ±13% on days the NIFTY moved under 1%, and +0.69% a night on average against the index's +0.06%.
Selling at the 09:20 close instead, a price you could actually get, loses money in both periods.</p></div>
{st_html}
{v1_html}
<h2>Finalists (picked on 2021-24)</h2>{''.join(fin_html)}
<p class="note">Research on past data, not a forecast and not investment advice.</p></main></body></html>"""
    path.write_text(body, encoding="utf-8")
    log(f"Report: {path}")
    return path, finals
