"""HTML report for the strategy library and the day-type router: data/research/results/library_report.html"""
import json

import numpy as np

from core.config.settings import DATA_DIR
from research.report import CSS, E, cls, inr

RES = DATA_DIR / "research" / "results"
SERIES = ["var(--series-1)", "#d9822b", "#2e9e8f", "#8a5cc2", "#8a8a85"]
EXTRA_CSS = """
.heat2 td{font-size:12px;text-align:center;padding:5px 6px;min-width:62px}
.heat2 th{font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin:6px 0}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.small{font-size:12px;color:var(--muted)}
.tag{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;background:var(--surface-2);margin:1px 2px}
details summary{cursor:pointer;font-weight:600;margin:6px 0}
"""


def _curves(appr, dates, split, w=900, h=260, label="2025-26: not used to choose"):
    pts = [np.array(a["curve"], float) for a in appr]
    lo = min(min(p.min() for p in pts), 0.0)
    hi = max(max(p.max() for p in pts), 1.0)
    span = hi - lo or 1
    n = len(dates)
    pl, pr, pt, pb = 74, 150, 10, 24
    X = lambda i: pl + i * (w - pl - pr) / max(1, n - 1)
    Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / span)
    k = int(np.searchsorted(np.array(dates), split, side="right"))
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative profit by approach">',
           f'<rect x="{X(k):.1f}" y="{pt}" width="{X(n - 1) - X(k):.1f}" height="{h - pt - pb}" fill="var(--surface-2)"/>',
           f'<text x="{X(k) + 6:.1f}" y="{pt + 14}" class="tick">{E(label)}</text>']
    for t in (lo, 0.0, hi):
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{inr(t)}</text>')
    step = max(1, n // 400)
    for j, (a, p) in enumerate(zip(appr, pts)):
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(p[i]):.1f}" for i in range(0, n, step))
        out.append(f'<path d="{path}" fill="none" stroke="{SERIES[j % len(SERIES)]}" stroke-width="{2.5 if j == 0 else 1.6}"/>')
        out.append(f'<text x="{w - pr + 6}" y="{Y(p[-1]) + 4:.1f}" class="tick">{E(a["name"][:22])}</text>')
    out.append(f'<text x="{pl}" y="{h - 6}" class="tick">{E(dates[0])}</text>'
               f'<text x="{w - pr}" y="{h - 6}" class="tick" text-anchor="end">{E(dates[-1])}</text></svg>')
    legend = "".join(f'<span><i style="background:{SERIES[j % len(SERIES)]}"></i>{E(a["name"])}</span>' for j, a in enumerate(appr))
    return "".join(out) + f'<div class="legend">{legend}</div>'


def _heat(h, title):
    vals = [v for row in h["grid"] for v in row if v is not None]
    mx = max([abs(v) for v in vals] + [0.05])
    head = "".join(f"<th>{E(c)}</th>" for c in h["cols"])
    rows = []
    for name, row in zip(h["rows"], h["grid"]):
        cells = []
        for v in row:
            if v is None:
                cells.append("<td class='small'>-</td>")
                continue
            pct = int(10 + 60 * min(1.0, abs(v) / mx))
            col = "var(--up)" if v >= 0 else "var(--down)"
            cells.append(f"<td style='background:color-mix(in srgb,{col} {pct}%,var(--surface))'>{v:+.2f}</td>")
        rows.append(f"<tr><th>{E(name)}</th>{''.join(cells)}</tr>")
    return (f"<h3>{E(title)}</h3><div class='scroll'><table class='heat2'><thead><tr><th></th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def write(summary=None, variants=None, leak=None):
    summary = summary or json.loads((RES / "library_summary.json").read_text())
    variants = variants or json.loads((RES / "library_variants.json").read_text())
    appr = summary["approaches"]
    split = summary["train_end"]
    dates = summary["dates_axis"]
    best = max(appr, key=lambda a: a["train"]["per_month"])
    router = appr[0]
    # per strategy: the base version and the best version picked on 2021-24 (account rupees a month)
    by_s = {}
    for r in variants:
        by_s.setdefault(r["strategy"], []).append(r)
    srows = []
    for key, rs in by_s.items():
        base = next(r for r in rs if r["filter"] == "none" and r["exit"] == "target 2R")
        pick = max(rs, key=lambda r: r["acct_train"]["per_month"])
        srows.append((base, pick))
    srows.sort(key=lambda x: -x[1]["acct_train"]["per_month"])
    strat_html = "".join(
        f"<tr><td><b>{E(b['name'])}</b><div class='small'>{E(b['family'])} · {E(b['source'])}</div></td>"
        f"<td class='r'>{b['train_trades']:,} / {b['test_trades']:,}</td>"
        f"<td class='r'>{b['train_win'] or 0:.0f}% / {b['test_win'] or 0:.0f}%</td>"
        f"<td class='r {cls(b['train_avgR'])}'>{b['train_avgR']:+.3f}</td><td class='r {cls(b['test_avgR'])}'>{b['test_avgR']:+.3f}</td>"
        f"<td class='small'>{E(p['filter'])} · {E(p['exit'])}</td>"
        f"<td class='r {cls(p['acct_train']['per_month'])}'>{inr(p['acct_train']['per_month'], True)}</td>"
        f"<td class='r {cls(p['acct_test']['per_month'])}'><b>{inr(p['acct_test']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(max(p['acct_train']['maxdd'], p['acct_test']['maxdd']))}</td></tr>" for b, p in srows)
    appr_html = "".join(
        f"<tr><td><b>{E(a['name'])}</b><div class='small'>{E(a['desc'])}</div></td>"
        f"<td class='r {cls(a['train']['per_month'])}'>{inr(a['train']['per_month'], True)}</td>"
        f"<td class='r {cls(a['test']['per_month'])}'><b>{inr(a['test']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(a['train']['maxdd'])} / {inr(a['test']['maxdd'])}</td>"
        f"<td class='r'>{a['train']['months_20k']}/{a['train']['months']} · {a['test']['months_20k']}/{a['test']['months']}</td>"
        f"<td class='r'>{a['train']['months_loss']} · {a['test']['months_loss']}</td></tr>" for a in appr)
    types_html = "".join(
        f"<tr><td><b>{E(t['type'])}</b></td><td class='r'>{t['train_days']} / {t['test_days']}</td>"
        f"<td>{''.join(f'<span class=tag>{E(x)}</span>' for x in t['picked']) or '<span class=small>nothing passed (no trades that day type)</span>'}</td>"
        f"<td class='r {cls(t['router_train'])}'>{inr(t['router_train'], True)}</td><td class='r {cls(t['router_test'])}'>{inr(t['router_test'], True)}</td></tr>"
        for t in summary["day_types"])
    open_html = "".join(
        f"<tr><td><b>{E(t['type'])}</b></td><td class='r'>{t['train_days']} / {t['test_days']}</td>"
        f"<td>{''.join(f'<span class=tag>{E(x)}</span>' for x in t['picked']) or '<span class=small>nothing passed</span>'}</td></tr>"
        for t in summary["open_types"])
    fil_html = "".join(
        f"<tr><td>{E(f['filter'])}</td><td class='r'>{f['strategies']}</td>"
        f"<td class='r {cls(f['train_dR'] or 0)}'>{(f['train_dR'] or 0):+.3f}</td><td class='r {cls(f['test_dR'] or 0)}'>{(f['test_dR'] or 0):+.3f}</td></tr>"
        for f in summary["filters"])
    cf = summary["confusion"]
    conf_html = ("<table><thead><tr><th>Router's type at 09:30</th>" + "".join(f"<th class='r'>{E(c)}</th>" for c in cf["cols"])
                 + "</tr></thead><tbody>" + "".join(f"<tr><td>{E(rn)}</td>" + "".join(f"<td class='r'>{v}</td>" for v in row) + "</tr>"
                                                  for rn, row in zip(cf["rows"], cf["grid"])) + "</tbody></table>")
    pos = summary["positive"]
    leak_txt = (f"{leak['cuts']} random cut points, {leak['rules']} rules (every strategy and filter): "
                f"{'no signal changed when the future was replaced' if not leak['failing'] else 'FAILING: ' + ', '.join(leak['failing'])}."
                if leak else "Run research/lib/leak.py to check.")
    yes = router["train"]["per_month"] >= 20000 and router["test"]["per_month"] >= 20000
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy Library Router</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>{summary['strategies']} book strategies, {summary['variants']:,} versions, and a day-type router</h1>
<p class="lede">{summary['stocks']} NIFTY 500 stocks · 5-minute bars {E(summary['dates'][0])} to {E(summary['dates'][1])} ·
signals use only bars already closed and enter at the next bar's open · chosen on data to {E(split)}, judged after ·
₹1 lakh account, all NSE charges and 3 bps slippage a side · built {E(summary['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>The day-type router made <b class="{cls(router['train']['per_month'])}">{inr(router['train']['per_month'], True)}</b> a month in 2021-24 and
<b class="{cls(router['test']['per_month'])}">{inr(router['test']['per_month'], True)}</b> a month in 2025-26, which it never saw.
{'That is above ₹20,000 in both periods.' if yes else 'That is not ₹20,000 a month in both periods.'}</li>
<li>Of {summary['variants']:,} strategy versions, {pos['train']} made money per trade in 2021-24, {pos['test']} in 2025-26, and {pos['both']} in both.
{summary['acct_20k_both']} reached ₹20,000 a month in both periods on the ₹1 lakh account.</li>
<li>No-lookahead check: {E(leak_txt)}</li></ul></div>
<h2>Router versus the alternatives</h2><div class="card scroll"><table><thead><tr><th>Approach</th><th class="r">2021-24 /mo</th>
<th class="r">2025-26 /mo</th><th class="r">Max DD</th><th class="r">Months ≥ ₹20k</th><th class="r">Losing months</th></tr></thead>
<tbody>{appr_html}</tbody></table>{_curves(appr, dates, split)}
<p class="note">Cumulative profit in rupees on the ₹1 lakh account. The shaded part was never used to choose anything.</p></div>
<h2>How the router reads a day</h2><div class="card scroll">
<p>At <b>09:15</b> (signals before 09:30): the NIFTY's gap (0.4% or more is a gap) and the India VIX against its last year (top 40% = volatile).
At <b>09:30</b>: also how the NIFTY moved in its first 15 minutes, the share of NIFTY 500 stocks above their open (breadth), and the size of the
first 15 minutes against the NIFTY's daily range. For each day type it trades the 5 strategy versions with the best 2021-24 record on that
type (at least {60} trades and a t-statistic of 2 or more).</p>
<table><thead><tr><th>Day type at 09:30</th><th class="r">Days 2021-24 / 2025-26</th><th>Strategies it trades</th>
<th class="r">Avg day 2021-24</th><th class="r">Avg day 2025-26</th></tr></thead><tbody>{types_html}</tbody></table>
<h3>At the open (signals before 09:30)</h3><table><thead><tr><th>Open type</th><th class="r">Days</th><th>Strategies</th></tr></thead><tbody>{open_html}</tbody></table></div>
<h2>Which families work on which days</h2><div class="card">
<p class="note">Average result per trade in R (1R = the planned loss), every strategy in the family, no filter, 2R target. Blue made money.</p>
{_heat(summary['heat']['router_train'], 'Day types known at 09:30, 2021-24')}
{_heat(summary['heat']['router_test'], 'Day types known at 09:30, 2025-26')}
{_heat(summary['heat']['hindsight_test'], 'What the day turned out to be (hindsight, for understanding only), 2025-26')}
<h3>Router's day type versus what the day became (all days)</h3><div class="scroll">{conf_html}</div></div>
<h2>Does confluence help?</h2><div class="card scroll"><table><thead><tr><th>Filter</th><th class="r">Strategies</th>
<th class="r">Change in R per trade, 2021-24</th><th class="r">2025-26</th></tr></thead><tbody>{fil_html}</tbody></table>
<p class="note">Against the same strategy with no filter (2R target). Positive = the filter improved the average trade.</p></div>
<h2>All {summary['strategies']} strategies</h2><div class="card scroll"><table><thead><tr><th>Strategy</th><th class="r">Trades 21-24 / 25-26</th>
<th class="r">Win %</th><th class="r">Avg R 21-24</th><th class="r">Avg R 25-26</th><th>Best version (picked on 2021-24)</th>
<th class="r">Account 21-24 /mo</th><th class="r">Account 25-26 /mo</th><th class="r">Max DD</th></tr></thead><tbody>{strat_html}</tbody></table>
<p class="note">Avg R: every trade of the plain version (no filter, 2R target), after charges and slippage. Account: the version with the best
2021-24 account result, its 5 earliest trades a day, under the ₹1 lakh account's limits.</p></div>
<p class="note">Research on past data, not a forecast and not investment advice. Survivorship: today's NIFTY 500 members. A stop-loss and a
target touched in the same 5-minute bar count as the stop.</p></main></body></html>"""
    path = RES / "library_report.html"
    path.write_text(body, encoding="utf-8")
    return path


def write_round2(summary=None):
    summary = summary or json.loads((RES / "library2_summary.json").read_text())
    appr = summary["approaches"]
    split, dates = summary["train_end"], summary["dates_axis"]

    def acct(a):
        return (f"<td class='r {cls(a['train']['per_month'])}'>{inr(a['train']['per_month'], True)}</td>"
                f"<td class='r {cls(a['test']['per_month'])}'><b>{inr(a['test']['per_month'], True)}</b></td>"
                f"<td class='r'>{inr(a['train']['maxdd'])} / {inr(a['test']['maxdd'])}</td>"
                f"<td class='r'>{a['train']['months_20k']}/{a['train']['months']} · {a['test']['months_20k']}/{a['test']['months']}</td>")
    appr_html = "".join(f"<tr><td><b>{E(a['name'])}</b></td>{acct(a)}<td class='r'>{a['train']['months_loss']} · {a['test']['months_loss']}</td></tr>" for a in appr)
    idea_html = ""
    for idea, top in summary["best"].items():
        rows = []
        for r in top:
            tw = summary["twins"].get(str(r["v"])) or summary["twins"].get(r["v"])
            worst = (f"<div class='small'>worst case in the entry bar: {inr(tw['acct_train']['per_month'], True)} / "
                     f"{inr(tw['acct_test']['per_month'], True)}</div>") if tw else ""
            rows.append(f"<tr><td><b>{E(r['name'])}</b><div class='small'>{E(r['universe'])} · {E(r['context'])} · {E(r['exit'])} · {E(r['entry'])}</div>{worst}</td>"
                        f"<td class='r'>{r['train_trades']:,} / {r['test_trades']:,}</td>"
                        f"<td class='r'>{(r['raw_R'] or 0):+.3f}</td><td class='r {cls(r['train_avgR'])}'>{r['train_avgR']:+.3f}</td>"
                        f"<td class='r {cls(r['test_avgR'])}'>{r['test_avgR']:+.3f}</td>"
                        f"<td class='r {cls(r['acct_train']['per_month'])}'>{inr(r['acct_train']['per_month'], True)}</td>"
                        f"<td class='r {cls(r['acct_test']['per_month'])}'><b>{inr(r['acct_test']['per_month'], True)}</b></td>"
                        f"<td class='r'>{inr(max(r['acct_train']['maxdd'], r['acct_test']['maxdd']))}</td></tr>")
        bi = summary["by_idea"][idea]
        idea_html += (f"<h2>{E(idea)}</h2><div class='card scroll'><p class='note'>{bi['variants']:,} versions; {bi['raw_positive']:,} with an "
                      f"edge before costs; {bi['net_positive_both']:,} positive after costs in both periods. The 10 best on 2021-24:</p>"
                      f"<table><thead><tr><th>Version</th><th class='r'>Trades</th><th class='r'>Raw R</th><th class='r'>Net R 21-24</th>"
                      f"<th class='r'>Net R 25-26</th><th class='r'>Account 21-24 /mo</th><th class='r'>Account 25-26 /mo</th>"
                      f"<th class='r'>Max DD</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    picks = "".join(f"<tr><td>{E(p['type'])}</td><td>{''.join(f'<span class=tag>{E(x)}</span>' for x in p['picked']) or '<span class=small>nothing passed</span>'}</td></tr>"
                    for p in summary["picks"])
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Library Round Two</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Round 2: big movers, stop-order entries and agreement</h1>
<p class="lede">{summary['stocks']} NIFTY 500 stocks · 5-minute bars {E(summary['dates'][0])} to {E(summary['dates'][1])} · {summary['variants']:,} versions ·
chosen on data to {E(split)}, judged after · ₹1 lakh account, NSE charges and 3 bps slippage a side · built {E(summary['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>{summary['positive_both']} versions made money per trade after costs in both periods ({summary['positive_train']} in 2021-24, {summary['positive_test']} in 2025-26).</li>
<li>{summary['acct_20k_both']} reached ₹20,000 a month on the ₹1 lakh account in both periods.</li></ul></div>
<h2>Router and the best of each idea</h2><div class="card scroll"><table><thead><tr><th>Approach</th><th class="r">2021-24 /mo</th>
<th class="r">2025-26 /mo</th><th class="r">Max DD</th><th class="r">Months ≥ ₹20k</th><th class="r">Losing months</th></tr></thead>
<tbody>{appr_html}</tbody></table>{_curves(appr, dates, split)}</div>
{idea_html}
<h2>What the router trades on each day type</h2><div class="card scroll"><table><thead><tr><th>Day type at 09:30</th><th>Versions</th></tr></thead>
<tbody>{picks}</tbody></table></div>
<p class="note">Research on past data, not investment advice. Stop orders: inside the entry bar the bar path decides a stop touch; the worst-case
line counts any touch as a loss. "Raw R": average result per trade before charges and slippage.</p></main></body></html>"""
    path = RES / "library2_report.html"
    path.write_text(body, encoding="utf-8")
    return path


def write_round3(summary=None):
    summary = summary or json.loads((RES / "library3_summary.json").read_text())
    rc, rt = summary["router_cal"], summary["router_test"]
    base = summary["base_win"]
    sp = summary["splits"]
    ok = (rt["win"] or 0) >= 75
    thr_html = "".join(
        f"<tr><td class='r'>{t['threshold']:.2f}</td><td class='r'>{t['cal']['trades']:,}</td><td class='r'>{t['cal']['win'] if t['cal']['win'] is not None else '-'}%</td>"
        f"<td class='r'>{t['test']['trades']:,}</td><td class='r {'pos' if (t['test']['win'] or 0) >= 75 else ''}'><b>{t['test']['win'] if t['test']['win'] is not None else '-'}%</b></td>"
        f"<td class='r {cls(t['test']['avgR'] or 0)}'>{(t['test']['avgR'] or 0):+.3f}</td></tr>" for t in summary["thresholds"])
    cov_html = "".join(
        f"<tr><td class='r'>top {c['top_pct']}%</td><td class='r'>{c['trades']:,}</td><td class='r'>{c['win']}%</td>"
        f"<td class='r'>{k['trades']:,}</td><td class='r {'pos' if k['win'] >= 75 else ''}'><b>{k['win']}%</b></td></tr>"
        for c, k in zip(summary["coverage"]["cal"], summary["coverage"]["test"]))
    acct_html = "".join(
        f"<tr><td><b>{E(n)}</b></td><td class='r'>{a['trades']:,}</td>"
        f"<td class='r {cls(a['train']['per_month'])}'>{inr(a['train']['per_month'], True)}</td>"
        f"<td class='r {cls(a['test']['per_month'])}'><b>{inr(a['test']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(a['train']['maxdd'])} / {inr(a['test']['maxdd'])}</td>"
        f"<td class='r'>{a['train']['months_loss']}/{a['train']['months']} · {a['test']['months_loss']}/{a['test']['months']}</td></tr>"
        for n, a in summary["accounts"].items())
    appr = [{"name": n, "curve": a["curve"]} for n, a in summary["accounts"].items()]
    imp_html = "".join(f"<tr><td>{E(i['feature'])}</td><td class='r'>{i['gain']:+.4f}</td></tr>" for i in summary["importance"][:12])
    picked_html = "".join(f"<tr><td>{E(n)}</td><td class='r'>{s['trades']:,}</td><td class='r'>{s['win']}%</td><td class='r {cls(s['avgR'] or 0)}'>{(s['avgR'] or 0):+.3f}</td></tr>"
                          for n, s in summary["picked"].items())
    strat_html = "".join(
        f"<tr><td><b>{E(s['name'])}</b><div class='small'>{E(s['family'])} · {E(s['source'])}</div></td>"
        f"<td class='r'>{s['early']['trades']:,} / {s['late']['trades']:,}</td><td class='r'>{s['early']['win'] if s['early']['win'] is not None else '-'}%</td>"
        f"<td class='r'><b>{s['late']['win'] if s['late']['win'] is not None else '-'}%</b></td>"
        f"<td class='r {cls(s['early']['avgR'] or 0)}'>{(s['early']['avgR'] or 0):+.3f}</td><td class='r {cls(s['late']['avgR'] or 0)}'>{(s['late']['avgR'] or 0):+.3f}</td></tr>"
        for s in summary["per_strategy"] if s["late"]["trades"] >= 200)
    months = summary["months"]
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Router At One To One</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Round 3: {summary['strategies']} strategies at 1:1 and a router aiming at 75% winners</h1>
<p class="lede">{summary['stocks']} NIFTY 500 stocks · 5-minute bars {E(summary['dates'][0])} to {E(summary['dates'][1])} · {summary['trades']:,} trades, target as far as the stop ·
the router learns on {E(sp['learn'][0])} to {E(sp['learn'][1])}, its cut-off is set on 2024, and it is judged once on {E(sp['judge'][0])} to {E(sp['judge'][1])} ·
NSE charges and 3 bps slippage a side · built {E(summary['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>Without the router, {base['test']['win']}% of all 1:1 trades won in 2025-26 (average {base['test']['avgR']:+.3f}R after costs).</li>
<li>The router's cut-off, set on 2024 where it gave <b>{rc['win']}%</b> winners on {rc['trades']:,} trades{'' if rc['reached_75'] else ' (75% was not reachable on 2024 with at least 150 trades; the best reachable cut-off was used)'},
gave <b class="{'pos' if ok else 'neg'}">{rt['win']}%</b> winners on <b>{rt['trades']:,}</b> trades in 2025-26, average <b class="{cls(rt['avgR'] or 0)}">{(rt['avgR'] or 0):+.3f}R</b> a trade after costs.
{'That meets 75% out of sample.' if ok else 'That does not meet 75% out of sample.'}</li>
<li>How well the router separates winners from losers (AUC, 0.5 = coin toss): {summary['auc']['cal']} on 2024, {summary['auc']['test']} on 2025-26.</li></ul></div>
<h2>Confidence cut-off: 2024 (used to choose) against 2025-26 (unseen)</h2><div class="card scroll"><table><thead><tr><th class="r">Cut-off</th>
<th class="r">2024 trades</th><th class="r">2024 winners</th><th class="r">2025-26 trades</th><th class="r">2025-26 winners</th><th class="r">2025-26 avg R</th></tr></thead>
<tbody>{thr_html}</tbody></table>
<h3>Taking only the most confident trades</h3><table><thead><tr><th class="r">Share taken</th><th class="r">2024 trades</th><th class="r">2024 winners</th>
<th class="r">2025-26 trades</th><th class="r">2025-26 winners</th></tr></thead><tbody>{cov_html}</tbody></table>
<p class="note">The second table picks each period's own top x% by confidence (hindsight on where the cut lands), to show the ceiling.</p></div>
<h2>The ₹1 lakh account</h2><div class="card scroll"><table><thead><tr><th>Approach</th><th class="r">Trades</th><th class="r">2021-24 /mo</th>
<th class="r">2025-26 /mo</th><th class="r">Max DD</th><th class="r">Losing months</th></tr></thead><tbody>{acct_html}</tbody></table>
{_curves(appr, summary['dates_axis'], sp['cutoff'][1])}
<p class="note">First 5 trades a day (or 10), one per stock, ₹1,000 risk a trade, ₹5 lakh intraday exposure, trading stops for the day at a ₹3,000 loss.
The 2021-24 column is in-sample for the router (it learned on 2021-23); only 2025-26 is a fair test.
The router's 2025-26 picks fall on {summary['spread']['days_with_trades']} of {summary['spread']['days']} days, {summary['spread']['mean_per_active_day']} a day on average
(up to {summary['spread']['max_in_a_day']}), {int(100 * summary['spread']['share_of_trades_on_top_20_days'])}% of them on the 20 busiest days: the account's 5-a-day limit leaves most of them untaken.
Months in 2025-26 for the router: {', '.join(f'{k}: {inr(v, True)}' for k, v in months.items())}.</p></div>
<div class="grid-2" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div class="card"><h3>What the router leans on (drop in AUC when the feature is shuffled)</h3><table><tbody>{imp_html}</tbody></table></div>
<div class="card"><h3>What it traded in 2025-26 at the cut-off</h3><table><thead><tr><th>Strategy</th><th class="r">Trades</th><th class="r">Winners</th><th class="r">Avg R</th></tr></thead><tbody>{picked_html or '<tr><td class=small>none</td></tr>'}</tbody></table></div></div>
<h2>Every strategy at 1:1, no router</h2><div class="card scroll"><table><thead><tr><th>Strategy</th><th class="r">Trades 21-24 / 25-26</th><th class="r">Winners 21-24</th>
<th class="r">Winners 25-26</th><th class="r">Avg R 21-24</th><th class="r">Avg R 25-26</th></tr></thead><tbody>{strat_html}</tbody></table>
<p class="note">Sorted by 2025-26 winners; strategies with under 200 trades in 2025-26 are left out. At 1:1 after costs a trade needs about 57% winners to break even.</p></div>
<p class="note">Research on past data, not investment advice. A bar touching the stop and the target counts as the stop.</p></main></body></html>"""
    path = RES / "library3_report.html"
    path.write_text(body, encoding="utf-8")
    return path
