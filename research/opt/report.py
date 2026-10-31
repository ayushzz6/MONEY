"""HTML report for round 5 (NIFTY / BANKNIFTY options at 1:2): data/research/results/opt5_report.html"""
import json

import numpy as np

from core.config.settings import DATA_DIR
from research.lib.report import EXTRA_CSS
from research.report import CSS, E, cls, inr

RES = DATA_DIR / "research" / "results"


def _v(v):
    L, J = v["learn"], v["judge"]
    w = lambda x: "-" if x is None else f"{x:.0f}%"
    r = lambda x: "-" if x is None else f"{x:+.2f}"
    return (f"<tr><td><b>{E(v['strategy'])}</b><div class='small'>{E(v['filter'])} · {E(v['contract'])} · {E(v['exit'])}</div></td>"
            f"<td class='r'>{L['trades']}</td><td class='r'>{w(L['win'])}</td><td class='r'>{r(L['avg_r'])}</td>"
            f"<td class='r {cls(L['per_month'])}'>{inr(L['per_month'], True)}</td>"
            f"<td class='r'>{J['trades']}</td><td class='r'>{w(J['win'])}</td><td class='r'>{r(J['avg_r'])}</td>"
            f"<td class='r {cls(J['per_month'])}'><b>{inr(J['per_month'], True)}</b></td><td class='r'>{inr(J['max_dd'])}</td></tr>")


HEAD = ("<thead><tr><th>Strategy · filter · contract · exit</th><th class='r'>Trades 21-24</th><th class='r'>Win 21-24</th>"
        "<th class='r'>Avg R 21-24</th><th class='r'>₹/month 21-24</th><th class='r'>Trades 25-26</th><th class='r'>Win 25-26</th>"
        "<th class='r'>Avg R 25-26</th><th class='r'>₹/month 25-26</th><th class='r'>Max DD 25-26</th></tr></thead>")


def write(summary=None, edge=None):
    s = summary or json.loads((RES / "opt5_summary.json").read_text())
    edge = edge or (json.loads((RES / "opt5_edge.json").read_text()) if (RES / "opt5_edge.json").exists() else None)
    c, a = s["counts"], s["account"]
    ru = a.get("ruin") or {}
    risk = s["risk_per_trade"]
    val = s["validation"]["modes"][s["iv_mode"]]
    val_html = "".join(
        f"<tr><td>{E(g)}</td><td class='r'>{r['n']:,}</td>" + "".join(
            f"<td class='r'>{r[k]['median_abs_pct']}%<div class='small'>bias {r[k]['bias_pct']:+.1f}%</div></td>" for k in ("open", "high", "low", "close"))
        + "</tr>" for g, r in val.items())
    size_html = "".join(
        f"<tr><td>{E(r['index'])}</td><td class='r'>{r['delta']:.2f}</td><td class='r'>{r['2021-24']['lot']:.0f}</td><td class='r'>{r['2021-24']['fits_pct']}%</td>"
        f"<td class='r'>{r['2025-26']['lot']:.0f}</td><td class='r'><b>{r['2025-26']['fits_pct']}%</b></td></tr>" for r in s["sizing"])
    months = a["judge"]["monthly"]
    stress_html = "".join(f"<tr><td>{E(k)}</td><td class='r {cls(v['learn']['per_month'])}'>{inr(v['learn']['per_month'], True)}</td>"
                          f"<td class='r {cls(v['judge']['per_month'])}'><b>{inr(v['judge']['per_month'], True)}</b></td><td class='r'>{inr(v['judge']['max_dd'])}</td></tr>"
                          for k, v in s["stress"].items())
    picks_html = "".join(f"<li>{E(p['strategy'])} · {E(p['filter'])} · {E(p['contract'])} · {E(p['exit'])}</li>" for p in s["picks"])
    edge_html = ""
    if edge:
        edge_html = f"""<h2>Is there any edge at all? (no account-size limit, at least 1 lot every time)</h2><div class="card">
<ul class="findings"><li>Of {edge['ok']:,} variants with enough trades in both periods, the median trade lost <b>{edge['median_learn_r']:+.2f}R</b> in 2021-24 and
<b>{edge['median_judge_r']:+.2f}R</b> in 2025-26 (R = the rupees at risk on the trade).</li>
<li>{edge['both_positive']:,} variants ({edge['both_positive'] / edge['ok'] * 100:.1f}%) were positive in both periods. The 50 best on 2021-24 averaged
{edge['top50_judge_mean_r']:+.3f}R in 2025-26 ({edge['top50_judge_positive']} of 50 positive).</li>
<li>The one family that held up: <b>NIFTY expiry days, afternoon trend at 13:00, buy next week's option</b>: about +0.15R to +0.19R a trade in 2025-26,
but only 34 trades in 22 months (about ₹300 to ₹1,100 a month with one lot).</li></ul></div>"""
    snap_note = ""
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Options At One To Two</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Round 5: NIFTY and BANKNIFTY options with a 1:2 stop and target</h1>
<p class="lede">Real NSE option chains for every session, Oct 2021 to Oct 2026 (3.2 million option rows: prices, open interest, lot sizes) ·
{len(s['signals'])} entry signals × 2 indices × {len(s['filters'])} option-chain filters × {len(s['contracts'])} contracts × {len(s['exits'])} exits = {c['variants']:,} variants ·
chosen on Oct 2021 to Dec 2024, judged once on Jan 2025 to Oct 2026 · ₹{risk:,.0f} risk a trade · NSE charges and slippage · built {E(s['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>No variant made ₹20,000 a month</b> in 2021-24 or in 2025-26 ({c['learn_20k']} and {c['judge_20k']} of {c['variants']:,}).</li>
<li>The 25 best on 2021-24 made at most {inr(s['top'][0]['learn']['per_month'], True)} a month there; in 2025-26 only {c['judge_positive_of_top']} of them made money.</li>
<li>An account trading the 5 best (at most 2 open trades, stops for the day at a ₹6,000 loss): <b class="{cls(a['learn']['per_month'])}">{inr(a['learn']['per_month'], True)}</b> a month in 2021-24,
<b class="{cls(a['judge']['per_month'])}">{inr(a['judge']['per_month'], True)}</b> a month in 2025-26, worst drawdown {inr(a['judge']['max_dd'])}, {a['judge']['losing_months']} losing months of {a['judge']['months']}.</li>
<li>Replaying 2025-26's days as random years: a losing year {ru.get('p_year_negative', 0) * 100:.0f}% of the time, a ₹50,000 drawdown {ru.get('p_drawdown_50k', 0) * 100:.0f}% of the time,
the whole ₹1 lakh lost {ru.get('p_lose_1L', 0) * 100:.1f}% of the time. Small, fixed risk per trade is what keeps the capital alive; it does not make the strategy profitable.</li>
<li>₹20,000 a month at ₹{risk:,.0f} risk a trade means +10R every month. The best consistent family earned about +0.17R a trade on 1.5 trades a month: roughly +0.25R a month.</li></ul></div>

<h2>The data, and how close the simulated option prices are to the real ones</h2><div class="card scroll">
<p>Each option is priced from the previous close's real implied volatility at its strike (from NSE's closing chain), the forward from put-call parity,
and the index's 5-minute bars. Checked against the real open, high, low and close of {s['validation']['contract_days']:,} contract-days
(front expiry, 5 strikes around the index, calls and puts). Intraday IV rule used: <b>{E(s['iv_mode'])}</b> (chosen on 2021-24).</p>
<table><thead><tr><th>Group</th><th class='r'>Contract-days</th><th class='r'>Open error</th><th class='r'>High error</th><th class='r'>Low error</th><th class='r'>Close error</th></tr></thead>
<tbody>{val_html}</tbody></table><p class="note">Median absolute error and median bias (simulated vs real). Out-of-the-money options are the least accurate.</p></div>

<h2>What ₹1 lakh can trade since SEBI's bigger lots (Nov 2024)</h2><div class="card scroll"><table><thead><tr><th>Index</th><th class='r'>Delta</th>
<th class='r'>Lot 21-24</th><th class='r'>Signals that fit 21-24</th><th class='r'>Lot 25-26</th><th class='r'>Signals that fit 25-26</th></tr></thead><tbody>{size_html}</tbody></table>
<p class="note">Share of signals where one lot risks at most ₹3,000 with a 25% premium stop and the premium paid is at most ₹1 lakh. BANKNIFTY lost its weekly expiry in Nov 2024,
so its nearest option is now a monthly one with a far bigger premium.</p></div>

<h2>The 25 best on 2021-24, judged on 2025-26</h2><div class="card scroll"><table>{HEAD}<tbody>{''.join(_v(v) for v in s['top'])}</tbody></table>
<p class="note">Ranked by ₹/month on 2021-24 among variants with 60+ trades and a profit in each of 2022, 2023 and 2024. "0 trades" in 2025-26 means the trade no longer fits
₹1 lakh with ₹3,000 at risk.</p></div>

<h2>With hindsight: the best of 2025-26</h2><div class="card scroll"><table>{HEAD}<tbody>{''.join(_v(v) for v in s['hindsight'])}</tbody></table>
<p class="note">Picked by looking at 2025-26 itself: every one of them lost money in 2021-24. This is what chasing last year's winner looks like.</p></div>
{edge_html}
<h2>The account of the 5 best and its stress tests</h2><div class="card scroll"><ul class="findings">{picks_html}</ul>
<table><thead><tr><th>Version</th><th class='r'>₹/month 21-24</th><th class='r'>₹/month 25-26</th><th class='r'>Max DD 25-26</th></tr></thead><tbody>
<tr><td><b>As tested</b></td><td class='r {cls(a['learn']['per_month'])}'>{inr(a['learn']['per_month'], True)}</td><td class='r {cls(a['judge']['per_month'])}'><b>{inr(a['judge']['per_month'], True)}</b></td><td class='r'>{inr(a['judge']['max_dd'])}</td></tr>
{stress_html}</tbody></table>
<p class="note">Months in 2025-26: {', '.join(f'{k}: {inr(v, True)}' for k, v in months.items())}.</p></div>
{snap_note}
<p class="note">Method: signals use only closed 5-minute bars and the previous close's option chain; entry at the next bar's open; a stop and a target hit in the same bar count as the stop;
whatever is open closes at 15:15. Charges: ₹20 brokerage an order, STT 0.1% of premium on sells, NSE 0.03503%, stamp, SEBI, GST; slippage max(₹0.5, 0.5% of premium)
a side, double on stops. Research on past data, not investment advice. SEBI found 93% of individual F&O traders lost money in FY22-FY24.</p></main></body></html>"""
    path = RES / "opt5_report.html"
    path.write_text(body, encoding="utf-8")
    return path
