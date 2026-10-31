"""HTML report for round 6 (price action + market-type router): data/research/results/library6_report.html"""
import json

from core.config.settings import DATA_DIR
from research.lib.report import EXTRA_CSS, _curves
from research.report import CSS, E, cls, inr

RES = DATA_DIR / "research" / "results"


def _r(x, nd=3):
    return "-" if x is None else f"{x:+.{nd}f}"


def write(summary=None):
    s = summary or json.loads((RES / "library6_summary.json").read_text())
    acc = s["accounts"]
    base = s["base"]
    names = list(acc)
    best_name = max(names, key=lambda n: acc[n]["judge"]["per_month"])
    rows = "".join(
        f"<tr><td><b>{E(n)}</b></td><td class='r'>{a['trades']:,}</td>"
        f"<td class='r {cls(a['learn']['per_month'])}'>{inr(a['learn']['per_month'], True)}</td>"
        f"<td class='r {cls(a['check']['per_month'])}'>{inr(a['check']['per_month'], True)}</td>"
        f"<td class='r {cls(a['judge']['per_month'])}'><b>{inr(a['judge']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(a['judge']['max_dd'])}</td><td class='r'>{a['judge']['losing_months']}/{a['judge']['months']}</td>"
        f"<td class='r'>{_r(a.get('judge_avg_r'))}</td></tr>" for n, a in acc.items())
    curves = _curves([{"name": n, "curve": a["curve"]} for n, a in acc.items()], s["dates_axis"], s["check_end"])
    held = sum(1 for t in s["regime_table"] if t["judge"]["avg_r"] is not None and t["judge"]["n"] >= 30 and t["judge"]["avg_r"] > 0)
    rated = sum(1 for t in s["regime_table"] if t["judge"]["avg_r"] is not None and t["judge"]["n"] >= 30)
    reg_rows = "".join(
        f"<tr><td><b>{E(t['regime'])}</b><div class='small'>{t['days']} sessions, {t['days_judge']} in 2025-26</div></td>"
        f"<td>{E(t['best'] or 'none with 80+ trades')}</td>"
        f"<td class='r {cls(t['learn']['avg_r'] or 0)}'>{_r(t['learn']['avg_r'])}<div class='small'>{t['learn']['n']:,}</div></td>"
        f"<td class='r {cls(t['check']['avg_r'] or 0)}'>{_r(t['check']['avg_r'])}<div class='small'>{t['check']['n']:,}</div></td>"
        f"<td class='r {cls(t['judge']['avg_r'] or 0)}'><b>{_r(t['judge']['avg_r'])}</b><div class='small'>{t['judge']['n']:,}</div></td>"
        f"<td class='small'>{E(t['hindsight_best'] or '-')} {_r(t['hindsight_avg_r'], 2)}</td></tr>" for t in s["regime_table"])
    trends = ["NIFTY uptrend", "NIFTY sideways", "NIFTY downtrend"]
    cols = [f"{t}, {sd}" for t in trends for sd in ("long", "short")]
    fam_rows = "".join(
        f"<tr><td><b>{E(f)}</b></td>" + "".join(
            f"<td class='r {cls(v[c]['judge_r'] or 0)}'>{_r(v[c]['judge_r'], 2)}<div class='small'>{_r(v[c]['learn_r'], 2)}</div></td>" for c in cols)
        + "</tr>" for f, v in s["family_trend"].items())
    top = s["per_setup"][:25]
    set_rows = "".join(
        f"<tr><td>{E(p['setup'])}</td><td class='r'>{p['learn']['n']:,}</td><td class='r {cls(p['learn']['avg_r'] or 0)}'>{_r(p['learn']['avg_r'])}</td>"
        f"<td class='r'>{p['judge']['n']:,}</td><td class='r'>{p['judge']['win']:.0f}%</td>"
        f"<td class='r {cls(p['judge']['avg_r'] or 0)}'><b>{_r(p['judge']['avg_r'])}</b></td></tr>" for p in top)
    ml = s["ml"]
    q_rows = "".join(f"<tr><td class='r'>top {q['top_pct']}%</td><td class='r {cls(q['check_per_month'])}'>{inr(q['check_per_month'], True)}</td></tr>"
                     for q in ml["q_table"])
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Price Action Router</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Round 6: price action on the NIFTY 500, routed by market type</h1>
<p class="lede">{s['stocks']} stocks · 5-minute bars {E(s['dates'][0])} to {E(s['dates'][1])} · {s['rules']} price-action rules × long / short × 3 exits = {s['setups']} setups ·
{s['trades']:,} trades · 18 market types from the NIFTY trend, India VIX and the open · routers learn on 2021-23, are checked on 2024 and judged once on 2025-26 ·
₹1,000 risk a trade, NSE charges and 3 bps slippage a side · built {E(s['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>Setups (300+ trades) that made money on their own: {s['setups_positive']['learn']} of {s['setups']} in 2021-23, {s['setups_positive']['check']} in 2024 and
<b>{s['setups_positive']['judge']}</b> in 2025-26.</li>
<li>All {s['setups']} setups together averaged <b>{base['judge']['avg_r']:+.3f}R</b> a trade in 2025-26 ({base['judge']['win']:.0f}% winners) after costs, against {base['learn']['avg_r']:+.3f}R in 2021-23.</li>
<li>Best approach in 2025-26: <b>{E(best_name)}</b>, {inr(acc[best_name]['judge']['per_month'], True)} a month (worst drawdown {inr(acc[best_name]['judge']['max_dd'])}).
No approach came near ₹20,000 a month.</li>
<li>The best setup for each market type, picked on 2021-23, still made money in 2025-26 in {held} of the {rated} types with 30+ trades.</li>
<li>The machine-learning router's predictions correlated {_r(ml['ic']['learn'], 3)} with real results on the years it learned from, {_r(ml['ic']['check'], 3)} on 2024 and {_r(ml['ic']['judge'], 3)} on 2025-26.</li></ul></div>

<h2>The ₹1 lakh account, approach by approach</h2><div class="card scroll"><table><thead><tr><th>Approach</th><th class='r'>Trades</th>
<th class='r'>₹/month 2021-23</th><th class='r'>₹/month 2024</th><th class='r'>₹/month 2025-26</th><th class='r'>Max DD 2025-26</th><th class='r'>Losing months</th><th class='r'>Avg R 2025-26</th></tr></thead>
<tbody>{rows}</tbody></table>{curves}
<p class="note">At most 5 trades and 5 open positions a day, one per stock, ₹5 lakh intraday exposure, trading stops for the day at a ₹3,000 loss.
The routers saw 2021-23 (and the regime tables and the ML cut-off saw 2024); only 2025-26 is a fair test for them.</p></div>

<h2>Which setup works best in which market</h2><div class="card scroll"><table><thead><tr><th>Market type</th><th>Best setup on 2021-23 (80+ trades)</th>
<th class='r'>Avg R 2021-23</th><th class='r'>Avg R 2024</th><th class='r'>Avg R 2025-26</th><th>With hindsight, best in 2025-26</th></tr></thead><tbody>{reg_rows}</tbody></table>
<p class="note">Avg R is the average result per trade in units of the risk taken, after costs; small numbers under each are trade counts. The last column is picked by
looking at 2025-26 itself, to show how much the winners change from one period to the next.</p></div>

<h2>Families by NIFTY trend and side (2025-26, with 2021-23 underneath)</h2><div class="card scroll"><table><thead><tr><th>Family</th>
{''.join(f'<th class="r">{E(c)}</th>' for c in cols)}</tr></thead><tbody>{fam_rows}</tbody></table></div>

<div class="grid-2" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div class="card"><h3>Machine-learning cut-off, set on 2024</h3><table><thead><tr><th class='r'>Trades taken</th><th class='r'>₹/month in 2024</th></tr></thead>
<tbody>{q_rows}</tbody></table><p class="note">The share with the best 2024 result (top {ml['top_pct']}%) was used for 2025-26.</p></div>
<div class="card"><h3>How the routers chose</h3><ul class="findings">
<li>Regime table, checked: {s['cells_selected']['checked']} market-type × setup pairs earned 0.05R+ on 2021-23 (t-stat 1.5+, 80+ trades) and stayed positive in 2024.</li>
<li>Regime table, pooled: {s['cells_selected']['pooled']} pairs earned 0.05R+ on 2021-24 with a t-stat of 2+.</li></ul></div></div>

<h2>The 25 best setups in 2025-26 (all market types, with hindsight)</h2><div class="card scroll"><table><thead><tr><th>Setup</th><th class='r'>Trades 2021-23</th>
<th class='r'>Avg R 2021-23</th><th class='r'>Trades 2025-26</th><th class='r'>Winners 2025-26</th><th class='r'>Avg R 2025-26</th></tr></thead><tbody>{set_rows}</tbody></table></div>

<p class="note">Method: 25 new rules (Brooks, Volman, Beggs, Bulkowski, Raschke, Crabel, Nison and the "smart money" vocabulary) plus 19 from round 1. Each rule decides at a bar's close
and enters at the next bar's open, with its structural stop; a test that replaces all later bars with a random walk changed none of the 44 rules' signals or stops.
A stop and a target hit in the same bar count as the stop. Market type uses only what is known by 09:15. Research on past data, not investment advice.</p></main></body></html>"""
    path = RES / "library6_report.html"
    path.write_text(body, encoding="utf-8")
    return path
