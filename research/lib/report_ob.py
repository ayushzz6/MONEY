"""HTML report for the order-block zone backtest: data/research/results/obtest_report.html"""
import json

from core.config.settings import DATA_DIR
from research.lib.report import EXTRA_CSS, _curves
from research.report import CSS, E, cls, inr

RES = DATA_DIR / "research" / "results"


def _r(x, nd=3):
    return "-" if x is None else f"{x:+.{nd}f}"


def write(summary=None):
    s = summary or json.loads((RES / "obtest_summary.json").read_text())
    names = list(s["main"])
    d0 = s["main"][names[0]]
    rows = "".join(
        f"<tr><td><b>{E(n)}</b></td><td class='r'>{v['learn']['n']:,}</td><td class='r {cls(v['account']['learn']['per_month'])}'>{inr(v['account']['learn']['per_month'], True)}</td>"
        f"<td class='r'>{v['test']['n']:,}</td><td class='r'>{v['test']['win']:.0f}%</td><td class='r {cls(v['test']['gross_r'] or 0)}'>{_r(v['test']['gross_r'])}</td>"
        f"<td class='r {cls(v['test']['avg_r'] or 0)}'>{_r(v['test']['avg_r'])}</td><td class='r {cls(v['account']['test']['per_month'])}'><b>{inr(v['account']['test']['per_month'], True)}</b></td>"
        f"<td class='r'>{inr(v['account']['test']['max_dd'])}</td><td class='r'>{v['account']['test']['losing_months']}/{v['account']['test']['months']}</td></tr>"
        for n, v in s["main"].items())
    curves = _curves([{"name": n.split(":")[0][:40], "curve": v["account"]["curve"]} for n, v in s["main"].items()],
                     s["dates_axis"], s["test_start"], label="Last 2 years")
    top = "".join(
        f"<tr><td>{E(g['filter'])} · {g['aligned']}+ aligned · exit {E(g['exit'])}{' · strong zones only' if g['strong_only'] else ''} · min risk {g['min_risk_pct']}%</td>"
        f"<td class='r'>{g['learn']['n']:,}</td><td class='r {cls(g['account']['learn']['per_month'])}'>{inr(g['account']['learn']['per_month'], True)}</td>"
        f"<td class='r'>{g['test']['n']:,}</td><td class='r {cls(g['test']['gross_r'] or 0)}'>{_r(g['test']['gross_r'])}</td>"
        f"<td class='r {cls(g['account']['test']['per_month'])}'><b>{inr(g['account']['test']['per_month'], True)}</b></td></tr>" for g in s["top"])
    sides = "".join(f"<tr><td>{E(k)}</td><td class='r'>{v['test']['n']}</td><td class='r'>{v['test']['win']:.0f}%</td>"
                    f"<td class='r'>{_r(v['test']['gross_r'])}</td><td class='r {cls(v['test']['avg_r'] or 0)}'>{_r(v['test']['avg_r'])}</td></tr>" for k, v in s["sides"].items())
    ex = "".join(f"<tr><td>{E(e['date'])} {E(e['time'])}</td><td><b>{E(e['stock'])}</b></td><td>{E(e['side'])}</td><td class='r'>{e['entry']:,.2f}</td>"
                 f"<td class='r'>{e['stop']:,.2f}</td><td class='r'>{e['risk_pct']}%</td><td class='r'>{e['exit']:,.2f}</td><td class='r {cls(e['r'])}'>{e['r']:+.2f}R</td></tr>"
                 for e in s["examples"])
    t = d0["test"]
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Order Block Backtest</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Order-block zones, backtested on NSE F&amp;O stocks</h1>
<p class="lede">The board on the dashboard, replayed bar by bar: each day's F&amp;O list, every stock's strength from the bars closed so far, picks, alignment and
zones as the live code finds them. {s['retests']['learn']:,} first retests in Oct 2022 to Oct 2024 and {s['retests']['test']:,} in the last 2 years ·
limit order at the zone level, stop at invalidation, target the 2nd liquidity level · ₹1,000 risk a trade, NSE charges, 3 bps slippage · built {E(s['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>As the site describes it</b> (a pick, 4 of 4 aligned, first retest of the zone): {t['n']:,} trades in the last 2 years, {t['win']:.0f}% winners,
<b>{t['avg_r']:+.3f}R a trade after costs</b>, which is <b class="neg">{inr(d0['account']['test']['per_month'], True)} a month</b> on the ₹1 lakh account,
losing in {d0['account']['test']['losing_months']} of {d0['account']['test']['months']} months. The 2 years before: {inr(d0['account']['learn']['per_month'], True)} a month.</li>
<li>Before costs the same trades averaged {t['gross_r']:+.3f}R: the zones hold slightly more often than chance, but an order block is one 5-minute candle, so the stop sits
only about {s['risk_mix']['median_risk_pct']:.2f}% from the entry (median) and charges and slippage take about a third of the risk on every trade.</li>
<li>None of the {s['variants_ok']} versions (board filter, alignment, 4 exits, strong zones only, minimum stop distance) made money in the last 2 years.</li>
<li>Its worst drawdown, {inr(d0['account']['test']['max_dd'])}, is far more than the capital: trading it would have emptied a ₹1 lakh account.</li></ul></div>
<h2>The versions</h2><div class="card scroll"><table><thead><tr><th>Version</th><th class='r'>Trades 22-24</th><th class='r'>₹/month 22-24</th><th class='r'>Trades last 2y</th>
<th class='r'>Winners</th><th class='r'>Avg R before costs</th><th class='r'>Avg R after costs</th><th class='r'>₹/month last 2y</th><th class='r'>Max DD</th><th class='r'>Losing months</th></tr></thead>
<tbody>{rows}</tbody></table>{curves}</div>
<div class="grid-2" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div class="card"><h3>Bullish vs bearish zones (as described, last 2 years)</h3><table><thead><tr><th>Side</th><th class='r'>Trades</th><th class='r'>Winners</th><th class='r'>Before costs</th><th class='r'>After costs</th></tr></thead><tbody>{sides}</tbody></table></div>
<div class="card"><h3>How the trade works</h3><p>When a stock is on the board on the zone's side, a limit order rests at the zone level Z for the zone's first retest. Stop at the
invalidation level; target the second liquidity level (the board's R:R target), else the first, else 2R; out at 15:15. A stop touched in the entry bar counts as a loss
and a target never counts in the entry bar. One trade per stock a day; at most 5 trades and 5 positions a day; stop for the day at a ₹3,000 loss.</p></div></div>
<h2>The best versions on Oct 2022 to Oct 2024, judged on the last 2 years</h2><div class="card scroll"><table><thead><tr><th>Version</th><th class='r'>Trades 22-24</th>
<th class='r'>₹/month 22-24</th><th class='r'>Trades last 2y</th><th class='r'>Before costs</th><th class='r'>₹/month last 2y</th></tr></thead><tbody>{top}</tbody></table></div>
<h2>The latest trades (as described)</h2><div class="card scroll"><table><thead><tr><th>When</th><th>Stock</th><th>Side</th><th class='r'>Entry</th><th class='r'>Stop</th>
<th class='r'>Stop distance</th><th class='r'>Exit</th><th class='r'>Result</th></tr></thead><tbody>{ex}</tbody></table></div>
<p class="note">Uses only bars closed before each decision; zone targets are recomputed from the bars known when the order rests. Research on past data, not investment advice.</p>
</main></body></html>"""
    path = RES / "obtest_report.html"
    path.write_text(body, encoding="utf-8")
    return path
