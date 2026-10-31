"""HTML report for the strong support / resistance breakout strategy: data/research/results/srbreak_report.html"""
import json

from core.config.settings import DATA_DIR
from research.lib.report import EXTRA_CSS, _curves
from research.report import CSS, E, cls, inr

RES = DATA_DIR / "research" / "results"


def _r(x, nd=3):
    return "-" if x is None else f"{x:+.{nd}f}"


def write(summary=None, swing=None):
    s = summary or json.loads((RES / "srbreak_summary.json").read_text())
    sw = swing or (json.loads((RES / "srbreak_swing.json").read_text()) if (RES / "srbreak_swing.json").exists() else None)
    d = s["described"]
    a = d["account"]
    st = d["stats"]
    curves = _curves([{"name": "As described", "curve": a["curve"]}] +
                     [{"name": f"Best on 2022-24 #{i + 1}", "curve": g["account"]["curve"]} for i, g in enumerate(s["top"][:2])],
                     s["dates_axis"], s["test_start"], label="Last 2 years: not used to choose")
    months = a["test"]["monthly"]
    one_rows = "".join(
        f"<tr><td>{E(o['factor'])}</td><td class='r'>{o['learn']['n']:,}</td><td class='r {cls(o['learn']['avg_r'] or 0)}'>{_r(o['learn']['avg_r'])}</td>"
        f"<td class='r'>{o['test']['n']:,}</td><td class='r'>{o['test']['win']:.0f}%</td><td class='r {cls(o['test']['avg_r'] or 0)}'><b>{_r(o['test']['avg_r'])}</b></td></tr>"
        for o in s["one_factor"])
    top_rows = "".join(
        f"<tr><td><b>{E(', '.join(g['factors']) or 'no extra factor')}</b><div class='small'>volume {g['vol']:g}x · level {g['strength']}+ touches · stop: {E(g['stop_label'])} · exit: {E(g['exit_label'])}</div></td>"
        f"<td class='r'>{g['learn']['n']:,}</td><td class='r {cls(g['account']['learn']['per_month'])}'>{inr(g['account']['learn']['per_month'], True)}</td>"
        f"<td class='r'>{g['test']['n']:,}</td><td class='r'>{(g['test']['win'] or 0):.0f}%</td>"
        f"<td class='r {cls(g['account']['test']['per_month'])}'><b>{inr(g['account']['test']['per_month'], True)}</b></td><td class='r'>{inr(g['account']['test']['max_dd'])}</td></tr>"
        for g in s["top"][:10])
    ex_rows = "".join(
        f"<tr><td>{E(e['date'])}</td><td><b>{E(e['stock'])}</b></td><td>{E(e['side'])}</td><td class='r'>{e['level']:,.2f}</td><td class='r'>{e['touches']}</td>"
        f"<td class='r'>{e['volume_x']}x</td><td class='r'>{e['entry']:,.2f}</td><td class='r'>{e['stop']:,.2f}</td><td class='r'>{e['exit']:,.2f}</td>"
        f"<td class='r {cls(e['r'])}'>{e['r']:+.2f}R</td></tr>" for e in d["examples"])
    side_rows = "".join(
        f"<tr><td>{E(k)}</td><td class='r'>{v['learn']['n']}</td><td class='r {cls(v['learn']['avg_r'] or 0)}'>{_r(v['learn']['avg_r'])}</td>"
        f"<td class='r'>{v['test']['n']}</td><td class='r {cls(v['test']['avg_r'] or 0)}'><b>{_r(v['test']['avg_r'])}</b></td></tr>" for k, v in d["sides"].items())
    swing_html = ""
    if sw:
        rows = "".join(
            f"<tr><td>{E(v['version'])}<div class='small'>{E(v['exit'])}</div></td><td class='r'>{v['test']['n']}</td>"
            f"<td class='r {cls(v['test']['gross_r'] or 0)}'>{_r(v['test']['gross_r'], 2)}</td><td class='r {cls(v['test']['avg_r'] or 0)}'>{_r(v['test']['avg_r'], 2)}</td>"
            f"<td class='r {cls(v['account']['test']['per_month'])}'><b>{inr(v['account']['test']['per_month'], True)}</b></td><td class='r'>{inr(v['account']['test']['max_dd'])}</td></tr>"
            for v in sw["versions"])
        swing_html = f"""<h2>Held for days instead (swing, buying only)</h2><div class="card scroll"><table><thead><tr><th>Version · exit</th>
<th class='r'>Breakouts, last 2 years</th><th class='r'>Avg R before costs</th><th class='r'>Avg R after costs</th><th class='r'>₹/month</th><th class='r'>Max DD</th></tr></thead>
<tbody>{rows}</tbody></table><p class="note">A daily close above strong resistance on heavy volume, bought at the next open with a stop 0.5 ATR under the level; ₹20,000 a position,
at most 5 at once, delivery charges (STT 0.1% each side, stamp, depository charge) and 5 bps slippage a side. Short swing trades need stock futures and are not tested.</p></div>"""
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Level Breakout Test</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Strong support / resistance breakouts on the NIFTY 500</h1>
<p class="lede">Every evening each liquid stock's strong levels are found from its swing highs and lows of the past year (3+ touches, 52-week extremes and round numbers
count extra); stocks within 1.5 ATR of one go on the next day's list. The trade is the first 5-minute close through the level, with volume and market checks ·
{s['breakouts']['learn']:,} breakouts in Oct 2022 to Oct 2024 (used to choose) and {s['breakouts']['test']:,} in the last 2 years (judged) · ₹1,000 risk a trade, NSE charges,
3 bps slippage a side · built {E(s['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>As you described it</b> (strong level, a breakout bar on 2x+ its usual volume with the day's volume running 1.2x+, NIFTY and sector moving the same way, the stock's
daily trend and 20-day strength agreeing, on VWAP's side, 2R+ room to the next level, not already stretched; stop 0.25 ATR back inside the level; target 2R):
<b>{st['test']['n']} trades in the last 2 years, {st['test']['win']:.0f}% winners, {st['test']['avg_r']:+.3f}R a trade after costs</b>.</li>
<li>On the ₹1 lakh account that is <b class="{cls(a['test']['per_month'])}">{inr(a['test']['per_month'], True)} a month</b>, worst drawdown {inr(a['test']['max_dd'])},
{a['test']['losing_months']} losing months of {a['test']['months']}. In the 2 years before: {inr(a['learn']['per_month'], True)} a month.</li>
<li>Before costs the same trades made about +0.09R each in both periods: the volume and market checks do pick better breakouts than a plain break
(about +0.01R in the last 2 years). Charges and slippage, about 0.13R a trade, are larger than that edge.</li>
<li>Of {s['variants_ok']:,} versions (every mix of the 8 checks, 3 volume levels, 2 level strengths, 2 stops, 4 exits), {s['variants_positive_test']} made money in the last 2 years.
The best ones on 2022-24 all lost in the last 2 years.</li></ul></div>

<h2>The ₹1 lakh account</h2><div class="card scroll">{curves}
<p class="note">Months in the last 2 years, as described: {', '.join(f'{k}: {inr(v, True)}' for k, v in months.items())}.
At most 5 trades and 5 positions a day, one per stock, ₹5 lakh intraday exposure, trading stops for the day at a ₹3,000 loss. The shaded part was not used to choose anything.</p></div>

<div class="grid-2" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div class="card"><h3>Buying breakouts vs selling breakdowns (as described)</h3><table><thead><tr><th>Side</th><th class='r'>Trades 22-24</th><th class='r'>Avg R 22-24</th>
<th class='r'>Trades last 2y</th><th class='r'>Avg R last 2y</th></tr></thead><tbody>{side_rows}</tbody></table></div>
<div class="card"><h3>How many candidates</h3><p>A median of {s['per_day']['median']:.0f} stocks a day break a strong level on 1.5x volume (up to {s['per_day']['max']}); the full set
of checks leaves about 11 trades a month.</p></div></div>

<h2>What each check adds (on top of a strong level + 1.5x volume, 2R target)</h2><div class="card scroll"><table><thead><tr><th>Check</th><th class='r'>Trades 22-24</th>
<th class='r'>Avg R 22-24</th><th class='r'>Trades last 2y</th><th class='r'>Winners last 2y</th><th class='r'>Avg R last 2y</th></tr></thead><tbody>{one_rows}</tbody></table>
<p class="note">Avg R after costs; a 2R target needs about 36% winners to break even before costs.</p></div>

<h2>The best versions on Oct 2022 to Oct 2024, judged on the last 2 years</h2><div class="card scroll"><table><thead><tr><th>Checks used</th><th class='r'>Trades 22-24</th>
<th class='r'>₹/month 22-24</th><th class='r'>Trades last 2y</th><th class='r'>Winners</th><th class='r'>₹/month last 2y</th><th class='r'>Max DD</th></tr></thead><tbody>{top_rows}</tbody></table></div>

<h2>The latest trades it would have taken (as described)</h2><div class="card scroll"><table><thead><tr><th>Date</th><th>Stock</th><th>Side</th><th class='r'>Level</th>
<th class='r'>Touches</th><th class='r'>Volume</th><th class='r'>Entry</th><th class='r'>Stop</th><th class='r'>Exit</th><th class='r'>Result</th></tr></thead><tbody>{ex_rows}</tbody></table></div>
{swing_html}
<p class="note">Levels use only daily bars up to the previous close (a swing point counts 5 days after it forms); breakouts use only closed 5-minute bars; a test that replaces
everything after a random bar with a random walk changed none of the decisions. A stop and a target hit in the same bar count as the stop. Research on past data, not investment advice.</p>
</main></body></html>"""
    path = RES / "srbreak_report.html"
    path.write_text(body, encoding="utf-8")
    return path
