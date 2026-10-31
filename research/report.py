"""Self-contained HTML report of the 5-year, 15-strategy backtest (inline SVG, no scripts needed)."""
import html
import json
from pathlib import Path

E = html.escape


def inr(v, signed=False):
    if v is None:
        return "–"
    s = "−" if v < 0 else ("+" if signed and v > 0 else "")
    v = abs(v)
    if v >= 1e7:
        txt = f"{v / 1e7:.2f} Cr"
    elif v >= 1e5:
        txt = f"{v / 1e5:.2f} L"
    else:
        txt = f"{v:,.0f}"
    return f"{s}₹{txt}"


def compact(v):
    s = "−" if v < 0 else "+"
    v = abs(v)
    if v >= 1e5:
        return f"{s}{v / 1e5:.1f}L"
    if v >= 1000:
        return f"{s}{v / 1000:.1f}k"
    return f"{s}{v:.0f}"


def cls(v):
    return "pos" if (v or 0) > 0 else ("neg" if (v or 0) < 0 else "")


def sparkline(points, w=160, h=36):
    if len(points) < 2:
        return ""
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1
    pts = " ".join(f"{i * (w - 2) / (len(ys) - 1) + 1:.1f},{h - 2 - (y - lo) / span * (h - 4):.1f}" for i, y in enumerate(ys))
    base = h - 2 - (100000 - lo) / span * (h - 4)
    base = min(h - 1, max(1, base))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
            f'<line x1="0" x2="{w}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--axis)" stroke-dasharray="2 3"/>'
            f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" stroke-width="1.5"/></svg>')


def equity_chart(points, w=860, h=240):
    if len(points) < 2:
        return "<p class='muted'>No data.</p>"
    pad_l, pad_r, pad_t, pad_b = 70, 12, 10, 24
    ys = [p[1] for p in points]
    lo, hi = min(ys + [100000]), max(ys + [100000])
    span = (hi - lo) or 1
    X = lambda i: pad_l + i * (w - pad_l - pad_r) / (len(points) - 1)
    Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / span)
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(p[1]):.1f}" for i, p in enumerate(points))
    ticks = [lo, (lo + hi) / 2, hi]
    years, marks = set(), []
    for i, (d, _) in enumerate(points):
        y = d[:4]
        if y not in years:
            years.add(y)
            if not marks or i - marks[-1][0] >= len(points) / 12:
                marks.append((i, y))
    hits = "".join(
        f'<rect x="{X(i) - 3:.1f}" y="{pad_t}" width="6" height="{h - pad_t - pad_b}" fill="transparent">'
        f'<title>{E(d)}: equity {inr(v)} ({inr(v - 100000, True)})</title></rect>'
        for i, (d, v) in enumerate(points) if i % 2 == 0)
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Equity curve">'
            + "".join(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)"/>'
                      f'<text x="{pad_l - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{inr(t)}</text>'
                      for t in ticks)
            + f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(100000):.1f}" y2="{Y(100000):.1f}" stroke="var(--axis)" stroke-dasharray="4 4"/>'
            + "".join(f'<text x="{X(i):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{y}</text>' for i, y in marks)
            + f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>'
            + hits + "</svg>")


def heatmap(monthly, target):
    if not monthly:
        return ""
    years = sorted({k[:4] for k in monthly})
    mx = max(abs(v) for v in monthly.values()) or 1
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for y in years:
        cells = []
        for m in range(1, 13):
            k = f"{y}-{m:02d}"
            if k not in monthly:
                cells.append("<td class='hm none'></td>")
                continue
            v = monthly[k]
            pct = int(12 + 58 * min(1.0, abs(v) / mx))
            col = "var(--up)" if v >= 0 else "var(--down)"
            hit = " hit" if v >= target else ""
            cells.append(f"<td class='hm{hit}' style='background:color-mix(in srgb,{col} {pct}%,var(--surface))' "
                         f"title='{k}: {inr(v, True)}'>{compact(v)}</td>")
        total = sum(monthly.get(f"{y}-{m:02d}", 0) for m in range(1, 13))
        rows.append(f"<tr><th>{y}</th>{''.join(cells)}<td class='r yr {cls(total)}'><b>{compact(total)}</b></td></tr>")
    return (f"<table class='heat'><thead><tr><th></th>{''.join(f'<th>{n}</th>' for n in names)}<th class='r'>Year</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"<p class='note'>Blue months made money, red months lost it (deeper = bigger). Bold = at least {inr(target)}. "
            f"k = thousand, L = lakh rupees.</p>")


def kv(pairs):
    return "<dl class='kv'>" + "".join(f"<dt>{E(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def findings(doc):
    S = doc["strategies"]
    t = doc["target"]
    daily = [s for s in S if s["avg_per_session"] >= t]
    monthly = [s for s in S if s["avg_per_month"] >= t]
    positive = [s for s in S if s["net_pnl"] > 0]
    robust = [s for s in positive if (s.get("out_of_sample") or {}).get("net", 0) > 0
              and (s.get("in_sample") or {}).get("net", 0) > 0 and s["max_drawdown_pct"] <= 30
              and not s.get("ruined_on")
              and all(v > 0 for v in (s.get("iv_sensitivity") or {"x": 1}).values())
              and all(v > 0 for v in (s.get("slippage_sensitivity") or {"x": 1}).values())]
    best = max(S, key=lambda s: s["net_pnl"])
    out = [
        f"<li><b>{len(daily)} of {len(S)}</b> strategies averaged {inr(t)} a day on {inr(doc['capital'])}. "
        + ("" if daily else "None came close: that is about 500% a year.") + "</li>",
        f"<li><b>{len(monthly)} of {len(S)}</b> averaged {inr(t)} a month (about 24% a year)"
        + (": " + ", ".join(f"{E(s['code'])} {E(s['name'])}" for s in monthly) if monthly else "") + ".</li>",
        f"<li><b>{len(positive)} of {len(S)}</b> made money over the 5 years after charges and slippage; "
        f"{len(S) - len(positive)} lost money.</li>",
        f"<li>Best total: {E(best['code'])} {E(best['name'])}, {inr(best['net_pnl'], True)} "
        f"(worst drawdown {inr(best['max_drawdown'])}, {best['max_drawdown_pct']}% of capital).</li>",
        "<li>Robust (positive in both 2021-24 and 2025-26, drawdown under 30% of capital, never wiped out, and still positive "
        "at 5 bps slippage or at every IV assumption): " +
        (", ".join(f"{E(s['code'])} {E(s['name'])}" + (" (expiry-day option prices unverified)" if s.get("unverified") else "")
                   for s in robust) if robust else "<b>none</b>") + ".</li>",
    ]
    return "<ul class='findings'>" + "".join(out) + "</ul>"


def strategy_section(s, target):
    ins, oos = s.get("in_sample") or {}, s.get("out_of_sample") or {}
    yearly = "".join(f"<tr><td>{y}</td><td class='r {cls(v['net'])}'>{inr(v['net'], True)}</td><td class='r'>{v['trades']}</td>"
                     f"<td class='r'>{v['win_rate_pct'] if v['win_rate_pct'] is not None else '–'}%</td></tr>"
                     for y, v in s["yearly"].items())
    sens = s.get("iv_sensitivity")
    sens_html = ""
    if sens:
        sens_html = ("<h4>Option-price sensitivity (5-year net)</h4>" + kv(
            [("IV 0.8 × VIX", f"<span class='{cls(sens.get('iv_0.8x_vix'))}'>{inr(sens.get('iv_0.8x_vix'), True)}</span>"),
             ("IV 0.9 × VIX (base)", f"<span class='{cls(s['net_pnl'])}'>{inr(s['net_pnl'], True)}</span>"),
             ("IV 1.0 × VIX", f"<span class='{cls(sens.get('iv_1x_vix'))}'>{inr(sens.get('iv_1x_vix'), True)}</span>")]))
    slip = s.get("slippage_sensitivity")
    if slip:
        sens_html += ("<h4>Cost sensitivity (5-year net)</h4>" + kv(
            [("No slippage (charges only)", f"<span class='{cls(slip.get('slip_0bps'))}'>{inr(slip.get('slip_0bps'), True)}</span>"),
             ("3 bps a side (base)", f"<span class='{cls(s['net_pnl'])}'>{inr(s['net_pnl'], True)}</span>"),
             ("5 bps a side", f"<span class='{cls(slip.get('slip_5bps'))}'>{inr(slip.get('slip_5bps'), True)}</span>"),
             ("Average R before any costs", f"{s.get('avg_R_before_costs')}")]))
    ruin = ""
    if s.get("ruined_on"):
        ruin = (f"<p class='warn'>The ₹1 L account would have been wiped out on {E(s['ruined_on'])}; the numbers after that "
                f"date assume you kept trading the same size, which you could not have.</p>")
    fut = ""
    if s.get("futures_equivalent_net") is not None:
        fut = kv([("Same signals with 1 futures lot (needs ~₹1.7-2 L margin)",
                   f"<span class='{cls(s['futures_equivalent_net'])}'>{inr(s['futures_equivalent_net'], True)}</span>")])
    verdict = "meets" if s["avg_per_month"] >= target else "misses"
    return f"""
<section class="card strat" id="{E(s['code'])}">
  <div class="shead"><h3><span class="code">{E(s['code'])}</span> {E(s['name'])}</h3>
    <span class="chip">{E(s.get('segment', ''))}</span>
    <span class="chip {'good' if s['net_pnl'] > 0 else 'bad'}">{'Profitable' if s['net_pnl'] > 0 else 'Lost money'} over 5 years</span>
    <span class="chip">{verdict} {inr(target)}/month</span></div>
  <p class="rules">{E(s.get('doc', ''))}</p>
  <p class="note">Sizing: {E(s.get('sizing', ''))}.</p>{ruin}
  <div class="grid3">
    {kv([("Net P&L, 5 years", f"<b class='{cls(s['net_pnl'])}'>{inr(s['net_pnl'], True)}</b>"),
         ("Return a year on ₹1 L", f"{s['annual_return_pct']}%"),
         ("Average a month", f"<span class='{cls(s['avg_per_month'])}'>{inr(s['avg_per_month'], True)}</span>"),
         ("Average a session", f"<span class='{cls(s['avg_per_session'])}'>{inr(s['avg_per_session'], True)}</span>"),
         ("Months at ≥ ₹2,000", f"{s['months_at_target_pct']}%"),
         ("Sessions at ≥ ₹2,000", f"{s['days_at_target_pct']}%")])}
    {kv([("Trades", f"{s['trades']:,}"), ("Win rate", f"{s['win_rate_pct']}%" if s['win_rate_pct'] is not None else "–"),
         ("Average win / loss", f"{inr(s['avg_win'])} / {inr(s['avg_loss'])}"),
         ("Expectancy a trade", f"<span class='{cls(s['expectancy_per_trade'])}'>{inr(s['expectancy_per_trade'], True)}</span>"),
         ("Profit factor", s['profit_factor'] if s['profit_factor'] is not None else "–"),
         ("Charges paid", inr(s['charges']))])}
    {kv([("Max drawdown", f"{inr(s['max_drawdown'])} ({s['max_drawdown_pct']}%)"),
         ("Longest drawdown", f"{s['longest_drawdown_sessions']} sessions"),
         ("Worst / best session", f"{inr(s['worst_day'], True)} / {inr(s['best_day'], True)}"),
         ("Worst / best month", f"{inr(s['worst_month'], True)} / {inr(s['best_month'], True)}"),
         ("Positive months", f"{s['positive_months_pct']}%"),
         ("Sharpe (daily, annualized)", s['sharpe'] if s['sharpe'] is not None else "–")])}
  </div>
  <h4>Equity of a ₹1 L account (no compounding)</h4>
  {equity_chart(s['equity'])}
  <div class="grid2">
    <div><h4>By year</h4><table><thead><tr><th>Year</th><th class='r'>Net</th><th class='r'>Trades</th><th class='r'>Win</th></tr></thead>
      <tbody>{yearly}</tbody></table>
      <h4>Stability</h4>{kv([("2021-10 to 2024-12", f"<span class='{cls(ins.get('net'))}'>{inr(ins.get('net'), True)}</span> ({ins.get('annual_return_pct')}%/yr)"),
                              ("2025-01 to 2026-10", f"<span class='{cls(oos.get('net'))}'>{inr(oos.get('net'), True)}</span> ({oos.get('annual_return_pct')}%/yr)")])}
      {sens_html}{fut}</div>
    <div><h4>Month by month</h4>{heatmap(s['monthly'], target)}</div>
  </div>
</section>"""


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f3f2ee;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6d68;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--up:#2a78d6;--down:#e34948;--good:#006300;--bad:#c33232}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#232321;
--ink:#f0efec;--ink-2:#c3c2b7;--muted:#9a9890;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;
--up:#3987e5;--down:#e66767;--good:#0ca30c;--bad:#ef7474}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px}
h3{font-size:16px;margin:0}h4{font-size:13px;color:var(--ink-2);margin:16px 0 6px;font-weight:600}
.lede{color:var(--ink-2);margin:0 0 16px}.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin:12px 0}
.findings li{margin:4px 0}.muted,.note{color:var(--muted);font-size:12px}.pos{color:var(--good)}.neg{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:left}
th{color:var(--ink-2);font-weight:600;font-size:12px}.r{text-align:right;font-variant-numeric:tabular-nums}td{font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0;font-size:13px}.kv dt{color:var(--ink-2)}
.kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px 24px;margin-top:8px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.6fr);gap:24px}@media (max-width:860px){.grid2{grid-template-columns:minmax(0,1fr)}}
.chip{display:inline-block;font-size:12px;padding:1px 9px;border-radius:999px;border:1px solid var(--border);background:var(--surface-2);color:var(--ink-2);margin:2px 4px 2px 0}
.chip.good{color:var(--good)}.chip.bad{color:var(--bad)}.shead{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}
.code{display:inline-block;min-width:30px;font-weight:700;color:var(--series-1)}.rules{margin:8px 0 2px}
.chart{width:100%;height:auto;display:block}.tick{font-size:11px;fill:var(--muted)}.spark{display:block}
.heat td.hm{text-align:center;font-size:11px;padding:5px 2px;border:2px solid var(--surface);color:var(--ink);white-space:nowrap}
.heat td.none{background:transparent}.heat td.hit{font-weight:700}.heat td.yr{white-space:nowrap}.heat th{text-align:center}.heat{table-layout:fixed}
.name{display:block;font-size:11.5px;color:var(--muted);font-weight:400}
.summary td,.summary th{white-space:nowrap}.summary a{color:var(--ink);text-decoration:none;font-weight:600}
details summary{cursor:pointer;font-weight:600}
.warn{border-left:3px solid var(--bad);padding:6px 10px;background:color-mix(in srgb,var(--down) 8%,var(--surface));font-size:13px}
"""


def write_report(doc, path: Path) -> Path:
    S = doc["strategies"]
    t = doc["target"]
    rows = "".join(f"""<tr><td><a href="#{E(s['code'])}">{E(s['code'])}</a></td><td>{E(s['name'])}</td>
      <td class="r">{s['trades']:,}</td><td class="r">{s['win_rate_pct'] if s['win_rate_pct'] is not None else '–'}%</td>
      <td class="r {cls(s['net_pnl'])}">{inr(s['net_pnl'], True)}</td><td class="r">{s['annual_return_pct']}%</td>
      <td class="r {cls(s['avg_per_month'])}">{inr(s['avg_per_month'], True)}</td>
      <td class="r {cls(s['avg_per_session'])}">{inr(s['avg_per_session'], True)}</td>
      <td class="r">{inr(s['max_drawdown'])}</td><td class="r">{s['profit_factor'] if s['profit_factor'] is not None else '–'}</td>
      <td class="r">{s['months_at_target_pct']}%</td><td>{sparkline(s['equity'][::4])}</td></tr>""" for s in S)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE Strategy Backtest</title><style>{CSS}</style></head><body><main>
<h1>15 NSE intraday strategies, 5 years of 5-minute data</h1>
<p class="lede">{E(doc['from'])} to {E(doc['to'])} · {doc['sessions']:,} sessions · {inr(doc['capital'])} account per strategy ·
target {inr(t)} profit · every number after brokerage, STT, exchange charges, GST, stamp duty and slippage · run {E(doc['run_at'])}</p>
<div class="card"><h2 style="margin-top:0">What the test says</h2>{findings(doc)}</div>
<h2>All 15 at a glance</h2>
<div class="card scroll"><table class="summary"><thead><tr><th></th><th>Strategy</th><th class="r">Trades</th><th class="r">Win</th>
<th class="r">Net, 5 yrs</th><th class="r">A year</th><th class="r">A month</th><th class="r">A session</th><th class="r">Max DD</th>
<th class="r">PF</th><th class="r">Months ≥ ₹2k</th><th>Equity</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">PF = profit factor (gross wins / gross losses). Max DD = largest fall from a high of the ₹1 L account. Equity line: dashed = starting ₹1 L.</p></div>
<h2>How it was tested</h2>
<div class="card"><details open><summary>Data, costs and assumptions</summary><ul>
<li><b>Data:</b> Angel One SmartAPI 5-minute bars (split and bonus adjusted) for the NIFTY, BANKNIFTY, India VIX and the current NIFTY 100
 stocks. Using today's NIFTY 100 adds survivorship bias in favour of the equity strategies.</li>
<li><b>Rules fixed in advance</b> (textbook parameters, no optimization on this data), all intraday and flat by 15:15 except O3 (held to expiry).
 Signals use only bars that had closed; stops are checked before targets inside a bar; gaps fill at the bar's open.</li>
<li><b>Equity (E1-E9):</b> NSE intraday (MIS), risk ₹1,000 a trade (1%), position size = risk / stop distance, at most ₹2.5 L a position,
 5 open positions, 8 trades and a ₹3,000 loss stop a day, MIS up to 5x. Slippage 3 bps a side (results at 0 and 5 bps shown). Charges: brokerage min(₹20, 0.03%),
 STT 0.025% on sells, NSE 0.00297%, SEBI, stamp 0.003% on buys, GST 18%.</li>
<li><b>Index F&O (I1-I3, O1-O3):</b> 1 lot (NIFTY 65, BANKNIFTY 30: today's lot sizes for the whole period). Futures need about ₹1.7-2 L margin
 a NIFTY lot, more than ₹1 L, so index views use options; the futures result of the same signals is shown for reference.
 Option charges: ₹20 an order, STT 0.1% of premium on sells, NSE 0.03503%, GST; slippage max(₹0.50, 0.5%) a side, 3x on stop-loss fills.</li>
<li><b>Option prices are modeled</b> (expired contracts' prices are not available): Black-Scholes on the index bars, IV = 0.9 × India VIX at that
 bar with a skew, trading-time to expiry (NIFTY weekly: Thursday, Tuesday from Sep 2025; BANKNIFTY monthly), BANKNIFTY IV = 1.25 × NIFTY's.
 The 0.9 comes from 64 real 5-minute NIFTY option prices (Oct 2026, 2 to 10 sessions to expiry): real premiums were 10% under the 1.0 × VIX model.
 Expiry-day (0DTE) prices could not be checked against real ones. Short-premium results swing hard with this one assumption, so each F&O
 strategy also shows IV 0.8x and 1.0x.</li>
<li>Not modeled: rejected or partial orders, freak trades, SEBI/broker margin changes over time, exchange-imposed weekly-expiry changes beyond the
 dates above, taxes on profits.</li></ul></details></div>
<h2>Strategy by strategy</h2>
{''.join(strategy_section(s, t) for s in S)}
<p class="note">Backtests describe the past under stated assumptions; they are not a forecast. This is research, not investment advice.</p>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")
    (path.parent / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    return path
