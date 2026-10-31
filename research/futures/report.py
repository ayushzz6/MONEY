"""HTML report: MCX commodities and NSE currency futures, intraday factory + positional trend following."""
import json
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from core.config.settings import DATA_DIR
from research.futures import factory as fx
from research.futures.data import SPECS, load
from research.futures.trend import backtest
from research.report import CSS, E, cls, compact, heatmap, inr, kv

OUT = DATA_DIR / "research" / "results"


def _equity(daily, dates, split, w=860, h=200):
    eq = 100000 + np.cumsum(daily)
    pts = list(zip([str(d) for d in dates[::3]], eq[::3]))
    lo, hi = min(min(eq), 100000), max(max(eq), 100000)
    span = (hi - lo) or 1
    pl, pr, pt, pb = 70, 10, 10, 20
    X = lambda i: pl + i * (w - pl - pr) / max(1, len(pts) - 1)
    Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / span)
    k = next((i for i, (d, _) in enumerate(pts) if d > split), len(pts) - 1)
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(pts))
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Equity">'
            f'<rect x="{X(k):.1f}" y="{pt}" width="{w - pr - X(k):.1f}" height="{h - pt - pb}" fill="var(--surface-2)"/>'
            f'<text x="{X(k) + 6:.1f}" y="{pt + 14}" class="tick">test period</text>'
            + "".join(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)"/>'
                      f'<text x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{inr(t)}</text>'
                      for t in sorted({lo, 100000, hi}) if t == 100000 or abs(Y(t) - Y(100000)) > 14)
            + f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2"/></svg>')


def _detail(title, desc, daily, dates, split, row):
    months = defaultdict(float)
    for d, v in zip(dates, daily):
        months[str(d)[:7]] += v
    return (f"<section class='card'><h3>{E(title)}</h3><p class='rules'>{E(desc)}</p>"
            + kv([("Before the split a month", f"<span class='{cls(row['train_per_month'])}'>{inr(row['train_per_month'], True)}</span>"),
                  ("After the split a month (unseen)", f"<b class='{cls(row['test_per_month'])}'>{inr(row['test_per_month'], True)}</b>"),
                  ("Max drawdown before / after", f"{inr(row['train_maxdd'])} / {inr(row['test_maxdd'])}")])
            + _equity(daily, dates, split) + heatmap({k: round(v) for k, v in months.items()}, 20000) + "</section>")


def write(path=OUT / "futures_report.html"):
    warnings.filterwarnings("ignore")
    intra = json.loads((OUT / "grid_futures.json").read_text())
    trend = json.loads((OUT / "grid_trend.json").read_text())
    rows = []
    for c in SPECS:
        R = [r for r in intra if r["contract"] == c]
        T = [r for r in trend if r["contract"] == c]
        a, b = np.array([r["train_per_month"] for r in R]), np.array([r["test_per_month"] for r in R])
        best = max((r for r in R if r["trades"] >= 150 and r["train_maxdd"] <= 40000), key=lambda r: r["train_per_month"], default=None)
        tb = max(T, key=lambda r: r["train_per_month"], default=None)
        sp = SPECS[c]
        trend_dd = f" <span class='muted'>(DD {compact(tb['test_maxdd'])})</span>" if tb else ""
        rows.append(f"<tr><td><b>{E(c)}</b><span class='name'>{E(sp.exchange)} · margin {100 * sp.margin_pct:.1f}% of notional</span></td>"
                    f"<td class='r'>{len(R):,}</td><td class='r'>{(a > 0).mean():.0%}</td><td class='r'>{(b > 0).mean():.0%}</td>"
                    f"<td class='r'>{spearmanr(a, b).correlation:.2f}</td>"
                    f"<td>{E(best['setup']) if best else '–'}</td>"
                    f"<td class='r {cls(best['train_per_month']) if best else ''}'>{inr(best['train_per_month'], True) if best else '–'}</td>"
                    f"<td class='r {cls(best['test_per_month']) if best else ''}'>{inr(best['test_per_month'], True) if best else '–'}</td>"
                    f"<td class='r {cls(tb['test_per_month']) if tb else ''}'>{inr(tb['test_per_month'], True) if tb else '–'}"
                    f"{trend_dd}</td></tr>")
    # details: the steadiest intraday finalist, and the gold trend as a warning
    ng = next(r for r in sorted(intra, key=lambda r: -r["train_per_month"])
              if r["contract"] == "NATGASMINI" and r["setup"].startswith("Evening") and r["filter"] == "trend" and r["trades"] >= 150)
    s = load("NATGASMINI")
    f = fx.features(s)
    cand = fx.setups_for(SPECS["NATGASMINI"])[ng["setup"]](s, f, SPECS["NATGASMINI"])
    ex = next(e for e in fx.EXITS if e.label == ng["exit"])
    daily, _, _ = fx.run_variant(s, f, SPECS["NATGASMINI"], cand, ng["filter"], ex, ng["risk"])
    d1 = _detail("Steadiest intraday finalist: natural gas mini, evening range breakout",
                 f"{ng['setup']} · filter {ng['filter']} · {ng['exit']} · risk ₹{ng['risk']:,.0f} a trade (avg {ng['avg_lots']} lots)",
                 daily, s.dates, "2024-12-31", ng)
    g = max((r for r in trend if r["contract"] == "GOLDPETAL"), key=lambda r: r["train_per_month"])
    sysk = g["system"].split()
    a_, b_ = map(int, sysk[1].split("/"))
    r2, gd, gp = backtest("GOLDPETAL", (sysk[0], a_, b_), g["risk"], "2024-12-31")
    d2 = _detail("Why the biggest number is not a strategy: Gold Petal trend following",
                 f"{g['system']} · risk ₹{g['risk']:,.0f} · {g['trades']} trade(s): one leveraged long through gold's rise from about ₹7,100 to ₹14,200 a gram",
                 gp, gd, "2024-12-31", r2)
    all_rows = intra + trend
    tr = np.array([r["train_per_month"] for r in all_rows])
    te = np.array([r["test_per_month"] for r in all_rows])
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Commodity &amp; Currency Search</title><style>{CSS}</style></head><body><main>
<h1>{len(all_rows):,} MCX commodity and NSE currency strategies on ₹1 lakh</h1>
<p class="lede">Crude oil mini, natural gas mini, silver micro, gold petal (MCX, real 5-minute bars Jun 2021 – Oct 2026) and USDINR, EURINR,
GBPINR, JPYINR (hourly from Jan 2024, daily for 10 years) · intraday and positional · chosen on data before 2025 (currency trend: before 2023),
judged after · real Angel One margins · all charges and slippage</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>{int(((tr >= 20000) & (te >= 20000)).sum())}</b> of {len(all_rows):,} variants averaged ₹20,000 a month in both periods; the best
 before-2025 month figure of any variant was {inr(tr.max(), True)}.</li>
<li>Margin sets the ceiling: ₹1 lakh carries about 3 crude mini lots (₹30,600 each), 4 natural gas mini (₹23,100), 3 silver micro (₹27,050),
 ~70 gold petal (₹1,320) or ~40 USDINR lots (₹2,280). Sized so a losing trade costs ₹1,000-2,000, the edges found are worth hundreds to a
 few thousand rupees a month.</li>
<li>Currency futures move little (USDINR about 0.2% a day), so a ₹20,000 month needs around 40 lots, where a few bad days erase it.</li>
<li>The two large 2025-26 figures (gold trend, ₹4,000-9,000 a month) came from holding leveraged gold through a doubling, with drawdowns
 of ₹75,000 to ₹1.5 lakh on a ₹1 lakh account.</li></ul></div>
<h2>By contract</h2><div class="card scroll"><table><thead><tr><th>Contract</th><th class="r">Intraday variants</th><th class="r">Before &gt; 0</th>
<th class="r">After &gt; 0</th><th class="r">Rank corr.</th><th>Best intraday setup (picked before 2025)</th><th class="r">Before /mo</th>
<th class="r">After /mo</th><th class="r">Best trend system, after /mo</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Detail</h2>{d1}{d2}
<h2>How it was tested</h2><div class="card"><ul>
<li>Intraday (flat by 23:00 MCX, 16:45 currency): session opening ranges (15/30/60 min), the 17:00-18:30 evening range and the first US
 hour (18:30-19:30 IST, an hour later outside US summer time), previous-day high/low, VWAP 2 and 2.5-sigma reversion, momentum at 17:00 and
 19:30, reversal at 21:00, gap follow/fade, EMA 9/21, Supertrend(10,3), the EIA crude (Wednesday) and gas (Thursday) reports; x filters
 (trend, volatility regime, weekday) x exits (structural, 0.25/0.5/1 ATR stops; 1/2/3R or none; breakeven) x risk ₹1,000 or ₹2,000 a trade.</li>
<li>Positional: Donchian 20/10 and 55/20, moving-average 10/50, 20/100, 50/200, time-series momentum 60/120/250 days; lots for a 2-ATR loss
 of ₹2,000 or ₹4,000; monthly roll costs; MCX roll jumps muted (the continuous series is not roll-adjusted); currency forward premium 1.5% a year.</li>
<li>Charges: ₹20 an order; MCX commodity transaction tax 0.01% on sells, exchange 0.0021%, stamp 0.002%; currency exchange 0.00035%,
 stamp 0.0001%; SEBI fee, GST; slippage the larger of one tick or 1 bp a side.</li></ul></div>
<p class="note">Research on past data, not a forecast and not investment advice.</p></main></body></html>"""
    path.write_text(body, encoding="utf-8")
    return path
