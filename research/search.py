"""The strategy search: run the three factories, pick finalists on the TRAIN period only, and
report how they did on the TEST period they never saw.

    ./research.sh search          # all factories (about 30 minutes), then the report
    ./research.sh search --report # rebuild the report from the saved grids
"""
import json
import time

import numpy as np
from scipy.stats import spearmanr

from core.config.settings import DATA_DIR, Settings
from research import data
from research.factory_equity import EXIT_GRID, Ctx, Exit, Universe, evaluate, run_search
from research.factory_index import IndexSearch, run_index_search
from research.factory_options import OptionFactory, run_grid
from research.factory_spreads import run_spread_grid, spread_daily
from research.report import heatmap, inr, kv, cls, compact, CSS, E

OUT = DATA_DIR / "research" / "results"
TARGET_MONTH = 20000.0
TARGET_DAY = 2000.0
CAPITAL = 100000.0
TRAIN_END = np.datetime64("2024-12-31")


def _grids(log):
    N, B, V = data.load("NIFTY"), data.load("BANKNIFTY"), data.load("INDIAVIX")
    syms = [s for s in data.nifty100() if (data.HIST_DIR / f"{s}.npz").exists()]
    t0 = time.time()
    log(f"Equity factory on {len(syms)} NIFTY 100 stocks")
    e1, e2 = run_search(syms, N, V, Settings().get("costs"), log=log)
    log(f"Index factory ({time.time() - t0:.0f}s)")
    i1, i2 = run_index_search(N, B, V, log=log)
    log(f"Option-selling factory ({time.time() - t0:.0f}s)")
    o = run_grid(N, V, log=log)
    log(f"Credit-spread factory ({time.time() - t0:.0f}s)")
    sp = run_spread_grid(N, V, log=log)
    doc = {"stocks": len(syms), "equity": e1 + e2, "index": i1 + i2, "options": o, "spreads": sp,
           "run_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}
    (OUT / "search_grids.json").write_text(json.dumps(doc, default=str))
    return doc


def _family_rows(doc):
    fam = {"Equity intraday (NIFTY 100, MIS)": doc["equity"],
           "Index option selling (NIFTY weekly, hedged)": doc["options"]}
    if doc.get("spreads"):
        fam["Directional credit spreads (NIFTY weekly)"] = doc["spreads"]
    for r in doc["index"]:
        fam.setdefault(f"Index {r['family'][6:]}", []).append(r)
    return fam


def _finalists(rows, min_trades, max_dd, n=5):
    ok = [r for r in rows if r["train_trades"] >= min_trades and r["train_maxdd"] <= max_dd and r["train_per_month"] > 0]
    ok.sort(key=lambda r: -r["train_per_month"])
    out, seen = [], set()
    for r in ok:
        key = (r.get("setup") or r.get("entry"), r.get("filter"), r.get("instrument"), r.get("structure"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- daily series for finalists
class Recompute:
    def __init__(self):
        self.N, self.B, self.V = data.load("NIFTY"), data.load("BANKNIFTY"), data.load("INDIAVIX")
        self.dates = self.N.dates
        self._uni = self._ctx = self._idx = self._opt = None
        self.costs = Settings().get("costs")

    def daily(self, r):
        fam = r["family"]
        if fam == "equity":
            if self._uni is None:
                self._ctx = Ctx(self.N, self.V)
                syms = [s for s in data.nifty100() if (data.HIST_DIR / f"{s}.npz").exists()]
                self._uni = Universe(syms, lambda m: None, self._ctx)
            ex = Exit(r["exit_stop"], r["exit_target"], r["exit_time"], r["exit_breakeven"])
            daily, trades, pt = evaluate(self._uni, self._ctx, r["setup"], r["filter"], ex, self.dates, self.costs)
            return daily, trades
        if fam.startswith("index"):
            if self._idx is None:
                self._idx = IndexSearch(self.N, self.B, self.V)
            ex = next(e for e in EXIT_GRID + [Exit("struct", 2.0, "15:15", False), Exit("struct", 0, "15:15", False),
                                              Exit("atr0.25", 2.0, "15:15", False)] if e.label == r["exit"])
            fut, opt, n = self._idx.evaluate(r["instrument"], r["setup"], r["filter"], ex)
            return (fut if "futures" in fam else opt), n
        if fam == "credit_spread":
            import research.factory_options as fo
            from research.factory_spreads import SPREAD_WINGS
            if getattr(self, "_spr", None) is None:
                self._spr, self._spr_pre = OptionFactory(self.N, self.V), {}
            key = (r["entry"], r["expiry"])
            if key not in self._spr_pre:
                keep, fo.WINGS = fo.WINGS, SPREAD_WINGS
                try:
                    self._spr_pre[key] = self._spr.precompute(*key)
                finally:
                    fo.WINGS = keep
            return spread_daily(self._spr, self._spr_pre[key], r)
        if self._opt is None:
            self._opt = OptionFactory(self.N, self.V)
            self._opt_filters = self._opt.filters()
            self._pre = {}
        key = (r["entry"], r["expiry"])
        if key not in self._pre:
            self._pre[key] = self._opt.precompute(*key)
        stop = ("none", 0) if r["stop"] == "none" else (r["stop"].split()[0], float(r["stop"].split()[1]))
        tgt = None if r["target"] == "none" else int(r["target"].split("%")[0]) / 100
        pnl, lots, credit = self._opt.evaluate(self._pre[key], r["structure"], r["wings"], stop, tgt, r["exit"])
        fm = self._opt_filters[r["filter"]]
        return np.where(fm, pnl, 0.0), (fm & (lots > 0)).astype(int)


def _scale_for_target(daily, dates):
    """How much bigger every position would have to be for the TEST period to average Rs 20k a
    month, and the drawdown that size would have carried over the whole 5 years."""
    test = dates > TRAIN_END
    months = len({str(d)[:7] for d in dates[test]})
    per_month = daily[test].sum() / months
    if per_month <= 0:
        return None
    k = TARGET_MONTH / per_month
    eq = np.cumsum(daily * k)
    dd = float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max())
    return {"multiple": round(k, 1), "max_drawdown_at_that_size": round(dd, 0)}


# ---------------------------------------------------------------- report
def _scatter(rows, w=520, h=360):
    pts = [(r["train_per_month"], r["test_per_month"]) for r in rows]
    if len(pts) > 6000:
        idx = np.random.default_rng(0).choice(len(pts), 6000, replace=False)
        pts = [pts[i] for i in idx]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    lo = min(min(xs), min(ys), -1000)
    hi = max(max(xs), max(ys), TARGET_MONTH * 1.05)
    pad = 56
    X = lambda v: pad + (v - lo) / (hi - lo) * (w - pad - 10)
    Y = lambda v: h - pad + 10 - (v - lo) / (hi - lo) * (h - pad - 10)
    dots = "".join(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="1.6" fill="var(--series-1)" fill-opacity="0.35"/>' for a, b in pts)
    ticks = [t for t in sorted({lo, 0, TARGET_MONTH}) if True] + ([hi] if hi > TARGET_MONTH * 1.25 else [])
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Train vs test monthly profit">'
            + "".join(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="10" y2="{h - pad + 10}" stroke="var(--grid)"/>'
                      f'<line x1="{pad}" x2="{w - 10}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)"/>'
                      f'<text x="{X(t):.1f}" y="{h - pad + 26}" class="tick" text-anchor="middle">{compact(t)}</text>'
                      f'<text x="{pad - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{compact(t)}</text>' for t in ticks)
            + f'<line x1="{X(TARGET_MONTH):.1f}" x2="{X(TARGET_MONTH):.1f}" y1="10" y2="{h - pad + 10}" stroke="var(--down)" stroke-dasharray="4 4"/>'
            + f'<line x1="{pad}" x2="{w - 10}" y1="{Y(TARGET_MONTH):.1f}" y2="{Y(TARGET_MONTH):.1f}" stroke="var(--down)" stroke-dasharray="4 4"/>'
            + dots
            + f'<text x="{(pad + w) / 2:.0f}" y="{h - 6}" class="tick" text-anchor="middle">2021-24 (train) ₹ a month</text>'
            + f'<text x="12" y="{h / 2:.0f}" class="tick" transform="rotate(-90 12 {h / 2:.0f})" text-anchor="middle">2025-26 (test) ₹ a month</text>'
            + "</svg>")


def _equity_svg(daily, dates, w=860, h=220):
    eq = CAPITAL + np.cumsum(daily)
    pts = list(zip([str(d) for d in dates[::5]], eq[::5]))
    lo, hi = min(min(eq), CAPITAL), max(max(eq), CAPITAL)
    span = (hi - lo) or 1
    pl, pr, pt, pb = 70, 10, 10, 22
    X = lambda i: pl + i * (w - pl - pr) / (len(pts) - 1)
    Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / span)
    split = next((i for i, (d, _) in enumerate(pts) if d > str(TRAIN_END)), len(pts) - 1)
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(pts))
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Equity">'
            f'<rect x="{X(split):.1f}" y="{pt}" width="{w - pr - X(split):.1f}" height="{h - pt - pb}" fill="var(--surface-2)"/>'
            f'<text x="{X(split) + 6:.1f}" y="{pt + 14}" class="tick">test period (not used to choose)</text>'
            + "".join(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)"/>'
                      f'<text x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{inr(t)}</text>' for t in (lo, CAPITAL, hi))
            + f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2"/></svg>')


def _describe(r):
    if r["family"] == "equity":
        return f"{r['setup']} · filter: {r['filter']} · {r['exit']} · risk ₹1,000 a trade, first 5 trades a day"
    if r["family"].startswith("index"):
        return f"{r['instrument']} {r['setup']} · filter: {r['filter']} · {r['exit']}"
    if r["family"] == "credit_spread":
        return (f"At {r['entry']}, direction by {r['signal']}: sell a {r['short_strike']} put spread (bullish) or call spread "
                f"(bearish), {r['wings']} hedge, stop {r['stop']}, target {r['target']}, out {r['exit']}, days: {r['filter']}, "
                f"{r['expiry'].replace('_', ' ')} · lots a ₹1 L account can carry (avg {r['avg_lots']})")
    return (f"Sell {r['structure']} at {r['entry']} ({r['expiry'].replace('_', ' ')}), {r['wings']} hedge, stop {r['stop']}, "
            f"target {r['target']}, out {r['exit']}, days: {r['filter']} · lots a ₹1 L account can carry (avg {r['avg_lots']})")


def write_search_report(doc, finals, path):
    fam = _family_rows(doc)
    summary_rows, sections = [], []
    for name, rows in fam.items():
        tr = np.array([r["train_per_month"] for r in rows])
        te = np.array([r["test_per_month"] for r in rows])
        rho = spearmanr(tr, te).correlation if len(rows) > 2 else float("nan")
        summary_rows.append(f"<tr><td>{E(name)}</td><td class='r'>{len(rows):,}</td><td class='r'>{(tr > 0).mean():.0%}</td>"
                            f"<td class='r'>{(te > 0).mean():.0%}</td><td class='r'>{rho:.2f}</td>"
                            f"<td class='r'>{inr(tr.max(), True)}</td><td class='r'>{inr(te.max(), True)}</td>"
                            f"<td class='r'>{int((tr >= TARGET_MONTH).sum())}</td><td class='r'>{int((te >= TARGET_MONTH).sum())}</td></tr>")
        sections.append(f"<div class='card'><h3>{E(name)}</h3><div class='grid2'><div>{_scatter(rows)}</div><div>"
                        f"<p class='note'>Each dot is one variant. Right of the dashed line: at least ₹20,000 a month in "
                        f"2021-24. Above it: at least ₹20,000 a month in 2025-26. Rank correlation between the two periods: "
                        f"{rho:.2f} (1 = the past ranking held, 0 = no relation).</p></div></div></div>")
    final_html = []
    for f in finals:
        r, daily, n = f["row"], f["daily"], f["trades"]
        dates = f["dates"]
        months = {}
        for d, v in zip(dates, daily):
            months[str(d)[:7]] = months.get(str(d)[:7], 0) + v
        test = dates > TRAIN_END
        sc = f.get("scale")
        need = (f"<p>To have averaged ₹20,000 a month in 2025-26, every position would have needed to be "
                f"<b>{sc['multiple']}×</b> bigger; at that size the worst drawdown over 5 years was <b>{inr(sc['max_drawdown_at_that_size'])}</b> "
                f"on a ₹1 L account.</p>") if sc else "<p>It lost money in 2025-26, so no size gets it to the target.</p>"
        final_html.append(f"""<section class='card'><h3>{E(f['family'])}</h3><p class='rules'>{E(_describe(r))}</p>
          <div class='grid3'>{kv([("2021-24 (train) a month", f"<span class='{cls(r['train_per_month'])}'>{inr(r['train_per_month'], True)}</span>"),
                                  ("2025-26 (test) a month", f"<b class='{cls(r['test_per_month'])}'>{inr(r['test_per_month'], True)}</b>"),
                                  ("Sessions at ≥ ₹2,000 (test)", f"{100 * float((daily[test] >= TARGET_DAY).mean()):.1f}%")])}
          {kv([("Trades train / test", f"{r['train_trades']:,} / {r['test_trades']:,}"),
               ("Max drawdown train / test", f"{inr(r['train_maxdd'])} / {inr(r['test_maxdd'])}"),
               ("Positive months train / test", f"{r.get('train_pos_months', '–')}% / {r.get('test_pos_months', '–')}%")])}</div>
          {need}{_equity_svg(daily, dates)}{heatmap({k: round(v) for k, v in months.items()}, TARGET_MONTH)}</section>""")
    total = sum(len(v) for v in fam.values())
    hit_both = sum(1 for rows in fam.values() for r in rows if r["train_per_month"] >= TARGET_MONTH and r["test_per_month"] >= TARGET_MONTH)
    best_test_final = max((f["row"]["test_per_month"] for f in finals), default=0)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE Strategy Search</title><style>{CSS}</style></head><body><main>
<h1>Searching {total:,} NSE strategy variants for ₹20,000 a month on ₹1 lakh</h1>
<p class="lede">5 years of 5-minute bars ({doc['stocks']} NIFTY 100 stocks, NIFTY, BANKNIFTY, India VIX) · chosen on Oct 2021 – Dec 2024 ·
judged on Jan 2025 – Oct 2026 · after brokerage, STT, exchange charges, GST, stamp duty and slippage · run {E(doc['run_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li><b>{hit_both}</b> of {total:,} variants averaged ₹20,000 a month in both periods.</li>
<li>The best finalist picked on 2021-24 data made <b>{inr(best_test_final, True)}</b> a month in 2025-26.</li>
<li>₹2,000 a day is about 500% a year and ₹20,000 a month about 240% a year on ₹1 lakh. Nothing here earned that without
 position sizes whose drawdowns exceed the account (see "what it would take" under each finalist).</li>
<li>Option selling needs about ₹86,000-96,000 of margin a hedged NIFTY lot (Angel One, Oct 2026), so ₹1 lakh carries one lot, and none on
 expiry days since the extra 2% expiry-day margin (Nov 2024).</li></ul></div>
<h2>Every family, train vs test</h2>
<div class="card scroll"><table><thead><tr><th>Family</th><th class="r">Variants</th><th class="r">Train &gt; 0</th><th class="r">Test &gt; 0</th>
<th class="r">Rank corr.</th><th class="r">Best train /mo</th><th class="r">Best test /mo</th><th class="r">Train ≥ 20k</th><th class="r">Test ≥ 20k</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>
<p class="note">"Best test" is chosen with hindsight across all variants: it is what luck looks like, not a strategy anyone could have picked.</p></div>
{''.join(sections)}
<h2>Finalists (picked on 2021-24 only)</h2>
<p class="note">Rule: the highest 2021-24 monthly profit with enough trades and a 2021-24 drawdown under ₹35,000 (₹60,000 for futures), one per setup.</p>
{''.join(final_html)}
<h2>How it was tested</h2><div class="card"><ul>
<li>Equity: 38 setups (opening ranges 5/15/30/60 min, stop-order and close breakouts; previous-day high/low, CPR, pivots, Camarilla
 H3/L3 and H4/L4, 20-day and weekly highs/lows as breakouts and rejections; VWAP crosses and 1.5/2/2.5-sigma reversion; gap follow and fade;
 momentum at 10:15/11:00/13:00; volume spikes; opening drives) x 12 filters (trend, NIFTY direction, opening volume, narrow CPR, NR7, VIX
 regime and combinations) x exits (structural, 0.1/0.25/0.5 ATR, 0.5%/1% stops; 1/1.5/2/3R or no target; out at 13:00 or 15:15;
 breakeven after 1R). Risk ₹1,000 a trade, at most ₹2.5 L a position, the first 5 trades a day, 3 bps slippage a side.</li>
<li>Index: the same price setups on NIFTY and BANKNIFTY, as 1 futures lot (needs more than ₹1 L margin) and as 1 lot of ATM options bought.</li>
<li>Option selling: 8 entry times x 4 strike choices x 2 hedge widths x same-day expiry or not x 12 stop rules x 3 profit targets x 2 exit
 times x 13 day filters (weekday, expiry day, VIX regime, gap size). Option prices modeled (IV 0.9 x India VIX, checked against real
 NIFTY option candles); stop-loss fills pay 3x slippage.</li>
<li>Selection used only the train period. Searching this many variants guarantees some that look great by chance; the test period is the check.</li>
</ul></div><p class="note">Research on past data, not a forecast and not investment advice.</p></main></body></html>"""
    path.write_text(body, encoding="utf-8")
    return path


def main(rebuild_only=False, log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    if rebuild_only and (OUT / "search_grids.json").exists():
        doc = json.loads((OUT / "search_grids.json").read_text())
        if not doc.get("spreads") and (OUT / "grid_spreads.json").exists():
            doc["spreads"] = json.loads((OUT / "grid_spreads.json").read_text())
    else:
        doc = _grids(log)
    fam = _family_rows(doc)
    rc = Recompute()
    finals = []
    for name, rows in fam.items():
        dd_cap = 60000 if "futures" in name else 35000
        mt = 200 if name.startswith("Equity") else 100
        for r in _finalists(rows, mt, dd_cap, n=2):
            daily, trades = rc.daily(r)
            finals.append({"family": name, "row": r, "daily": daily, "trades": trades, "dates": rc.dates,
                           "scale": _scale_for_target(daily, rc.dates)})
    path = write_search_report(doc, finals, OUT / "search_report.html")
    log(f"Report: {path}")
    return doc, finals
