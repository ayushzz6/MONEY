"""HTML report for round 4 (leading information): data/research/results/lead_report.html"""
import json

from core.config.settings import DATA_DIR
from research.lib.report import EXTRA_CSS
from research.report import CSS, E

RES = DATA_DIR / "research" / "results"
P3 = ["2021-23 (choose)", "2024 (check)", "2025-26 (judge)"]


def _c(v):
    return "" if v is None else ("pos" if v > 0 else "neg" if v < 0 else "")


def _f(v, nd=1, plus=True):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def _rows(rows, trade_key=None, show_ic=True):
    out = []
    for r in rows:
        p = r["periods"]
        j = p["2025-26 (judge)"]
        cells = [f"<td><b>{E(r['signal'])}</b><div class='small'>{E(r['use'])} · {E(r.get(trade_key) or r.get('horizon') or '')}</div></td>"]
        if show_ic:
            cells += [f"<td class='r'>{_f(p[k]['ic'], 3)}</td>" for k in P3]
        cells += [f"<td class='r {_c(p[k]['gross_bps'])}'>{_f(p[k]['gross_bps'])}</td>" for k in P3]
        cells += [f"<td class='r {_c(j['net_bps'])}'><b>{_f(j['net_bps'])}</b></td>",
                  f"<td class='r'>{_f(j.get('win_pct'), 1, False)}%</td>",
                  f"<td class='r'>{j['trades']:,}</td>"]
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out)


def _head(show_ic=True):
    ic = "".join(f"<th class='r'>IC {k[:7]}</th>" for k in P3) if show_ic else ""
    return (f"<thead><tr><th>Signal · use · horizon</th>{ic}"
            + "".join(f"<th class='r'>Gross bps {k[:7]}</th>" for k in P3)
            + "<th class='r'>Net bps 2025-26</th><th class='r'>Profitable trades 2025-26</th><th class='r'>Trades 2025-26</th></tr></thead>")


def _comb(rows, key):
    return "".join(
        f"<tr><td><b>{E(c[key])}</b></td>" + "".join(f"<td class='r'>{_f(c['periods'][k]['ic'], 3)}</td>" for k in P3)
        + "".join(f"<td class='r {_c(c['periods'][k]['gross_bps'])}'>{_f(c['periods'][k]['gross_bps'])}</td>" for k in P3)
        + f"<td class='r {_c(c['periods'][P3[2]]['net_bps'])}'><b>{_f(c['periods'][P3[2]]['net_bps'])}</b></td>"
        + f"<td class='r'>{_f(c['periods'][P3[2]]['win_pct'], 1, False)}%</td></tr>" for c in rows)


def _survivors(rows):
    """Signals that beat costs in 2024 AND 2025-26 with the direction chosen on 2021-23."""
    return [r for r in rows if (r["periods"]["2024 (check)"]["net_bps"] or -1) > 0
            and (r["periods"]["2025-26 (judge)"]["net_bps"] or -1) > 0]


def write(intra=None, daily=None):
    intra = intra or json.loads((RES / "lead_intraday.json").read_text())
    daily = daily or json.loads((RES / "lead_daily.json").read_text())
    cost = intra["cost_bps"]
    s_i, s_m = _survivors(intra["stock"]), _survivors(intra["market"])
    s_d, s_dm = _survivors(daily["stock"]), _survivors(daily["market"])
    best_i = max(intra["stock"], key=lambda r: r["periods"]["2025-26 (judge)"]["gross_bps"])
    best_d = max(daily["stock"], key=lambda r: r["periods"]["2025-26 (judge)"]["net_bps"] or -1e9)
    n_tested = len(intra["stock"]) + len(intra["market"]) + len(daily["stock"]) + len(daily["market"])
    surv = s_i + s_m + s_d + s_dm
    ci = {c["horizon"]: c for c in intra["combined"]}
    cd = {c["trade"]: c for c in daily["combined"]}["next day, 09:20 to 15:15"]["periods"]
    surv_html = "".join(f"<li><b>{E(r['signal'])}</b> ({E(r['use'])}, {E(r.get('trade') or r.get('horizon'))}): "
                        f"{_f(r['periods']['2024 (check)']['net_bps'])} bps in 2024, {_f(r['periods']['2025-26 (judge)']['net_bps'])} bps in 2025-26 after costs, "
                        f"{r['periods']['2025-26 (judge)']['trades']:,} trades, {_f(r['periods']['2025-26 (judge)'].get('win_pct'), 1, False)}% profitable</li>" for r in surv)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leading Signals Test</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Round 4: is there a leading indicator that beats costs?</h1>
<p class="lede">{intra['stocks']} NIFTY 500 stocks · {intra['sessions']} sessions of 5-minute bars, Oct 2021 to Oct 2026 · NSE delivery, F&amp;O open interest and
FII / DII positions for every session · each signal's direction and horizon chosen on 2021-23, checked on 2024, judged once on 2025-26 · built {E(intra['built_at'])}</p>
<div class="card"><h2 style="margin-top:0">The answer</h2><ul class="findings">
<li>{n_tested} candidate leading signals were tested. A trade costs about <b>{cost:.0f} bps</b> (0.{int(round(cost)):02d}%) of its value in charges and slippage
on ₹1 lakh intraday, and about <b>{daily['costs_bps']['5-day swing']:.0f} bps</b> for a 5-day delivery swing. A signal has to predict a bigger move than that, on average, to make money.</li>
<li>Strongest intraday stock signal in 2025-26: <b>{E(best_i['signal'])}</b> ({E(best_i['use'])}, {E(best_i['horizon'])}),
{_f(best_i['periods']['2025-26 (judge)']['gross_bps'])} bps a trade before costs, {_f(best_i['periods']['2025-26 (judge)']['net_bps'])} bps after.</li>
<li>Strongest positioning signal in 2025-26: <b>{E(best_d['signal'])}</b> ({E(best_d['use'])}, {E(best_d['trade'])}),
{_f(best_d['periods']['2025-26 (judge)']['gross_bps'])} bps before costs, {_f(best_d['periods']['2025-26 (judge)']['net_bps'])} bps after.</li>
<li>All intraday signals combined in one model (30 minutes): {_f(ci['30 min']['periods']['2025-26 (judge)']['gross_bps'])} bps before costs,
{_f(ci['30 min']['periods']['2025-26 (judge)']['net_bps'])} bps after, {_f(ci['30 min']['periods']['2025-26 (judge)']['win_pct'], 1, False)}% of trades profitable.</li>
<li>All positioning signals combined in one model (next session, intraday): {_f(cd['2021-23 (choose)']['net_bps'])} bps after costs on 2021-23 (the years it learned on),
{_f(cd['2024 (check)']['net_bps'])} bps in 2024 and {_f(cd['2025-26 (judge)']['net_bps'])} bps in 2025-26, {_f(cd['2025-26 (judge)']['win_pct'], 1, False)}% of trades profitable: the closest to break-even of anything here,
and fading every year (rank correlation {cd['2021-23 (choose)']['ic']} → {cd['2024 (check)']['ic']} → {cd['2025-26 (judge)']['ic']}).</li>
<li>Signals that beat costs in both 2024 and 2025-26: <b>{len(surv)}</b> of {n_tested}.</li></ul>
{('<h3>The ones that held up</h3><ul class="findings">' + surv_html + '</ul>') if surv else ''}</div>

<h2>Intraday: each stock against the others</h2><div class="card scroll"><table>{_head()}<tbody>{_rows(intra['stock'])}</tbody></table>
<p class="note">Every 15 minutes from 09:30 to 14:30: buy the {intra['k']} stocks with the strongest signal and short the {intra['k']} weakest, entering at the next bar's open.
IC is the rank correlation between the signal and the next move across stocks (0 = none; 0.02-0.05 is typical of real but small effects).
Gross and net are basis points per trade (100 bps = 1%); net subtracts {cost:.1f} bps. "use" says whether the signal is followed or faded, chosen on 2021-23.</p></div>

<h2>Intraday: market-wide signals (NIFTY, traded through NIFTYBEES)</h2><div class="card scroll"><table>{_head()}<tbody>{_rows(intra['market'])}</tbody></table>
<p class="note">Trades only when the signal is in its most extreme 10% (thresholds from 2021-23). IC is with the NIFTY's next move over all signal times.</p></div>

<h2>Intraday: all signals together</h2><div class="card scroll"><table><thead><tr><th>Horizon</th>{''.join(f"<th class='r'>IC {k[:7]}</th>" for k in P3)}
{''.join(f"<th class='r'>Gross bps {k[:7]}</th>" for k in P3)}<th class='r'>Net bps 2025-26</th><th class='r'>Profitable 2025-26</th></tr></thead>
<tbody>{_comb(intra['combined'], 'horizon')}</tbody></table>
<p class="note">A gradient-boosted model of every signal above, trained on 2021-23 to predict each stock's move relative to the others.</p></div>

<h2>Next day: delivery, open interest and options positioning</h2><div class="card scroll"><table>{_head()}<tbody>{_rows(daily['stock'], 'trade')}</tbody></table>
<p class="note">Known after the close from NSE's daily files ({daily['data']['fo_stocks']} F&amp;O stocks for open interest). Each day: buy the {daily['k']} strongest and short
the {daily['k']} weakest. Intraday trades run 09:20 to 15:15 the next session ({daily['costs_bps']['next day, 09:20 to 15:15']} bps cost); swings run from the next
09:20 to the 5th session's close ({daily['costs_bps']['5-day swing']} bps cost; shorting for 5 days needs futures, so the short side is indicative).</p></div>

<h2>Next day: FII / DII positions, NIFTY options and the US market</h2><div class="card scroll"><table>{_head()}<tbody>{_rows(daily['market'], 'trade')}</tbody></table>
<p class="note">Traded through NIFTYBEES on the most extreme 20% of days (thresholds from 2021-23), 09:20 to 15:15 the next session.</p></div>

<h2>Next day: all positioning signals together</h2><div class="card scroll"><table><thead><tr><th>Trade</th>{''.join(f"<th class='r'>IC {k[:7]}</th>" for k in P3)}
{''.join(f"<th class='r'>Gross bps {k[:7]}</th>" for k in P3)}<th class='r'>Net bps 2025-26</th><th class='r'>Profitable 2025-26</th></tr></thead>
<tbody>{_comb(daily['combined'], 'trade')}</tbody></table></div>

<p class="note">Method: every signal uses only bars that have closed (a test that blanks all later bars changes none of them) and files NSE publishes the same evening.
Entries are at the next bar's open, or 09:20 for next-day trades. Costs: Angel One / NSE intraday charges on ₹1 lakh plus 5 bps slippage a side.
{n_tested} signals were tried, so one or two passing by luck is expected. Research on past data, not investment advice.</p></main></body></html>"""
    path = RES / "lead_report.html"
    path.write_text(body, encoding="utf-8")
    return path
