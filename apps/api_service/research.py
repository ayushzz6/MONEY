"""The research results (data/research/results) in the shape the dashboard's Research page needs."""
import json
import re
from pathlib import Path

from core.config.settings import DATA_DIR

RESULTS = DATA_DIR / "research" / "results"
REPORT_NAME = re.compile(r"[a-z0-9_]+\.html")

REPORTS = [
    ("obtest_report.html", "Order-block zones backtested",
     "The dashboard's order-block board replayed bar by bar on NSE F&O stocks: first retests of the zones, chosen on 2022-24, judged on the last 2 years."),
    ("srbreak_report.html", "Strong support / resistance breakouts",
     "Daily scan for stocks at strong levels, traded on a high-volume breakout with market, sector, trend and room checks; chosen on 2022-24, judged on the last 2 years."),
    ("library6_report.html", "Round 6: price action routed by market type",
     "44 price-action rules on the NIFTY 500 in 18 market types (NIFTY trend, VIX, the open), with regime-table and machine-learning routers judged on 2025-26."),
    ("opt5_report.html", "Round 5: NIFTY / BANKNIFTY options at 1:2",
     "29,361 option-buying variants on real NSE option chains (IV, delta, open interest, PCR, max pain), stop and target at 1:2, judged on 2025-26."),
    ("lead_report.html", "Round 4: leading indicators against costs",
     "Lead-lag, order flow, breadth, delivery, open interest, FII / DII positions and the US market as predictors of the next move, judged on 2025-26."),
    ("library3_report.html", "Round 3: 1:1 trades and a 75%-accuracy router",
     "123 strategies with the target as far as the stop, and a machine-learning router judged on 2025-26."),
    ("library2_report.html", "Round 2: big movers, stop orders, agreement",
     "The book strategies on stocks with big expected moves, breakouts with stop-order entries, and multi-strategy agreement, plus the router."),
    ("library_report.html", "102 book strategies + day-type router",
     "1,428 versions of strategies from the classic trading books on the NIFTY 500, a router that picks strategies by day type, all after costs."),
    ("equity3_report_n500.html", "Equity cash, round 3: NIFTY 500",
     "Every intraday, swing, pair and ETF strategy on the 500 most liquid NSE stocks, 5-minute bars, 2021-26."),
    ("equity3_report.html", "Equity cash, round 3: NIFTY 200",
     "17 new strategy ideas plus pairs, ETFs and swing on the NIFTY 200, with the exact ₹1 lakh account."),
    ("equity_report.html", "Equity cash, round 2: NIFTY 100",
     "8,082 combinations of setup, filter and exit on 5-minute bars."),
    ("report.html", "The first 15 strategies: equity and F&O",
     "Equity cash, index options and futures on 5 years of data, with real margins."),
    ("futures_report.html", "Commodities, currencies and gold",
     "MCX crude, natural gas, silver and gold, NSE currency pairs and gold (XAUUSD via MCX)."),
]

_cache = {}


def _load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    key = (name, p.stat().st_mtime)
    if key not in _cache:
        _cache[key] = json.loads(p.read_text())
    return _cache[key]


def _families(grid):
    out = []
    for fam in ("intraday", "swing", "pairs", "etf"):
        rows = grid.get(fam) or []
        if not rows:
            continue
        out.append({"family": fam, "variants": len(rows),
                    "positive_train": sum(r["train_per_month"] > 0 for r in rows),
                    "positive_test": sum(r["test_per_month"] > 0 for r in rows),
                    "both_20k": sum(r["train_per_month"] >= 20000 and r["test_per_month"] >= 20000 for r in rows)})
    return out


def summary() -> dict:
    universes = []
    for key, label, exact, grid in (("nifty500", "NIFTY 500", "equity3_exact_n500.json", "equity3_grid_n500.json"),
                                    ("nifty200", "NIFTY 200", "equity3_exact.json", "equity3_grid.json")):
        ex = _load(exact)
        if not ex:
            continue
        g = _load(grid) or {}
        universes.append({"key": key, "label": label, "run_at": ex.get("run_at"), "stocks": g.get("stocks"),
                          "board": ex.get("board") or [], "families": _families(g),
                          "report": f"/reports/{'equity3_report_n500.html' if key == 'nifty500' else 'equity3_report.html'}"})
    reports = [{"file": f, "title": t, "desc": d, "url": f"/reports/{f}"} for f, t, d in REPORTS if (RESULTS / f).exists()]
    notice = RESULTS / "NOTICE.txt"          # a warning shown above everything while results are being redone
    return {"universes": universes, "reports": reports,
            "notice": notice.read_text().strip() if notice.exists() else None,
            "rules": "Chosen on Oct 2021 - Dec 2024 only, judged on Jan 2025 - Oct 2026. ₹1 lakh capital, ₹5 lakh "
                     "intraday (MIS) exposure, ₹1,000 risk a trade, at most 5 positions and 5 trades a day, trading "
                     "stops for the day at a ₹3,000 loss, 3 bps slippage a side, all NSE charges."}


def report_path(name: str):
    if not REPORT_NAME.fullmatch(name):
        return None
    p = RESULTS / name
    return p if p.exists() else None
