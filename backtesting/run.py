"""Backtest the live pipeline on recent NSE data.

    ./backtest.sh                          # top 100 NIFTY 500 stocks by turnover, all strategies
    ./backtest.sh --universe-size 200 --no-tm
    ./backtest.sh --symbols RELIANCE,TCS,INFY --json

Writes data/backtests/latest.json (the dashboard's Strategy Lab reads it).
"""
import argparse
import json
import logging
import sys
import time

from backtesting.engine import Backtester
from core.config.settings import DATA_DIR, Settings
from market.calendar import now_ist
from market.history import HistoryStore
from market.instruments.universe import Universe
from scanner.features.daily import compute_daily_stats
from strategies.ensemble import build_strategies

OUT_DIR = DATA_DIR / "backtests"


def pick_universe(daily, size, today):
    rows = []
    for sym, bars in daily.items():
        st = compute_daily_stats(sym, bars, before=today)
        if st:
            rows.append((st.turnover20, sym))
    rows.sort(reverse=True)
    return [s for _, s in rows[:size]]


def run_backtest(settings=None, universe=None, history=None, symbols=None, size=None, use_tm=True, log=print):
    settings = settings or Settings()
    universe = universe or Universe().load()
    history = history or HistoryStore()
    today = now_ist().date().isoformat()
    t0 = time.time()
    if not symbols:
        daily = history.daily(universe.symbols)
        size = size or int(settings.get("backtest.universe_size", 100))
        symbols = pick_universe(daily, size, today)
        log(f"Backtest universe: the {len(symbols)} most traded NIFTY 500 stocks (20-day turnover)")
    daily = history.daily(symbols)
    intraday = history.intraday60(symbols)
    index = history.intraday60([settings.get("market_filter.symbol", "^NSEI")])
    idx_bars = next(iter(index.values()), None)
    log(f"Data: {len(intraday)} of {len(symbols)} stocks with 5-minute history, index {'yes' if idx_bars else 'no'}")
    bt = Backtester(settings, universe, build_strategies(settings))
    res = bt.run(intraday, daily, idx_bars, use_tm=use_tm, progress=log)
    res["run_at"] = now_ist().strftime("%Y-%m-%d %H:%M")
    res["seconds"] = round(time.time() - t0, 1)
    res["config"] = {k: settings.get(k) for k in ("capital_inr", "max_loss_per_trade_inr", "max_daily_loss_inr",
                                                  "max_trades_per_day", "min_reward_risk", "flat_by", "slippage_bps")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "latest.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT_DIR / f"{now_ist().strftime('%Y%m%d-%H%M')}.json").write_text(json.dumps(res, default=str))
    return res


def print_report(res):
    if res.get("error"):
        print("Backtest could not run:", res["error"])
        return
    o = res["overall"]
    print(f"\nBacktest {res['from']} to {res['to']}: {res['sessions']} sessions, {res['stocks']} stocks, "
          f"Time Machine gate {'on' if res['time_machine'] else 'off'}. After slippage and charges.")
    if not o.get("trades"):
        print("No trades.")
    else:
        print(f"{'strategy':<16}{'trades':>7}{'win%':>7}{'avgR':>8}{'netR':>8}{'net Rs':>12}{'charges':>10}{'PF':>6}{'maxDD':>10}")
        rows = list(res["by_strategy"].items()) + [("ALL", o)]
        for name, m in rows:
            pf = f"{m['profit_factor']:.2f}" if m.get("profit_factor") else "-"
            print(f"{name:<16}{m['trades']:>7}{100 * m['win_rate']:>6.0f}%{m['avg_R']:>8.2f}{m['avg_net_R']:>8.2f}"
                  f"{m['net_pnl']:>12,.0f}{m['charges']:>10,.0f}{pf:>6}{m['max_drawdown_inr']:>10,.0f}")
    if res.get("skipped"):
        print("Signals not taken:", ", ".join(f"{k} {v}" for k, v in res["skipped"].items()))
    print("A short sample of recent sessions. Past results do not predict future ones.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", help="comma-separated NSE symbols (default: most traded NIFTY 500)")
    ap.add_argument("--universe-size", type=int)
    ap.add_argument("--no-tm", action="store_true", help="skip the Time Machine gate")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    syms = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    res = run_backtest(symbols=syms, size=a.universe_size, use_tm=not a.no_tm,
                       log=(lambda m: None) if a.json else print)
    if a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "trades"}, indent=1, default=str))
    else:
        print_report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
