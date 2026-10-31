"""Replay a past session through the LIVE engine components, bar by bar, on a simulated clock.

    ./replay.sh 2026-10-23            # the morning scan for that day, every strategy, paper fills
    ./replay.sh 2026-10-23 --no-auto  # ignore the auto-select bench (show every strategy's trades)

Unlike the backtest (which uses its own loop), this drives the real DecisionPipeline,
OrderManager and PaperBroker exactly as ./run.sh does during the session, so it shows
what the engine itself would have done. Paper only; nothing is sent anywhere.
"""
import argparse
import datetime as dt
import logging
import sys
import time

from apps.decision_service.main import DecisionPipeline
from apps.scanner_service.main import ScannerService
from backtesting.selection import load_latest, strategy_allowed
from brokers.paper import PaperBroker
from core.config.settings import Settings
from market.calendar import IST
from market.candles.bars import split_sessions
from market.history import HistoryStore
from market.instruments.universe import Universe
from market.state.manager import MarketState
from oms.order_manager import OrderManager
from portfolio.positions import TradeBook
from risk.kill_switch import KillSwitch
from risk.limits import RiskGate
from strategies.ensemble import build_strategies


class _Clock:
    def __init__(self):
        self.t = time.time()
        self._real = time.time

    def install(self):
        time.time = lambda: self.t

    def uninstall(self):
        time.time = self._real


class _NoNews:
    @staticmethod
    def block_reason(symbol):
        return None

    @staticmethod
    def for_symbol(symbol, hours=24):
        return []


class _MemJournal:
    def save_trade(self, t):
        pass

    def save_signal(self, date, row):
        pass


class _MemKill(KillSwitch):
    def __init__(self):
        self.state = {"global": None, "strategies": {}, "symbols": {}}
        import threading
        self._lock = threading.Lock()

    def _save(self):
        pass


def replay(date: str, use_auto: bool = True, log=print):
    settings = Settings()
    universe = Universe().load()
    history = HistoryStore()
    scanner = ScannerService(settings, universe, history)
    res = scanner.run_daily(date)
    wl = [r["symbol"] for r in res["watchlist"]]
    idx_sym = settings.get("market_filter.symbol", "^NSEI")
    hist = history.intraday60(wl + [idx_sym])
    day = {s: split_sessions(b).get(date, []) for s, b in hist.items()}
    have = [s for s in wl if day.get(s)]
    if not have:
        log(f"No 5-minute bars for {date} (Yahoo keeps about 60 sessions; weekends and holidays have none).")
        return None
    log(f"Replay {date}: watchlist {len(wl)} stocks ({len(have)} with bars), index {'yes' if day.get(idx_sym) else 'no'}")

    state = MarketState()
    broker = PaperBroker(state, float(settings.get("slippage_bps", 5)))
    book, journal, kill = TradeBook(), _MemJournal(), _MemKill()
    events = []
    oms = OrderManager(settings, broker, book, journal, state, on_event=lambda k, t, tr=None: events.append((k, t)))
    strategies = build_strategies(settings)
    latest = load_latest()

    def gate_fn(name):
        if use_auto:
            return strategy_allowed(settings, latest, name)
        return True, "replay: auto-select ignored"
    pipe = DecisionPipeline(settings, universe, strategies, state, history, scanner, _NoNews(), RiskGate(settings, kill, universe),
                            kill, oms, book, journal, gate_fn)
    clock = _Clock()
    clock.install()
    try:
        for k in range(75):
            stamp = None
            for sym in have + [idx_sym]:
                bars = [b for b in day.get(sym, []) if b.slot <= k]
                if not bars:
                    continue
                stamp = bars[-1].start + 300
                clock.t = stamp + 20
                state.set_bars(sym, bars, day=date)
                state.on_price(sym, bars[-1].c, clock.t, build_bars=False)
            if stamp is None:
                continue
            now = dt.datetime.fromtimestamp(clock.t, IST)
            for sym in have:
                bars = state.bars(sym)
                if bars and bars[-1].slot == k:
                    oms.on_price(sym, bars[-1].c, bar=bars[-1])
                    pipe.on_bar(sym, now, date)
            oms.step(now)
    finally:
        clock.uninstall()

    log("\nSignals:")
    for s in reversed(pipe.signals):
        log(f"  {s['time']} {s['symbol']:<12} {s['strategy']:<15} {s['side']:<4} {s['decision']:<17} {s['why'][:110]}")
    log("\nTrades:")
    net = 0.0
    for t in sorted(book.trades.values(), key=lambda x: x.created_at):
        net += t.net_pnl or 0
        log(f"  {t.symbol:<12} {t.strategy:<15} {t.side} {t.qty:>4} @ {t.entry_price}  exit {t.exit_price} "
            f"({t.exit_reason})  net Rs {t.net_pnl:,.2f}  R {t.R}  [{t.status}]")
    log(f"\n{len(book.trades)} trades, net Rs {net:,.2f} after charges (paper replay; Yahoo 5-minute bars).")
    return {"signals": pipe.signals, "trades": [t.to_dict() for t in book.trades.values()], "net": net}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", help="a past NSE session, YYYY-MM-DD (within Yahoo's ~60 sessions of 5-minute bars)")
    ap.add_argument("--no-auto", action="store_true", help="trade every enabled strategy, ignoring the auto-select bench")
    ap.add_argument("-v", action="store_true", help="show the engine's log lines")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING, format="%(name)s: %(message)s")
    replay(a.date, use_auto=not a.no_auto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
