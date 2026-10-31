"""Daily scan (trader-scanner, NSE): NIFTY 500 daily bars -> setups -> today's watchlist.

Run before the open (or any time: it always scans the sessions before the trading day it
is for). Writes data/state/watchlist.json.
"""
import json
import logging
import time
from collections import Counter
from typing import Dict, List

from core.config.settings import STATE_DIR
from scanner.features.daily import DailyStats, compute_daily_stats
from scanner.ranking.watchlist import liquid, rank_watchlist, score

logger = logging.getLogger("ScannerService")

WATCHLIST_PATH = STATE_DIR / "watchlist.json"


class ScannerService:
    def __init__(self, settings, universe, history):
        self.settings = settings
        self.universe = universe
        self.history = history
        self.stats: Dict[str, DailyStats] = {}
        self.watchlist: List[dict] = []
        self.result: dict = {}

    def run_daily(self, trade_date: str) -> dict:
        t0 = time.time()
        syms = self.universe.symbols
        daily = self.history.daily(syms)
        near = float(self.settings.get("strategies.level_breakout.near_pct", 3.0))
        stats = {}
        for s, bars in daily.items():
            st = compute_daily_stats(s, bars, before=trade_date, near_pct=near)
            if st:
                stats[s] = st
        u = self.settings.get("universe", {}) or {}
        min_price, min_to = float(u.get("min_price", 50)), float(u.get("min_turnover_cr", 15))
        ranked = rank_watchlist(stats, int(u.get("watchlist_size", 50)), min_price, min_to)
        self.stats = stats
        self.watchlist = [self._row(stats[s], sc) for s, sc in ranked]
        n_liquid = sum(1 for st in stats.values() if liquid(st, min_price, min_to))
        setups = Counter(x for st in stats.values() if liquid(st, min_price, min_to) for x in st.setups)
        as_of = max((st.as_of for st in stats.values()), default=None)
        self.result = {
            "trade_date": trade_date, "as_of": as_of, "universe": len(syms), "fetched": len(daily),
            "scanned": len(stats), "liquid": n_liquid, "setups": dict(setups.most_common()),
            "watchlist": self.watchlist, "seconds": round(time.time() - t0, 1),
            "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        WATCHLIST_PATH.write_text(json.dumps(self.result, indent=1))
        logger.info("Daily scan for %s: %d of %d charts, %d liquid, watchlist %d", trade_date, len(daily),
                    len(syms), n_liquid, len(self.watchlist))
        return self.result

    def _row(self, st: DailyStats, sc: float) -> dict:
        return {
            "symbol": st.symbol, "name": self.universe.name(st.symbol), "score": round(sc, 2),
            "close": st.close, "change_pct": st.change_pct, "atr_pct": round(st.atr_pct or 0, 2),
            "rsi14": round(st.rsi14, 1) if st.rsi14 is not None else None,
            "sma20": round(st.sma20, 2) if st.sma20 else None, "sma50": round(st.sma50, 2) if st.sma50 else None,
            "high20": round(st.high20, 2) if st.high20 else None,
            "turnover_cr": round(st.turnover20 / 1e7, 1), "setups": st.setups,
            "sector": (self.universe.get(st.symbol).industry if self.universe.get(st.symbol) else ""),
        }

    def stats_for(self, symbol: str):
        return self.stats.get(symbol)

    @staticmethod
    def score(st: DailyStats) -> float:
        return score(st)
