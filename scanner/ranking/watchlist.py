"""Turns a day's DailyStats into the watchlist the intraday strategies watch.

Liquidity first (price floor, average turnover), then a score: daily ATR percent (room to
move inside a day) lifted by the setups that the strategies act on. The same function
picks the watchlist live and in the backtest, so both trade the same kind of list.
"""
from typing import Dict, List, Tuple

from scanner.features.daily import DailyStats

SETUP_BONUS = {
    "breakout_20d": 0.30,
    "near_high20": 0.30,
    "bull_flag": 0.25,
    "reclaim_ma20": 0.20,
    "oversold_bounce": 0.20,
    "trend_up": 0.20,
}


def score(s: DailyStats) -> float:
    if not s.atr_pct:
        return 0.0
    return s.atr_pct * (1.0 + sum(SETUP_BONUS.get(x, 0.0) for x in s.setups))


def liquid(s: DailyStats, min_price: float, min_turnover_cr: float) -> bool:
    return s.close >= min_price and s.turnover20 >= min_turnover_cr * 1e7


def rank_watchlist(stats: Dict[str, DailyStats], size: int, min_price: float = 50,
                   min_turnover_cr: float = 15) -> List[Tuple[str, float]]:
    rows = [(sym, score(s)) for sym, s in stats.items() if liquid(s, min_price, min_turnover_cr)]
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows[:size]
