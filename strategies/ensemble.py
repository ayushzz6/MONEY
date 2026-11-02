"""Strategy registry: builds every strategy from configs/trading.json -> strategies.*"""
from typing import Dict, List

from strategies.base import BaseStrategy
from strategies.breakout.level import LevelBreakout
from strategies.breakout.orb import OpeningRangeBreakout
from strategies.smc.order_block import OrderBlockFirstStage, OrderBlockRetest
from strategies.vwap.reclaim import VwapReclaim

REGISTRY = {cls.name: cls for cls in (OpeningRangeBreakout, LevelBreakout, VwapReclaim, OrderBlockRetest, OrderBlockFirstStage)}


def build_strategies(settings) -> List[BaseStrategy]:
    out = []
    for name, cls in REGISTRY.items():
        params = settings.strategy(name)
        if params:
            out.append(cls(params))
    return out


def by_name(strategies: List[BaseStrategy]) -> Dict[str, BaseStrategy]:
    return {s.name: s for s in strategies}
