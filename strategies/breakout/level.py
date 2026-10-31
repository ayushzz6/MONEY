"""20-day high breakout: trader-scanner's breakout_20d, traded intraday on the level.

Only for stocks that closed yesterday within `near_pct` of their 20-session high without
breaking it. The level is that 20-session high. The trigger is the first bar in the
entry window that closes above it. A gap that opens more than `max_gap_pct` above the
level is skipped: the move already happened before the bell.
"""
from typing import List, Optional

from market.candles.bars import Bar
from scanner.features.daily import DailyStats
from strategies.base import BaseStrategy, DayContext, Signal, in_window


class LevelBreakout(BaseStrategy):
    name = "level_breakout"
    label = "20-day high breakout"

    def wants(self, daily: Optional[DailyStats]) -> bool:
        if not daily or not daily.high20:
            return False
        near = float(self.params.get("near_pct", 3.0))
        return daily.high20 * (1 - near / 100.0) <= daily.close < daily.high20

    def first_trigger(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[Signal]:
        if not bars or not self.wants(daily):
            return None
        level = daily.high20
        gap = 100.0 * (bars[0].o / level - 1.0)
        if gap > float(self.params.get("max_gap_pct", 1.5)):
            return None
        for i, b in enumerate(bars):
            if b.close_minute > ctx.entry_until:
                break
            if not in_window(b, ctx):
                continue
            if b.c > level:
                swing = self._swing(bars, i)
                return self._signal(ctx, "buy", i, b, level, stop_hint=swing,
                                    context={"high20": round(level, 2), "gap_pct": round(gap, 2)},
                                    reason=f"closed {100 * (b.c / level - 1):.2f}% above the 20-day high {level:.2f}")
        return None

    def _swing(self, bars: List[Bar], i: int) -> float:
        rule = str(self.params.get("stop", "swing:3"))
        n = int(rule.split(":")[1]) if rule.startswith("swing:") else 3
        return min(b.l for b in bars[max(0, i - n + 1):i + 1])

    def watch_state(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[dict]:
        if not self.wants(daily) or not bars:
            return None
        dist = 100.0 * (bars[-1].c / daily.high20 - 1.0)
        return {"state": "waiting", "level": round(daily.high20, 2), "distance_pct": round(dist, 2),
                "note": f"20-day high {daily.high20:.2f}"}
