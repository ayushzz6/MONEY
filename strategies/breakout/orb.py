"""Opening-range breakout (trader-entry's rule, on NSE hours).

The opening range is the high and low of the first `or_minutes` (15: the 09:15, 09:20
and 09:25 bars). The trigger is the first completed bar inside the entry window that
CLOSES above the range high (long) or below the range low (short, when allowed). A wick
through the level is not a trigger, and a close exactly at it is not beyond it.

Filters: the range must be between min_or_pct and max_or_pct of the price, and the
range's volume must be at least min_or_rvol times its average over prior sessions
("in play"). An optional min_volume_x needs the trigger bar's volume at that multiple of
the range bars' average.
"""
from typing import List, Optional

from market.candles.bars import Bar, opening_range
from scanner.features.daily import DailyStats
from strategies.base import BaseStrategy, DayContext, Signal, in_window


class OpeningRangeBreakout(BaseStrategy):
    name = "orb"
    label = "Opening-range breakout"

    def _range(self, bars: List[Bar], ctx: DayContext):
        n = int(self.params.get("or_minutes", 15)) // 5
        orr = opening_range(bars, n)
        if orr is None:
            return None, "range forming"
        oh, ol, ovol = orr
        mid = (oh + ol) / 2.0
        width = 100.0 * (oh - ol) / mid if mid else 0.0
        if width < float(self.params.get("min_or_pct", 0.0)):
            return None, f"range {width:.2f}% is narrower than {self.params.get('min_or_pct')}%"
        if width > float(self.params.get("max_or_pct", 100.0)):
            return None, f"range {width:.2f}% is wider than {self.params.get('max_or_pct')}%"
        rvol = (ovol / ctx.or_volume_avg) if ctx.or_volume_avg else None
        need = float(self.params.get("min_or_rvol", 0) or 0)
        if need and rvol is not None and rvol < need:
            return None, f"not in play: opening-range volume {rvol:.2f}x its average (needs {need:g}x)"
        return {"n": n, "or_high": oh, "or_low": ol, "or_volume": ovol, "or_width_pct": round(width, 2),
                "or_rvol": round(rvol, 2) if rvol is not None else None}, None

    def first_trigger(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[Signal]:
        rng, _ = self._range(bars, ctx)
        if rng is None:
            return None
        n, oh, ol = rng["n"], rng["or_high"], rng["or_low"]
        vol_x = float(self.params.get("min_volume_x", 0) or 0)
        bar_avg = rng["or_volume"] / max(1, n - 1)
        shorts = ctx.allow_short and bool(self.params.get("short", True))
        for i, b in enumerate(bars):
            if b.slot < n or not in_window(b, ctx):
                if b.close_minute > ctx.entry_until:
                    break
                continue
            if vol_x and bar_avg > 0 and b.v < vol_x * bar_avg:
                continue
            ctxd = dict(rng, trigger_volume_x=round(b.v / bar_avg, 2) if bar_avg else None)
            if b.c > oh:
                return self._signal(ctx, "buy", i, b, oh, context=ctxd,
                                    reason=f"closed {100 * (b.c / oh - 1):.2f}% above the {n * 5}-minute range high")
            if shorts and b.c < ol:
                return self._signal(ctx, "sell", i, b, ol, context=ctxd,
                                    reason=f"closed {100 * (1 - b.c / ol):.2f}% below the {n * 5}-minute range low")
        return None

    def watch_state(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[dict]:
        rng, why = self._range(bars, ctx)
        if rng is None:
            return {"state": "forming" if why == "range forming" else "filtered", "note": why}
        last = bars[-1]
        dist = 100.0 * (last.c / rng["or_high"] - 1.0)
        return {"state": "waiting", "level": round(rng["or_high"], 2), "low": round(rng["or_low"], 2),
                "distance_pct": round(dist, 2), "or_rvol": rng["or_rvol"],
                "note": f"range {rng['or_low']:.2f} to {rng['or_high']:.2f}"}
