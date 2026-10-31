"""VWAP reclaim: a pullback inside a daily uptrend (or, for shorts, a pop inside a downtrend).

Long: the stock is in a daily uptrend (close above its 20-day average, which is above the
50-day). After the opening range, a bar closes below the session VWAP (the dip). The
trigger is the first later bar, from `not_before` on and inside the entry window, that
closes back above VWAP and above the previous bar's high, within two bars of the last
close under VWAP (a reclaim, not a drift hours later). The structural stop is the lowest
low of the dip. Short is the mirror image in a daily downtrend.
"""
from typing import List, Optional

from market.calendar import parse_hhmm
from market.candles.bars import Bar, vwap_series
from scanner.features.daily import DailyStats
from strategies.base import BaseStrategy, DayContext, Signal, in_window


class VwapReclaim(BaseStrategy):
    name = "vwap_reclaim"
    label = "VWAP reclaim (trend pullback)"

    def wants(self, daily: Optional[DailyStats]) -> bool:
        return bool(daily and (daily.trend_up or daily.trend_down))

    def first_trigger(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[Signal]:
        if not bars or not daily:
            return None
        long_ok = daily.trend_up
        short_ok = daily.trend_down and ctx.allow_short
        if not (long_ok or short_ok):
            return None
        not_before = parse_hhmm(self.params.get("not_before", "10:00"))
        vw = vwap_series(bars)
        dipped, extreme, last_dip = False, None, -10
        for i, b in enumerate(bars):
            if b.slot < 3:
                continue
            if b.close_minute > ctx.entry_until:
                break
            below = b.c < vw[i]
            if long_ok:
                if below:
                    extreme = b.l if i - last_dip > 2 else min(extreme, b.l)   # a new dip starts fresh
                    dipped, last_dip = True, i
                    continue
                if dipped and i - last_dip <= 2 and b.c > bars[i - 1].h and b.close_minute >= not_before \
                        and in_window(b, ctx):
                    stop = min(extreme, b.l)
                    return self._signal(ctx, "buy", i, b, vw[i], stop_hint=stop,
                                        context={"vwap": round(vw[i], 2), "dip_low": round(extreme, 2)},
                                        reason=f"reclaimed VWAP {vw[i]:.2f} after a dip to {extreme:.2f} in a daily uptrend")
            else:
                if not below:
                    extreme = b.h if i - last_dip > 2 else max(extreme, b.h)
                    dipped, last_dip = True, i
                    continue
                if dipped and i - last_dip <= 2 and b.c < bars[i - 1].l and b.close_minute >= not_before \
                        and in_window(b, ctx):
                    stop = max(extreme, b.h)
                    return self._signal(ctx, "sell", i, b, vw[i], stop_hint=stop,
                                        context={"vwap": round(vw[i], 2), "pop_high": round(extreme, 2)},
                                        reason=f"lost VWAP {vw[i]:.2f} after a pop to {extreme:.2f} in a daily downtrend")
        return None

    def watch_state(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[dict]:
        if not bars or not self.wants(daily):
            return None
        vw = vwap_series(bars)[-1]
        dist = 100.0 * (bars[-1].c / vw - 1.0)
        trend = "uptrend" if daily.trend_up else "downtrend"
        return {"state": "waiting", "level": round(vw, 2), "distance_pct": round(dist, 2),
                "note": f"daily {trend}; VWAP {vw:.2f}"}
