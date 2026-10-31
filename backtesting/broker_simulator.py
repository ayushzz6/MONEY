"""How a bracket (entry, stop, target, flat-by) ends, bar by bar. One implementation, used by
the backtest, the Time Machine's history and the live paper broker.

Per bar: the stop is checked first, then the target, so a bar that touches both counts
as the stop (conservative). A bar that OPENS through the stop fills at its open (a gap
fills worse than the stop); a bar that opens through the target fills at its open (a
resting limit fills better). Anything still open is closed at the flat-by bar's close.
"""
from typing import List, Optional, Tuple

from market.candles.bars import Bar


def bar_exit(long: bool, stop: float, target: float, b: Bar) -> Optional[Tuple[str, float]]:
    if long:
        if b.l <= stop:
            return "stop", (b.o if b.o < stop else stop)
        if b.h >= target:
            return "target", (b.o if b.o > target else target)
    else:
        if b.h >= stop:
            return "stop", (b.o if b.o > stop else stop)
        if b.l <= target:
            return "target", (b.o if b.o < target else target)
    return None


def tick_exit(long: bool, stop: float, target: float, ltp: float) -> Optional[Tuple[str, float]]:
    if long:
        if ltp <= stop:
            return "stop", ltp
        if ltp >= target:
            return "target", target
    else:
        if ltp >= stop:
            return "stop", ltp
        if ltp <= target:
            return "target", target
    return None


def simulate_bracket(long: bool, stop: float, target: float, bars_after: List[Bar], flat_slot: int) -> dict:
    """bars_after: the session's bars after the entry bar. Returns outcome (stop, target or
    flat), the exit price (before slippage) and the exit bar's slot."""
    last = None
    for b in bars_after:
        if b.slot > flat_slot:
            break
        hit = bar_exit(long, stop, target, b)
        if hit:
            return {"outcome": hit[0], "exit": hit[1], "exit_slot": b.slot}
        last = b
    if last is None:
        return {"outcome": "flat", "exit": None, "exit_slot": None}
    return {"outcome": "flat", "exit": last.c, "exit_slot": last.slot}
