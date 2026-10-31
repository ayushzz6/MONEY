"""Strategy contract.

A strategy looks at one stock's closed 5-minute bars for one session (plus the daily
context from the sessions before it) and returns the FIRST bar of that session that
triggered its rule, or None. Live, the engine calls it after every bar close: a trigger
on the newest bar is a fresh signal, an older one was already seen. A backtest calls it
once per historical session with the whole day. Same code, same answer.

Strategies never size, check risk or place orders.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from market.calendar import hhmm
from market.candles.bars import Bar
from scanner.features.daily import DailyStats


@dataclass
class DayContext:
    symbol: str
    date: str
    entry_from: int                  # minutes after midnight, IST (no_entries_before)
    entry_until: int                 # no_entries_after
    allow_short: bool = False
    or_volume_avg: Optional[float] = None   # average opening-range volume of prior sessions


@dataclass
class Signal:
    symbol: str
    strategy: str
    side: str                 # "buy" or "sell"
    bar_index: int            # index into the session's bar list
    slot: int                 # 0 = the 09:15 bar
    time: str                 # the trigger bar's close time, HH:MM IST
    entry: float              # the trigger bar's close
    level: float              # the level it closed beyond
    stop_rule: str
    target_r: float
    stop_hint: Optional[float] = None    # a structural stop the strategy found (swing low...)
    target_hint: Optional[float] = None  # a target level the strategy found (else target_r x the risk)
    context: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def long(self) -> bool:
        return self.side == "buy"

    def to_dict(self) -> dict:
        return asdict(self)


class BaseStrategy(ABC):
    name = "base"
    version = "v1"
    label = "Base"
    live_trigger = "bars"     # "bars": the engine asks after every bar close; "board": the engine fires it itself

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})

    @property
    def enabled(self) -> bool:
        return bool(self.params.get("enabled", True))

    def wants(self, daily: Optional[DailyStats]) -> bool:
        """Cheap pre-filter on the daily context: can this stock trigger today at all?"""
        return True

    @abstractmethod
    def first_trigger(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[Signal]:
        ...

    def watch_state(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[dict]:
        """What the dashboard shows while waiting: {'state', 'level', 'distance_pct', 'note'}."""
        return None

    def _signal(self, ctx: DayContext, side: str, i: int, b: Bar, level: float, stop_hint: Optional[float] = None,
                context: Optional[dict] = None, reason: str = "") -> Signal:
        return Signal(
            symbol=ctx.symbol, strategy=self.name, side=side, bar_index=i, slot=b.slot,
            time=hhmm(b.close_minute), entry=round(b.c, 2), level=round(level, 2),
            stop_rule=str(self.params.get("stop", "or_mid")),
            target_r=float(self.params.get("target_r", 2.0)),
            stop_hint=stop_hint, context=context or {}, reason=reason,
        )


def in_window(b: Bar, ctx: DayContext) -> bool:
    return ctx.entry_from <= b.close_minute <= ctx.entry_until
