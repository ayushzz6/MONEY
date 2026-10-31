"""Exit planner (trader-exit, on NSE ticks): the stop and the target, set before the entry.

Stop rules: or_mid (opening-range midpoint), or_low (the far side of the range), swing /
swing:N (the strategy's structural stop, or the extreme of the last N bars), zone (an
order block's invalidation level), pct:X, atr:K (K x the 14-bar ATR of the session up to
the entry), price:X.

The stop rounds AWAY from the entry to the stock's tick (down for a long). The target is
the strategy's target level when it names one, else entry + R x risk, rounded to the nearest tick. Refused, with the reason: a stop on the
wrong side, a risk under min_risk_pct of the price (too tight to survive the spread), a
risk over max_risk_pct, a target that does not clear the entry, or an entry at or after
the flat-by time. Nothing here is ever loosened to make a plan pass.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from market.calendar import OPEN_MIN, SESSION_BARS, parse_hhmm
from market.candles.bars import Bar, atr, round_to_tick
from strategies.base import Signal


@dataclass
class Plan:
    symbol: str
    strategy: str
    side: str
    date: str
    entry: float
    entry_time: str
    entry_slot: int
    stop: float
    stop_rule: str
    target: float
    target_r: float
    risk_per_share: float
    risk_pct: float
    reward_risk: float
    flat_by: str
    tick: float
    level: float
    context: dict = field(default_factory=dict)

    @property
    def long(self) -> bool:
        return self.side == "buy"

    def to_dict(self) -> dict:
        return asdict(self)


def _raw_stop(sig: Signal, bars: List[Bar]) -> Tuple[Optional[float], str]:
    rule = sig.stop_rule or "or_mid"
    c = sig.context
    long = sig.long
    if rule == "or_mid":
        if "or_high" not in c:
            return None, "or_mid needs an opening range (this setup has none): name a stop rule"
        return (c["or_high"] + c["or_low"]) / 2.0, "opening-range midpoint"
    if rule in ("or_low", "or_far"):
        if "or_high" not in c:
            return None, f"{rule} needs an opening range"
        return (c["or_low"] if long else c["or_high"]), "far side of the opening range"
    if rule.startswith("swing"):
        if sig.stop_hint is not None:
            return sig.stop_hint, "structural swing"
        n = int(rule.split(":")[1]) if ":" in rule else 3
        rows = bars[max(0, sig.bar_index - n + 1):sig.bar_index + 1]
        return (min(b.l for b in rows) if long else max(b.h for b in rows)), f"extreme of the last {n} bars"
    if rule == "zone":
        if sig.stop_hint is None:
            return None, "zone stop needs the order block's invalidation level"
        return sig.stop_hint, "order-block invalidation"
    if rule.startswith("pct:"):
        x = float(rule.split(":")[1]) / 100.0
        return (sig.entry * (1 - x) if long else sig.entry * (1 + x)), f"{rule[4:]}% from the entry"
    if rule.startswith("atr:"):
        k = float(rule.split(":")[1])
        a = atr(bars[:sig.bar_index + 1], 14)
        if not a:
            return None, "no bars for an ATR stop"
        return (sig.entry - k * a if long else sig.entry + k * a), f"{k:g} x ATR(14) of {a:.2f}"
    if rule.startswith("price:"):
        return float(rule.split(":")[1]), "your price"
    return None, f"unknown stop rule {rule!r}"


def plan_exit(sig: Signal, bars: List[Bar], tick: float, cfg, date: str) -> Tuple[Optional[Plan], Optional[str]]:
    """(plan, None) or (None, refusal reason)."""
    long = sig.long
    raw, rule_text = _raw_stop(sig, bars)
    if raw is None:
        return None, rule_text
    stop = round_to_tick(raw, tick, "down" if long else "up")
    entry = sig.entry
    risk = (entry - stop) if long else (stop - entry)
    if risk <= 0:
        return None, f"stop {stop:.2f} is not {'below' if long else 'above'} the entry {entry:.2f}"
    risk_pct = 100.0 * risk / entry
    min_pct, max_pct = float(cfg.get("min_risk_pct", 0.1)), float(cfg.get("max_risk_pct", 3.0))
    if risk_pct < min_pct:
        return None, f"stop {stop:.2f} is {risk_pct:.2f}% from the entry: under {min_pct}%, too tight to survive the spread"
    if risk_pct > max_pct:
        return None, f"stop {stop:.2f} is {risk_pct:.2f}% from the entry: over the {max_pct}% limit"
    r = float(sig.target_r)
    raw_target = sig.target_hint if sig.target_hint is not None else (entry + r * risk if long else entry - r * risk)
    target = round_to_tick(raw_target, tick, "nearest")
    reward = (target - entry) if long else (entry - target)
    if reward <= 0:
        return None, "the target does not clear the entry"
    if sig.target_hint is not None:
        r = round(reward / risk, 2)
    flat_by = str(cfg.get("flat_by", "15:10"))
    entry_min = parse_hhmm(sig.time)
    if entry_min >= parse_hhmm(flat_by):
        return None, f"entry at {sig.time} is at or after the flat-by time {flat_by}"
    return Plan(
        symbol=sig.symbol, strategy=sig.strategy, side=sig.side, date=date,
        entry=entry, entry_time=sig.time, entry_slot=sig.slot,
        stop=stop, stop_rule=f"{sig.stop_rule}: {rule_text}", target=target,
        target_r=r, risk_per_share=round(risk, 4), risk_pct=round(risk_pct, 3),
        reward_risk=round(reward / risk, 2), flat_by=flat_by, tick=tick, level=sig.level,
        context=dict(sig.context, reason=sig.reason),
    ), None


def flat_by_slot(flat_by: str) -> int:
    """Index of the last bar that closes at or before the flat-by time."""
    m = parse_hhmm(flat_by)
    return max(0, min(SESSION_BARS - 1, (m - OPEN_MIN) // 5 - 1))
