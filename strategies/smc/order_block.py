"""Order-block retest: the order-block board's setups, traded on paper (paper_only in configs/trading.json).

Live, the engine arms every board pick (the board's "pick" rule and at least `min_aligned` of its 4 checks) whose
current zone price has not come back to since it formed, and enters the moment the live price reaches the zone's
entry level Z: the board's limit order at Z, sent as a market order when the price gets there. Stop at the zone's
invalidation, target its 2nd liquidity level (T2; `target: T1` for the first; with none, `target_r` x the risk),
flat by the flat-by time, one trade per stock a day. Bearish zones (short sales) only when allow_short is true.
Zones, picks and alignment: market/orderblocks.py.

first_trigger() is the same rule on one stock's 5-minute bars, for the Time Machine's record and the morning
re-test. The board ranks every F&O stock together, so one stock's history cannot say whether it was a pick: there
it is the first return to the stock's current zone on the side its own strength points, filled at Z (or at the
bar's open when it opened past Z) inside that bar.
"""
from typing import List, Optional, Tuple

from market import orderblocks as OB
from market.calendar import OPEN_MIN, SESSION_BARS, hhmm, minute_of, to_ist
from market.candles.bars import Bar
from scanner.features.daily import DailyStats
from strategies.base import BaseStrategy, DayContext, Signal

SIDE = {1: "buy", -1: "sell"}


class OrderBlockRetest(BaseStrategy):
    name = "order_block"
    label = "Order-block retest"
    live_trigger = "board"

    def wants(self, daily: Optional[DailyStats]) -> bool:
        return bool(daily and daily.atr14 and daily.close)

    def _target(self, targets: List[float], side: int, entry: float, risk: float) -> float:
        ahead = [t for t in targets if (t - entry) * side > 0]          # nearest first
        if ahead:
            return ahead[0] if str(self.params.get("target", "T2")).upper() == "T1" else ahead[-1]
        return entry + side * float(self.params.get("target_r", 2.0)) * risk

    # ------------------------------------------------------------ live, from the board
    def armed(self, row: dict, ctx: DayContext) -> bool:
        """A board row that enters when its live price comes back to Z."""
        return self.why_not(row, ctx) is None

    def why_not(self, row: dict, ctx: DayContext) -> Optional[str]:
        """Why a board row is not armed, or None when it is."""
        z, side = row.get("zone"), row.get("side") or 0
        if self.params.get("picks_only", True) and row.get("rule") != "pick":
            return "only the board's picks are traded"
        need = int(self.params.get("min_aligned", 4))
        if (row.get("aligned") or 0) < need:
            return f"{row.get('aligned') or 0} of 4 checks aligned, needs {need}"
        if not z or not side:
            return "no order block yet"
        if z.get("status") == "broken":
            return "its zone broke"
        if z.get("entry_at"):
            return f"price already came back to Z at {to_ist(z['entry_at']).strftime('%H:%M')} (only the first return is traded)"
        if side < 0 and not ctx.allow_short:
            return "a bearish zone is a short sale, and short selling is off (allow_short)"
        return None

    def from_board(self, row: dict, price: float, bars: List[Bar], now) -> Tuple[Optional[Signal], Optional[str]]:
        """(signal, None): the live price is back at Z. (signal, why): it is already past the invalidation.
        (None, None): still away, or the candles closed since the board last looked touched or broke the zone."""
        z, side = row["zone"], row["side"]
        for b in bars:
            if b.start < z["known_at"]:
                continue
            if ((b.l <= z["z"]) if side > 0 else (b.h >= z["z"])) or \
                    ((b.c < z["bottom"]) if side > 0 else (b.c > z["top"])):
                return None, None
        if (price > z["z"]) if side > 0 else (price < z["z"]):
            return None, None
        risk = abs(price - z["invalid"])
        given = to_ist(z["known_at"]).strftime("%H:%M")
        sig = Signal(
            symbol=row["symbol"], strategy=self.name, side=SIDE[side], bar_index=max(0, len(bars) - 1),
            slot=max(0, min(SESSION_BARS - 1, (minute_of(now) - OPEN_MIN) // 5)), time=now.strftime("%H:%M"),
            entry=round(price, 2), level=z["z"], stop_rule=str(self.params.get("stop", "zone")),
            target_r=float(self.params.get("target_r", 2.0)), stop_hint=z["invalid"],
            target_hint=self._target(z.get("targets") or [], side, price, risk),
            context={"fill": "market at Z", "zone_bottom": z["bottom"], "zone_top": z["top"], "z": z["z"],
                     "invalid": z["invalid"], "targets": z.get("targets") or [], "given_at": given,
                     "board_rank": row.get("rank"), "aligned": row.get("aligned"), "strong": z.get("strong")},
            reason=f"price {price:.2f} came back to Z {z['z']:.2f} of the {'bullish' if side > 0 else 'bearish'} "
                   f"order block given at {given} (pick #{row.get('rank')}, {row.get('aligned')}/4 aligned)")
        if (price <= z["invalid"]) if side > 0 else (price >= z["invalid"]):
            return sig, f"price {price:.2f} was already past the invalidation {z['invalid']:.2f} when the engine saw it"
        return sig, None

    # ------------------------------------------------------------ one stock's history
    def first_trigger(self, bars: List[Bar], daily: Optional[DailyStats], ctx: DayContext) -> Optional[Signal]:
        if len(bars) < 2 or not self.wants(daily):
            return None
        ph, pl = daily.prev_high, daily.prev_low
        at = {b.start: i for i, b in enumerate(bars)}
        retests = [(at[z.entry_at], z) for z in OB.detect_zones(bars, ph, pl) if z.entry_at is not None]
        for j, zz in sorted(retests, key=lambda x: x[0]):
            side = zz.side
            if side < 0 and not ctx.allow_short:
                continue
            if not ctx.entry_from <= bars[j - 1].close_minute <= ctx.entry_until:     # the order rests from here
                continue
            st = OB.strength(bars[:j], daily.close, daily.atr14)
            if OB.event_risk(st) or (1 if st["score"] > 0 else -1 if st["score"] < 0 else 0) != side:
                continue
            cur = OB.current_setup(OB.detect_zones(bars[:j], ph, pl), side)
            if cur is None or cur.formed_at != zz.formed_at:
                continue
            b = bars[j]
            if (b.o <= zz.invalid) if side > 0 else (b.o >= zz.invalid):
                continue                                                           # opened through the stop
            fill = min(zz.z, b.o) if side > 0 else max(zz.z, b.o)
            given = hhmm(bars[at[zz.formed_at]].close_minute)
            return Signal(
                symbol=ctx.symbol, strategy=self.name, side=SIDE[side], bar_index=j, slot=b.slot,
                time=hhmm(bars[j - 1].close_minute), entry=round(fill, 2), level=round(zz.z, 2),
                stop_rule=str(self.params.get("stop", "zone")), target_r=float(self.params.get("target_r", 2.0)),
                stop_hint=zz.invalid, target_hint=self._target(cur.targets, side, fill, abs(fill - zz.invalid)),
                context={"fill": "limit", "zone_bottom": round(zz.bottom, 2), "zone_top": round(zz.top, 2),
                         "invalid": round(zz.invalid, 2), "given_at": given},
                reason=f"came back to Z {zz.z:.2f} of the order block given at {given}")
        return None
