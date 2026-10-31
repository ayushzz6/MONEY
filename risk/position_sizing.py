"""Position sizing (trader-budget, in rupees): the whole shares that fit inside four limits.

  max loss      max_loss_per_trade_inr / (risk a share + round-trip charges a share)
  position cap  capital x max_position_pct / entry
  max order     max_order_inr / entry
  cash          (cash - buffer) x intraday_leverage / entry

The size is the smallest; the result says which limit decided it. A stop too far for
the max loss is a refusal (zero shares), never a smaller stop.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from portfolio.pnl import round_trip_charges


@dataclass
class SizeResult:
    qty: int
    binding: Optional[str]
    limits: Dict[str, int] = field(default_factory=dict)
    entry: float = 0.0
    value: float = 0.0
    max_loss: float = 0.0          # at the stop, charges included
    charges_at_stop: float = 0.0
    refusal: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def size_position(entry: float, stop: float, long: bool, cfg, cash_available: float) -> SizeResult:
    risk = (entry - stop) if long else (stop - entry)
    if risk <= 0:
        return SizeResult(0, None, entry=entry, refusal="the stop is not on the losing side of the entry")
    costs = cfg.get("costs", {}) or {}
    capital = float(cfg.get("capital_inr", 0))
    max_loss = float(cfg.get("max_loss_per_trade_inr", 0))
    buffer = capital * float(cfg.get("cash_buffer_pct", 0)) / 100.0
    lev = max(1.0, float(cfg.get("intraday_leverage", 1) or 1))
    # charges per share for a round trip that ends at the stop (they scale with price)
    per_share_cost = round_trip_charges(entry, stop, 100, long, costs) / 100.0
    limits = {
        "max loss": int(max_loss // (risk + per_share_cost)) if max_loss > 0 else 0,
        "position cap": int(capital * float(cfg.get("max_position_pct", 100)) / 100.0 // entry),
        "max order": int(float(cfg.get("max_order_inr", 0)) // entry),
        "cash": int(max(0.0, (cash_available - buffer) * lev) // entry),
    }
    binding = min(limits, key=lambda k: limits[k])
    qty = max(0, limits[binding])
    res = SizeResult(qty=qty, binding=binding, limits=limits, entry=entry, value=round(qty * entry, 2))
    if qty <= 0:
        if binding == "max loss":
            res.refusal = (f"one share risks Rs {risk + per_share_cost:.2f} with charges, more than the "
                           f"Rs {max_loss:.0f} max loss per trade")
        else:
            res.refusal = f"zero shares: bound by {binding}"
        return res
    # the per-share estimate above can undershoot (brokerage is a flat fee up to a cap), so
    # check the real charges for this quantity and step down until the loss fits
    charges = round_trip_charges(entry, stop, qty, long, costs)
    while qty > 0 and max_loss > 0 and qty * risk + charges > max_loss:
        qty -= 1
        charges = round_trip_charges(entry, stop, qty, long, costs) if qty else 0.0
    if qty != res.qty:
        res.qty, res.binding = qty, "max loss"
        res.limits["max loss"] = qty
        res.value = round(qty * entry, 2)
        if qty <= 0:
            res.refusal = f"one share risks more than the Rs {max_loss:.0f} max loss per trade with charges"
            return res
    res.charges_at_stop = round(charges, 2)
    res.max_loss = round(qty * risk + charges, 2)
    min_order = float(cfg.get("min_order_inr", 0))
    if res.value < min_order:
        res.refusal = f"order Rs {res.value:,.0f} is under the Rs {min_order:,.0f} minimum"
    return res
