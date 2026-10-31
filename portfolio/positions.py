"""Trades (one bracket each: entry, stop, target, flat-by) and today's book of them."""
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from portfolio.pnl import gross_pnl, round_trip_charges

OPEN_STATES = ("ENTRY_PLACED", "OPEN", "EXITING")
PENDING_STATES = ("PENDING_APPROVAL", "APPROVED")
FINAL_STATES = ("CLOSED", "REJECTED", "EXPIRED", "CANCELLED", "ERROR")


@dataclass
class Trade:
    symbol: str
    strategy: str
    side: str
    qty: int
    date: str
    mode: str
    plan: dict
    stop: float
    target: float
    max_loss: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "PENDING_APPROVAL"
    created_at: float = field(default_factory=time.time)
    approved_at: Optional[float] = None
    opened_at: Optional[float] = None
    closed_at: Optional[float] = None
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    charges: Optional[float] = None
    net_pnl: Optional[float] = None
    R: Optional[float] = None
    verdict: dict = field(default_factory=dict)
    size: dict = field(default_factory=dict)
    tm: dict = field(default_factory=dict)
    news: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @property
    def long(self) -> bool:
        return self.side == "buy"

    @property
    def exit_side(self) -> str:
        return "sell" if self.long else "buy"

    def log(self, text: str):
        self.events.append({"t": time.time(), "text": text})

    def unrealized(self, ltp: Optional[float], costs: dict) -> Optional[float]:
        if self.status not in ("OPEN", "EXITING") or ltp is None or self.entry_price is None:
            return None
        return round(gross_pnl(self.entry_price, ltp, self.qty, self.long)
                     - round_trip_charges(self.entry_price, ltp, self.qty, self.long, costs), 2)

    def open_risk(self) -> float:
        """Loss at the stop from the fill (or the plan, before the fill), charges included."""
        if self.status in FINAL_STATES:
            return 0.0
        return float(self.max_loss)

    def close(self, price: float, reason: str, costs: dict):
        self.exit_price = round(price, 2)
        self.exit_reason = reason
        self.closed_at = time.time()
        self.status = "CLOSED"
        entry = self.entry_price or self.plan.get("entry")
        self.gross_pnl = round(gross_pnl(entry, price, self.qty, self.long), 2)
        self.charges = round(round_trip_charges(entry, price, self.qty, self.long, costs), 2)
        self.net_pnl = round(self.gross_pnl - self.charges, 2)
        risk = abs(entry - self.plan.get("stop", entry)) * self.qty
        self.R = round(self.net_pnl / risk, 3) if risk else None
        self.log(f"closed at {price:.2f} ({reason}): net Rs {self.net_pnl:,.2f} after Rs {self.charges:,.2f} charges")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["long"] = self.long
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


class TradeBook:
    def __init__(self):
        self.trades: Dict[str, Trade] = {}

    def add(self, t: Trade):
        self.trades[t.id] = t

    def get(self, trade_id: str) -> Optional[Trade]:
        return self.trades.get(trade_id)

    def today(self, date: str) -> List[Trade]:
        return [t for t in self.trades.values() if t.date == date]

    def open(self) -> List[Trade]:
        return [t for t in self.trades.values() if t.status in OPEN_STATES]

    def pending(self) -> List[Trade]:
        return [t for t in self.trades.values() if t.status in PENDING_STATES]

    def realized(self, date: str) -> float:
        return round(sum(t.net_pnl or 0.0 for t in self.today(date) if t.status == "CLOSED"), 2)

    def entries_today(self, date: str) -> List[Trade]:
        """Entries that reached the broker today (placed, open or closed)."""
        return [t for t in self.today(date) if t.entry_order_id]

    def committed_value(self) -> float:
        return sum((t.entry_price or t.plan.get("entry", 0)) * t.qty
                   for t in self.trades.values() if t.status in OPEN_STATES + PENDING_STATES)
