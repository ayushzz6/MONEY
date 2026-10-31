"""Broker interface the order manager talks to. PaperBroker and AngelOneBroker implement it."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, Optional


class BrokerError(Exception):
    pass


@dataclass
class OrderState:
    order_id: str
    status: str                 # open, filled, rejected, cancelled
    filled_qty: int = 0
    avg_price: Optional[float] = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker(ABC):
    name = "broker"
    paper = True
    # True when the broker can hold the target as a resting order next to the stop and
    # cancel one when the other fills (one-cancels-other). Otherwise the order manager
    # watches the target itself and exits at market when it is reached.
    native_oco = False

    @abstractmethod
    def place(self, symbol: str, side: str, qty: int, order_type: str = "MARKET",
              price: Optional[float] = None, trigger: Optional[float] = None,
              tag: str = "", oco: Optional[str] = None) -> str:
        """order_type: MARKET, LIMIT, SL-M. Returns the broker's order id; raises BrokerError."""

    @abstractmethod
    def cancel(self, order_id: str):
        ...

    @abstractmethod
    def status(self, order_id: str) -> OrderState:
        ...

    def refresh(self):
        """Called once per engine loop before status() calls (a live broker reads its order book)."""

    def funds(self) -> Optional[float]:
        return None

    def positions(self) -> Dict[str, int]:
        return {}

    def on_price(self, symbol: str, price: float, bar=None):
        """Paper brokers fill against prices; live brokers ignore this."""
