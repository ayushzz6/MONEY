"""Paper broker: simulated fills against the live prices and closed bars.

- MARKET fills at the next price with slippage (a buy pays up, a sell gets less).
- LIMIT fills at its price when the price trades through it, or at the bar's open if the
  bar opened beyond it.
- SL-M triggers when the price reaches the trigger and fills there with slippage, or at
  the bar's open with slippage if the bar gapped through it.
- Orders sharing an `oco` id cancel each other when one fills (the bracket's stop and
  target). Within one bar the stop is checked first, as in the backtest.
"""
import itertools
import threading
import time
from typing import Dict, Optional

from brokers.base import Broker, BrokerError, OrderState
from portfolio.pnl import slip


class PaperBroker(Broker):
    name = "paper"
    paper = True
    native_oco = True

    def __init__(self, market_state, slippage_bps: float = 5.0):
        self.state = market_state
        self.bps = float(slippage_bps)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.orders: Dict[str, dict] = {}

    def place(self, symbol, side, qty, order_type="MARKET", price=None, trigger=None, tag="", oco=None) -> str:
        if side not in ("buy", "sell") or qty <= 0:
            raise BrokerError(f"bad order: {side} {qty}")
        if order_type == "LIMIT" and not price:
            raise BrokerError("a LIMIT order needs a price")
        if order_type == "SL-M" and not trigger:
            raise BrokerError("an SL-M order needs a trigger price")
        oid = f"P{int(time.time())}-{next(self._ids)}"
        with self._lock:
            self.orders[oid] = {"id": oid, "symbol": symbol, "side": side, "qty": int(qty), "type": order_type,
                                "price": price, "trigger": trigger, "tag": tag, "oco": oco, "status": "open",
                                "filled_qty": 0, "avg_price": None, "created": time.time(), "message": ""}
        if order_type == "MARKET":
            p, _ = self.state.price(symbol)
            if p:
                self._fill(oid, slip(p, self.bps, adverse_for_buy=(side == "buy")))
        return oid

    def cancel(self, order_id):
        with self._lock:
            o = self.orders.get(order_id)
            if o and o["status"] == "open":
                o["status"] = "cancelled"

    def status(self, order_id) -> OrderState:
        with self._lock:
            o = self.orders.get(order_id)
            if not o:
                return OrderState(order_id, "rejected", message="unknown order")
            return OrderState(order_id, o["status"], o["filled_qty"], o["avg_price"], o["message"])

    def _fill(self, oid, price):
        with self._lock:
            o = self.orders[oid]
            if o["status"] != "open":
                return
            o.update(status="filled", filled_qty=o["qty"], avg_price=round(price, 2), filled_at=time.time())
            if o["oco"]:
                for other in self.orders.values():
                    if other["oco"] == o["oco"] and other["id"] != oid and other["status"] == "open":
                        other["status"] = "cancelled"
                        other["message"] = "one-cancels-other"

    def _open_for(self, symbol):
        with self._lock:
            rows = [o for o in self.orders.values() if o["symbol"] == symbol and o["status"] == "open"]
        # stops before targets, so a bar that touches both fills the stop
        return sorted(rows, key=lambda o: 0 if o["type"] == "SL-M" else 1)

    def on_price(self, symbol, price, bar=None):
        for o in self._open_for(symbol):
            if self.status(o["id"]).status != "open":
                continue
            buy = o["side"] == "buy"
            if bar is not None and bar.start + 300 <= o["created"]:
                continue        # that bar ended before the order existed
            if o["type"] == "MARKET":
                self._fill(o["id"], slip(price, self.bps, adverse_for_buy=buy))
            elif o["type"] == "SL-M":
                t = o["trigger"]
                if bar is not None:
                    hit = bar.h >= t if buy else bar.l <= t
                    gap = (bar.o > t) if buy else (bar.o < t)
                    if hit:
                        self._fill(o["id"], slip(bar.o if gap else t, self.bps, adverse_for_buy=buy))
                elif (price >= t) if buy else (price <= t):
                    self._fill(o["id"], slip(price, self.bps, adverse_for_buy=buy))
            elif o["type"] == "LIMIT":
                lp = o["price"]
                if bar is not None:
                    hit = bar.l <= lp if buy else bar.h >= lp
                    better = (bar.o < lp) if buy else (bar.o > lp)
                    if hit:
                        self._fill(o["id"], bar.o if better else lp)
                elif (price <= lp) if buy else (price >= lp):
                    self._fill(o["id"], lp)

    def open_orders(self):
        with self._lock:
            return [dict(o) for o in self.orders.values() if o["status"] == "open"]
