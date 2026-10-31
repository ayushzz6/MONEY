"""Angel One SmartAPI order execution (LIVE). Intraday product, NSE cash segment.

- Entries and exits are MARKET orders; the protective stop is a STOPLOSS_MARKET order held
  at the broker, so the position stays protected even if this program stops.
- Angel One has no one-cancels-other for intraday orders, so the target is watched by the
  order manager on live quotes, which cancels the stop and exits at market.
- Angel One now ties order APIs to a registered static IP (SEBI's retail algo rules);
  placing orders from an unregistered IP is rejected.
"""
import logging
import threading
import time
from typing import Dict, Optional

from brokers.base import Broker, BrokerError, OrderState

logger = logging.getLogger("AngelOneBroker")

ORDER_TYPES = {"MARKET": "MARKET", "LIMIT": "LIMIT", "SL-M": "STOPLOSS_MARKET", "SL": "STOPLOSS_LIMIT"}
STATUS_MAP = {"complete": "filled", "rejected": "rejected", "cancelled": "cancelled"}


class AngelOneBroker(Broker):
    name = "angelone"
    paper = False
    native_oco = False

    def __init__(self, smart_api, universe):
        self.api = smart_api
        self.universe = universe
        self._lock = threading.Lock()
        self._book: Dict[str, dict] = {}
        self._book_at = 0.0
        self._variety: Dict[str, str] = {}

    def place(self, symbol, side, qty, order_type="MARKET", price=None, trigger=None, tag="", oco=None) -> str:
        inst = self.universe.get(symbol)
        if inst is None or not inst.angel_token:
            raise BrokerError(f"no Angel One token for {symbol}")
        variety = "STOPLOSS" if order_type in ("SL-M", "SL") else "NORMAL"
        params = {
            "variety": variety,
            "tradingsymbol": inst.trading_symbol,
            "symboltoken": inst.angel_token,
            "transactiontype": "BUY" if side == "buy" else "SELL",
            "exchange": "NSE",
            "ordertype": ORDER_TYPES[order_type],
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": f"{price:.2f}" if price else "0",
            "triggerprice": f"{trigger:.2f}" if trigger else "0",
            "quantity": str(int(qty)),
            "ordertag": (tag or "moneyv2")[:20],
        }
        try:
            resp = self.api.placeOrderFullResponse(params)
        except Exception as e:
            raise BrokerError(f"placeOrder failed: {e}") from e
        data = (resp or {}).get("data") or {}
        oid = data.get("orderid")
        if not (resp or {}).get("status") or not oid:
            raise BrokerError(f"order rejected by Angel One: {(resp or {}).get('message')} "
                              f"({(resp or {}).get('errorcode')})")
        with self._lock:
            self._variety[oid] = variety
        logger.info("Placed %s %s %s x%d -> %s", order_type, side, symbol, qty, oid)
        return oid

    def cancel(self, order_id):
        variety = self._variety.get(order_id)
        if variety is None:          # after a restart: read it from the order book
            self._book_at = 0.0
            self.refresh()
            variety = (self._book.get(order_id) or {}).get("variety") or "NORMAL"
        try:
            resp = self.api.cancelOrder(order_id, variety)
        except Exception as e:
            raise BrokerError(f"cancel failed: {e}") from e
        if not (resp or {}).get("status"):
            st = self.status(order_id)
            if st.status == "open":
                raise BrokerError(f"cancel refused: {(resp or {}).get('message')}")

    def refresh(self):
        """Read the order book at most once a second."""
        if time.time() - self._book_at < 1.0:
            return
        try:
            resp = self.api.orderBook()
        except Exception as e:
            logger.warning("orderBook failed: %s", e)
            return
        rows = (resp or {}).get("data") or []
        with self._lock:
            self._book = {r.get("orderid"): r for r in rows if r.get("orderid")}
            self._book_at = time.time()

    def status(self, order_id) -> OrderState:
        with self._lock:
            r = self._book.get(order_id)
        if r is None:
            return OrderState(order_id, "open", message="not in the order book yet")
        raw = str(r.get("status") or r.get("orderstatus") or "").lower()
        st = STATUS_MAP.get(raw, "open")
        filled = int(float(r.get("filledshares") or 0))
        avg = float(r.get("averageprice") or 0) or None
        return OrderState(order_id, st, filled, avg, r.get("text") or raw)

    def funds(self) -> Optional[float]:
        try:
            data = (self.api.rmsLimit() or {}).get("data") or {}
            return float(data.get("availablecash") or data.get("net") or 0)
        except Exception as e:
            logger.warning("rmsLimit failed: %s", e)
            return None

    def positions(self) -> Dict[str, int]:
        try:
            rows = (self.api.position() or {}).get("data") or []
        except Exception as e:
            logger.warning("position failed: %s", e)
            return {}
        out = {}
        for r in rows:
            if r.get("producttype") != "INTRADAY" or r.get("exchange") != "NSE":
                continue
            sym = str(r.get("tradingsymbol", "")).replace("-EQ", "")
            qty = int(float(r.get("netqty") or 0))
            if qty:
                out[sym] = qty
        return out
