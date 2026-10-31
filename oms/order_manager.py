"""Order manager: runs each trade's bracket from approval to flat.

PAPER: an entry that passed the risk gate is placed at once.
LIVE: it waits in PENDING_APPROVAL for your Approve on the dashboard (it expires after
approval_timeout_seconds), and the risk gate runs again at the moment you approve.

After the entry fills, the stop goes to the broker as an SL-M order. With a broker that
supports one-cancels-other (the paper broker) the target rests there too; with Angel One
the target is watched on live quotes and the position is closed at market when it is
reached. Exits are never blocked: target, stop, flat-by time, kill switch, manual.
"""
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from brokers.base import Broker, BrokerError
from market.calendar import minute_of, parse_hhmm
from oms.state_machine import transition
from portfolio.positions import FINAL_STATES, Trade, TradeBook
from portfolio.pnl import round_trip_charges

logger = logging.getLogger("OMS")

ENTRY_FILL_TIMEOUT_S = 60


class OrderManager:
    def __init__(self, settings, broker: Broker, book: TradeBook, journal, market_state,
                 on_event: Optional[Callable[[str, str, Optional[Trade]], None]] = None):
        self.settings = settings
        self.broker = broker
        self.book = book
        self.journal = journal
        self.state = market_state
        self.on_event = on_event or (lambda kind, text, trade=None: None)
        self._lock = threading.RLock()
        self._exit_stage = {}       # trade id -> (reason, stage)
        self._cancel_since = {}     # trade id -> when we started cancelling its stop/target

    # ---------------------------------------------------------------- helpers
    def _save(self, t: Trade):
        self.journal.save_trade(t.to_dict())

    def _emit(self, kind, text, t=None):
        try:
            self.on_event(kind, text, t)
        except Exception as e:
            logger.error("event handler failed: %s", e)

    @property
    def costs(self):
        return self.settings.get("costs", {}) or {}

    # ---------------------------------------------------------------- entries
    def submit(self, t: Trade):
        """A trade that passed every gate. Paper: place now. Live: wait for approval."""
        with self._lock:
            self.book.add(t)
            if self.settings.live:
                t.log("waiting for your approval")
                self._save(t)
                self._emit("approval", f"APPROVE? {t.side.upper()} {t.qty} {t.symbol} at ~{t.plan['entry']:.2f}, "
                                       f"stop {t.stop:.2f}, target {t.target:.2f}, max loss Rs {t.max_loss:,.0f}", t)
                return
            transition(t, "APPROVED", "paper trading: no approval needed")
            self._place_entry(t)

    def approve(self, trade_id: str, recheck: Callable[[Trade], Optional[str]]) -> str:
        with self._lock:
            t = self.book.get(trade_id)
            if not t or t.status != "PENDING_APPROVAL":
                return f"trade {trade_id} is not waiting for approval"
            why = recheck(t)
            if why:
                transition(t, "REJECTED", f"blocked when approved: {why}")
                self._save(t)
                self._emit("blocked", f"{t.symbol}: not placed, {why}", t)
                return f"blocked: {why}"
            t.approved_at = time.time()
            transition(t, "APPROVED", "approved on the dashboard")
            self._place_entry(t)
            return "placed" if t.status == "ENTRY_PLACED" else t.status.lower()

    def reject(self, trade_id: str) -> str:
        with self._lock:
            t = self.book.get(trade_id)
            if not t or t.status != "PENDING_APPROVAL":
                return f"trade {trade_id} is not waiting for approval"
            transition(t, "REJECTED", "rejected on the dashboard")
            self._save(t)
            return "rejected"

    def _place_entry(self, t: Trade):
        try:
            t.entry_order_id = self.broker.place(t.symbol, t.side, t.qty, "MARKET", tag=f"mv2-{t.strategy}")
            transition(t, "ENTRY_PLACED", f"market {t.side} {t.qty} sent ({t.entry_order_id})")
            self._emit("order", f"{t.side.upper()} {t.qty} {t.symbol} market order sent", t)
        except BrokerError as e:
            transition(t, "ERROR", f"entry failed: {e}")
            self._emit("error", f"{t.symbol}: entry order failed: {e}", t)
        self._save(t)
        self._step_trade(t, time.time())

    # ---------------------------------------------------------------- the loop
    def on_price(self, symbol: str, price: float, bar=None):
        """Paper fills follow the prices; then react to whatever filled."""
        if self.broker.paper:
            self.broker.on_price(symbol, price, bar)
        with self._lock:
            for t in self.book.open():
                if t.symbol == symbol:
                    self._step_trade(t, time.time(), price)

    def step(self, now: datetime):
        with self._lock:
            self.broker.refresh()
            ts = time.time()
            timeout = float(self.settings.get("approval_timeout_seconds", 120))
            for t in self.book.pending():
                if t.status == "PENDING_APPROVAL" and ts - t.created_at > timeout:
                    transition(t, "EXPIRED", f"no approval within {timeout:.0f}s: the price has moved on")
                    self._save(t)
                    self._emit("expired", f"{t.symbol}: approval expired", t)
            flat = parse_hhmm(self.settings.get("flat_by", "15:10"))
            past_flat = minute_of(now) >= flat
            for t in list(self.book.open()):
                p, _ = self.state.price(t.symbol)
                self._step_trade(t, ts, p)
                if past_flat and t.status == "OPEN":
                    self.exit_now(t.id, "flat-by")
            if past_flat:
                for t in self.book.pending():
                    transition(t, "EXPIRED" if t.status == "PENDING_APPROVAL" else "CANCELLED", "past the flat-by time")
                    self._save(t)

    def _step_trade(self, t: Trade, ts: float, price: Optional[float] = None):
        if t.status == "ENTRY_PLACED":
            st = self.broker.status(t.entry_order_id)
            if st.status == "filled":
                self._on_entry_fill(t, st.avg_price, st.filled_qty)
            elif st.status in ("rejected", "cancelled"):
                transition(t, "REJECTED", f"entry {st.status} by the broker: {st.message}")
                self._save(t)
                self._emit("error", f"{t.symbol}: entry {st.status}: {st.message}", t)
            elif ts - (t.approved_at or t.created_at) > ENTRY_FILL_TIMEOUT_S:
                try:
                    self.broker.cancel(t.entry_order_id)
                except BrokerError:
                    pass
                transition(t, "ERROR", "the market entry did not fill within a minute: cancelled")
                self._save(t)
                self._emit("error", f"{t.symbol}: entry not filled, cancelled", t)
            return
        if t.status == "OPEN":
            for oid, reason in ((t.stop_order_id, "stop"), (t.target_order_id, "target")):
                if not oid:
                    continue
                st = self.broker.status(oid)
                if st.status == "filled":
                    self._closed(t, st.avg_price, reason)
                    return
            if not self.broker.native_oco and price is not None:
                hit = price >= t.target if t.long else price <= t.target
                if hit:
                    self.exit_now(t.id, "target")
            return
        if t.status == "EXITING":
            self._advance_exit(t)

    def _on_entry_fill(self, t: Trade, price: float, qty: int):
        t.entry_price = round(price, 2)
        t.qty = qty or t.qty
        t.opened_at = time.time()
        risk = (t.entry_price - t.stop) if t.long else (t.stop - t.entry_price)
        t.max_loss = round(max(risk, 0) * t.qty + round_trip_charges(t.entry_price, t.stop, t.qty, t.long, self.costs), 2)
        transition(t, "OPEN", f"filled {t.qty} at {t.entry_price:.2f}")
        self._emit("fill", f"{t.symbol}: {t.side} {t.qty} filled at {t.entry_price:.2f}; stop {t.stop:.2f}, "
                           f"target {t.target:.2f}", t)
        self._place_exits(t)
        self._save(t)

    def _place_exits(self, t: Trade):
        try:
            t.stop_order_id = self.broker.place(t.symbol, t.exit_side, t.qty, "SL-M", trigger=t.stop,
                                                tag="mv2-stop", oco=t.id)
            t.log(f"stop {t.stop:.2f} at the broker ({t.stop_order_id})")
        except BrokerError as e:
            t.log(f"stop order failed: {e}")
            self._emit("error", f"{t.symbol}: STOP ORDER FAILED ({e}): exiting at market", t)
            self.exit_now(t.id, "no stop at the broker")
            return
        if self.broker.native_oco:
            try:
                t.target_order_id = self.broker.place(t.symbol, t.exit_side, t.qty, "LIMIT", price=t.target,
                                                      tag="mv2-target", oco=t.id)
                t.log(f"target {t.target:.2f} resting at the broker ({t.target_order_id})")
            except BrokerError as e:
                t.log(f"target order failed: {e}; the target is watched instead")

    # ---------------------------------------------------------------- exits
    def exit_now(self, trade_id: str, reason: str) -> str:
        with self._lock:
            t = self.book.get(trade_id)
            if not t:
                return "unknown trade"
            if t.status == "ENTRY_PLACED":
                try:
                    self.broker.cancel(t.entry_order_id)
                except BrokerError:
                    pass
                self._step_trade(t, time.time())
                if t.status == "ENTRY_PLACED":
                    transition(t, "CANCELLED", f"cancelled: {reason}")
                    self._save(t)
                    return "entry cancelled"
            if t.status in ("PENDING_APPROVAL", "APPROVED"):
                transition(t, "CANCELLED", reason)
                self._save(t)
                return "cancelled"
            if t.status != "OPEN":
                return f"nothing to exit ({t.status.lower()})"
            transition(t, "EXITING", reason)
            self._exit_stage[t.id] = (reason, "cancelling")
            self._cancel_since[t.id] = time.time()
            for oid in (t.stop_order_id, t.target_order_id):
                if oid:
                    try:
                        self.broker.cancel(oid)
                    except BrokerError as e:
                        t.log(f"cancel {oid} failed: {e}")
            self.broker.refresh()
            self._advance_exit(t)
            self._save(t)
            return "exiting"

    def _advance_exit(self, t: Trade):
        reason, stage = self._exit_stage.get(t.id, ("manual", "cancelling"))
        if stage == "cancelling":
            # if the stop or target filled while we were cancelling, that fill closed the trade
            for oid, why in ((t.stop_order_id, "stop"), (t.target_order_id, "target")):
                if oid:
                    st = self.broker.status(oid)
                    if st.status == "filled":
                        self._closed(t, st.avg_price, why)
                        return
                    if st.status == "open" and not self.broker.paper:
                        waited = time.time() - self._cancel_since.get(t.id, time.time())
                        if waited > 15 and not t.events[-1]["text"].startswith("STUCK"):
                            t.log("STUCK: the broker has not confirmed the cancel")
                            self._emit("error", f"{t.symbol}: the broker has not confirmed cancelling order {oid} "
                                                f"after {waited:.0f}s. Check the Angel One app.", t)
                        return          # wait for the cancel to show in the order book
            try:
                t.exit_order_id = self.broker.place(t.symbol, t.exit_side, t.qty, "MARKET", tag="mv2-exit")
                self._exit_stage[t.id] = (reason, "placed")
                t.log(f"market exit sent ({t.exit_order_id}): {reason}")
            except BrokerError as e:
                transition(t, "ERROR", f"EXIT FAILED: {e}")
                self._emit("error", f"{t.symbol}: EXIT ORDER FAILED ({e}). Close it in the Angel One app.", t)
                self._save(t)
                return
        st = self.broker.status(t.exit_order_id)
        if st.status == "filled":
            self._closed(t, st.avg_price, reason)
        elif st.status in ("rejected", "cancelled"):
            transition(t, "ERROR", f"exit {st.status}: {st.message}")
            self._emit("error", f"{t.symbol}: EXIT {st.status.upper()} ({st.message}). Close it in the Angel One app.", t)
            self._save(t)

    def _closed(self, t: Trade, price: float, reason: str):
        if t.status == "OPEN":
            transition(t, "EXITING", reason)
        t.close(price, reason, self.costs)
        self._exit_stage.pop(t.id, None)
        self._cancel_since.pop(t.id, None)
        for oid in (t.stop_order_id, t.target_order_id):
            if oid and self.broker.status(oid).status == "open":
                try:
                    self.broker.cancel(oid)
                except BrokerError:
                    pass
        self._save(t)
        self._emit("closed", f"{t.symbol} closed at {price:.2f} ({reason}): net Rs {t.net_pnl:,.2f} ({t.R:+.2f}R)", t)

    def flatten_all(self, reason: str) -> int:
        n = 0
        with self._lock:
            for t in list(self.book.trades.values()):
                if t.status not in FINAL_STATES and t.status != "EXITING":
                    self.exit_now(t.id, reason)
                    n += 1
        return n

    # ---------------------------------------------------------------- restart
    def restore(self, rows):
        """Reload today's trades after a restart. Paper: re-create the resting stop and target.
        Live: keep the broker's order ids; reconciliation checks the positions."""
        for d in rows:
            t = Trade.from_dict(d)
            self.book.add(t)
            if t.status in FINAL_STATES:
                continue
            if t.status == "PENDING_APPROVAL":
                transition(t, "EXPIRED", "restarted: approvals do not survive a restart")
            elif self.broker.paper and t.status in ("ENTRY_PLACED", "APPROVED"):
                transition(t, "CANCELLED", "restarted before the paper entry filled")
            elif self.broker.paper and t.status in ("OPEN", "EXITING"):
                t.status = "OPEN"
                t.log("restarted: paper stop and target re-created")
                self._place_exits(t)
            self._save(t)
