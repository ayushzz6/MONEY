"""Order manager + paper broker, end to end, and the LIVE approval path with a fake broker."""
import time

import pytest

from brokers.base import Broker, OrderState
from brokers.paper import PaperBroker
from helpers import FakeSettings, ist, make_day
from market.state.manager import MarketState
from oms.order_manager import OrderManager
from portfolio.positions import Trade, TradeBook

D = "2026-10-23"


class MemJournal:
    def __init__(self):
        self.saved = {}

    def save_trade(self, t):
        self.saved[t["id"]] = t


def trade(**kw):
    base = dict(symbol="TEST", strategy="orb", side="buy", qty=10, date=D, mode="PAPER",
                plan={"entry": 100.0, "stop": 98.0, "target": 104.0}, stop=98.0, target=104.0, max_loss=25.0)
    base.update(kw)
    return Trade(**base)


def setup(live=False, broker=None):
    s = FakeSettings(live=live)
    st = MarketState()
    st.on_price("TEST", 100.0, time.time(), build_bars=False)
    b = broker or PaperBroker(st, 5)
    book, j, events = TradeBook(), MemJournal(), []
    om = OrderManager(s, b, book, j, st, on_event=lambda k, txt, t=None: events.append((k, txt)))
    return s, st, b, book, om, events


def future_bar(o, h, l, c):
    """A bar that starts after the orders exist (paper fills ignore bars that ended before)."""
    t0 = int(time.time()) + 1
    bar = make_day(D, [(o, h, l, c)])[0]
    bar.start = t0
    return bar


def test_paper_entry_fills_and_target_closes_it():
    s, st, b, book, om, events = setup()
    t = trade()
    om.submit(t)
    assert t.status == "OPEN" and t.entry_price == pytest.approx(100.05)     # 5 bps slippage
    assert t.stop_order_id and t.target_order_id
    om.on_price("TEST", 104.2, bar=future_bar(101, 104.5, 100.5, 104.2))
    assert t.status == "CLOSED" and t.exit_reason == "target" and t.exit_price == 104.0
    assert t.net_pnl == pytest.approx((104.0 - 100.05) * 10 - t.charges)
    assert b.status(t.stop_order_id).status == "cancelled"                  # one-cancels-other


def test_paper_stop_wins_when_a_bar_touches_both():
    s, st, b, book, om, events = setup()
    t = trade()
    om.submit(t)
    om.on_price("TEST", 101, bar=future_bar(100, 105, 97, 101))
    assert t.status == "CLOSED" and t.exit_reason == "stop" and t.exit_price < 98.0


def test_flat_by_closes_open_trades():
    s, st, b, book, om, events = setup()
    t = trade()
    om.submit(t)
    st.on_price("TEST", 101.0, time.time(), build_bars=False)
    om.step(ist(D, "15:11"))
    assert t.status == "CLOSED" and t.exit_reason == "flat-by"


def test_manual_exit_and_flatten():
    s, st, b, book, om, events = setup()
    t1, t2 = trade(), trade(symbol="OTHER")
    st.on_price("OTHER", 50.0, time.time(), build_bars=False)
    t2.plan, t2.stop, t2.target = {"entry": 50, "stop": 49, "target": 52}, 49.0, 52.0
    om.submit(t1)
    om.submit(t2)
    assert om.exit_now(t1.id, "manual") == "exiting" and t1.status == "CLOSED"
    assert om.flatten_all("test") == 1 and t2.status == "CLOSED"


class FakeLive(Broker):
    """A live-like broker: no one-cancels-other, fills only when told."""
    paper = False
    native_oco = False
    name = "fake-live"

    def __init__(self):
        self.orders = {}
        self.n = 0

    def place(self, symbol, side, qty, order_type="MARKET", price=None, trigger=None, tag="", oco=None):
        self.n += 1
        oid = f"L{self.n}"
        self.orders[oid] = {"symbol": symbol, "side": side, "qty": qty, "type": order_type, "trigger": trigger,
                            "status": "filled" if order_type == "MARKET" else "open",
                            "price": 100.0 if order_type == "MARKET" else None}
        return oid

    def cancel(self, oid):
        if self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "cancelled"

    def status(self, oid):
        o = self.orders[oid]
        return OrderState(oid, o["status"], o["qty"] if o["status"] == "filled" else 0, o["price"])


def test_live_waits_for_approval_and_rechecks():
    fb = FakeLive()
    s, st, b, book, om, events = setup(live=True, broker=fb)
    t = trade(mode="LIVE")
    om.submit(t)
    assert t.status == "PENDING_APPROVAL" and not fb.orders and events[0][0] == "approval"
    assert om.approve(t.id, lambda tr: "price ran away") .startswith("blocked")
    assert t.status == "REJECTED" and not fb.orders
    t2 = trade(mode="LIVE")
    om.submit(t2)
    assert om.approve(t2.id, lambda tr: None) == "placed" or t2.status == "OPEN"
    assert t2.status == "OPEN"
    assert [o["type"] for o in fb.orders.values()] == ["MARKET", "SL-M"]        # stop at the broker, no target
    om.on_price("TEST", 104.1)                                                   # target watched on quotes
    assert t2.status == "CLOSED" and t2.exit_reason == "target"
    assert fb.orders[t2.stop_order_id]["status"] == "cancelled"


def test_live_approval_expires():
    fb = FakeLive()
    s, st, b, book, om, events = setup(live=True, broker=fb)
    t = trade(mode="LIVE")
    om.submit(t)
    t.created_at -= 1000
    om.step(ist(D, "10:30"))
    assert t.status == "EXPIRED" and not fb.orders


def test_live_stop_fill_is_recorded():
    fb = FakeLive()
    s, st, b, book, om, events = setup(live=True, broker=fb)
    t = trade(mode="LIVE")
    om.submit(t)
    om.approve(t.id, lambda tr: None)
    fb.orders[t.stop_order_id].update(status="filled", price=97.9)
    om.step(ist(D, "11:00"))
    assert t.status == "CLOSED" and t.exit_reason == "stop" and t.exit_price == 97.9
