import datetime as dt

import pytest

from helpers import FakeSettings, flat_day, ist, make_day
from market import calendar as cal
from market.candles.bars import opening_range, round_to_tick, vwap_series
from portfolio.pnl import order_charges, round_trip_charges
from risk.kill_switch import KillSwitch
from risk.limits import AccountView, RiskGate
from risk.position_sizing import size_position
from strategies.base import Signal
from strategies.exits import flat_by_slot, plan_exit


# ------------------------------------------------------------------ calendar
def test_calendar_holidays_and_sessions():
    assert not cal.is_trading_day(dt.date(2026, 10, 20))   # Dussehra
    assert not cal.is_trading_day(dt.date(2026, 10, 24))   # Saturday
    assert cal.is_trading_day(dt.date(2026, 10, 23))
    assert cal.session_state(ist("2026-10-23", "09:14")) == "PRE_OPEN"
    assert cal.session_state(ist("2026-10-23", "09:15")) == "OPEN"
    assert cal.session_state(ist("2026-10-23", "15:30")) == "CLOSED"
    assert cal.next_open(ist("2026-10-23", "16:00")).date() == dt.date(2026, 10, 26)
    assert cal.next_open(ist("2026-11-09", "16:00")).date() == dt.date(2026, 11, 11)  # Nov 10 holiday


def test_bar_slots_and_close_times():
    bars = flat_day("2026-10-23")
    assert bars[0].slot == 0 and bars[0].close_minute == 9 * 60 + 20
    assert bars[-1].slot == 74 and bars[-1].close_minute == 15 * 60 + 30
    assert cal.last_closed_slot(ist("2026-10-23", "09:35")) == 3
    assert flat_by_slot("15:10") == 70


def test_tick_rounding():
    assert round_to_tick(1187.33, 0.10, "down") == 1187.3
    assert round_to_tick(1187.37, 0.10, "up") == 1187.4
    assert round_to_tick(8432.6, 1.0, "nearest") == 8433.0
    assert round_to_tick(101.026, 0.05, "down") == 101.0


def test_opening_range_and_vwap():
    bars = make_day("2026-10-23", [(100, 101, 99, 100, 0), (100, 102, 99.5, 101, 500), (101, 101.5, 100, 101, 500)])
    assert opening_range(bars, 3) == (102, 99, 1000)      # 09:15 volume left out
    assert opening_range(bars[:2], 3) is None
    vw = vwap_series(bars)
    assert len(vw) == 3 and 100 < vw[-1] < 101.5


# ------------------------------------------------------------------ charges
def test_charges_are_positive_and_sell_side_pays_stt():
    costs = FakeSettings().get("costs")
    buy, sell = order_charges(100000, "buy", costs), order_charges(100000, "sell", costs)
    assert sell > buy > 0
    assert 75 < round_trip_charges(1000, 1000, 100, True, costs) < 90   # Rs 2 lakh turnover: Rs 40 brokerage, Rs 25 STT...


# ------------------------------------------------------------------ exit planner
def _sig(**kw):
    base = dict(symbol="TEST", strategy="orb", side="buy", bar_index=5, slot=5, time="09:45", entry=1000.0,
                level=995.0, stop_rule="or_mid", target_r=2.0, context={"or_high": 995.0, "or_low": 985.0})
    base.update(kw)
    return Signal(**base)


def test_exit_plan_or_mid_rounds_away_and_sets_2r():
    s = FakeSettings()
    plan, why = plan_exit(_sig(), [], 0.10, s, "2026-10-23")
    assert why is None
    assert plan.stop == 990.0 and plan.risk_per_share == pytest.approx(10.0)
    assert plan.target == 1020.0 and plan.reward_risk == 2.0


def test_exit_plan_refuses_tight_wide_and_late():
    s = FakeSettings()
    _, why = plan_exit(_sig(context={"or_high": 999.9, "or_low": 999.7}), [], 0.05, s, "2026-10-23")
    assert "too tight" in why
    _, why = plan_exit(_sig(context={"or_high": 990.0, "or_low": 900.0}), [], 0.05, s, "2026-10-23")
    assert "over the" in why
    _, why = plan_exit(_sig(time="15:15"), [], 0.05, s, "2026-10-23")
    assert "flat-by" in why
    _, why = plan_exit(_sig(stop_rule="or_mid", context={}), [], 0.05, s, "2026-10-23")
    assert "opening range" in why


def test_exit_plan_short_side():
    s = FakeSettings()
    plan, why = plan_exit(_sig(side="sell", entry=980.0, context={"or_high": 995.0, "or_low": 985.0}), [], 0.05, s,
                          "2026-10-23")
    assert why is None and plan.stop == 990.0 and plan.target == 960.0


# ------------------------------------------------------------------ sizing
def test_sizing_binds_on_max_loss():
    s = FakeSettings()
    r = size_position(1000.0, 980.0, True, s, 100000)     # 24 shares by max loss, 30 by the position cap
    assert r.binding == "max loss" and r.qty == r.limits["max loss"]
    assert r.max_loss <= s.get("max_loss_per_trade_inr")


def test_sizing_refuses_when_one_share_is_too_risky():
    s = FakeSettings()
    r = size_position(20000.0, 19400.0, True, s, 100000)
    assert r.qty == 0 and "max loss" in r.refusal


def test_sizing_binds_on_cash():
    s = FakeSettings()
    r = size_position(1000.0, 999.0, True, s, 15000)
    assert r.binding == "cash" and r.qty == int((15000 - 10000) // 1000)


# ------------------------------------------------------------------ risk gate
class _NoKill:
    @staticmethod
    def blocked(strategy=None, symbol=None):
        return None


def _order(qty=25, entry=1000.0, stop=990.0, target=1020.0, side="buy"):
    return {"symbol": "TEST", "side": side, "qty": qty, "order_type": "market", "entry": entry, "stop": stop,
            "target": target, "strategy": "orb", "date": "2026-10-23"}


def _acct(**kw):
    a = AccountView(capital=100000, cash_available=100000)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_risk_pass():
    g = RiskGate(FakeSettings(), _NoKill())
    v = g.check_entry(_order(), _acct(), ist("2026-10-23", "09:45"), fresh_price=1000.5, price_age_s=5)
    assert v.passed, v.blocks


def test_risk_blocks_each_rule():
    g = RiskGate(FakeSettings(), _NoKill())
    t = ist("2026-10-23", "09:45")
    assert any("loss:" in b for b in g.check_entry(_order(qty=60), _acct(), t).blocks)
    assert any("daily loss" in b for b in g.check_entry(_order(), _acct(realized_today=-1800), t).blocks)
    assert any("trades:" in b for b in g.check_entry(_order(), _acct(entries_today=6), t).blocks)
    assert any("window" in b for b in g.check_entry(_order(), _acct(), ist("2026-10-23", "09:20")).blocks)
    assert any("window" in b for b in g.check_entry(_order(), _acct(), ist("2026-10-23", "14:40")).blocks)
    assert any("market is" in b for b in g.check_entry(_order(), _acct(), ist("2026-10-24", "10:00")).blocks)
    assert any("chase" in b for b in g.check_entry(_order(), _acct(), t, fresh_price=1004.0).blocks)
    assert any("stale" in b for b in g.check_entry(_order(), _acct(), t, fresh_price=1000.0, price_age_s=999).blocks)
    assert any("short" in b for b in g.check_entry(_order(side="sell", stop=1010, target=980), _acct(), t).blocks)
    assert any("duplicate" in b for b in g.check_entry(_order(), _acct(active_symbols={"TEST"}), t).blocks)
    assert any("reward:risk" in b for b in g.check_entry(_order(target=1010.0), _acct(), t).blocks)
    assert any("cash" in b for b in g.check_entry(_order(), _acct(cash_available=20000), t).blocks)


def test_profits_do_not_extend_the_daily_limit():
    g = RiskGate(FakeSettings(), _NoKill())
    v = g.check_entry(_order(), _acct(realized_today=5000, open_risk=1800), ist("2026-10-23", "10:00"))
    assert any("daily loss" in b for b in v.blocks)


def test_exits_are_never_blocked_by_day_limits():
    assert RiskGate.check_exit("TEST", 10, 10).passed
    assert not RiskGate.check_exit("TEST", 11, 10).passed


def test_kill_switch_blocks_and_persists(tmp_path):
    k = KillSwitch(tmp_path / "k.json")
    g = RiskGate(FakeSettings(), k)
    k.trip("test")
    v = g.check_entry(_order(), _acct(), ist("2026-10-23", "10:00"))
    assert any("kill switch" in b for b in v.blocks)
    assert KillSwitch(tmp_path / "k.json").active
    k.reset()
    assert not KillSwitch(tmp_path / "k.json").active
