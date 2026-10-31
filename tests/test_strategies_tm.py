import pytest

from backtesting import timemachine as tm
from backtesting.broker_simulator import simulate_bracket
from backtesting.engine import market_ok, run_day
from helpers import FakeSettings, flat_day, make_day
from market.candles.bars import full_session
from scanner.features.daily import DailyStats, wilder_rsi
from strategies.base import DayContext
from strategies.breakout.level import LevelBreakout
from strategies.breakout.orb import OpeningRangeBreakout
from strategies.vwap.reclaim import VwapReclaim

D = "2026-10-23"


def ctx(**kw):
    base = dict(symbol="TEST", date=D, entry_from=9 * 60 + 30, entry_until=14 * 60 + 30, allow_short=False,
                or_volume_avg=None)
    base.update(kw)
    return DayContext(**base)


def orb_day():
    # range 99..101 (09:15-09:25), a wick above at 09:30, the 09:35 bar closes above
    rows = [(100, 101, 99, 100), (100, 100.8, 99.2, 100.5), (100.5, 100.9, 99.5, 100.2),
            (100.2, 101.6, 100.0, 100.9),       # wick to 101.6, close 100.9: not a trigger
            (100.9, 101.8, 100.8, 101.5)]       # closes 101.5 > 101: the trigger (closes 09:40)
    rows += [(101.5, 102.0, 101.2, 101.8)] * 70
    return make_day(D, rows)


def daily(**kw):
    base = dict(symbol="TEST", as_of="2026-10-22", close=100, change_pct=0, sma20=95, sma50=90, rsi14=55,
                atr14=2, atr_pct=2, high20=103, close_high20=102, avg_volume20=1e6, turnover20=5e8,
                trend_up=True, trend_down=False, near_high20=True, setups=[])
    base.update(kw)
    return DailyStats(**base)


def test_orb_waits_for_a_close_not_a_wick():
    s = OpeningRangeBreakout({"or_minutes": 15, "min_or_pct": 0.3, "max_or_pct": 3, "min_or_rvol": 0})
    sig = s.first_trigger(orb_day(), None, ctx())
    assert sig and sig.time == "09:40" and sig.entry == 101.5 and sig.level == 101
    assert sig.context["or_low"] == 99


def test_orb_in_play_filter_and_volume_confirm():
    s = OpeningRangeBreakout({"or_minutes": 15, "min_or_rvol": 1.5})
    assert s.first_trigger(orb_day(), None, ctx(or_volume_avg=10000)) is None      # range volume 2000: not in play
    assert s.first_trigger(orb_day(), None, ctx(or_volume_avg=1000)) is not None
    s2 = OpeningRangeBreakout({"or_minutes": 15, "min_volume_x": 3})
    assert s2.first_trigger(orb_day(), None, ctx()) is None                        # no bar has 3x volume


def test_orb_short_only_when_allowed():
    rows = [(100, 101, 99, 100)] * 3 + [(99.5, 99.6, 98.0, 98.5)] + [(98.5, 98.7, 98.2, 98.4)] * 71
    s = OpeningRangeBreakout({"or_minutes": 15, "short": True})
    assert s.first_trigger(make_day(D, rows), None, ctx()) is None
    sig = s.first_trigger(make_day(D, rows), None, ctx(allow_short=True))
    assert sig.side == "sell" and sig.level == 99


def test_level_breakout_needs_near_high_and_skips_big_gaps():
    s = LevelBreakout({"near_pct": 3, "max_gap_pct": 1.5, "stop": "swing:3"})
    rows = [(101, 102, 100.5, 101.5)] * 3 + [(101.5, 103.5, 101.4, 103.2)] + [(103.2, 103.6, 103, 103.4)] * 71
    sig = s.first_trigger(make_day(D, rows), daily(), ctx())
    assert sig and sig.time == "09:35" and sig.level == 103 and sig.stop_hint == 100.5
    assert s.first_trigger(make_day(D, rows), daily(close=95), ctx()) is None      # not near the high
    gap = [(105.5, 106, 105, 105.8)] * 75
    assert s.first_trigger(make_day(D, gap), daily(), ctx()) is None               # gapped 2.4% over


def test_vwap_reclaim():
    rows = [(100, 100.5, 99.5, 100.2)] * 3
    rows += [(100.2, 100.3, 99.0, 99.2)] * 6       # 09:30-09:55 under VWAP (dip low 99.0)
    rows += [(99.2, 100.6, 99.1, 100.5)]           # 10:00 bar closes above VWAP and the previous high
    rows += [(100.5, 100.8, 100.3, 100.6)] * 65
    s = VwapReclaim({"not_before": "10:00", "stop": "swing"})
    sig = s.first_trigger(make_day(D, rows), daily(), ctx())
    assert sig and sig.time == "10:05" and sig.stop_hint == 99.0
    assert s.first_trigger(make_day(D, rows), daily(trend_up=False), ctx()) is None


def test_wilder_rsi_bounds():
    up = wilder_rsi([float(i) for i in range(30)])
    assert up[-1] == 100.0
    mixed = wilder_rsi([100 + (i % 2) for i in range(30)])
    assert 40 < mixed[-1] < 60


def test_bracket_stop_first_and_gaps():
    day = make_day(D, [(100, 100, 100, 100)] * 5 + [(100, 103, 97, 101)] + [(101, 101, 101, 101)] * 69)
    r = simulate_bracket(True, 98.0, 102.0, day[5:], 70)
    assert r["outcome"] == "stop" and r["exit"] == 98.0           # touched both: the stop
    gap = make_day(D, [(100, 100, 100, 100)] * 5 + [(96, 96.5, 95, 96)] + [(96, 96, 96, 96)] * 69)
    r = simulate_bracket(True, 98.0, 102.0, gap[5:], 70)
    assert r["exit"] == 96                                         # a gap fills at the open, worse
    r = simulate_bracket(True, 90.0, 110.0, flat_day(D)[5:], 70)
    assert r["outcome"] == "flat" and r["exit_slot"] == 70


def test_run_day_end_to_end():
    s = FakeSettings()
    strat = OpeningRangeBreakout({"or_minutes": 15, "stop": "or_mid", "target_r": 1.0})
    r = run_day(strat, ctx(), orb_day(), None, s, 0.05, None)
    assert r["plan"].stop == 100.0 and r["plan"].target == 103.0
    assert r["outcome"] == "flat" and r["R"] == pytest.approx((101.8 - 101.5) / 1.5, abs=1e-3)


def test_market_filter():
    up = make_day(D, [(100, 101, 99.8, 100.5)] * 5)
    assert market_ok("buy", up, 4) and not market_ok("sell", up, 4)
    assert market_ok("buy", None, 4)


def test_time_machine_is_seeded_and_sane():
    sessions = []
    for k in range(25):
        rows = []
        p = 100.0
        for i in range(75):
            o = p
            p = p * (1 + (0.002 if (i + k) % 3 == 0 else -0.0009))
            rows.append((o, max(o, p) * 1.001, min(o, p) * 0.999, p))
        sessions.append(full_session(make_day(f"2026-09-{(k % 28) + 1:02d}", rows)))
    pool = tm.build_pool(sessions)
    a = tm.simulate(pool, 100, 99, 102, 5, 70, replays=2000, seed=7)
    b = tm.simulate(pool, 100, 99, 102, 5, 70, replays=2000, seed=7)
    assert a == b
    assert abs(a["p_target"] + a["p_stop"] + a["p_neither"] - 1) < 1e-6
    far = tm.simulate(pool, 100, 50, 200, 5, 70, replays=2000, seed=7)
    assert far["p_neither"] == 1.0
    ok, why = tm.gate(far, {"triggered": 0, "avg_R": None}, FakeSettings())
    assert not ok and "neither" in why
    assert "R" in tm.verdict(a, "15:10")
