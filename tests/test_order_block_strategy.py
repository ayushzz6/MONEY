"""Order-block retest and first-stage strategies: the one-stock rule, the live board trigger, paper only."""
from backtesting.engine import run_day
from backtesting.selection import strategy_allowed
from helpers import FakeSettings, ist, make_day, session_start
from market import orderblocks as OB
from scanner.features.daily import DailyStats
from strategies.base import DayContext
from strategies.ensemble import build_strategies
from strategies.smc.order_block import OrderBlockFirstStage, OrderBlockRetest

D = "2026-10-26"
PARAMS = {"enabled": True, "paper_only": True, "picks_only": True, "min_aligned": 4, "stop": "zone", "target": "T2"}

# a swing high at 09:25 (101.3), the 09:35 red candle (100.5-100.9) is the order block, the 09:40 candle closes
# through 101.3 (zone given 09:45), a run to 102.7, a pullback, and the 10:15 candle comes back to Z = 100.9
BASE = [(100.0, 100.6, 99.9, 100.5), (100.5, 101.1, 100.4, 101.0), (101.0, 101.3, 100.8, 101.1),
        (101.1, 101.2, 100.7, 100.8), (100.8, 100.9, 100.5, 100.6), (100.6, 101.6, 100.6, 101.5),
        (101.5, 102.0, 101.4, 101.9), (101.9, 102.4, 101.8, 102.3), (102.3, 102.6, 102.1, 102.5),
        (102.5, 102.7, 102.2, 102.4), (102.4, 102.5, 101.8, 101.9), (101.9, 102.0, 101.3, 101.4)]
RETEST = (101.4, 101.5, 100.85, 101.2)
AFTER = [(101.2, 102.0, 101.1, 101.9), (101.9, 103.6, 101.8, 103.5)] + [(103.5, 103.6, 103.3, 103.4)] * 10


def daily():
    return DailyStats(symbol="TEST", as_of="2026-10-23", close=100.0, change_pct=0, sma20=95, sma50=90, rsi14=55,
                      atr14=2.0, atr_pct=2.0, high20=110, close_high20=109, avg_volume20=1e6, turnover20=5e8,
                      prev_high=103.5, prev_low=99.0)


def ctx(**kw):
    base = dict(symbol="TEST", date=D, entry_from=9 * 60 + 30, entry_until=14 * 60 + 30, allow_short=False)
    base.update(kw)
    return DayContext(**base)


def strat(**kw):
    return OrderBlockRetest(dict(PARAMS, **kw))


def test_the_first_return_to_z_is_a_limit_fill_at_z_with_the_zone_stop_and_t2():
    sig = strat().first_trigger(make_day(D, BASE + [RETEST] + AFTER), daily(), ctx())
    assert sig and sig.side == "buy" and sig.bar_index == 12 and sig.time == "10:15"
    assert sig.entry == 100.9 and sig.context["fill"] == "limit" and sig.context["given_at"] == "09:45"
    assert 100.4 < sig.stop_hint < 100.5 and sig.target_hint == 103.5          # T2 = yesterday's high
    assert strat(target="T1").first_trigger(make_day(D, BASE + [RETEST] + AFTER), daily(), ctx()).target_hint == 102.7


def test_the_backtest_takes_the_target_and_counts_a_stop_inside_the_entry_bar():
    S = FakeSettings()
    r = run_day(strat(), ctx(), make_day(D, BASE + [RETEST] + AFTER), daily(), S, 0.05, None)
    assert r["outcome"] == "target" and r["plan"].target == 103.5 and r["R"] > 5
    deep = (101.4, 101.5, 100.3, 100.6)                                          # touches Z, then the stop, closes inside
    r = run_day(strat(), ctx(), make_day(D, BASE + [deep] + AFTER), daily(), S, 0.05, None)
    assert r["outcome"] == "stop" and r["R"] == -1.0 and r["exit_slot"] == 12


def test_no_trade_when_the_day_runs_the_other_way_or_shorts_are_off():
    d = daily()
    d.close = 106.0                                         # far below yesterday's close: the stock's side is bearish
    assert strat().first_trigger(make_day(D, BASE + [RETEST] + AFTER), d, ctx()) is None
    assert strat().first_trigger(make_day(D, BASE + [RETEST]), daily(), ctx(entry_from=10 * 60 + 30)) is None


def board_row(**kw):
    bars = make_day(D, BASE)
    z = OB.current_setup(OB.detect_zones(bars, 103.5, 99.0), 1).to_dict()
    row = {"symbol": "TEST", "side": 1, "rule": "pick", "rank": 2, "aligned": 4, "zone": z}
    row.update(kw)
    return row, bars


def test_the_board_trigger_fires_when_the_live_price_reaches_z():
    s = strat()
    row, bars = board_row()
    assert s.why_not(row, ctx()) is None and s.armed(row, ctx())
    now = ist(D, "10:17")
    assert s.from_board(row, 101.3, bars, now) == (None, None)                    # still above Z
    sig, why = s.from_board(row, 100.88, bars, now)
    assert why is None and sig.side == "buy" and sig.entry == 100.88 and sig.time == "10:17" and sig.slot == 12
    assert sig.stop_rule == "zone" and sig.target_hint == 103.5 and sig.context["given_at"] == "09:45"
    sig, why = s.from_board(row, 100.3, bars, now)
    assert sig and "past the invalidation" in why
    touched = bars + make_day(D, [(0, 0, 0, 0)] * 12 + [RETEST])[12:]            # a closed candle already came back
    assert s.from_board(row, 100.88, touched, now) == (None, None)


def test_only_aligned_picks_with_an_untouched_zone_are_armed():
    s = strat()
    row, _ = board_row(rule="not picked · rank 4")
    assert "picks" in s.why_not(row, ctx())
    row, _ = board_row(aligned=3)
    assert s.why_not(row, ctx()) == "3 of 4 checks aligned, needs 4"
    row, _ = board_row(side=-1)
    assert "short selling is off" in s.why_not(row, ctx()) and s.why_not(row, ctx(allow_short=True)) is None
    row, _ = board_row()
    row["zone"] = dict(row["zone"], entry_at=session_start(D) + 12 * 300)
    assert "only the first return" in s.why_not(row, ctx())


def test_paper_only_trades_on_paper_whatever_the_retest_and_never_live():
    bt = {"overall": {"trades": 40}, "by_strategy": {"order_block": {"trades": 40, "avg_net_R": -0.3}}}
    ok, why = strategy_allowed(FakeSettings(), bt, "order_block")
    assert ok and why.startswith("paper trial") and "-0.30R" in why
    ok, why = strategy_allowed(FakeSettings(live=True), bt, "order_block")
    assert not ok and why.startswith("paper only")


def test_the_pipeline_journals_a_board_signal_once():
    from apps.decision_service.main import DecisionPipeline

    class Journal:
        rows = []

        def save_signal(self, date, row):
            self.rows.append(row)

    p = DecisionPipeline(FakeSettings(), None, [], None, None, None, None, None, None, None, None, Journal(),
                         lambda name: (True, ""))
    s = strat()
    row, bars = board_row()
    sig, why = s.from_board(row, 100.3, bars, ist(D, "10:17"))
    out = p.on_signal(sig, s, ist(D, "10:17"), D, skip=why)
    assert out["decision"] == "SKIPPED" and out["strategy"] == "order_block" and p.signals[0] is out
    assert p.on_signal(sig, s, ist(D, "10:18"), D, skip=why) is None              # one try per stock a day


# ------------------------------------------------------------ first stage: enter when the zone is given
def first(**kw):
    return OrderBlockFirstStage(dict(PARAMS, window_bars=1, **kw))


def test_the_first_stage_enters_at_the_close_that_gives_the_zone_and_aims_for_t2_as_given():
    sig = first().first_trigger(make_day(D, BASE + [RETEST] + AFTER), daily(), ctx())
    assert sig and sig.side == "buy" and sig.bar_index == 5 and sig.time == "09:45" and sig.context["fill"] == "market"
    assert sig.entry == 101.5 and 100.4 < sig.stop_hint < 100.5
    assert sig.target_hint == 103.5                                              # T2 as given (yesterday's high)
    r = run_day(first(), ctx(), make_day(D, BASE + [RETEST] + AFTER), daily(), FakeSettings(), 0.05, None)
    assert r["outcome"] == "target" and r["plan"].entry == 101.5 and r["plan"].target == 103.5 and 1.7 < r["R"] < 1.9
    late = first().first_trigger(make_day(D, BASE + [RETEST] + AFTER), daily(), ctx(entry_from=9 * 60 + 50))
    assert late.time == "10:30" and late.entry == 103.5                          # the next zone given inside the hours
    assert first().first_trigger(make_day(D, BASE + [RETEST] + AFTER), daily(), ctx(entry_until=9 * 60 + 40)) is None


def test_the_first_stage_is_armed_only_in_the_candle_after_the_zone_is_given():
    s = first()
    row, _ = board_row()
    z = row["zone"]
    assert z["given_targets"] == [101.6, 103.5] and z["targets"] == [102.7, 103.5]   # frozen vs the moving levels
    assert s.why_not(row, ctx(), ist(D, "09:47").timestamp()) is None
    assert "first stage ended at 09:50" in s.why_not(row, ctx(), ist(D, "09:51").timestamp())
    row["zone"] = dict(z, entry_at=session_start(D) + 7 * 300)
    assert "came back to Z at 09:50" in s.why_not(row, ctx(), ist(D, "09:47").timestamp())
    row, _ = board_row(aligned=3)
    assert s.why_not(row, ctx(), ist(D, "09:47").timestamp()) == "3 of 4 checks aligned, needs 4"
    assert "until 09:50" in s.armed_text(board_row()[0])


def test_the_first_stage_board_trigger_buys_at_the_live_price_before_any_retest():
    s = first()
    row, bars = board_row()
    now = ist(D, "09:47")
    sig, why = s.from_board(row, 101.8, bars[:6], now)
    assert why is None and sig.strategy == "order_block_first" and sig.entry == 101.8 and sig.time == "09:47"
    assert sig.stop_rule == "zone" and sig.stop_hint == row["zone"]["invalid"] and sig.target_hint == 103.5
    assert first(target="T1").from_board(row, 101.55, bars[:6], now)[0].target_hint == 101.6
    assert s.from_board(row, 100.9, bars[:6], now) == (None, None)                 # back at Z: the retest's trade
    assert s.from_board(row, 101.8, bars[:6], ist(D, "09:51")) == (None, None)     # the window ended
    assert s.from_board(row, 101.8, make_day(D, BASE + [RETEST]), now) == (None, None)   # a candle came back to Z


def test_the_first_stage_is_a_paper_only_board_strategy_from_the_config():
    built = {x.name: x for x in build_strategies(FakeSettings())}
    assert built["order_block_first"].live_trigger == "board" and built["order_block_first"].params["window_bars"] == 1
    bt = {"overall": {"trades": 40}, "by_strategy": {"order_block_first": {"trades": 40, "avg_net_R": -0.3}}}
    assert strategy_allowed(FakeSettings(), bt, "order_block_first")[0]
    assert not strategy_allowed(FakeSettings(live=True), bt, "order_block_first")[0]
