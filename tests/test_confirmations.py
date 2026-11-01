"""Order-block board confirmations: CPR, its narrowness, volume by time of day, and each side's checks."""
import datetime as dt

from helpers import make_day

from market import confirmations as CF
from market.candles.bars import DailyBar


def _daily(widths):
    """Daily bars whose CPR widths grow with the high-low range (close at the midpoint gives a zero-width CPR)."""
    out, d = [], dt.date(2026, 9, 1)
    for w in widths:
        out.append(DailyBar(d.isoformat(), 100, 100 + w, 100 - w, 100 + w / 2))
        d += dt.timedelta(days=1)
    return out


def test_cpr_levels():
    c = CF.cpr(110, 90, 105)
    assert round(c["pivot"], 4) == round(305 / 3, 4)
    assert c["bc"] == 100 and round(c["tc"], 4) == round(2 * 305 / 3 - 100, 4)
    assert c["bc"] <= c["tc"] and c["width_pct"] > 0


def test_the_narrowest_cpr_in_20_sessions_ranks_1():
    before = _daily([5.0 + i % 3 for i in range(20)] + [0.5])
    assert CF.cpr_rank(before) == 1.0
    wide = _daily([1.0] * 20 + [9.0])
    assert CF.cpr_rank(wide) == 0.0
    assert CF.cpr_rank(before[-10:]) is None                       # not enough history


def test_volume_against_the_same_time_of_day():
    past = []
    for i in range(20):
        past += make_day(f"2026-10-{i + 1:02d}", [(100, 101, 99, 100)] * 75, volume=1000.0)
    prof = CF.volume_profile(past, "2026-10-27")
    assert prof[0] == 1000.0 and prof[9] == 10000.0
    today = make_day("2026-10-27", [(100, 101, 99, 100)] * 10, volume=2000.0)
    assert CF.rel_volume(today, prof) == 2.0
    assert CF.volume_profile(past[:75 * 3], "2026-10-27") is None  # under 5 sessions


def test_checks_follow_the_side():
    x = {"last": 105.0, "change_pct": 1.2, "fut_buildup": "long build-up", "fut_oi_chg": 3.0, "live_oi_chg": 2.5,
         "pe_chg": 50000.0, "ce_chg": 10000.0, "pcr": 1.1, "cpr": CF.cpr(104, 100, 102), "cpr_rank": 0.9,
         "rvol": 1.8, "upto": "11:00", "prev_high": 104.0, "prev_low": 100.0, "sma": 101.0, "index_chg": 0.4}
    bull = CF.evaluate(1, x)
    assert bull["of"] == len(CF.RULES) and bull["score"] == len(CF.RULES)
    bear = CF.evaluate(-1, x)
    on = {k for k, v in bear["checks"].items() if v["ok"]}
    assert on == {"narrow_cpr", "volume"}                         # the two checks that do not depend on the side


def test_missing_data_is_not_counted():
    r = CF.evaluate(1, {"last": 100.0, "change_pct": 0.5, "rvol": 1.0})
    assert r["of"] == 1 and r["score"] == 0
    assert r["checks"]["live_oi"]["ok"] is None and r["checks"]["futures"]["ok"] is None


def test_live_oi_needs_a_real_build_up():
    base = {"last": 100.0, "change_pct": 1.0}
    assert CF.evaluate(1, dict(base, live_oi_chg=0.4))["checks"]["live_oi"]["ok"] is False     # under 1%: flat
    assert CF.evaluate(1, dict(base, live_oi_chg=-3.0))["checks"]["live_oi"]["ok"] is False    # short covering
    assert CF.evaluate(-1, dict(base, change_pct=-1.0, live_oi_chg=3.0))["checks"]["live_oi"]["ok"] is True


def test_the_board_fills_sessions_yahoo_skipped():
    from apps.trading_engine.obboard import OrderBlockBoard
    board = OrderBlockBoard(None)
    board._days5m = {"X": {"2026-10-26": DailyBar("2026-10-26", 10, 12, 9, 11)}}
    daily = [DailyBar("2026-10-23", 9, 10, 8, 9.5), DailyBar("2026-10-27", 11, 13, 10, 12)]
    assert [b.date for b in board._filled("X", daily)] == ["2026-10-23", "2026-10-26", "2026-10-27"]
    assert board._filled("Y", daily) is daily


class _FakeAngel:
    def __init__(self, oi, pcr=None):
        self.oi, self.pcr, self.calls = oi, pcr or [], 0

    def getMarketData(self, mode, tokens):
        self.calls += 1
        return {"status": True, "data": {"fetched": [{"symbolToken": t, "opnInterest": self.oi[t]} for t in tokens["NFO"] if t in self.oi]}}

    def putCallRatio(self):
        return {"status": True, "data": self.pcr}


def test_live_oi_sums_every_expiry_and_checks_units(monkeypatch):
    from market.instruments import fno
    monkeypatch.setattr(fno.time, "sleep", lambda s: None)
    contracts = {f"S{i}": [{"token": f"{i}a", "symbol": f"S{i}OCTFUT"}, {"token": f"{i}b", "symbol": f"S{i}NOVFUT"}] for i in range(60)}
    oi = {t["token"]: 500.0 for cs in contracts.values() for t in cs}
    del oi["0b"]                                                    # one expiry did not answer: that stock is left out
    base = {s: 1000.0 for s in contracts}
    api = _FakeAngel(oi, [{"tradingSymbol": "S1OCTFUT", "pcr": 0.8}])
    live = fno.LiveOI()
    live.refresh(api, contracts, base, "2026-10-27")
    assert api.calls == 3 and live.oi["S1"] == 1000.0 and "S0" not in live.oi
    assert live.pcr == {"S1": 0.8} and live.day == "2026-10-27" and not live.error
    lots = fno.LiveOI()                                             # OI in lots, not shares: refused
    lots.refresh(_FakeAngel({k: v / 100 for k, v in oi.items()}), contracts, base, "2026-10-27")
    assert lots.oi == {} and "units" in lots.error
