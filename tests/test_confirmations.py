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
         "rvol": 1.8, "upto": "11:00", "prev_high": 104.0, "prev_low": 100.0, "sma": 101.0, "index_chg": 0.4,
         "opt_pe_chg": 40000.0, "opt_ce_chg": 10000.0, "opt_pcr": 1.2, "opt_expiry": "2026-11-24"}
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


def test_live_oi_uses_the_websocket_once_it_matches_the_quote_api(monkeypatch):
    from market.instruments import fno
    monkeypatch.setattr(fno.time, "sleep", lambda s: None)
    contracts = {f"S{i}": [{"token": f"{i}a", "symbol": f"S{i}NOVFUT"}] for i in range(12)}
    oi = {f"{i}a": 1000.0 for i in range(12)}
    base = {s: 1000.0 for s in contracts}
    live = fno.LiveOI()
    for i in range(12):
        live.on_stream(f"{i}a", 1100.0)                              # streamed before the first quote-API read
    assert live.oi == {}                                             # nothing until the quote API has read every contract
    live.refresh(_FakeAngel(oi), contracts, base, "2026-11-02")
    assert live.stream_ok is True and live.oi["S3"] == 1100.0       # the stream is newer and in the same units
    live.on_stream("3a", 1250.0)
    assert live.oi["S3"] == 1250.0
    bad = fno.LiveOI()
    for i in range(12):
        bad.on_stream(f"{i}a", 11.0)                                 # lots, not shares
    bad.refresh(_FakeAngel(oi), contracts, base, "2026-11-02")
    assert bad.stream_ok is False and bad.oi["S3"] == 1000.0 and "websocket" in bad.error


def test_live_options_against_the_same_expiry_in_the_bhavcopy(monkeypatch):
    from market.instruments import fno
    monkeypatch.setattr(fno.time, "sleep", lambda s: None)
    chains = {"A": {"expiry": "2026-11-24", "CE": ["a1", "a2"], "PE": ["a3"]},
              "B": {"expiry": "2026-11-24", "CE": ["b1"], "PE": ["b2"]},
              "C": {"expiry": "2026-11-24", "CE": ["c1"], "PE": ["c2"]}}
    oi = {"a1": 300.0, "a2": 200.0, "a3": 900.0, "b1": 100.0, "b2": 100.0, "c1": 50.0}      # c2 did not answer
    base = {"A": {"2026-11-24": [400.0, 600.0]}, "B": {"2026-11-24": [100.0, 120.0]}}
    lo = fno.LiveOptions()
    lo.refresh(_FakeAngel(oi), chains, base, "2026-11-02")
    a = lo.data["A"]
    assert (a["ce"], a["pe"], a["ce_chg"], a["pe_chg"], a["pcr"]) == (500.0, 900.0, 100.0, 300.0, 1.8)
    assert lo.data["B"]["pe_chg"] == -20.0 and "C" not in lo.data and not lo.error
    base3 = dict(base, B2={"2026-11-24": [100.0, 100.0]})
    chains3 = dict(chains, B2={"expiry": "2026-11-24", "CE": ["d1"], "PE": ["d2"]})
    lots3 = fno.LiveOptions()
    lots3.refresh(_FakeAngel({k: v / 100 for k, v in dict(oi, d1=100.0, d2=100.0).items()}), chains3, base3, "2026-11-02")
    assert lots3.data == {} and "units" in lots3.error              # three stocks in lots, not shares: refused


def test_buildup_lists_name_each_contract_s_stock(monkeypatch):
    from market.instruments import fno
    monkeypatch.setattr(fno.time, "sleep", lambda s: None)

    class Api:
        def oIBuildup(self, params):
            if params["datatype"] == "Short Covering":
                return {"status": False, "message": "rate limit"}
            return {"status": True, "data": [{"tradingSymbol": "M&M25NOV26FUT", "ltp": "3100.5", "percentChange": "2.1",
                                              "opnInterest": "123456", "netChangeOpnInterest": "-5000"}]}
    b = fno.BuildUpLists()
    b.refresh(Api(), {"M&M25NOV26FUT": None}.get)
    row = b.lists["long build-up"][0]
    assert row["symbol"] == "M&M" and row["chg_pct"] == 2.1 and row["oi_chg"] == -5000.0
    assert "short covering" not in b.lists and "rate limit" in b.error


def test_the_feed_routes_futures_oi_and_keeps_each_socket_s_own_subscriptions():
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    from brokers.angelone import market_data as M
    md = M.AngelOneMarketData("jwt", "key", "client", "feed")
    assert md.sws.input_request_dict is not SmartWebSocketV2.input_request_dict
    got, quotes = [], []
    md.on_oi = lambda t, oi, ts: got.append((t, oi))
    md.on_quote = lambda *a: quotes.append(a[0])
    md.token_to_symbol = {"123": "STOCK"}
    md.subscribe_fo({"123": "STOCKNOVFUT"})                          # the same token number on NFO
    md._on_data(None, {"exchange_type": 2, "token": "123", "open_interest": 5000, "last_traded_price": 1})
    md._on_data(None, {"exchange_type": 1, "token": "123", "last_traded_price": 10000})
    assert got == [("123", 5000.0)] and quotes == ["STOCK"]
    md.token_to_symbol = {str(i): f"S{i}" for i in range(M.MAX_TOKENS - 3)}
    md.fo_tokens = {}
    assert md.subscribe_fo({f"f{i}": f"F{i}" for i in range(10)}) == 3     # only what fits on the socket


def test_live_view_rechecks_prices_every_call():
    import time as _t

    from apps.trading_engine.obboard import OrderBlockBoard
    from market.state.manager import MarketState
    from tests.helpers import FakeSettings

    class E:
        settings = FakeSettings()
        state = MarketState()
    e = E()
    board = OrderBlockBoard(e)
    bars = make_day("2026-11-02", [(100, 101, 99.5, 100.5)] * 6)
    ctx = {"prev_close": 100.0, "prev_high": 102.0, "prev_low": 98.0, "cpr": CF.cpr(102, 98, 100), "cpr_rank": 0.5, "sma": 99.0}
    row = {"symbol": "X", "side": 1, "last": 100.5, "change_pct": 0.5, "oi_buildup": None, "rank": 1}
    board.board = {"session": "2026-11-02", "live": True, "updated_at": _t.time(), "map": [{"symbol": "X"}], "picks": [],
                   "bullish": [row], "bearish": [], "index_chg": None}
    board.details = {"X": {"bars": bars, "ctx": ctx}}
    e.state.on_price("X", 101.0, _t.time(), 5000)
    v = board.live_view()
    assert v["rows"]["X"]["last"] == 101.0 and v["chg"]["X"] == 1.0
    assert v["rows"]["X"]["confirm"]["checks"]["prev_day"]["ok"] is False
    board._live["at"] = 0                                            # older than LIVE_S: re-read
    e.state.on_price("X", 102.5, _t.time(), 6000)
    v = board.live_view("X", after=0)
    assert v["rows"]["X"]["confirm"]["checks"]["prev_day"]["ok"] is True and v["rows"]["X"]["chg"] == 2.5
    assert v["tail"]["symbol"] == "X" and v["tail"]["bars"] and v["tail"]["forming"]


def test_volume_so_far_against_the_usual_volume_by_now():
    from apps.trading_engine.obboard import OrderBlockBoard
    from market.calendar import IST
    prof = [1000.0 * (k + 1) for k in range(75)]                     # 1,000 a candle
    at = dt.datetime(2026, 11, 2, 9, 27, 30, tzinfo=IST)            # half way through the third candle
    assert OrderBlockBoard._rvol_now(5000.0, prof, at) == 2.0       # usual by now 2,500
    assert OrderBlockBoard._rvol_now(None, prof, at) is None
