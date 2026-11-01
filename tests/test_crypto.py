"""Crypto & XAUUSD board: Binance messages, the UTC-day board, confirmations, cheap-coin prices. No network."""
import time

from helpers import FakeSettings

from apps.trading_engine import cxboard as CX
from market import orderblocks as OB
from market.candles.bars import DailyBar
from market.crypto import binance as B
from market.crypto import confirmations as CC

DAY0 = CX.day0()


def _daily(close=100.0, n=25):
    out = []
    for i in range(n, 0, -1):
        d = time.strftime("%Y-%m-%d", time.gmtime(DAY0 - 86400 * i))
        out.append(DailyBar(d, close - 1, close + 1.5, close - 1.5, close + (0.2 if i % 2 else -0.2), 1000.0))
    return out


def _trend(start=100.0, n=40, step=0.08):
    """A clean up-move from 00:00 UTC with a pullback in the middle, so an order block forms and is retested."""
    bars, p = [], start
    for i in range(n):
        d = -2.5 * step if 18 <= i < 22 else step
        o, c = p, p + d
        bars.append(B.CBar(DAY0 + 300 * i, o, max(o, c) + 0.03, min(o, c) - 0.03, c, 10.0, tb=6.0))
        p = c
    return bars


def board(**cfg):
    cx = CX.CryptoBoard(FakeSettings(crypto=dict({"enabled": True, "symbols": ["BTCUSDT", "ETHUSDT"], "gold": "XAUUSDT"}, **cfg)))
    for s, bars in (("BTCUSDT", _trend()), ("ETHUSDT", _trend(step=-0.08)), ("XAUUSDT", _trend(step=0.0))):
        cx.closed[s], cx.daily[s], cx.hourly[s] = bars, _daily(), []
    cx.oi = {"BTCUSDT": {"chg": 2.5}, "ETHUSDT": {"chg": -0.5}}
    return cx


def test_the_board_ranks_coins_by_strength_and_keeps_gold_apart():
    cx = board()
    cx.refresh()
    v = cx.view()
    assert v["ready"] and v["coins"] == 2 and v["gold"]["symbol"] == "XAUUSDT"
    assert [r["symbol"] for r in v["bullish"]] == ["BTCUSDT"] and [r["symbol"] for r in v["bearish"]] == ["ETHUSDT"]
    btc = v["bullish"][0]
    assert btc["checks"]["btc"] is None and btc["aligned_of"] == 3          # BTC is not aligned with itself
    assert btc["confirm"]["checks"]["oi"]["ok"] is True                     # OI up 2.5% with price up: long build-up
    assert btc["confirm"]["checks"]["taker"]["ok"] is True                  # 60% of volume bought by takers
    assert v["gold"]["side"] == 0 and v["gold"]["rule"] == "flat or choppy" and v["gold"]["confirm"] is None
    assert v["live"]["BTCUSDT"]["price"] == cx.closed["BTCUSDT"][-1].c


def test_websocket_candles_update_the_forming_bar_then_close_it():
    cx = board()
    k = {"t": (DAY0 + 300 * 40) * 1000, "o": "103", "h": "103.5", "l": "102.9", "c": "103.2", "v": "5", "V": "3", "x": False}
    cx._on_msg("BTCUSDT", "kline", {"k": k})
    assert cx.forming["BTCUSDT"].c == 103.2 and len(cx.closed["BTCUSDT"]) == 40
    assert cx.live()["BTCUSDT"]["price"] == 103.2
    cx._on_msg("BTCUSDT", "kline", {"k": dict(k, c="103.4", x=True)})
    assert cx.forming["BTCUSDT"] is None and cx.closed["BTCUSDT"][-1].c == 103.4 and cx.closed["BTCUSDT"][-1].tb == 3.0
    # the next candle's closing message is missed: its last update stands in when the one after starts
    nxt = dict(k, t=(DAY0 + 300 * 41) * 1000, c="103.6", x=False)
    cx._on_msg("BTCUSDT", "kline", {"k": nxt})
    cx._on_msg("BTCUSDT", "kline", {"k": dict(k, t=(DAY0 + 300 * 42) * 1000, c="103.7", x=False)})
    assert cx.closed["BTCUSDT"][-1].start == DAY0 + 300 * 41 and cx.closed["BTCUSDT"][-1].c == 103.6
    cx._on_msg("BTCUSDT", "ticker", {"c": "103.7", "o": "100", "h": "104", "l": "99", "q": "1000000"})
    assert abs(cx.live()["BTCUSDT"]["chg24"] - 3.7) < 1e-9


def test_open_interest_since_midnight_and_relative_volume():
    rows = [(DAY0 - 600, 100.0), (DAY0 - 300, 101.0), (DAY0 + 300, 102.0), (DAY0 + 3600, 103.02)]
    assert abs(CX.oi_change(rows, DAY0)["chg"] - 2.0) < 1e-9                # from the last reading before 00:00
    hourly = [B.CBar(DAY0 - 86400 * d + 3600 * h, 1, 1, 1, 1, 10.0) for d in range(1, 8) for h in range(24)]
    assert abs(CC.rel_volume(30.0, hourly, DAY0, 2 * 3600) - 1.5) < 1e-9    # 30 against 20 by 02:00 UTC
    assert CC.rel_volume(30.0, hourly[:24 * 3], DAY0, 3600) is None         # under 5 days of history


def test_confirmations_disagree_when_the_evidence_points_the_other_way():
    x = {"last": 98.0, "day_chg": -1.0, "oi_chg": 3.0, "taker": 0.45, "rvol": 2.0, "prev_high": 101, "prev_low": 99,
         "sma": 100.0, "cpr": {"pivot": 100, "bc": 99.8, "tc": 100.2, "width_pct": 0.4}, "cpr_rank": 0.9}
    bull, bear = CC.evaluate(1, x), CC.evaluate(-1, x)
    assert bull["checks"]["oi"]["ok"] is False and bear["checks"]["oi"]["ok"] is True   # OI up with price down: short build-up
    assert bear["checks"]["taker"]["ok"] and not bull["checks"]["taker"]["ok"]
    assert bear["checks"]["prev_day"]["ok"] and bear["checks"]["trend"]["ok"] and bear["checks"]["cpr_side"]["ok"]
    assert bear["score"] == 7 and bear["of"] == 7 and CC.evaluate(1, {})["of"] == 0


def test_cheap_coin_zones_keep_their_digits():
    assert OB._r(0.0758879) == 0.0758879 and OB._r(8.84853) == 8.8485 and OB._r(3893.456) == 3893.46
    z = OB.Zone(side=1, formed_at=0, ob_start=0, top=0.07589, bottom=0.07577, z=0.07589, invalid=0.0757580, targets=[0.07626])
    assert z.to_dict()["invalid"] == 0.075758 and z.to_dict()["targets"] == [0.07626]
    info = {"symbols": [{"symbol": "DOGEUSDT", "status": "TRADING", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00001000"}]},
                        {"symbol": "BCHUSDT", "status": "TRADING", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10000000"}]}]}
    assert B._decimals(info) == {"DOGEUSDT": 5, "BCHUSDT": 1}


def test_switched_off_and_refused_location():
    assert CX.CryptoBoard(FakeSettings(crypto={"enabled": False})).view()["error"].startswith("crypto is switched off")

    class E(Exception):
        response = type("R", (), {"status_code": 451})()
    assert "refused requests from this location" in CX.why(E("451"))
