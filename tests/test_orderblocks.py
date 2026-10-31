"""Order-block board rules: strength, choppiness, zones, breaks, retests, picks."""
from helpers import make_day, session_start

from market import orderblocks as OB

DAY = "2026-10-26"


def _trend_day():
    """Drops into a base, a red candle, then a push that closes through the swing high, then a pullback."""
    rows = [
        (100.0, 100.6, 99.8, 100.4), (100.4, 101.0, 100.2, 100.8), (100.8, 101.2, 100.5, 100.6),   # swing high 101.2
        (100.6, 100.7, 100.0, 100.1), (100.1, 100.3, 99.6, 99.7), (99.7, 99.9, 99.4, 99.8),
        (99.8, 99.9, 99.3, 99.4),                                                                  # last red candle = order block
        (99.4, 100.6, 99.4, 100.5), (100.5, 101.6, 100.4, 101.5),                                 # push, closes above 101.2
        (101.5, 102.2, 101.3, 102.0), (102.0, 102.4, 101.7, 101.9), (101.9, 102.0, 101.2, 101.4),
        (101.4, 101.6, 100.6, 100.8), (100.8, 101.9, 100.7, 101.8), (101.8, 102.6, 101.7, 102.5),
    ]
    return make_day(DAY, rows)


def test_a_push_through_the_swing_high_marks_the_last_red_candle():
    zones = OB.detect_zones(_trend_day())
    bull = [z for z in zones if z.side > 0]
    assert bull, "a bullish order block should form"
    z = bull[0]
    assert (z.bottom, z.top) == (99.3, 99.9)                    # the last red candle before the push
    assert z.z == z.top and z.invalid < z.bottom
    assert z.status in ("fresh", "retested")


def test_a_close_through_the_far_edge_breaks_the_zone():
    bars = _trend_day() + make_day(DAY, [(99.5, 99.6, 98.8, 99.0)])[:1]
    bars[-1] = make_day(DAY, [(0, 0, 0, 0)] * 15 + [(100.0, 100.1, 98.9, 99.0)])[-1]   # bar 15 closes under 99.3
    bull = [z for z in OB.detect_zones(bars) if z.side > 0]
    assert bull[0].status == "broken"
    assert OB.current_setup(OB.detect_zones(bars), 1) is not bull[0]
    t0 = session_start(DAY)
    assert bull[0].entry_at == t0 + 15 * 300 and bull[0].broken_at == t0 + 16 * 300   # touched Z in the candle that broke it


def test_zone_times_say_when_the_entry_was_given_and_when_price_came_back():
    t0 = session_start(DAY)
    fresh = next(z for z in OB.detect_zones(_trend_day()) if z.side > 0)
    assert fresh.known_at == t0 + 9 * 300                   # the 09:55 candle closed through 101.2: Z given at 10:00
    assert fresh.entry_at is None and fresh.broken_at is None
    bars = _trend_day() + make_day(DAY, [(0, 0, 0, 0)] * 15 + [(100.8, 100.9, 99.8, 100.2)])[15:]
    z = next(z for z in OB.detect_zones(bars) if z.side > 0)
    assert z.entry_at == t0 + 15 * 300                      # back at Z (99.9) in the 10:30 candle
    assert (z.status, z.visits, z.broken_at) == ("retested", 1, None)
    assert z.to_dict()["known_at"] == fresh.known_at


def test_strength_is_zero_when_the_day_is_choppy():
    zig = make_day(DAY, [(100, 101, 99, 101 if i % 2 else 99) for i in range(20)])
    s = OB.strength(zig, prev_close=100.0, atr=2.0)
    assert s["clean"] < OB.CHOPPY and s["score"] == 0.0


def test_a_clean_trend_scores_by_its_move_in_atrs():
    up = make_day(DAY, [(100 + 0.2 * i, 100.25 + 0.2 * i, 99.95 + 0.2 * i, 100.2 + 0.2 * i) for i in range(20)])
    s = OB.strength(up, prev_close=100.0, atr=2.0)
    assert s["clean"] > 0.95 and s["place"] > 0.9
    assert 1.8 < s["move_atr"] < 2.1 and s["score"] > 1.5


def test_event_risk_and_picks_respect_the_sector_cap():
    rows = [{"symbol": f"S{i}", "sector": "Banks" if i < 3 else "IT", "score": 2.0 - 0.1 * i, "event": False} for i in range(6)]
    rows.append({"symbol": "JUMP", "sector": "Pharma", "score": 5.0, "event": True})
    out = OB.rank_board(rows)
    picks = [r["symbol"] for r in out["bullish_picks"]]
    assert "JUMP" not in picks and len(picks) == 3
    assert sum(1 for r in out["bullish_picks"] if r["sector"] == "Banks") <= 2
    assert next(r for r in rows if r["symbol"] == "S2")["rule"] == "not picked · sector cap"
    assert next(r for r in rows if r["symbol"] == "JUMP")["rule"] == "not picked · event risk"
    assert OB.event_risk({"move_atr": 3.5, "change_pct": 2.0}) and not OB.event_risk({"move_atr": 1.0, "change_pct": 2.0})


def test_backtest_strength_matches_the_live_board():
    """research.lib.obtest rebuilds the board for every bar; its scores must equal market.orderblocks.strength."""
    from collections import namedtuple
    import numpy as np
    from research.lib import obtest as T
    rng = np.random.default_rng(3)
    S, D = 2, 18
    C = (100 * np.exp(np.cumsum(rng.normal(0, 0.002, (S, D, 75)), axis=2))).astype(np.float32)
    O = np.concatenate([C[:, :, :1] * 0.999, C[:, :, :-1]], axis=2)
    H, L = np.maximum(O, C) * 1.001, np.minimum(O, C) * 0.999
    Bd = T.board_arrays({"o": O, "h": H, "l": L, "c": C, "v": np.ones_like(C)}, ["A", "B"], np.ones((S, D), bool))
    Bar = namedtuple("Bar", "start o h l c")
    for s in range(S):
        for d in (16, 17):
            for k in (4, 40, 74):
                bars = [Bar(i, O[s, d, i], H[s, d, i], L[s, d, i], C[s, d, i]) for i in range(k + 1)]
                ref = OB.strength(bars, float(C[s, d - 1, -1]), float(Bd["atr"][s, d]))["score"]
                assert abs(ref - float(Bd["score"][s, d, k])) < 2e-3
