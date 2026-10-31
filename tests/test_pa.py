"""Round 6 price-action helpers: rolling windows, same-session memory and level tracking only look back."""
import numpy as np

from research.lib import pa


def test_roll_covers_the_window_ending_lag_bars_back():
    A = np.arange(1, 151, dtype=float).reshape(2, 75)
    m = pa.roll(A, 3, "max", lag=1)
    assert m[0, 5] == A[0, 4]                         # bars 2-4 before bar 5
    assert pa.roll(A, 3, "min", lag=0)[0, 5] == A[0, 3]
    B = A.copy()
    B[0, 6:] = 1e9                                   # the future must not matter
    assert pa.roll(B, 3, "max", lag=1)[0, 5] == m[0, 5]


def test_any_before_today_resets_each_session_and_skips_the_current_bar():
    B = np.zeros((2, 75), bool)
    B[0, 10] = True
    out = pa.any_before_today(B, lag=1)
    assert not out[0, 10] and out[0, 11] and out[0, 74]
    assert not out[1].any()                          # the next session starts clean


def test_ffill_day_carries_a_level_forward_within_the_session_only():
    A = np.full((2, 75), np.nan)
    A[0, 5] = 100.0
    out = pa.ffill_day(A)
    assert np.isnan(out[0, 4]) and out[0, 5] == 100.0 and out[0, 74] == 100.0
    assert np.isnan(out[1]).all()


def test_prev_level_is_the_value_before_the_latest_change():
    A = np.full((1, 75), np.nan)
    A[0, 3:10] = 50.0
    A[0, 10:] = 55.0
    out = pa.prev_level(A)
    assert np.isnan(out[0, 9]) and out[0, 10] == 50.0 and out[0, 74] == 50.0


def test_every_rule_returns_full_shaped_signals_and_stops():
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    D = 4
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, (D, 75)), axis=1))
    O = np.concatenate([C[:, :1], C[:, :-1]], axis=1)
    H, L = np.maximum(O, C) * 1.001, np.minimum(O, C) * 0.999
    lv = SimpleNamespace(wk_low=np.full(D, 95.0), wk_high=np.full(D, 105.0))
    f = SimpleNamespace(prev_close=np.full(D, 100.0), prev_high=np.full(D, 101.0), prev_low=np.full(D, 99.0))
    run = np.maximum.accumulate
    X = SimpleNamespace(O=O, H=H, L=L, C=C, V=np.ones((D, 75)), f=f, lv=lv, rv=np.ones((D, 75)), atr5=np.full((D, 75), 0.2),
                        ema9=C, ema20=C, ema21=C, ema50=C, vwap=C, day_hi=run(H, axis=1), day_lo=-run(-L, axis=1),
                        swing_hi=run(H, axis=1), swing_lo=-run(-L, axis=1), or15_hi=np.full((D, 75), 101.0), or15_lo=np.full((D, 75), 99.0))
    for k, name, fam, src, rule, fn in pa.NEW:
        Lg, Sh, stL, stS = fn(X)
        assert np.shape(Lg) == (D, 75) and np.shape(Sh) == (D, 75), k
        assert (np.asarray(stL, float) * np.ones((D, 75))).shape == (D, 75), k
