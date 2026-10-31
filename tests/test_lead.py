"""Round 4 helpers: nothing from today or later leaks into 'known before' values; costs are sane."""
import numpy as np

from research.lead import intraday as I


def test_prior_mean_excludes_today_and_later():
    x = np.arange(10, dtype=np.float32)[None, :]          # one stock, 10 days
    m = I.prior_mean(x, 3)
    assert np.isnan(m[0, 0])
    assert m[0, 3] == np.float32((0 + 1 + 2) / 3)          # day 3 sees days 0-2 only
    y = x.copy()
    y[0, 5:] = 1e6                                         # change the future
    assert np.allclose(I.prior_mean(y, 3)[0, :6], m[0, :6], equal_nan=True)


def test_last_k_bar_return_uses_only_closed_bars():
    C = np.linspace(100, 174, 75, dtype=np.float32)[None, None, :]
    O = C.copy()
    r = I._retk(C, O, 3)
    t = I.EVAL[5]
    assert np.isclose(r[0, 0, 5], C[0, 0, t] / C[0, 0, t - 3] - 1)
    C2 = C.copy()
    C2[..., t + 1:] = 1.0                                  # later bars must not matter
    assert np.isclose(I._retk(C2, O, 3)[0, 0, 5], r[0, 0, 5])


def test_forward_return_starts_at_the_next_bar_open():
    O = np.arange(75, dtype=np.float32)[None, :] + 100
    C = O + 0.5
    f = I.forward(C, O)["15 min"]
    t = I.EVAL[0]
    assert np.isclose(f[0, 0], C[0, t + 3] / O[0, t + 1] - 1)


def test_round_trip_cost_on_one_lakh():
    c = I.cost_bps(100_000, slip_bps=0)
    assert 7.5 < c < 9.0                                    # ~Rs 82 of charges on Rs 1 lakh
    assert np.isclose(I.cost_bps(100_000, slip_bps=5) - c, 10)


def test_cross_sectional_ic_and_picks():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 200)).astype(np.float32)
    assert np.allclose(I._ic(y, y), 1.0, atol=1e-5)
    g = I._picks(y, y, k=3)                                # picking on the outcome itself: all winners
    assert (g > 0).all()
