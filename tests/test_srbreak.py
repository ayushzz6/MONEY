"""Strong-level finder: repeated swing highs make a level; future swing points never leak in."""
import numpy as np

from research.lib import srbreak as SB


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    base = 105 + rng.normal(0, 0.05, n)
    # price keeps rising into 110 and failing there: a strong resistance
    c = np.clip(base + 6 * np.sin(np.arange(n) / 6.0), None, 109.6)
    h = np.where(np.isclose(c, 109.6), 110.0, c + 0.5)
    l = c - 0.5
    atr = np.full(n, 1.0)
    return h, l, c, atr


def test_repeated_highs_form_a_strong_resistance():
    h, l, c, atr = _series()
    lv = SB.daily_levels(h, l, c, atr)
    days = np.flatnonzero(np.isfinite(lv["res"]))
    assert days.size > 0
    near = np.abs(lv["res"][days] - 110.0) <= 0.6
    assert near.mean() > 0.5                      # mostly the 110 ceiling
    assert (lv["res_t"][days] >= 3).all()          # only strong levels are reported


def test_levels_ignore_the_future():
    h, l, c, atr = _series(seed=1)
    full = SB.daily_levels(h, l, c, atr)
    d = 250
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[d:] *= 1.5                                   # rewrite everything from day d on
    l2[d:] *= 0.5
    c2[d:] *= 1.3
    cut = SB.daily_levels(h2, l2, c2, atr)
    for k in ("res", "res_t", "sup", "sup_t"):
        assert np.allclose(full[k][:d], cut[k][:d], equal_nan=True), k


def test_swing_cost_is_about_half_a_percent_on_twenty_thousand():
    c = SB.swing_cost(20000.0, slip_bps=0)
    assert 90 < c < 140                             # STT both sides + brokerage + stamp + DP + GST
