"""Round 5 option maths: pricing, implied volatility, trading time, charges, signal timing."""
import numpy as np

from research.costs import option_order_charges
from research.opt import grid as G
from research.opt import model as M
from research.opt import signals as SG


def test_implied_vol_round_trip_and_parity():
    F, T = 24000.0, 2 * 375 / M.YEAR_MIN
    K = np.array([23400.0, 24000.0, 24600.0])
    for cp in (1, -1):
        p = M.black(F, K, T, 0.14, cp)
        assert np.allclose(M.implied_vol(p, F, K, T, cp), 0.14, atol=1e-4)
    c, p = M.black(F, 24000.0, T, 0.14, 1), M.black(F, 24000.0, T, 0.14, -1)
    assert np.isclose(24000.0 + np.exp(M.R * T) * (c - p), F)


def test_price_below_intrinsic_has_no_iv():
    assert np.isnan(M.implied_vol(50.0, 24200.0, 24000.0, 0.01, 1))   # intrinsic is 200


def test_trading_time_runs_to_1530_on_expiry_day():
    assert np.isclose(M.t_years(0, 0) * M.YEAR_MIN, 375)               # expiry day, at 09:15
    assert np.isclose(M.t_years(1, 375) * M.YEAR_MIN, 375)             # the close before expiry day
    assert M.t_years(0, 375) * M.YEAR_MIN >= 1                         # never zero


def test_vectorised_charges_match_the_cost_model():
    for v in (5000.0, 25000.0, 150000.0):
        for side in ("buy", "sell"):
            assert np.isclose(G.charges(v, side), option_order_charges(v, side))


def test_first_signal_takes_the_earliest_bar_in_the_window():
    up = np.zeros((2, 75), bool)
    dn = np.zeros((2, 75), bool)
    up[0, 10], dn[0, 7] = True, True          # day 0: the short comes first
    up[1, 2] = True                            # day 1: before the window opens
    slot, side = SG._first(up, dn, start=5)
    assert slot[0] == 7 and side[0] == -1
    assert slot[1] == -1 and side[1] == 0
