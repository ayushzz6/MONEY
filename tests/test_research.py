import datetime as dt

import numpy as np
import pytest

from research import options as op
from research.costs import option_order_charges
from research.data import DayMatrix
from research.sim import bracket_exits, stop_entries


def test_put_call_parity_and_monotonic_in_vol():
    S, K, T = 22600.0, 22600.0, 5 / 252
    c, p = float(op.bs(S, K, T, 0.13, True)), float(op.bs(S, K, T, 0.13, False))
    assert c - p == pytest.approx(S - K * np.exp(-op.R * T), rel=1e-6)
    assert float(op.bs(S, K, T, 0.20, True)) > c
    # ATM straddle ~ 0.8 x S x sigma x sqrt(T)
    assert c + p == pytest.approx(0.8 * S * 0.13 * np.sqrt(T), rel=0.08)


def test_skew_makes_otm_puts_richer():
    assert float(op.iv_at(22000, 22600, 0.13)) > 0.13 > float(op.iv_at(23200, 22600, 0.13))


def test_nifty_weekly_expiry_calendar():
    days = [dt.date(2025, 8, 1) + dt.timedelta(days=i) for i in range(500)]
    days = [d for d in days if d.weekday() < 5 and d != dt.date(2026, 10, 20)]
    exps = op.nifty_weekly_expiries(days)
    assert dt.date(2025, 8, 28) in exps          # last Thursday expiry
    assert dt.date(2025, 9, 2) in exps           # first Tuesday expiry
    assert dt.date(2026, 10, 19) in exps         # Tuesday Oct 20 2026 holiday -> Monday
    clock = op.ExpiryClock(days, exps)
    assert clock.expiry_for(dt.date(2026, 10, 19), 0) == dt.date(2026, 10, 19)
    assert clock.expiry_for(dt.date(2026, 10, 19), 1) == dt.date(2026, 10, 27)
    # one full session plus nothing left today
    assert clock.T(dt.date(2026, 10, 26), 15 * 60 + 30, dt.date(2026, 10, 27)) == pytest.approx(375 / op.YEAR_MIN)


def test_option_charges_include_stt_on_sells_only():
    assert option_order_charges(10000, "sell") - option_order_charges(10000, "buy") == pytest.approx(10 - 0.3, abs=0.01)


def _mat(rows):
    a = np.array(rows, float)
    o, h, l, c = (a[:, i][None, :].repeat(1, 0) for i in range(4))
    pad = lambda x: np.concatenate([x, np.repeat(x[:, -1:], 75 - x.shape[1], axis=1)], axis=1)
    o, h, l, c = pad(o), pad(h), pad(l), pad(c)
    return DayMatrix("T", np.array(["2026-10-23"], dtype="datetime64[D]"), o, h, l, c, np.ones_like(o), np.array([0]))


def test_stop_entry_and_bar_path_rule():
    # bar 1: green bar trades through 101 (entry) with a low of 99.5 that came BEFORE the high
    m = _mat([(100, 100.5, 99.8, 100.2), (100.2, 101.5, 99.5, 101.4), (101.4, 102, 101.3, 101.9)])
    k, fill, ok = stop_entries(m, [0], [101.0], [1], 1, 70)
    assert ok[0] and k[0] == 1 and fill[0] == 101.0
    es, px, oc = bracket_exits(m, [0], k, [1], [99.9], [np.nan], 71, include_entry_bar=True)
    assert oc[0] == 0                             # the low was before the fill: not stopped
    # same bar but red (high first, then the low): stopped in the entry bar
    m2 = _mat([(100, 100.5, 99.8, 100.2), (100.6, 101.5, 99.5, 99.7), (99.7, 99.9, 99.6, 99.8)])
    k, fill, ok = stop_entries(m2, [0], [101.0], [1], 1, 70)
    es, px, oc = bracket_exits(m2, [0], k, [1], [99.9], [np.nan], 71, include_entry_bar=True)
    assert oc[0] == -1 and es[0] == 1 and px[0] == 99.9


def test_filters_never_use_volume_before_it_is_known():
    """top10 rvol ranks on the 09:20 + 09:25 bars (known at 09:30); a stop order filled in the 09:20 or 09:25
    bar must not use it. top10 rvol5 ranks on the 09:15 bar (known at 09:20)."""
    from types import SimpleNamespace
    from research.factory_equity import filter_mask
    n = 3
    lv = SimpleNamespace(rvol_rank=np.ones(n, int), rvol5_rank=np.ones(n, int), rvol15=np.full(n, 3.0),
                         nifty_dir=np.ones((n, 75)))
    rows, sides = np.arange(n), np.ones(n, int)
    slots = np.array([1, 2, 3])
    args = (None, None, lv, None, rows, slots, sides)
    assert list(filter_mask("top10 rvol", *args, touch=True)) == [False, False, True]
    assert list(filter_mask("top10 rvol", *args, touch=False)) == [False, True, True]
    assert list(filter_mask("top10 rvol5", *args, touch=True)) == [True, True, True]
    assert list(filter_mask("rvol", *args, touch=True)) == [False, False, True]
    # the NIFTY's direction: a stop fill inside bar 0 cannot know bar 0's close
    lv.nifty_dir[:, 0], lv.nifty_dir[:, 1:] = -1, 1
    assert list(filter_mask("nifty", None, None, lv, None, rows, np.array([0, 1, 2]), sides, touch=True)) == [False, False, True]
