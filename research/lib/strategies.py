"""The strategy library: rules from the classic trading and technical-analysis books, on 5-minute NSE bars.

Each strategy returns, per session x slot: a long signal, a short signal (both decided at that bar's CLOSE,
using only that bar and earlier ones) and the protective stop the book uses for it. The engine enters at the
NEXT bar's open, so no signal ever trades on the bar that created it.

Sources (rules paraphrased, adapted to NSE 5-minute bars):
  Crabel - Day Trading with Short Term Price Patterns and Opening Range Breakout (1990)
  Fisher - The Logical Trader (ACD method, 2002)
  Williams - Long-Term Secrets to Short-Term Trading (volatility breakout, 1999)
  Dalton - Mind Over Markets (opening types, 1990/2013)
  Shannon - Technical Analysis Using Multiple Timeframes / Maximum Trading Gains with Anchored VWAP
  Murphy - Technical Analysis of the Financial Markets (S/R, moving averages, oscillators, 1999)
  Floor-trader pivots, CPR (Frank Ochoa, Secrets of a Pivot Boss, 2010), Camarilla (Nick Stott)
  Raschke & Connors - Street Smarts (Holy Grail, Turtle Soup, 80-20, 1995)
  Connors & Alvarez - Short Term Trading Strategies That Work (RSI(2), 2008)
  Wilder - New Concepts in Technical Trading Systems (RSI, ADX, Parabolic SAR, 1978)
  Appel - Technical Analysis: Power Tools for Active Investors (MACD, 2005)
  Elder - Trading for a Living / Come Into My Trading Room (Triple Screen, Impulse, 1993/2002)
  Bollinger - Bollinger on Bollinger Bands (2001); Carter - Mastering the Trade (TTM squeeze, 2005)
  Nison - Japanese Candlestick Charting Techniques (1991); Brooks - Trading Price Action Trends (2011)
  Wyckoff / Pruden - The Three Skills of Top Trading; Williams (Tom) - Master the Markets (VSA, 1993)
  Granville - New Key to Stock Market Profits (OBV, 1963); Morales & Kacher - Trade Like an O'Neil Disciple
  O'Neil - How to Make Money in Stocks (relative strength, leadership); sector rotation
  Gao, Han, Li & Zhou - Market Intraday Momentum (Journal of Financial Economics, 2018)
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from research.data import SLOTS
from research.sim import slot_of_close

S = np.arange(SLOTS)[None, :]
T0930, T0945, T1000, T1100, T1200, T1330, T1400, T1430 = (slot_of_close(t) for t in
                                                          ("09:30", "09:45", "10:00", "11:00", "12:00", "13:30", "14:00", "14:30"))


@dataclass(frozen=True)
class Strat:
    key: str
    name: str
    family: str
    source: str
    rule: str
    fn: Callable


def prev(A, k=1):
    """The value k bars earlier (the previous session's last bars for the first slots)."""
    flat = A.reshape(-1)
    out = np.empty_like(flat, dtype=float)
    out[:k] = np.nan
    out[k:] = flat[:-k]
    return out.reshape(A.shape)


def xup(a, b):
    return (a > b) & (prev(a) <= prev(b))


def xdn(a, b):
    return (a < b) & (prev(a) >= prev(b))


def col(x):
    return np.asarray(x, float)[:, None]


def colb(x):
    return np.asarray(x, bool)[:, None]


def window(a, b):
    return (S >= a) & (S <= b)


def none_like(X):
    return np.zeros(X.C.shape, bool)


def nan_like(X):
    return np.full(X.C.shape, np.nan)


def pair(L, S_, stL, stS):
    return L, S_, stL, stS


def mirror(fn):
    """Build long and short from one long rule written on (X, sign): sign=+1 long, -1 short."""
    def both(X):
        L, stL = fn(X, 1)
        Sx, stS = fn(X, -1)
        return L, Sx, stL, stS
    return both


# side-aware views: for a short, "high" is the low and comparisons flip
def hi(X, s):
    return X.H if s > 0 else X.L


def lo(X, s):
    return X.L if s > 0 else X.H


def gt(a, b, s):
    return a > b if s > 0 else a < b


def ge(a, b, s):
    return a >= b if s > 0 else a <= b


# ---------------------------------------------------------------- A. opening range and volatility
def orb(k_name, k_slot, stop_mid=True, day_mask=None):
    @mirror
    def fn(X, s):
        top = getattr(X, f"{k_name}_hi") if s > 0 else getattr(X, f"{k_name}_lo")
        bot = getattr(X, f"{k_name}_lo") if s > 0 else getattr(X, f"{k_name}_hi")
        sig = gt(X.C, top, s) & (S >= k_slot)
        if day_mask is not None:
            sig &= col(day_mask(X)).astype(bool)
        stop = (top + bot) / 2 if stop_mid else bot
        return sig, stop
    return fn


@mirror
def acd(X, s):
    """Fisher: A-up = opening range high + A (0.1 x daily ATR); valid once two closes hold beyond it."""
    a = 0.1 * col(X.atr_d)
    lvl = (X.or15_hi + a) if s > 0 else (X.or15_lo - a)
    sig = gt(X.C, lvl, s) & gt(prev(X.C), lvl, s) & (S >= 4)
    return sig, (X.or15_hi + X.or15_lo) / 2


def acd_fix(X):
    """ACD A-up confirmed by the daily pivot range: price on the same side of the CPR."""
    L, S_, stL, stS = acd(X)
    cL, cS = col(X.lv.TC), col(X.lv.BC)
    return L & (X.C > cL), S_ & (X.C < cS), stL, stS


@mirror
def williams_vb(X, s):
    lvl = col(X.o0 + s * 0.6 * X.prev_range)
    return gt(X.C, lvl, s) & (S >= 1), np.repeat(col(X.o0), SLOTS, 1)


@mirror
def gap_and_go(X, s):
    g = col(X.gap)
    first_dir = col(np.sign(X.C[:, 0] - X.O[:, 0]))
    sig = (g * s >= 0.01) & (first_dir == s) & gt(X.C, X.or5_hi if s > 0 else X.or5_lo, s) & (S >= 1) & (S <= T1100)
    return sig, (X.or5_lo if s > 0 else X.or5_hi)


@mirror
def gap_fade(X, s):
    """Gap of 1%+ against the trade that fails: price closes back through the first bar's far side."""
    g = col(X.gap)
    far = X.or5_hi if s > 0 else X.or5_lo                # fading a gap down: buy above the first bar's high
    sig = (g * s <= -0.01) & gt(X.C, far, s) & (S >= 2) & (S <= T1000)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def open_drive(X, s):
    """Dalton open-drive: the first three bars all close in one direction on heavy volume; enter at 09:30."""
    c = X.C
    seq = gt(c[:, 0], X.O[:, 0], s) & gt(c[:, 1], c[:, 0], s) & gt(c[:, 2], c[:, 1], s)
    heavy = np.nan_to_num(X.crv[:, 2]) >= 1.5
    sig = np.zeros(c.shape, bool)
    sig[:, T0930] = seq & heavy
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def open_test_drive(X, s):
    """Dalton open-test-drive: an early probe beyond yesterday's low (high) fails and price drives back."""
    ref = col(X.f.prev_low if s > 0 else X.f.prev_high)
    probed = (X.day_lo <= ref) if s > 0 else (X.day_hi >= ref)
    sig = probed & gt(X.C, (X.or15_hi + X.or15_lo) / 2, s) & (S >= T0930) & (S <= T1000)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def open_rejection(X, s):
    """Dalton open-rejection-reverse: opens beyond yesterday's range and falls back inside it."""
    ref = col(X.f.prev_low if s > 0 else X.f.prev_high)
    opened_out = (col(X.o0) < ref) if s > 0 else (col(X.o0) > ref)
    sig = opened_out & gt(X.C, ref, s) & (S >= 1) & (S <= T1100)
    return sig, (X.day_lo if s > 0 else X.day_hi)


def nr7_mask(X):
    return X.lv.nr7


def inside_mask(X):
    return X.lv.inside


# ---------------------------------------------------------------- B. VWAP
@mirror
def vwap_reclaim(X, s):
    sig = gt(X.C, X.vwap, s) & ~gt(prev(X.C), prev(X.vwap), s) & (S >= T1000) & \
        (col(X.o0) < X.vwap if s > 0 else col(X.o0) > X.vwap)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def vwap_pullback(X, s):
    """Shannon: trend above VWAP for an hour, a pullback that tags VWAP and closes back in the trend."""
    held = (X.above_vwap_n >= 12) if s > 0 else ((S + 1 - X.above_vwap_n) >= 12)
    tag = (X.L <= X.vwap * 1.001) if s > 0 else (X.H >= X.vwap * 0.999)
    sig = held & tag & gt(X.C, X.vwap, s) & gt(X.C, X.O, s) & (S >= T1000)
    return sig, (X.L if s > 0 else X.H)


@mirror
def vwap_band_revert(X, s):
    band = X.vwap - s * 2 * X.vsd
    sig = gt(X.C, band, s) & ~gt(prev(X.C), prev(band), s) & (S >= T1000)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def vwap_band_break(X, s):
    band = X.vwap + s * X.vsd
    sig = gt(X.C, band, s) & ~gt(prev(X.C), prev(band), s) & (np.nan_to_num(X.rv) >= 2) & (S >= T1000)
    return sig, X.vwap


@mirror
def orb_vwap_retest(X, s):
    """After a 15-minute range break, the first pullback to VWAP that holds."""
    broke = np.maximum.accumulate(gt(X.C, X.or15_hi if s > 0 else X.or15_lo, s) & (S >= 3), 1)
    tag = (X.L <= X.vwap * 1.001) if s > 0 else (X.H >= X.vwap * 0.999)
    sig = prev(broke.astype(float)).astype(bool) & tag & gt(X.C, X.vwap, s) & (S >= 4)
    return sig, (X.or15_lo if s > 0 else X.or15_hi)


# ---------------------------------------------------------------- C. support and resistance
@mirror
def pdh_break(X, s):
    ref = col(X.f.prev_high if s > 0 else X.f.prev_low)
    inside_open = (col(X.o0) < ref) if s > 0 else (col(X.o0) > ref)
    sig = inside_open & gt(X.C, ref, s) & ~gt(prev(X.C), ref, s) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def turtle_soup(X, s):
    """Raschke: a break of yesterday's low (high) that closes back inside within two bars: fade it."""
    ref = col(X.f.prev_low if s > 0 else X.f.prev_high)
    pierced = (np.minimum(prev(X.L), X.L) < ref) if s > 0 else (np.maximum(prev(X.H), X.H) > ref)
    sig = pierced & gt(X.C, ref, s) & (S >= 1)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def pivot_bounce(X, s):
    lvl = col(X.lv.S1 if s > 0 else X.lv.R1)
    touch = (X.L <= lvl) if s > 0 else (X.H >= lvl)
    sig = touch & gt(X.C, lvl, s) & gt(X.C, X.O, s) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def pivot_break(X, s):
    lvl = col(X.lv.R1 if s > 0 else X.lv.S1)
    sig = gt(X.C, lvl, s) & ~gt(prev(X.C), lvl, s) & (np.nan_to_num(X.rv) >= 1.5) & (S >= 1)
    return sig, np.repeat(col(X.lv.P), SLOTS, 1)


@mirror
def cpr_narrow(X, s):
    """Ochoa: a narrow central pivot range marks a likely trend day; trade the break of it after 09:30."""
    narrow = colb(X.lv.cpr_w < 0.0025)
    lvl = col(X.lv.TC if s > 0 else X.lv.BC)
    sig = narrow & gt(X.C, lvl, s) & ~gt(prev(X.C), lvl, s) & (S >= T0930)
    return sig, np.repeat(col(X.lv.BC if s > 0 else X.lv.TC), SLOTS, 1)


@mirror
def cpr_wide_revert(X, s):
    """A wide CPR marks a likely range day: fade the first stretch to S1 (R1)."""
    wide = colb(X.lv.cpr_w > 0.006)
    lvl = col(X.lv.S1 if s > 0 else X.lv.R1)
    touch = (X.L <= lvl) if s > 0 else (X.H >= lvl)
    sig = wide & touch & gt(X.C, lvl, s) & (S >= 1)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def camarilla_revert(X, s):
    lvl = col(X.lv.L3 if s > 0 else X.lv.H3)
    touch = (X.L <= lvl) if s > 0 else (X.H >= lvl)
    sig = touch & gt(X.C, lvl, s) & (S >= 1)
    return sig, np.repeat(col(X.lv.L4 if s > 0 else X.lv.H4), SLOTS, 1)


@mirror
def camarilla_break(X, s):
    lvl = col(X.lv.H4 if s > 0 else X.lv.L4)
    sig = gt(X.C, lvl, s) & ~gt(prev(X.C), lvl, s) & (S >= 1)
    return sig, np.repeat(col(X.lv.H3 if s > 0 else X.lv.L3), SLOTS, 1)


@mirror
def week_break(X, s):
    lvl = col(X.lv.wk_high if s > 0 else X.lv.wk_low)
    sig = gt(X.C, lvl, s) & ~gt(prev(X.C), lvl, s) & (S >= 1)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def fractal_break(X, s):
    """Bill Williams: a close beyond the last confirmed 5-bar swing, on volume."""
    lvl = X.swing_hi if s > 0 else X.swing_lo
    sig = gt(X.C, lvl, s) & ~gt(prev(X.C), prev(lvl), s) & (np.nan_to_num(X.rv) >= 1.5) & (S >= 2)
    return sig, (X.swing_lo if s > 0 else X.swing_hi)


@mirror
def late_day_high(X, s):
    """A new high of the day after noon, on volume."""
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    sig = gt(X.C, ref, s) & (np.nan_to_num(X.rv) >= 1.5) & (S >= T1200)
    return sig, X.vwap


@mirror
def sr_confluence(X, s):
    """A bounce off a cluster of supports (yesterday's low, S1, last week's low, VWAP - 2 sd) within 0.2%."""
    levels = [col(X.f.prev_low), col(X.lv.S1), col(X.lv.wk_low)] if s > 0 else \
        [col(X.f.prev_high), col(X.lv.R1), col(X.lv.wk_high)]
    band = X.vwap - s * 2 * X.vsd
    ext = lo(X, s)
    near = sum((np.abs(ext - lv_) <= 0.002 * X.C).astype(int) for lv_ in levels) + \
        (np.abs(ext - band) <= 0.002 * X.C).astype(int)
    sig = (near >= 2) & gt(X.C, X.O, s) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


# ---------------------------------------------------------------- D. trend indicators
@mirror
def ema_cross(X, s):
    sig = (xup(X.ema9, X.ema21) if s > 0 else xdn(X.ema9, X.ema21)) & gt(X.C, X.vwap, s) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def holy_grail(X, s):
    """Raschke: ADX > 30, a pullback to the 20 EMA, then a close beyond the prior bar's extreme."""
    trend = (X.adx > 30) & (X.pdi > X.ndi if s > 0 else X.ndi > X.pdi)
    touched = (prev(X.L) <= prev(X.ema20)) if s > 0 else (prev(X.H) >= prev(X.ema20))
    sig = trend & touched & gt(X.C, prev(hi(X, s)), s) & (S >= 2)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def macd_cross(X, s):
    sig = (xup(X.macd, X.macd_sig) if s > 0 else xdn(X.macd, X.macd_sig)) & gt(X.macd, 0, s) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def macd_zero(X, s):
    zero = np.zeros_like(X.macd)
    sig = (xup(X.macd, zero) if s > 0 else xdn(X.macd, zero)) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def supertrend_flip(X, s):
    sig = (X.st_dir == s) & (prev(X.st_dir) == -s) & (S >= 1)
    return sig, X.st_line


@mirror
def dmi_cross(X, s):
    sig = (xup(X.pdi, X.ndi) if s > 0 else xup(X.ndi, X.pdi)) & (X.adx > 20) & (X.adx > prev(X.adx)) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def psar_flip(X, s):
    flip = (X.psar < X.C) & (prev(X.psar) > prev(X.C)) if s > 0 else (X.psar > X.C) & (prev(X.psar) < prev(X.C))
    sig = flip & gt(X.C, X.ema50, s) & (S >= 1)
    return sig, X.psar


@mirror
def triple_screen(X, s):
    """Elder: the higher timeframe's trend (a 240-bar EMA rising, price above it) and a 5-minute stochastic
    turning up from oversold (down from overbought)."""
    tide = gt(X.ema240, prev(X.ema240, 12), s) & gt(X.C, X.ema240, s)
    wave = (xup(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) < 30)) if s > 0 else (xdn(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) > 70))
    return tide & wave & (S >= 2), (X.low3 if s > 0 else X.high3)


@mirror
def elder_impulse(X, s):
    green = gt(X.ema13, prev(X.ema13), s) & gt(X.macd_hist, prev(X.macd_hist), s)
    was = gt(prev(X.ema13), prev(X.ema13, 2), s) & gt(prev(X.macd_hist), prev(X.macd_hist, 2), s)
    sig = green & ~was & gt(X.C, X.vwap, s) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def ichimoku_tk(X, s):
    cross = xup(X.tenkan, X.kijun) if s > 0 else xdn(X.tenkan, X.kijun)
    sig = cross & gt(X.C, X.cloud_top if s > 0 else X.cloud_bot, s) & (S >= 2)
    return sig, X.kijun


@mirror
def donchian_break(X, s):
    sig = gt(X.C, X.don_hi20 if s > 0 else X.don_lo20, s) & (X.adx > 20) & (S >= 2)
    return sig, (X.don_lo10 if s > 0 else X.don_hi10)


@mirror
def keltner_break(X, s):
    band = X.kc_up if s > 0 else X.kc_lo
    sig = gt(X.C, band, s) & ~gt(prev(X.C), prev(band), s) & (S >= 2)
    return sig, X.ema20


@mirror
def ttm_squeeze(X, s):
    """Carter: Bollinger Bands inside the Keltner Channel for 6+ bars, then they release with momentum."""
    sq = (X.bb_up < X.kc_up) & (X.bb_lo > X.kc_lo)
    held = sum(prev(sq.astype(float), k) for k in range(1, 7)) >= 6
    released = ~sq & held
    mom = X.C - X.bb_mid
    sig = released & gt(mom, 0, s) & gt(mom, prev(mom), s) & (S >= 2)
    return sig, X.bb_mid


@mirror
def momentum_burst(X, s):
    burst = s * (X.C - prev(X.C, 3)) > 2 * X.atr5
    sig = burst & (np.nan_to_num(X.rv) >= 3) & (S >= 3)
    return sig, (X.low3 if s > 0 else X.high3)


# ---------------------------------------------------------------- E. mean reversion
@mirror
def rsi2(X, s):
    """Connors: RSI(2) under 5 (over 95) while price is on the trend side of the 200 EMA."""
    sig = ((X.rsi2 < 5) if s > 0 else (X.rsi2 > 95)) & gt(X.C, X.ema200, s) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def rsi_divergence(X, s):
    ll = gt(prev(lo(X, s), 12), lo(X, s), s)            # a new extreme against the trade vs 12 bars ago
    hl = gt(X.rsi14, prev(X.rsi14, 12), s)
    zone = (X.rsi14 < 35) if s > 0 else (X.rsi14 > 65)
    sig = ll & hl & zone & gt(X.C, X.O, s) & (S >= 3)
    return sig, (X.L if s > 0 else X.H)


@mirror
def bb_reentry(X, s):
    band = X.bb_lo if s > 0 else X.bb_up
    sig = gt(X.C, band, s) & ~gt(prev(X.C), prev(band), s) & (S >= 2)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def stoch_cross(X, s):
    cross = (xup(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) < 20)) if s > 0 else (xdn(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) > 80))
    return cross & gt(X.C, X.ema50, s) & (S >= 2), (X.low3 if s > 0 else X.high3)


@mirror
def willr_rev(X, s):
    lvl = np.full_like(X.willr, -80.0 if s > 0 else -20.0)
    cross = xup(X.willr, lvl) if s > 0 else xdn(X.willr, lvl)
    return cross & col(X.trend_up if s > 0 else X.trend_dn).astype(bool) & (S >= 2), (X.low5 if s > 0 else X.high5)


@mirror
def cci_rev(X, s):
    lvl = np.full_like(X.cci, -100.0 if s > 0 else 100.0)
    cross = xup(X.cci, lvl) if s > 0 else xdn(X.cci, lvl)
    return cross & col(X.trend_up if s > 0 else X.trend_dn).astype(bool) & (S >= 2), (X.low5 if s > 0 else X.high5)


@mirror
def overextension_fade(X, s):
    """By 09:45 the stock has run more than 0.8 daily ATR against the trade: fade back toward VWAP."""
    run = -s * (X.C - col(X.o0)) > 0.8 * col(X.atr_d)
    sig = np.zeros(X.C.shape, bool)
    sig[:, T0945] = run[:, T0945] & gt(X.C, X.O, s)[:, T0945]
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def raschke_8020(X, s):
    """Raschke 80-20: yesterday opened in the top 20% of its range and closed in the bottom 20% (for a long);
    today trades below yesterday's low, then closes back above it."""
    o_, c_ = X.f.d_open, X.f.d_close
    po, pc_ = np.r_[np.nan, o_[:-1]], np.r_[np.nan, c_[:-1]]
    ph, pl = X.f.prev_high, X.f.prev_low
    r = np.maximum(ph - pl, 1e-9)
    setup = ((po - pl) / r >= 0.8) & ((pc_ - pl) / r <= 0.2) if s > 0 else ((po - pl) / r <= 0.2) & ((pc_ - pl) / r >= 0.8)
    ref = col(pl if s > 0 else ph)
    pierced = (X.day_lo < ref) if s > 0 else (X.day_hi > ref)
    sig = col(setup).astype(bool) & pierced & gt(X.C, ref, s) & (S >= 1)
    return sig, (X.day_lo if s > 0 else X.day_hi)


# ---------------------------------------------------------------- F. price action and candlesticks
def _support(X, s):
    """Distance-to-support test used by the candle patterns: within 0.25% of a known level."""
    ext = lo(X, s)
    levels = [col(X.f.prev_low), col(X.lv.S1), X.vwap - 2 * X.vsd, X.prior_day_lo] if s > 0 else \
        [col(X.f.prev_high), col(X.lv.R1), X.vwap + 2 * X.vsd, X.prior_day_hi]
    return np.logical_or.reduce([np.abs(ext - lv_) <= 0.0025 * X.C for lv_ in levels])


@mirror
def engulfing_sr(X, s):
    po, pc_ = prev(X.O), prev(X.C)
    eng = gt(X.C, X.O, s) & gt(pc_, po, -s) & gt(X.C, po, s) & gt(pc_, X.O, s)
    return eng & _support(X, s) & (S >= 2), (np.minimum(X.L, prev(X.L)) if s > 0 else np.maximum(X.H, prev(X.H)))


@mirror
def hammer_sr(X, s):
    body = np.abs(X.C - X.O)
    rng = np.maximum(X.H - X.L, 1e-9)
    tail = (np.minimum(X.O, X.C) - X.L) if s > 0 else (X.H - np.maximum(X.O, X.C))
    pos = (X.C - X.L) / rng if s > 0 else (X.H - X.C) / rng
    return (tail >= 2 * body) & (pos >= 0.66) & _support(X, s) & (S >= 1), (X.L if s > 0 else X.H)


@mirror
def inside_bar_trend(X, s):
    inside = (prev(X.H) <= prev(X.H, 2)) & (prev(X.L) >= prev(X.L, 2))
    trend = gt(X.C, X.ema20, s) & gt(X.ema20, prev(X.ema20, 3), s)
    sig = inside & gt(X.C, prev(hi(X, s)), s) & trend & (S >= 3)
    return sig, prev(lo(X, s))


@mirror
def brooks_h2(X, s):
    """Brooks high 2: in a trend and still inside a pullback (under the high of the 8 bars before), the second
    bar whose high exceeds the prior bar's, with a pause bar after the first."""
    trend = gt(X.ema20, prev(X.ema20, 6), s) & gt(X.C, X.vwap, s)
    h = hi(X, s)
    up = gt(h, prev(h), s)
    upf = up.astype(float)
    first = (prev(upf, 2) + prev(upf, 3) + prev(upf, 4)) >= 1
    pause = prev(upf) == 0
    ref = np.maximum.reduce([prev(X.H, k) for k in range(1, 9)]) if s > 0 else \
        np.minimum.reduce([prev(X.L, k) for k in range(1, 9)])
    in_pullback = gt(ref, h, s)
    return up & first & pause & in_pullback & trend & (S >= 4), (X.low5 if s > 0 else X.high5)


@mirror
def three_bar_rev(X, s):
    b2 = gt(prev(X.O, 2), prev(X.C, 2), s)               # bar t-2 against the trade
    b1 = gt(prev(lo(X, s), 2), prev(lo(X, s)), s)        # t-1 makes a further extreme
    sig = b2 & b1 & gt(X.C, prev(hi(X, s)), s) & gt(X.C, prev(hi(X, s), 2), s) & (S >= 3)
    return sig, prev(lo(X, s))


@mirror
def outside_rev(X, s):
    outside = (X.H > prev(X.H)) & (X.L < prev(X.L))
    rng = np.maximum(X.H - X.L, 1e-9)
    pos = (X.C - X.L) / rng if s > 0 else (X.H - X.C) / rng
    leg = gt(prev(X.C, 6), X.C, s)                       # after a move against the trade
    return outside & (pos >= 0.75) & leg & (S >= 6), (X.L if s > 0 else X.H)


@mirror
def wrb(X, s):
    rng = X.H - X.L
    pos = (X.C - X.L) / np.maximum(rng, 1e-9) if s > 0 else (X.H - X.C) / np.maximum(rng, 1e-9)
    sig = (rng > 2 * X.rng_avg20) & (pos >= 0.8) & (np.nan_to_num(X.rv) >= 2) & (S >= 1)
    return sig, (X.H + X.L) / 2


# ---------------------------------------------------------------- G. volume
@mirror
def volume_breakout(X, s):
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    rng = np.maximum(X.H - X.L, 1e-9)
    pos = (X.C - X.L) / rng if s > 0 else (X.H - X.C) / rng
    sig = (np.nan_to_num(X.rv) >= 3) & gt(X.C, ref, s) & (pos >= 0.75) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def climax_reversal(X, s):
    """VSA selling climax: a wide bar against the trade on ultra-high volume that closes off its extreme,
    after a 1.5% move from the day's extreme."""
    rng = X.H - X.L
    against = gt(X.O, X.C, s)
    pos = (X.C - X.L) / np.maximum(rng, 1e-9) if s > 0 else (X.H - X.C) / np.maximum(rng, 1e-9)
    ext = (X.prior_day_hi - X.L) / X.C if s > 0 else (X.H - X.prior_day_lo) / X.C
    sig = against & (rng > 2 * X.rng_avg20) & (np.nan_to_num(X.rv) >= 3) & (pos >= 0.4) & (ext >= 0.015) & (S >= 2)
    return sig, (X.L if s > 0 else X.H)


@mirror
def obv_lead(X, s):
    """Granville: on-balance volume makes a 50-bar extreme before price does."""
    if s > 0:
        sig = (X.obv > X.obv_hi50) & (X.C < X.c_hi50) & (X.C > X.vwap)
    else:
        sig = (X.obv < X.obv_lo50) & (X.C > X.c_lo50) & (X.C < X.vwap)
    return sig & (S >= 3), (X.don_lo10 if s > 0 else X.don_hi10)


@mirror
def wyckoff_spring(X, s):
    ref = np.where(np.isnan(X.or15_lo), col(X.f.prev_low), X.or15_lo) if s > 0 else \
        np.where(np.isnan(X.or15_hi), col(X.f.prev_high), X.or15_hi)
    pierce = (X.L < ref) if s > 0 else (X.H > ref)
    sig = pierce & gt(X.C, ref, s) & (np.nan_to_num(X.rv) >= 1.5) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def in_play_momentum(X, s):
    """Stocks in play: cumulative volume 2.5x normal for the time of day, above VWAP, a new high of the day."""
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    sig = (np.nan_to_num(X.crv) >= 2.5) & gt(X.C, X.vwap, s) & gt(X.C, ref, s) & (S >= T0930)
    return sig, X.vwap


@mirror
def pocket_pivot(X, s):
    """Morales & Kacher: an up bar whose volume beats every down bar of the last 10, above the 20 EMA."""
    down = (prev(X.C) > X.C).astype(float) * X.V if s > 0 else (prev(X.C) < X.C).astype(float) * X.V
    flat = down.reshape(-1)
    mx = np.r_[np.full(10, np.nan), np.lib.stride_tricks.sliding_window_view(flat, 10).max(1)[:-1]].reshape(X.C.shape)
    sig = gt(X.C, prev(X.C), s) & (X.V > mx) & gt(X.C, X.ema20, s) & (S >= 2)
    return sig, (X.low5 if s > 0 else X.high5)


# ---------------------------------------------------------------- H. relative strength and sectors
@mirror
def rs_pullback(X, s):
    rs = s * (X.ret_open - X.nifty_ret) > 0.01
    sec = s * X.sector_ret > 0
    tag = (X.L <= X.vwap * 1.002) if s > 0 else (X.H >= X.vwap * 0.998)
    sig = rs & sec & tag & gt(X.C, X.vwap, s) & (S >= T1000)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def sector_leader(X, s):
    rank_ok = (X.sector_rank <= 2) if s > 0 else (X.sector_rank >= 13)
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    sig = rank_ok & gt(X.C, X.vwap, s) & gt(X.C, ref, s) & (S >= T0945)
    return sig, X.vwap


@mirror
def sector_catchup(X, s):
    strong = s * X.sector_ret > 0.008
    lag = s * (X.ret_open - X.sector_ret) < -0.005
    sig = strong & lag & gt(X.C, X.vwap, s) & ~gt(prev(X.C), prev(X.vwap), s) & (S >= T1000)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def sector_divergence_fade(X, s):
    """A stock up 2%+ while its sector is down 0.5%+ (for a short; mirrored for a long) after 11:00."""
    sig = (-s * X.ret_open > 0.02) & (s * X.sector_ret > 0.005) & gt(X.C, X.O, s) & (S >= T1100)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def market_beta(X, s):
    mkt = (s * X.nifty_ret > 0.005) & (X.nifty_above if s > 0 else X.nifty_below)
    beta = col(np.nan_to_num(X.beta) > 1.2).astype(bool)
    sig = mkt & beta & gt(X.C, X.vwap, s) & ~gt(prev(X.C), prev(X.vwap), s) & (S >= T1000)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def rs_new_high(X, s):
    """O'Neil: a new 20-day high (low) while the NIFTY is flat or against it."""
    ref = col(X.f.high20) if s > 0 else col(X.f.low20)
    sig = gt(X.C, ref, s) & ~gt(prev(X.C), ref, s) & (s * X.nifty_ret <= 0.002) & (S >= 1)
    return sig, (X.low5 if s > 0 else X.high5)


# ---------------------------------------------------------------- I. time of day
@mirror
def lunch_break(X, s):
    a, b = T1200 + 1, T1330
    hi_ = after_range(X.H, a, b, np.max)
    lo_ = after_range(X.L, a, b, np.min)
    lvl = hi_ if s > 0 else lo_
    sig = gt(X.C, lvl, s) & (np.nan_to_num(X.rv) >= 1.3) & (S > b)
    return sig, (hi_ + lo_) / 2


def after_range(A, a, b, fn):
    v = fn(A[:, a:b + 1], axis=1)
    out = np.repeat(v[:, None], SLOTS, 1)
    out[:, :b + 1] = np.nan
    return out


@mirror
def intraday_momentum(X, s):
    """Gao et al.: the first half hour's return and the return to 14:30 point the same way: ride the last hour."""
    r1 = X.C[:, slot_of_close("09:45")] / X.O[:, 0] - 1
    r2 = X.C[:, T1430] / X.C[:, slot_of_close("09:45")] - 1
    ok = (s * r1 > 0.003) & (s * r2 > 0)
    sig = np.zeros(X.C.shape, bool)
    sig[:, T1430] = ok
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def afternoon_trend(X, s):
    held = (X.above_vwap_n >= 0.8 * (S + 1)) if s > 0 else ((S + 1 - X.above_vwap_n) >= 0.8 * (S + 1))
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    sig = held & gt(X.C, ref, s) & (S >= T1330)
    return sig, X.vwap


# ---------------------------------------------------------------- J. indicator combinations
def _orb15_side(X, s):
    return gt(X.C, X.or15_hi if s > 0 else X.or15_lo, s) & (S >= 3), (X.or15_hi + X.or15_lo) / 2


@mirror
def orb_vwap_side(X, s):
    sig, st = _orb15_side(X, s)
    return sig & gt(X.C, X.vwap, s), st


@mirror
def orb_rsi(X, s):
    sig, st = _orb15_side(X, s)
    return sig & ((X.rsi14 > 55) if s > 0 else (X.rsi14 < 45)), st


@mirror
def orb5_supertrend(X, s):
    sig = gt(X.C, X.or5_hi if s > 0 else X.or5_lo, s) & (S >= 1) & (X.st_dir == s)
    return sig, (X.or5_lo if s > 0 else X.or5_hi)


@mirror
def orb_cpr(X, s):
    sig, st = _orb15_side(X, s)
    return sig & col(X.lv.cpr_w < 0.0035).astype(bool), st


@mirror
def range_rules_pullback(X, s):
    """Constance Brown's RSI range rules: in an uptrend RSI(14) holds 40-50 on pullbacks; buy the turn up."""
    trend = gt(X.ema20, X.ema50, s) & gt(X.C, X.ema50, s)
    zone = ((prev(X.rsi14) >= 40) & (prev(X.rsi14) <= 50)) if s > 0 else ((prev(X.rsi14) >= 50) & (prev(X.rsi14) <= 60))
    turn = gt(X.rsi14, prev(X.rsi14), s) & gt(X.C, prev(hi(X, s)), s)
    return trend & zone & turn & (S >= 3), (X.low5 if s > 0 else X.high5)


@mirror
def macd_adx(X, s):
    cross = xup(X.macd, X.macd_sig) if s > 0 else xdn(X.macd, X.macd_sig)
    return cross & (X.adx > 25) & (X.pdi > X.ndi if s > 0 else X.ndi > X.pdi) & (S >= 2), (X.low5 if s > 0 else X.high5)


@mirror
def squeeze_volume(X, s):
    sig, st = ttm_squeeze(X)[0:3:2] if s > 0 else (ttm_squeeze(X)[1], ttm_squeeze(X)[3])
    return sig & (np.nan_to_num(X.rv) >= 2), st


@mirror
def supertrend_vwap(X, s):
    flip = (X.st_dir == s) & (prev(X.st_dir) == -s)
    return flip & gt(X.C, X.vwap, s) & (S >= 1), X.st_line


@mirror
def heikin_ashi_turn(X, s):
    """Heikin-Ashi: the first strong HA candle (no lower shadow for a long) after two opposite ones."""
    o, h, l, c = (a.reshape(-1) for a in (X.O, X.H, X.L, X.C))
    hc = (o + h + l + c) / 4
    from scipy.signal import lfilter
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2
    ho[1:], _ = lfilter([0.5], [1, -0.5], hc[:-1], zi=[ho[0] * 0.5])
    hh, hl = np.maximum(h, np.maximum(ho, hc)), np.minimum(l, np.minimum(ho, hc))
    HC, HO, HH, HL = (a.reshape(X.C.shape) for a in (hc, ho, hh, hl))
    strong = (HC > HO) & (np.abs(np.minimum(HO, HC) - HL) <= 1e-9 * HC + 0.0001 * HC) if s > 0 else \
        (HC < HO) & (np.abs(HH - np.maximum(HO, HC)) <= 0.0001 * HC)
    opp = gt(prev(HO), prev(HC), s) & gt(prev(HO, 2), prev(HC, 2), s)
    return strong & opp & gt(X.C, X.vwap, s) & (S >= 3), (X.low3 if s > 0 else X.high3)


@mirror
def ribbon_pullback(X, s):
    """EMA 9 / 21 / 50 ribbon fanned in order; a pullback that tags the 9 and closes back beyond it."""
    fan = gt(X.ema9, X.ema21, s) & gt(X.ema21, X.ema50, s)
    tag = (X.L <= X.ema9) if s > 0 else (X.H >= X.ema9)
    return fan & tag & gt(X.C, X.ema9, s) & gt(X.C, X.O, s) & (S >= 2), (X.L if s > 0 else X.H)


@mirror
def keltner_rsi(X, s):
    band = X.kc_lo if s > 0 else X.kc_up
    stretch = (prev(X.C) < prev(band)) if s > 0 else (prev(X.C) > prev(band))
    return stretch & gt(X.C, band, s) & ((X.rsi14 < 35) if s > 0 else (X.rsi14 > 65)) & (S >= 2), (X.low3 if s > 0 else X.high3)


@mirror
def adx_day_high(X, s):
    ref = X.prior_day_hi if s > 0 else X.prior_day_lo
    return gt(X.C, ref, s) & (X.adx > 25) & (X.adx > prev(X.adx, 3)) & (S >= T0945), (X.low5 if s > 0 else X.high5)


@mirror
def orb_retest(X, s):
    """The 15-minute range breaks, price comes back to the range edge and holds it."""
    edge = X.or15_hi if s > 0 else X.or15_lo
    broke = np.maximum.accumulate(gt(X.C, edge, s) & (S >= 3), 1)
    tag = (X.L <= edge * 1.001) if s > 0 else (X.H >= edge * 0.999)
    return prev(broke.astype(float)).astype(bool) & tag & gt(X.C, edge, s) & (S >= 4), (X.or15_hi + X.or15_lo) / 2


@mirror
def no_supply(X, s):
    """VSA no supply: in an uptrend, a down bar on volume under half its normal; buy the next bar beyond it."""
    trend = gt(X.C, X.vwap, s) & gt(X.ema20, prev(X.ema20, 6), s)
    quiet = gt(prev(X.O), prev(X.C), s) & (np.nan_to_num(prev(X.rv), nan=9) < 0.5)
    return trend & quiet & gt(X.C, prev(hi(X, s)), s) & (S >= 3), prev(lo(X, s))


@mirror
def expansion_after_contraction(X, s):
    """Crabel: after an NR4 day, the first wide-range 5-minute bar in either direction starts the expansion."""
    rng = X.H - X.L
    pos = (X.C - X.L) / np.maximum(rng, 1e-9) if s > 0 else (X.H - X.C) / np.maximum(rng, 1e-9)
    sig = col(X.lv.nr4).astype(bool) & (rng > 1.8 * X.rng_avg20) & (pos >= 0.7) & (S >= 1)
    return sig, (X.H + X.L) / 2


@mirror
def gap_vwap(X, s):
    g = col(X.gap)
    return (g * s >= 0.005) & gt(X.C, X.vwap, s) & gt(X.C, X.or15_hi if s > 0 else X.or15_lo, s) & (S >= 3), X.vwap


@mirror
def first_pullback_gap(X, s):
    """After a gap-and-go, the first pullback of three bars that resumes in the gap's direction."""
    g = col(X.gap)
    pull = gt(prev(X.C, 3), prev(X.C), s)
    sig = (g * s >= 0.01) & pull & gt(X.C, prev(hi(X, s)), s) & gt(X.C, X.vwap, s) & (S >= 4) & (S <= T1100)
    return sig, (X.low5 if s > 0 else X.high5)


def _bars15(X):
    """Completed 15-minute bars, known at the close of their last 5-minute bar (slots 2, 5, 8, ...)."""
    D = X.C.shape[0]
    o = np.full(X.C.shape, np.nan); h = o.copy(); l = o.copy(); c = o.copy()
    for k in range(2, SLOTS, 3):
        o[:, k] = X.O[:, k - 2]
        h[:, k] = X.H[:, k - 2:k + 1].max(1)
        l[:, k] = X.L[:, k - 2:k + 1].min(1)
        c[:, k] = X.C[:, k]
    return o, h, l, c


@mirror
def engulf15(X, s):
    o, h, l, c = _bars15(X)
    po, pc_ = prev(o, 3), prev(c, 3)
    eng = gt(c, o, s) & gt(pc_, po, -s) & gt(c, po, s) & gt(pc_, o, s)
    return eng & (S >= 5) & gt(X.C, X.vwap, s), (np.minimum(l, prev(l, 3)) if s > 0 else np.maximum(h, prev(h, 3)))


@mirror
def inside15(X, s):
    o, h, l, c = _bars15(X)
    inside = (prev(h, 3) <= prev(h, 6)) & (prev(l, 3) >= prev(l, 6))
    return inside & gt(c, prev(h if s > 0 else l, 3), s) & (S >= 8), prev(l if s > 0 else h, 3)


@mirror
def vwap_two_bar(X, s):
    tag = (prev(X.L) <= prev(X.vwap)) if s > 0 else (prev(X.H) >= prev(X.vwap))
    rev = gt(prev(X.O), prev(X.C), s) & gt(X.C, X.O, s) & gt(X.C, prev(X.O), s)
    return tag & rev & gt(X.C, X.vwap, s) & (S >= T1000), (np.minimum(X.L, prev(X.L)) if s > 0 else np.maximum(X.H, prev(X.H)))


@mirror
def midday_reversal(X, s):
    """11:30-13:00: a new day extreme against the trade with RSI(14) not confirming, then a reversal bar."""
    ext = (X.L <= X.prior_day_lo) if s > 0 else (X.H >= X.prior_day_hi)
    div = gt(X.rsi14, prev(X.rsi14, 12), s)
    win = (S >= slot_of_close("11:30")) & (S <= slot_of_close("13:00"))
    return prev(ext.astype(float)).astype(bool) & div & gt(X.C, X.O, s) & win, (X.day_lo if s > 0 else X.day_hi)


@mirror
def last_hour_fade(X, s):
    """At 14:30 a move of 1.5 daily ATR from the open against the trade: fade into the close."""
    run = -s * (X.C - col(X.o0)) > 1.5 * col(X.atr_d) * 0.5
    sig = np.zeros(X.C.shape, bool)
    sig[:, T1430] = run[:, T1430] & gt(X.C, X.O, s)[:, T1430]
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def donchian55(X, s):
    flat = (X.H if s > 0 else X.L).reshape(-1)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(flat, 55)
    ref = np.r_[np.full(55, np.nan), (w.max(1) if s > 0 else w.min(1))[:-1]].reshape(X.C.shape)
    return gt(X.C, ref, s) & ~gt(prev(X.C), prev(ref), s) & (S >= 2), (X.don_lo20 if s > 0 else X.don_hi20)


@mirror
def pivot_rsi(X, s):
    lvl = col(X.lv.P)
    cross = (X.C > lvl) & (prev(X.C) <= lvl) if s > 0 else (X.C < lvl) & (prev(X.C) >= lvl)
    return cross & ((X.rsi14 > 50) if s > 0 else (X.rsi14 < 50)) & (S >= 2), (X.low5 if s > 0 else X.high5)


@mirror
def stoch_vwap(X, s):
    cross = (xup(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) < 25)) if s > 0 else (xdn(X.stoch_k, X.stoch_d) & (prev(X.stoch_k) > 75))
    return cross & gt(X.C, X.vwap, s) & (S >= T1000), (X.low3 if s > 0 else X.high3)


@mirror
def bb_band_walk(X, s):
    """Bollinger: two closes outside the band on volume = a band walk; ride it."""
    band = X.bb_up if s > 0 else X.bb_lo
    sig = gt(X.C, band, s) & gt(prev(X.C), prev(band), s) & (np.nan_to_num(X.rv) >= 1.5) & (S >= 3)
    return sig, X.bb_mid


@mirror
def obv_vwap(X, s):
    rising = gt(X.obv, prev(X.obv, 6), s)
    return rising & gt(X.C, X.vwap, s) & ~gt(prev(X.C), prev(X.vwap), s) & (S >= T1000), (X.low5 if s > 0 else X.high5)


@mirror
def cam_orb(X, s):
    """Camarilla + opening range: the 15-minute range breaks while price is already beyond H3 (L3)."""
    sig, st = _orb15_side(X, s)
    return sig & gt(X.C, col(X.lv.H3 if s > 0 else X.lv.L3), s), st


# ---------------------------------------------------------------- K. high-probability reversion to a magnet
@mirror
def va_80(X, s):
    """Dalton's 80% rule: opened outside yesterday's value area, two closes back inside it: it tends to cross
    the whole area. Long when the open was below value; stop just outside the edge re-entered."""
    vah, val = col(X.pv_vah), col(X.pv_val)
    o = col(X.o0)
    outside = (o < val) if s > 0 else (o > vah)
    inside = (X.C > val) & (X.C < vah)
    two = inside & prev(inside.astype(float)).astype(bool)
    first = two & ~prev(two.astype(float)).astype(bool)
    sig = outside & first & (S >= 2) & (S <= T1200)
    return sig, (val - 0.0015 * X.C) if s > 0 else (vah + 0.0015 * X.C)


@mirror
def va_edge_fade(X, s):
    """Opened inside yesterday's value area, price tags its edge and a reversal bar closes back inside."""
    edge = col(X.pv_val) if s > 0 else col(X.pv_vah)
    opened_inside = (col(X.o0) < col(X.pv_vah)) & (col(X.o0) > col(X.pv_val))
    tag = (X.L <= edge) if s > 0 else (X.H >= edge)
    sig = opened_inside & tag & gt(X.C, edge, s) & gt(X.C, X.O, s) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def poc_magnet(X, s):
    """Half a daily ATR away from yesterday's point of control with no trend (ADX < 20): a reversal bar toward it."""
    poc = col(X.pv_poc)
    away = (poc - X.C) * s >= 0.5 * col(X.atr_d)
    sig = away & (X.adx < 20) & gt(X.C, X.O, s) & gt(X.C, prev(X.C), s) & (S >= T1000) & (S <= T1400)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def gap_fill_small(X, s):
    """A 0.3-1% gap whose first bar closes against it: trade toward yesterday's close (the gap fill)."""
    g = col(X.gap)
    sized = (-s * g >= 0.003) & (-s * g <= 0.01)
    sig = np.zeros(X.C.shape, bool)
    sig[:, 0] = sized[:, 0] & gt(X.C[:, 0], X.O[:, 0], s)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def vwap_magnet(X, s):
    """1.5 sd from VWAP with no trend (ADX < 20) and a reversal bar: back toward VWAP."""
    away = (X.vwap - X.C) * s >= 1.5 * X.vsd
    sig = away & (X.adx < 20) & gt(X.C, X.O, s) & gt(X.C, prev(X.C), s) & (S >= T1000)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def bb_mid_revert(X, s):
    """Touched the outer Bollinger band, closed back inside with a reversal bar: toward the middle band."""
    band = X.bb_lo if s > 0 else X.bb_up
    touched = (prev(X.L) <= prev(band)) if s > 0 else (prev(X.H) >= prev(band))
    sig = touched & gt(X.C, band, s) & gt(X.C, X.O, s) & (S >= 2)
    return sig, (np.minimum(X.L, prev(X.L)) if s > 0 else np.maximum(X.H, prev(X.H)))


@mirror
def ib_ext_fade(X, s):
    """Dalton: the first hour's range extended by half its width on quiet volume, then a reversal bar: back in."""
    w = X.ib_hi - X.ib_lo
    ext = (X.L <= X.ib_lo - 0.5 * w) if s > 0 else (X.H >= X.ib_hi + 0.5 * w)
    quiet = np.nan_to_num(X.rv, nan=9) < 0.8
    sig = ext & quiet & gt(X.C, X.O, s) & (S >= 12)
    return sig, (X.L if s > 0 else X.H)


@mirror
def orb_fail(X, s):
    """A close beyond the 15-minute range that fails: back inside within three bars (fade the breakout)."""
    hi_, lo_ = X.or15_hi, X.or15_lo
    if s > 0:
        broke = (prev(X.C) < prev(lo_)) | (prev(X.C, 2) < prev(lo_, 2)) | (prev(X.C, 3) < prev(lo_, 3))
    else:
        broke = (prev(X.C) > prev(hi_)) | (prev(X.C, 2) > prev(hi_, 2)) | (prev(X.C, 3) > prev(hi_, 3))
    back = (X.C < hi_) & (X.C > lo_)
    sig = broke & back & (S >= 4) & (S <= T1200)
    return sig, (X.day_lo if s > 0 else X.day_hi)


@mirror
def double_bottom(X, s):
    """A second test of the day's low (within 0.1%, at least six bars after the first) with a reversal bar."""
    ext = X.day_lo if s > 0 else X.day_hi
    far = prev(ext, 6)
    near = np.abs(lo(X, s) - far) <= 0.001 * X.C
    held = (ext >= far - 0.001 * X.C) if s > 0 else (ext <= far + 0.001 * X.C)
    sig = near & held & gt(X.C, X.O, s) & gt(X.C, prev(X.C), s) & (S >= 8)
    return sig, (lo(X, s) - 0.001 * X.C) if s > 0 else (lo(X, s) + 0.001 * X.C)


@mirror
def three_down(X, s):
    """Connors: three lower closes in a row above the 50 EMA, then a higher close."""
    d3 = gt(prev(X.C, 2), prev(X.C), s) & gt(prev(X.C, 3), prev(X.C, 2), s) & gt(prev(X.C, 4), prev(X.C, 3), s)
    sig = d3 & gt(X.C, prev(X.C), s) & gt(X.C, X.ema50, s) & (S >= 4)
    return sig, (X.low5 if s > 0 else X.high5)


@mirror
def pm_vwap_revert(X, s):
    """After 13:30, 2 sd from VWAP and a reversal bar: back toward VWAP."""
    away = (X.vwap - X.C) * s >= 2 * X.vsd
    sig = away & gt(X.C, X.O, s) & (S >= T1330)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def cpr_touch(X, s):
    """Ochoa: opened above the central pivot range, comes back to its top and bounces."""
    tc, bc = col(X.lv.TC), col(X.lv.BC)
    edge = tc if s > 0 else bc
    above = (col(X.o0) > tc) if s > 0 else (col(X.o0) < bc)
    tag = (X.L <= edge) if s > 0 else (X.H >= edge)
    sig = above & tag & gt(X.C, edge, s) & gt(X.C, X.O, s) & (S >= 1)
    return sig, np.repeat(bc if s > 0 else tc, SLOTS, 1)


@mirror
def pivot_p_bounce(X, s):
    """Opened above the daily pivot, tags it and bounces with a reversal bar."""
    p = col(X.lv.P)
    above = (col(X.o0) > p) if s > 0 else (col(X.o0) < p)
    tag = (X.L <= p) if s > 0 else (X.H >= p)
    sig = above & tag & gt(X.C, p, s) & gt(X.C, X.O, s) & (S >= 1)
    return sig, (X.L if s > 0 else X.H)


@mirror
def prev_close_magnet(X, s):
    """Midday, 0.7 daily ATR from yesterday's close with no trend: a reversal bar toward it."""
    pc = col(X.f.prev_close)
    away = (pc - X.C) * s >= 0.7 * col(X.atr_d)
    sig = away & (X.adx < 20) & gt(X.C, X.O, s) & gt(X.C, prev(X.C), s) & (S >= T1100) & (S <= T1400)
    return sig, (X.low3 if s > 0 else X.high3)


def _overext(at_slot, k):
    @mirror
    def fn(X, s):
        run = -s * (X.C - col(X.f.prev_close)) > k * col(X.atr_d)
        sig = np.zeros(X.C.shape, bool)
        sig[:, at_slot] = run[:, at_slot] & gt(X.C, X.O, s)[:, at_slot]
        return sig, (X.day_lo if s > 0 else X.day_hi)
    return fn


@mirror
def pdc_retest(X, s):
    """Above yesterday's close, a pullback that tags it and holds with a reversal bar."""
    pc = col(X.f.prev_close)
    above = (col(X.o0) > pc) if s > 0 else (col(X.o0) < pc)
    tag = (X.L <= pc) if s > 0 else (X.H >= pc)
    sig = above & gt(prev(X.C, 3), pc, s) & tag & gt(X.C, pc, s) & gt(X.C, X.O, s) & (S >= 3)
    return sig, (X.L if s > 0 else X.H)


@mirror
def vwap_sd_reclaim(X, s):
    """Below VWAP - 1 sd, closes back above the band but still under VWAP: toward VWAP."""
    band = X.vwap - s * X.vsd
    sig = gt(X.C, band, s) & ~gt(prev(X.C), prev(band), s) & ~gt(X.C, X.vwap, s) & (S >= T1000)
    return sig, (X.low3 if s > 0 else X.high3)


@mirror
def morning_star(X, s):
    """Nison: a long bar against the trade, a small-bodied bar, then a bar past the first bar's midpoint, at support."""
    o2, c2 = prev(X.O, 2), prev(X.C, 2)
    o1, c1, h1, l1 = prev(X.O), prev(X.C), prev(X.H), prev(X.L)
    big = gt(o2, c2, s) & (np.abs(o2 - c2) >= 0.8 * X.rng_avg20)
    small = np.abs(o1 - c1) <= 0.3 * np.maximum(h1 - l1, 1e-9)
    sig = big & small & gt(X.C, (o2 + c2) / 2, s) & gt(X.C, X.O, s) & _support(X, s) & (S >= 3)
    return sig, (np.minimum(X.L, l1) if s > 0 else np.maximum(X.H, h1))


@mirror
def tweezer(X, s):
    """Nison: two bars with the same low (within 0.05%) after a decline, the second closing up, at support."""
    same = np.abs(lo(X, s) - prev(lo(X, s))) <= 0.0005 * X.C
    sig = same & gt(prev(X.C, 5), X.C, s) & gt(X.C, X.O, s) & gt(prev(X.O), prev(X.C), s) & _support(X, s) & (S >= 5)
    return sig, (np.minimum(X.L, prev(X.L)) if s > 0 else np.maximum(X.H, prev(X.H)))


@mirror
def rsi14_30cross(X, s):
    """RSI(14) comes back out of oversold (over 30) with no strong trend (ADX < 25)."""
    lvl = np.full_like(X.rsi14, 30.0 if s > 0 else 70.0)
    cross = xup(X.rsi14, lvl) if s > 0 else xdn(X.rsi14, lvl)
    return cross & (X.adx < 25) & (S >= 2), (X.low5 if s > 0 else X.high5)


# ---------------------------------------------------------------- the catalogue
def _lib():
    A, B, C_, D, E, F, G, H, I, J, K = ("Opening range & volatility", "VWAP", "Support & resistance", "Trend indicators",
                                        "Mean reversion", "Price action & candles", "Volume", "Relative strength & sectors",
                                        "Time of day", "Indicator combinations", "High-probability reversion")
    rows = [
        ("orb5", "Opening-range breakout, 5 min", A, "Crabel", "Close beyond the first 5-minute bar; stop at its far side", orb("or5", 1, stop_mid=False)),
        ("orb15", "Opening-range breakout, 15 min", A, "Crabel", "Close beyond the 09:15-09:30 range; stop mid-range", orb("or15", 3)),
        ("orb30", "Opening-range breakout, 30 min", A, "Crabel", "Close beyond the 09:15-09:45 range; stop mid-range", orb("or30", 6)),
        ("orb15_nr7", "ORB 15 after an NR7 day", A, "Crabel", "15-minute ORB only after the narrowest range in 7 days", orb("or15", 3, day_mask=nr7_mask)),
        ("orb15_inside", "ORB 15 after an inside day", A, "Crabel", "15-minute ORB only after an inside day", orb("or15", 3, day_mask=inside_mask)),
        ("acd", "ACD A-up / A-down", A, "Fisher", "Two closes beyond the 15-minute range +/- 0.1 ATR", acd),
        ("acd_pivot", "ACD with the pivot range", A, "Fisher", "ACD signal on the same side of the CPR", acd_fix),
        ("williams_vb", "Volatility breakout", A, "Williams", "Close beyond open +/- 0.6 x yesterday's range; stop at the open", williams_vb),
        ("gap_go", "Gap and go", A, "Common (Crabel)", "Gap 1%+, first bar in the gap's direction, close beyond it", gap_and_go),
        ("gap_fade", "Gap fade", A, "Raschke", "Gap 1%+ that fails: close back through the first bar", gap_fade),
        ("open_drive", "Open-drive", A, "Dalton", "First three bars drive one way on heavy volume: enter at 09:30", open_drive),
        ("open_test_drive", "Open-test-drive", A, "Dalton", "Early probe beyond yesterday's range fails, price drives back", open_test_drive),
        ("open_reject", "Open-rejection-reverse", A, "Dalton", "Opens outside yesterday's range, falls back inside", open_rejection),
        ("vwap_reclaim", "VWAP reclaim / loss", B, "Shannon", "Opened on one side of VWAP, crosses to the other after 10:00", vwap_reclaim),
        ("vwap_pullback", "VWAP pullback in trend", B, "Shannon", "An hour on one side of VWAP, tags it and closes back in trend", vwap_pullback),
        ("vwap_revert", "VWAP 2-sigma reversion", B, "Shannon / Bollinger", "Close back inside VWAP +/- 2 sd", vwap_band_revert),
        ("vwap_band_break", "VWAP 1-sigma breakout", B, "Shannon", "Close beyond VWAP + 1 sd on 2x volume", vwap_band_break),
        ("orb_vwap", "ORB then VWAP retest", B, "Shannon", "After a 15-minute range break, first pullback to VWAP that holds", orb_vwap_retest),
        ("pdh_break", "Yesterday's high / low break", C_, "Murphy", "Close beyond yesterday's high (low) after opening inside", pdh_break),
        ("turtle_soup", "Turtle Soup", C_, "Raschke & Connors", "Break of yesterday's low (high) that closes back inside", turtle_soup),
        ("pivot_bounce", "Pivot S1 / R1 bounce", C_, "Floor pivots", "Tags S1 (R1) and closes back with a reversal bar", pivot_bounce),
        ("pivot_break", "Pivot R1 / S1 breakout", C_, "Floor pivots", "Close beyond R1 (S1) on 1.5x volume; stop at the pivot", pivot_break),
        ("cpr_narrow", "Narrow CPR breakout", C_, "Ochoa", "Narrow central pivot range day: close beyond it after 09:30", cpr_narrow),
        ("cpr_wide", "Wide CPR reversion", C_, "Ochoa", "Wide central pivot range day: fade S1 / R1", cpr_wide_revert),
        ("cam_revert", "Camarilla L3 / H3 reversal", C_, "Stott", "Tags L3 (H3) and closes back; stop at L4 (H4)", camarilla_revert),
        ("cam_break", "Camarilla H4 / L4 breakout", C_, "Stott", "Close beyond H4 (L4); stop at H3 (L3)", camarilla_break),
        ("week_break", "Last week's high / low break", C_, "Murphy", "Close beyond last week's high (low)", week_break),
        ("fractal_break", "Swing (fractal) breakout", C_, "B. Williams", "Close beyond the last confirmed 5-bar swing, on volume", fractal_break),
        ("late_high", "New day high after noon", C_, "Common", "New high (low) of the day after 12:00 on volume", late_day_high),
        ("sr_cluster", "Support cluster bounce", C_, "Murphy", "Bounce off two or more nearby levels", sr_confluence),
        ("ema_cross", "EMA 9/21 cross above VWAP", D, "Murphy", "9 EMA crosses the 21 EMA with price on the same side of VWAP", ema_cross),
        ("holy_grail", "Holy Grail", D, "Raschke & Connors", "ADX > 30, pullback to the 20 EMA, close beyond the prior bar", holy_grail),
        ("macd_cross", "MACD signal cross", D, "Appel", "MACD crosses its signal line on the trend side of zero", macd_cross),
        ("macd_zero", "MACD zero cross", D, "Appel", "MACD crosses zero", macd_zero),
        ("supertrend", "Supertrend (10, 3) flip", D, "Seban", "Supertrend turns", supertrend_flip),
        ("dmi_cross", "DMI cross with rising ADX", D, "Wilder", "+DI crosses -DI with ADX above 20 and rising", dmi_cross),
        ("psar", "Parabolic SAR flip", D, "Wilder", "SAR flips, price on the trend side of the 50 EMA", psar_flip),
        ("triple_screen", "Triple Screen", D, "Elder", "Higher-timeframe trend + 5-minute stochastic turn", triple_screen),
        ("impulse", "Impulse system turns green", D, "Elder", "13 EMA and MACD histogram both rising, first bar", elder_impulse),
        ("ichimoku", "Ichimoku TK cross", D, "Hosoda", "Tenkan crosses Kijun beyond the cloud", ichimoku_tk),
        ("donchian", "Donchian 20-bar breakout", D, "Donchian / Turtles", "Close beyond the 20-bar channel, ADX > 20", donchian_break),
        ("keltner", "Keltner breakout", D, "Keltner", "Close beyond EMA20 +/- 1.5 ATR", keltner_break),
        ("squeeze", "TTM squeeze release", D, "Carter", "Bands leave the Keltner channel after 6+ bars, with momentum", ttm_squeeze),
        ("burst", "Momentum burst", D, "Common", "3-bar move over 2 ATR on 3x volume", momentum_burst),
        ("rsi2", "RSI(2) pullback", E, "Connors", "RSI(2) < 5 (> 95) on the trend side of the 200 EMA", rsi2),
        ("rsi_div", "RSI divergence", E, "Wilder / Murphy", "New price extreme vs 12 bars ago, RSI(14) not confirming", rsi_divergence),
        ("bb_reentry", "Bollinger re-entry", E, "Bollinger", "Close back inside the band", bb_reentry),
        ("stoch", "Stochastic cross from oversold", E, "Lane", "%K crosses %D below 20 (above 80) with the 50 EMA", stoch_cross),
        ("willr", "Williams %R reversal", E, "L. Williams", "%R leaves oversold in a daily uptrend", willr_rev),
        ("cci", "CCI -100 / +100 reversal", E, "Lambert", "CCI crosses back through -100 (+100) in the daily trend", cci_rev),
        ("overext", "09:45 overextension fade", E, "Common", "Ran 0.8 daily ATR by 09:45: fade with a reversal bar", overextension_fade),
        ("r8020", "80-20 reversal", E, "Raschke & Connors", "Yesterday closed at its extreme; today fails beyond it", raschke_8020),
        ("engulf", "Engulfing at support / resistance", F, "Nison", "Engulfing bar at a known level", engulfing_sr),
        ("hammer", "Hammer / shooting star at S/R", F, "Nison", "Long tail rejection at a known level", hammer_sr),
        ("inside_bar", "Inside-bar breakout in trend", F, "Brooks", "Close beyond an inside bar, with the 20 EMA trend", inside_bar_trend),
        ("h2", "High 2 / Low 2 pullback", F, "Brooks", "Second push out of a pullback in a trend", brooks_h2),
        ("three_bar", "Three-bar reversal", F, "Raschke", "Close beyond the two prior bars after a new extreme", three_bar_rev),
        ("outside", "Outside-bar reversal", F, "Common", "Outside bar closing at its extreme after a counter move", outside_rev),
        ("wrb", "Wide-range bar continuation", F, "Crabel", "Bar over 2x average range, closing at its extreme, 2x volume", wrb),
        ("vol_break", "Volume breakout", G, "O'Neil", "3x volume, new high of the day, close near the high", volume_breakout),
        ("climax", "Selling / buying climax", G, "T. Williams (VSA)", "Wide bar on 3x volume closing off its extreme after a 1.5% run", climax_reversal),
        ("obv", "OBV leads price", G, "Granville", "OBV makes a 50-bar high before price", obv_lead),
        ("spring", "Wyckoff spring / upthrust", G, "Wyckoff / Pruden", "Break of the range or yesterday's low that closes back, on volume", wyckoff_spring),
        ("in_play", "Stocks-in-play momentum", G, "Zarattini et al.", "2.5x volume for the time of day, above VWAP, new day high", in_play_momentum),
        ("pocket", "Pocket pivot", G, "Morales & Kacher", "Up bar outvolumes every down bar of the last 10", pocket_pivot),
        ("rs_pull", "Relative-strength VWAP pullback", H, "O'Neil", "1% ahead of the NIFTY, sector up, holds VWAP after 10:00", rs_pullback),
        ("sector_leader", "Leading sector breakout", H, "Sector rotation", "Stock in the top-2 sector makes a new day high", sector_leader),
        ("sector_catchup", "Sector catch-up", H, "Sector rotation", "Sector up 0.8%, stock lags 0.5% and reclaims VWAP", sector_catchup),
        ("sector_fade", "Sector divergence fade", H, "Sector rotation", "Stock up 2% while its sector is down 0.5%: fade", sector_divergence_fade),
        ("beta", "High-beta with a trending market", H, "Common", "NIFTY 0.5% up above its average, beta > 1.2 stock reclaims VWAP", market_beta),
        ("rs_high", "New 20-day high on a flat market", H, "O'Neil", "New 20-day high while the NIFTY is flat", rs_new_high),
        ("lunch", "Lunch-range breakout", I, "Crabel", "Break of the 12:00-13:30 range on volume", lunch_break),
        ("last_hour", "Intraday momentum (last hour)", I, "Gao, Han, Li & Zhou", "First half hour and the day so far agree: ride 14:30-15:15", intraday_momentum),
        ("pm_trend", "Afternoon trend continuation", I, "Common", "Above VWAP 80% of the day, new high after 13:30", afternoon_trend),
        ("orb_vwap_side", "ORB 15 + VWAP side", J, "Crabel + Shannon", "15-minute ORB with price on the same side of VWAP", orb_vwap_side),
        ("orb_rsi", "ORB 15 + RSI(14) momentum", J, "Crabel + Wilder", "15-minute ORB with RSI(14) above 55 (below 45)", orb_rsi),
        ("orb5_st", "ORB 5 + Supertrend", J, "Crabel + Seban", "5-minute ORB in the Supertrend's direction", orb5_supertrend),
        ("orb_cpr", "ORB 15 on a narrow-CPR day", J, "Crabel + Ochoa", "15-minute ORB when the central pivot range is narrow", orb_cpr),
        ("range_rules", "RSI range-rules pullback", J, "Brown", "Uptrend, RSI(14) pulls back to 40-50 and turns up", range_rules_pullback),
        ("macd_adx", "MACD cross with ADX > 25", J, "Appel + Wilder", "MACD crosses its signal with a strong ADX trend", macd_adx),
        ("squeeze_vol", "Squeeze release on volume", J, "Carter + O'Neil", "TTM squeeze release on 2x volume", squeeze_volume),
        ("st_vwap", "Supertrend flip + VWAP", J, "Seban + Shannon", "Supertrend turns with price beyond VWAP", supertrend_vwap),
        ("heikin", "Heikin-Ashi trend change", J, "Valcu (Heikin-Ashi)", "First strong HA candle after two opposite ones, beyond VWAP", heikin_ashi_turn),
        ("ribbon", "EMA ribbon pullback", J, "Murphy", "9 > 21 > 50 EMA, pullback tags the 9 and closes back", ribbon_pullback),
        ("keltner_rsi", "Keltner stretch + RSI", J, "Keltner + Wilder", "Back inside the Keltner channel from an RSI extreme", keltner_rsi),
        ("adx_high", "New day high with rising ADX", J, "Wilder", "New day extreme after 09:45 with ADX above 25 and rising", adx_day_high),
        ("orb_retest", "Opening-range retest", J, "Crabel", "15-minute range breaks, retests its edge and holds", orb_retest),
        ("no_supply", "No-supply pullback (VSA)", J, "T. Williams (VSA)", "Quiet down bar in an uptrend, then a bar beyond it", no_supply),
        ("expansion", "Expansion after contraction", J, "Crabel", "After an NR4 day, the first wide-range bar", expansion_after_contraction),
        ("gap_vwap", "Gap + VWAP + ORB", J, "Common", "Gap 0.5%+, beyond VWAP and the 15-minute range", gap_vwap),
        ("gap_pullback", "First pullback after a gap", J, "Common", "Gap 1%+, a 3-bar pullback that resumes", first_pullback_gap),
        ("engulf15", "15-minute engulfing beyond VWAP", J, "Nison (15-min reference)", "Engulfing 15-minute candle beyond VWAP", engulf15),
        ("inside15", "15-minute inside-bar breakout", J, "Brooks (15-min reference)", "Close beyond a 15-minute inside bar", inside15),
        ("vwap_2bar", "Two-bar reversal at VWAP", J, "Shannon + Nison", "Tags VWAP, then a two-bar reversal", vwap_two_bar),
        ("midday_rev", "Midday reversal with divergence", J, "Murphy", "11:30-13:00 new extreme, RSI not confirming, reversal bar", midday_reversal),
        ("last_hour_fade", "Last-hour exhaustion fade", J, "Common", "At 14:30, 0.75 daily ATR from the open: fade", last_hour_fade),
        ("donchian55", "Donchian 55-bar breakout", J, "Turtles", "Close beyond the 55-bar channel", donchian55),
        ("pivot_rsi", "Pivot cross + RSI", J, "Floor pivots + Wilder", "Cross of the daily pivot with RSI(14) on the same side of 50", pivot_rsi),
        ("stoch_vwap", "Stochastic turn above VWAP", J, "Lane + Shannon", "Stochastic turns from an extreme beyond VWAP after 10:00", stoch_vwap),
        ("band_walk", "Bollinger band walk", J, "Bollinger", "Two closes outside the band on volume", bb_band_walk),
        ("obv_vwap", "OBV rising + VWAP reclaim", J, "Granville + Shannon", "OBV rising over 30 minutes, price reclaims VWAP", obv_vwap),
        ("cam_orb", "Camarilla + ORB", J, "Stott + Crabel", "15-minute ORB beyond Camarilla H3 (L3)", cam_orb),
        ("va_80", "Value-area 80% rule", K, "Dalton", "Opened outside yesterday's value area, two closes back inside", va_80),
        ("va_edge", "Value-area edge fade", K, "Dalton", "Tags the value-area edge from inside, reversal bar back in", va_edge_fade),
        ("poc_magnet", "Point-of-control magnet", K, "Dalton / Steidlmayer", "0.5 ATR from yesterday's POC, no trend, reversal bar", poc_magnet),
        ("gap_fill", "Small gap fill", K, "Common", "0.3-1% gap whose first bar closes against it", gap_fill_small),
        ("vwap_magnet", "VWAP magnet", K, "Shannon", "1.5 sd from VWAP, no trend, reversal bar", vwap_magnet),
        ("bb_mid", "Bollinger mid-band reversion", K, "Bollinger", "Outer-band touch, close back inside with a reversal bar", bb_mid_revert),
        ("ib_fade", "Initial-balance extension fade", K, "Dalton", "First hour extended by half its width on quiet volume, reversal bar", ib_ext_fade),
        ("orb_fail", "Failed opening-range breakout", K, "Brooks", "Close beyond the 15-minute range, back inside within 3 bars", orb_fail),
        ("double_bottom", "Double bottom / top of the day", K, "Murphy", "Second test of the day's extreme with a reversal bar", double_bottom),
        ("three_down", "Three down closes", K, "Connors", "Three lower closes above the 50 EMA, then a higher close", three_down),
        ("pm_vwap", "Afternoon VWAP reversion", K, "Shannon", "After 13:30, 2 sd from VWAP with a reversal bar", pm_vwap_revert),
        ("cpr_touch", "CPR support / resistance touch", K, "Ochoa", "Opened above the CPR, tags its top and bounces", cpr_touch),
        ("pivot_p", "Daily pivot bounce", K, "Floor pivots", "Opened above the pivot, tags it and bounces", pivot_p_bounce),
        ("pc_magnet", "Previous-close magnet", K, "Common", "Midday 0.7 ATR from yesterday's close, no trend, reversal bar", prev_close_magnet),
        ("overext_0930", "09:30 overextension fade", K, "Common", "0.6 daily ATR from yesterday's close by 09:30, reversal bar", _overext(T0930, 0.6)),
        ("overext_1000", "10:00 overextension fade", K, "Common", "0.8 daily ATR from yesterday's close by 10:00, reversal bar", _overext(T1000, 0.8)),
        ("pdc_retest", "Previous-close retest", K, "Common", "Above yesterday's close, pullback tags it and holds", pdc_retest),
        ("vwap_sd", "VWAP band reclaim", K, "Shannon", "Back above VWAP - 1 sd, still under VWAP", vwap_sd_reclaim),
        ("morning_star", "Morning / evening star at S/R", K, "Nison", "Three-bar star pattern at a known level", morning_star),
        ("tweezer", "Tweezer bottom / top at S/R", K, "Nison", "Two equal lows after a decline, at a known level", tweezer),
        ("rsi14_30", "RSI(14) leaves oversold", K, "Wilder", "RSI(14) back over 30 (under 70) with ADX under 25", rsi14_30cross),
    ]
    return [Strat(k, n, fam, src, rule, fn) for k, n, fam, src, rule, fn in rows]


LIBRARY = _lib()
