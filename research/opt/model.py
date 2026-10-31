"""Option maths for round 5: Black-76 on the implied forward, implied volatility, trading time.

- Forward: each expiry's forward comes from put-call parity at the strikes nearest the index,
  F = K + e^{rT} (C - P), so dividends and the futures basis are in it.
- Time: trading time, 375 minutes a session, 252 sessions a year, to 15:30 on expiry day.
- IV: bisection on Black-76 for every row at once.
"""
import numpy as np
from scipy.special import ndtr

R = 0.065
MIN_PER_DAY = 375.0
YEAR_MIN = MIN_PER_DAY * 252


def black(F, K, T, sigma, cp):
    """Black-76 price; cp = +1 call, -1 put. Arrays broadcast."""
    F, K, T, sigma, cp = (np.asarray(x, float) for x in (F, K, T, sigma, cp))
    T = np.maximum(T, 1.0 / YEAR_MIN)
    sigma = np.maximum(sigma, 1e-4)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    return np.exp(-R * T) * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))


def delta(F, K, T, sigma, cp):
    F, K, T, sigma, cp = (np.asarray(x, float) for x in (F, K, T, sigma, cp))
    T = np.maximum(T, 1.0 / YEAR_MIN)
    sq = np.maximum(sigma, 1e-4) * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    return np.exp(-R * T) * np.where(cp > 0, ndtr(d1), ndtr(d1) - 1)


def implied_vol(price, F, K, T, cp, lo=0.01, hi=3.0, iters=60):
    """NaN where the price is below intrinsic value or above the no-arbitrage bound."""
    price, F, K, T, cp = (np.asarray(x, float) for x in (price, F, K, T, cp))
    disc = np.exp(-R * np.maximum(T, 1.0 / YEAR_MIN))
    intrinsic = disc * np.maximum(cp * (F - K), 0)
    ok = np.isfinite(price) & (price > intrinsic + 1e-9) & (price < disc * np.where(cp > 0, F, K))
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        high = black(F, K, T, m, cp) > price
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def t_years(sessions_left, minutes_into_session):
    """Trading time to 15:30 on expiry: sessions_left = expiry index - today index (0 on expiry day)."""
    left = np.maximum(MIN_PER_DAY - np.asarray(minutes_into_session, float), 0) + MIN_PER_DAY * np.asarray(sessions_left, float)
    return np.maximum(left, 1.0) / YEAR_MIN
