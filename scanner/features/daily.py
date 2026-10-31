"""Daily-bar context for a stock, computed only from sessions before the day being traded.

The setups follow trader-scanner's rules on daily closes (breakout_20d, bull_flag,
reclaim_ma20, oversold, oversold_bounce) plus trend and nearness to the 20-day high.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from market.candles.bars import DailyBar


@dataclass
class DailyStats:
    symbol: str
    as_of: str                 # date of the last session used
    close: float               # that session's close: "yesterday's close" for the trading day
    change_pct: float
    sma20: Optional[float]
    sma50: Optional[float]
    rsi14: Optional[float]
    atr14: Optional[float]
    atr_pct: Optional[float]
    high20: Optional[float]    # highest high of the last 20 sessions
    close_high20: Optional[float]  # highest close of the 20 sessions before the last
    avg_volume20: float
    turnover20: float          # average close x volume, rupees
    trend_up: bool = False
    trend_down: bool = False
    near_high20: bool = False
    setups: List[str] = field(default_factory=list)
    prev_high: Optional[float] = None  # that session's high and low
    prev_low: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _sma(xs: List[float], n: int) -> Optional[float]:
    return sum(xs[-n:]) / n if len(xs) >= n else None


def wilder_rsi(closes: List[float], n: int = 14) -> List[Optional[float]]:
    """RSI with Wilder's smoothing, one value per close (None until there are n changes)."""
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / n, losses / n
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def daily_atr(bars: List[DailyBar], n: int = 14) -> Optional[float]:
    if len(bars) < n + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        b, pc = bars[i], bars[i - 1].c
        trs.append(max(b.h - b.l, abs(b.h - pc), abs(b.l - pc)))
    atr = sum(trs[:n]) / n
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


def compute_daily_stats(symbol: str, bars: List[DailyBar], before: Optional[str] = None,
                        near_pct: float = 3.0) -> Optional[DailyStats]:
    """Stats as of the close of the last session strictly before `before` (ISO date)."""
    if before:
        bars = [b for b in bars if b.date < before]
    if len(bars) < 22:
        return None
    closes = [b.c for b in bars]
    last, prev = bars[-1], bars[-2]
    sma20, sma50 = _sma(closes, 20), _sma(closes, 50)
    sma20_prev = sum(closes[-21:-1]) / 20
    rsis = wilder_rsi(closes, 14)
    rsi, rsi_prev = rsis[-1], rsis[-2]
    atr = daily_atr(bars[-60:], 14)
    high20 = max(b.h for b in bars[-20:])
    close_high20 = max(closes[-21:-1])
    vols = [b.v for b in bars[-20:]]
    turnover = sum(b.c * b.v for b in bars[-20:]) / 20

    setups = []
    if last.c > close_high20:
        setups.append("breakout_20d")
    if len(closes) >= 16:
        run_up = closes[-6] / closes[-16] - 1.0
        tail = closes[-5:]
        if run_up > 0.10 and (max(tail) - min(tail)) / min(tail) <= 0.05:
            setups.append("bull_flag")
    if prev.c < sma20_prev and last.c > sma20:
        setups.append("reclaim_ma20")
    if rsi is not None and rsi < 30:
        setups.append("oversold")
    if rsi_prev is not None and rsi_prev < 30 and last.c > prev.c:
        setups.append("oversold_bounce")

    trend_up = bool(sma50 and last.c > sma20 > sma50)
    trend_down = bool(sma50 and last.c < sma20 < sma50)
    near = high20 * (1 - near_pct / 100.0) <= last.c < high20
    if near:
        setups.append("near_high20")
    if trend_up:
        setups.append("trend_up")

    return DailyStats(
        symbol=symbol, as_of=last.date, close=last.c,
        change_pct=round(100.0 * (last.c / prev.c - 1.0), 2),
        sma20=sma20, sma50=sma50, rsi14=rsi, atr14=atr,
        atr_pct=(100.0 * atr / last.c) if atr else None,
        high20=high20, close_high20=close_high20,
        avg_volume20=sum(vols) / len(vols), turnover20=turnover,
        trend_up=trend_up, trend_down=trend_down, near_high20=near, setups=setups,
        prev_high=last.h, prev_low=last.l,
    )
