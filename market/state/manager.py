"""In-memory market state: last price per symbol and today's 5-minute bars."""
import threading
import time
from typing import Dict, List, Optional

from core.events.types import CandleClosedEvent, MarketTickEvent
from market.calendar import now_ist
from market.candles.bars import Bar, BarBuilder


class MarketState:
    def __init__(self):
        self._lock = threading.Lock()
        self._ltp: Dict[str, float] = {}
        self._ltp_ts: Dict[str, float] = {}
        self._volume: Dict[str, int] = {}
        self._prev_close: Dict[str, float] = {}
        self._builders: Dict[str, BarBuilder] = {}
        self._day = now_ist().date().isoformat()
        self._candles: Dict[str, Dict[str, list]] = {}  # instrument_id -> timeframe -> candles (legacy)

    def _check_day(self):
        today = now_ist().date().isoformat()
        if today != self._day:
            self._day = today
            self._builders = {}

    # ---------------------------------------------------------------- ticks
    def update_tick(self, tick: MarketTickEvent):
        sym = tick.instrument_id.replace("NSE_", "")
        self.on_price(sym, tick.ltp, tick.timestamp / 1000.0, tick.volume)

    def on_price(self, symbol: str, price: float, ts: Optional[float] = None, cum_volume: Optional[float] = None,
                 build_bars: bool = True) -> Optional[Bar]:
        ts = ts or time.time()
        with self._lock:
            self._check_day()
            if ts >= self._ltp_ts.get(symbol, 0):
                self._ltp[symbol] = price
                self._ltp_ts[symbol] = ts
            if cum_volume is not None:
                self._volume[symbol] = int(cum_volume)
            if build_bars:
                return self._builders.setdefault(symbol, BarBuilder()).on_tick(ts, price, cum_volume)
        return None

    def set_prev_close(self, symbol: str, price: Optional[float]):
        if price:
            with self._lock:
                self._prev_close[symbol] = float(price)

    # ---------------------------------------------------------------- bars
    def set_bars(self, symbol: str, bars: List[Bar], day: Optional[str] = None):
        """Replace today's (or `day`'s) closed bars (polling feeds deliver whole sessions)."""
        today = day or now_ist().date().isoformat()
        with self._lock:
            self._check_day()
            b = self._builders.setdefault(symbol, BarBuilder())
            b.closed = [x for x in bars if x.date == today]

    def seed_bars(self, symbol: str, bars: List[Bar]):
        today = now_ist().date().isoformat()
        with self._lock:
            self._check_day()
            self._builders.setdefault(symbol, BarBuilder()).seed([x for x in bars if x.date == today])

    def roll(self, now_ts: Optional[float] = None) -> List[str]:
        """Close bars whose time is up; returns the symbols that got a new closed bar."""
        now_ts = now_ts or time.time()
        out = []
        with self._lock:
            for sym, b in self._builders.items():
                if b.roll(now_ts):
                    out.append(sym)
        return out

    def bars(self, symbol: str) -> List[Bar]:
        with self._lock:
            b = self._builders.get(symbol)
            return list(b.closed) if b else []

    def forming_bar(self, symbol: str) -> Optional[Bar]:
        with self._lock:
            b = self._builders.get(symbol)
            return b.forming if b else None

    # ---------------------------------------------------------------- prices
    def get_ltp(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._ltp.get(symbol.replace("NSE_", ""))

    def price(self, symbol: str):
        """(price, age in seconds) or (None, None)."""
        with self._lock:
            p = self._ltp.get(symbol)
            ts = self._ltp_ts.get(symbol)
        if p is None:
            return None, None
        return p, max(0.0, time.time() - ts)

    def prev_close(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prev_close.get(symbol)

    def snapshot(self, symbols) -> Dict[str, dict]:
        out = {}
        now = time.time()
        with self._lock:
            for s in symbols:
                p = self._ltp.get(s)
                pc = self._prev_close.get(s)
                out[s] = {
                    "ltp": p, "age_s": round(now - self._ltp_ts[s], 1) if s in self._ltp_ts else None,
                    "prev_close": pc,
                    "change_pct": round(100.0 * (p / pc - 1.0), 2) if p and pc else None,
                    "volume": self._volume.get(s),
                }
        return out

    # ---------------------------------------------------------------- legacy candle API
    def update_candle(self, candle: CandleClosedEvent):
        with self._lock:
            tf = self._candles.setdefault(candle.instrument_id, {}).setdefault(candle.timeframe, [])
            tf.append(candle)
            if len(tf) > 1000:
                tf.pop(0)

    def get_candles(self, instrument_id: str, timeframe: str) -> list:
        with self._lock:
            return self._candles.get(instrument_id, {}).get(timeframe, [])
