"""Cached history for the scan, the Time Machine and the backtest.

- daily(): about a year of daily bars per stock (one cache file per day).
- intraday60(): about 60 sessions of 5-minute bars per stock, sessions before today only.
Both come from Yahoo (free, no login) and are fetched once per day per stock.
"""
import gzip
import json
import logging
import threading
from typing import Dict, Iterable, List

from core.config.settings import CACHE_DIR
from market.calendar import now_ist
from market.candles import yahoo
from market.candles.bars import Bar, DailyBar

logger = logging.getLogger("History")


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


class HistoryStore:
    def __init__(self, workers: int = 6):
        self.workers = workers
        self._lock = threading.Lock()
        self._daily: Dict[str, List[DailyBar]] = {}
        self._intra: Dict[str, List[Bar]] = {}
        self._day = None

    def _roll_day(self):
        today = now_ist().date().isoformat()
        if today != self._day:
            self._day, self._daily, self._intra = today, {}, {}
        return today

    # ---------------------------------------------------------------- daily bars
    def daily(self, symbols: Iterable[str]) -> Dict[str, List[DailyBar]]:
        with self._lock:
            today = self._roll_day()
            path = CACHE_DIR / "daily" / f"{today}.json.gz"
            if not self._daily and path.exists():
                raw = _read(path)
                self._daily = {s: [DailyBar(*row) for row in rows] for s, rows in raw.items()}
            want = [s for s in symbols if s not in self._daily]
        if want:
            logger.info("Fetching daily bars for %d stocks", len(want))
            got = yahoo.fetch_many(yahoo.daily, want, workers=self.workers, rng="1y")
            with self._lock:
                self._daily.update(got)
                _write(path, {s: [[b.date, b.o, b.h, b.l, b.c, b.v] for b in rows] for s, rows in self._daily.items()})
            logger.info("Daily bars: %d of %d fetched", len(got), len(want))
        with self._lock:
            return {s: self._daily[s] for s in symbols if s in self._daily}

    # ---------------------------------------------------------------- 60 days of 5-minute bars
    def intraday60(self, symbols: Iterable[str]) -> Dict[str, List[Bar]]:
        symbols = list(symbols)
        with self._lock:
            today = self._roll_day()
        folder = CACHE_DIR / "intraday60" / today
        out, want = {}, []
        for s in symbols:
            with self._lock:
                if s in self._intra:
                    out[s] = self._intra[s]
                    continue
            p = folder / f"{s.replace('^', 'IDX_')}.json.gz"
            if p.exists():
                bars = [Bar(*row) for row in _read(p)]
                with self._lock:
                    self._intra[s] = bars
                out[s] = bars
            else:
                want.append(s)
        if want:
            logger.info("Fetching 60 days of 5-minute bars for %d symbols", len(want))
            got = yahoo.fetch_many(yahoo.intraday, want, workers=self.workers, rng="60d", interval="5m")
            for s, bars in got.items():
                bars = [b for b in bars if b.date < today]
                _write(folder / f"{s.replace('^', 'IDX_')}.json.gz", [[b.start, b.o, b.h, b.l, b.c, b.v] for b in bars])
                with self._lock:
                    self._intra[s] = bars
                out[s] = bars
        return out
