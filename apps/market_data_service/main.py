"""Market data gateway: live quotes and today's 5-minute bars for the watched symbols.

- angel: Angel One SmartAPI WebSocket (real time). Bars are built from ticks; today's
  earlier bars are backfilled from Yahoo when a symbol is first watched.
- yahoo: free Yahoo polling. Quotes every data.poll_seconds; bars re-read once per bar,
  data.bar_delay_seconds after it closes. Delayed and unofficial.

data.provider "auto" uses Angel One when the login works, Yahoo otherwise.
"""
import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional

from market.calendar import last_closed_slot, now_ist, session_state
from market.candles import yahoo
from market.state.manager import MarketState

logger = logging.getLogger("MarketDataGateway")


class FeedManager:
    def __init__(self, settings, universe, state: MarketState):
        self.settings = settings
        self.universe = universe
        self.state = state
        self.mode = "yahoo"
        self.adapter = None
        self.symbols = set()
        self._lock = threading.Lock()
        self._running = False
        self._last_quotes = 0.0
        self._last_bar_slot: Dict[str, int] = {}
        self._retry = ([], -1, 0, 0.0)
        self.last_quotes_at: Optional[float] = None
        self.last_bars_at: Optional[float] = None
        self.errors = 0
        self.note = ""
        self._last_reconnect = 0.0
        self._reconnect_tries = 0

    # ---------------------------------------------------------------- start
    def start(self, adapter=None):
        if adapter is not None and getattr(adapter, "market_data", None) is not None:
            self.mode = "angel"
            self.adapter = adapter
            adapter.market_data.set_callback(self._on_angel_quote)
            try:
                import logzero
                logzero.loglevel(logging.WARNING)     # SmartApi logs every packet at INFO
            except Exception:
                pass
            self.note = "Angel One WebSocket (real time)"
        else:
            self.mode = "yahoo"
            self.note = "Yahoo Finance polling (free, delayed, unofficial)"
        self._running = True
        threading.Thread(target=self._loop, name="feed", daemon=True).start()
        logger.info("Market data: %s", self.note)

    def stop(self):
        self._running = False
        if self.adapter and self.adapter.market_data:
            self.adapter.market_data.close()

    def subscribe(self, symbols: Iterable[str]):
        new = [s for s in symbols if s not in self.symbols]
        if not new:
            return
        with self._lock:
            self.symbols.update(new)
        if self.mode == "angel":
            self.adapter.market_data.subscribe({s: self.universe.token(s) for s in new})
        # today's bars so far, and the previous close
        threading.Thread(target=self._backfill, args=(new,), daemon=True).start()

    def subscribe_fo(self, contracts: Dict[str, str], on_oi) -> int:
        """Stream futures contracts' open interest (token -> trading symbol, nearest expiry first) to on_oi(token,
        oi, ts). Angel One only; returns how many contracts are streamed (the rest are read every few minutes)."""
        md = self.adapter.market_data if (self.mode == "angel" and self.adapter) else None
        if md is None:
            return 0
        md.on_oi = on_oi
        md.subscribe_fo(contracts)
        return len(md.fo_tokens)

    # ---------------------------------------------------------------- angel
    def _on_angel_quote(self, symbol, ltp, ts, cum_volume, prev_close):
        self.state.on_price(symbol, ltp, ts, cum_volume, build_bars=True)
        if prev_close:
            self.state.set_prev_close(symbol, prev_close)
        self.last_quotes_at = time.time()

    # ---------------------------------------------------------------- yahoo
    def _backfill(self, symbols):
        self._quotes(symbols)
        self._bars(symbols, seed=(self.mode == "angel"))

    def _quotes(self, symbols):
        try:
            q = yahoo.quotes(list(symbols))
        except Exception as e:
            self.errors += 1
            logger.warning("quotes failed: %s", e)
            return
        for sym, row in q.items():
            if self.mode == "yahoo":
                self.state.on_price(sym, row["price"], row.get("time") or time.time(), build_bars=False)
            self.state.set_prev_close(sym, row.get("prev_close"))
        if q:
            self.last_quotes_at = time.time()

    def _bars(self, symbols, seed=False, expect_slot=None):
        """Read today's bars; returns the symbols whose newest bar is older than expect_slot
        (Yahoo had not caught up yet)."""
        lagging = []

        def one(sym):
            try:
                return sym, yahoo.intraday(sym, "1d", "5m")
            except Exception as e:
                logger.debug("bars failed %s: %s", sym, e)
                return sym, None

        with ThreadPoolExecutor(max_workers=6) as ex:
            for sym, bars in ex.map(one, list(symbols)):
                if bars is None:
                    self.errors += 1
                    continue
                if seed:
                    self.state.seed_bars(sym, bars)
                else:
                    self.state.set_bars(sym, bars)
                if expect_slot is not None and (not bars or bars[-1].slot < expect_slot):
                    lagging.append(sym)
        self.last_bars_at = time.time()
        return lagging

    def _loop(self):
        poll = float(self.settings.get("data.poll_seconds", 15))
        delay = float(self.settings.get("data.bar_delay_seconds", 20))
        while self._running:
            try:
                now = now_ist()
                st = session_state(now)
                with self._lock:
                    syms = sorted(self.symbols)
                if self.mode == "angel":
                    self.state.roll()
                    self._watch_angel(now)
                elif syms and st == "OPEN":
                    if time.time() - self._last_quotes >= poll:
                        self._last_quotes = time.time()
                        self._quotes(syms)
                    # once per bar, a little after it closes, read the bars again; retry the
                    # symbols Yahoo has not caught up on, every 15 s, three times
                    slot = last_closed_slot(now - dt.timedelta(seconds=delay))
                    if slot >= 0 and slot > self._last_bar_slot.get("all", -1):
                        self._last_bar_slot["all"] = slot
                        self._retry = (self._bars(syms, expect_slot=slot), slot, 0, time.time())
                    elif self._retry[0] and time.time() - self._retry[3] >= 15 and self._retry[2] < 3:
                        lag, rslot, n, _ = self._retry
                        self._retry = (self._bars(lag, expect_slot=rslot), rslot, n + 1, time.time())
                elif syms and time.time() - self._last_quotes >= 300:
                    self._last_quotes = time.time()
                    self._quotes(syms)
            except Exception as e:
                self.errors += 1
                logger.error("feed loop error: %s", e)
            time.sleep(1.0)

    # ---------------------------------------------------------------- angel watchdog
    # Angel One drops idle sockets outside market hours and SmartApi gives up after a few
    # retries, so from 09:00 to the close on trading days: when the socket is down, or silent
    # for SILENT_S while the market is open, log in again and open a new one.
    WATCH_FROM_MIN = 9 * 60
    SILENT_S = 90.0

    @classmethod
    def _angel_window(cls, now: dt.datetime) -> bool:
        st = session_state(now)
        return st == "OPEN" or (st == "PRE_OPEN" and now.hour * 60 + now.minute >= cls.WATCH_FROM_MIN)

    @staticmethod
    def _silence(md, now: dt.datetime) -> Optional[float]:
        """Seconds without a sign of life; while the market is open, counted from 09:15 at most
        (the pre-open lull before the bell is not a dead socket)."""
        silent = md.silent_for(now.timestamp())
        if silent is not None and session_state(now) == "OPEN":
            bell = now.replace(hour=9, minute=15, second=0, microsecond=0)
            silent = min(silent, (now - bell).total_seconds())
        return silent

    def _watch_angel(self, now: dt.datetime):
        md = self.adapter.market_data if self.adapter else None
        if md is None:
            return
        if md.connected and md.opened_at and md.opened_at > self._last_reconnect:
            self._reconnect_tries = 0
        if not self._angel_window(now):
            return
        silent = self._silence(md, now)
        if md.connected:
            # a quiet socket only means something once there is something to hear
            dead = (session_state(now) == "OPEN" and bool(md.token_to_symbol)
                    and silent is not None and silent > self.SILENT_S)
        else:
            raw = md.silent_for(now.timestamp())    # give a socket that is opening 30 s
            dead = md.gave_up or raw is None or raw > 30
        if not dead:
            return
        wait = min(300.0, 30.0 * 2 ** self._reconnect_tries)
        if time.time() - self._last_reconnect < wait:
            return
        self._last_reconnect = time.time()
        self._reconnect_tries += 1
        why = f"silent for {silent:.0f}s" if md.connected else "down"
        try:
            self.adapter.reconnect_feed()
            logger.warning("Angel One feed %s: logged in again and opened a new socket", why)
        except Exception as e:
            self.errors += 1
            logger.error("Angel One feed %s; reconnect failed: %s", why, e)

    def status(self) -> dict:
        md = self.adapter.market_data if self.adapter else None
        connected = bool(md and md.connected)
        silent = self._silence(md, now_ist()) if md else None
        return {
            "mode": self.mode, "note": self.note, "symbols": len(self.symbols),
            "last_quotes_age_s": round(time.time() - self.last_quotes_at, 1) if self.last_quotes_at else None,
            "last_bars_age_s": round(time.time() - self.last_bars_at, 1) if self.last_bars_at else None,
            "angel_connected": connected, "errors": self.errors,
            "angel_silent_s": round(silent, 1) if silent is not None else None,
            "reconnects": md.reconnects if md else 0,
            # down outside 09:00-close is normal: the watchdog reopens it before the open
            "standby": self.mode == "angel" and not connected and not self._angel_window(now_ist()),
        }

