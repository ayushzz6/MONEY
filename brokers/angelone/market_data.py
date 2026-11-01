"""Angel One SmartAPI WebSocket V2 feed (QUOTE mode: LTP, day volume, previous close).

The socket runs in its own daemon thread (SmartWebSocketV2.connect() blocks). Tokens are
subscribed when the socket opens, and again whenever new ones are added.

Two SmartApi quirks are handled here:
- its _on_close takes (wsapp) but websocket-client calls it with (wsapp, code, reason), so the
  close raised inside the SDK and never reached us: the feed looked connected after it died;
- its retry counter never resets, so after a few drops over the socket's life it gives up for
  good. reconnect() opens a brand-new socket (FeedManager calls it, after a fresh login).
"""
import logging
import threading
import time
from typing import Callable, Dict, Iterable, Optional

from SmartApi.smartWebSocketV2 import SmartWebSocketV2

logger = logging.getLogger(__name__)

QUOTE_MODE = 2
NSE_CM = 1

# on_quote(symbol, ltp, ts_seconds, cum_volume, prev_close)
QuoteCallback = Callable[[str, float, float, Optional[float], Optional[float]], None]


class AngelOneMarketData:
    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str):
        self.api_key = api_key
        self.client_code = client_code
        self.on_quote: Optional[QuoteCallback] = None
        self.token_to_symbol: Dict[str, str] = {}
        self.connected = False
        self.gave_up = False                    # the SDK ran out of retries on this socket
        self.last_message_at: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.started_at: Optional[float] = None
        self.reconnects = 0
        self._gen = 0                           # callbacks from replaced sockets are ignored
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.sws = self._socket(auth_token, feed_token)

    def _socket(self, auth_token: str, feed_token: str) -> SmartWebSocketV2:
        sws = SmartWebSocketV2(auth_token, self.api_key, self.client_code, feed_token,
                               max_retry_attempt=5, retry_strategy=1, retry_delay=5)
        gen = self._gen

        def live():
            return gen == self._gen

        sws.on_data = lambda wsapp, msg: live() and self._on_data(wsapp, msg)
        sws.on_open = lambda wsapp: live() and self._on_open(wsapp)
        sws.on_error = lambda *args: live() and self._on_error(*args)
        sws.on_close = lambda wsapp, *args: live() and self._on_close(wsapp)
        # SDK connect() passes self._on_close to websocket-client, which calls it with
        # (wsapp, code, reason); the SDK's own version accepts only (wsapp).
        sws._on_close = lambda wsapp, *args: sws.on_close(wsapp)
        return sws

    def set_callback(self, callback: QuoteCallback):
        self.on_quote = callback

    def _on_data(self, wsapp, msg):
        try:
            self.last_message_at = time.time()
            token = str(msg.get("token", "")).strip("\x00")
            sym = self.token_to_symbol.get(token)
            if not sym or "last_traded_price" not in msg:
                return
            ltp = msg["last_traded_price"] / 100.0
            ts = (msg.get("exchange_timestamp") or time.time() * 1000) / 1000.0
            vol = msg.get("volume_trade_for_the_day")
            pc = msg.get("closed_price")
            if self.on_quote:
                self.on_quote(sym, ltp, ts, float(vol) if vol is not None else None, pc / 100.0 if pc else None)
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")

    def _on_open(self, wsapp):
        self.connected = True
        self.gave_up = False
        self.opened_at = time.time()
        logger.info("Angel One WebSocket open")
        self._subscribe(list(self.token_to_symbol))

    def _on_error(self, *args):
        logger.error(f"WebSocket error: {args}")
        if args and "Max retry" in str(args[0]):
            self.connected = False
            self.gave_up = True

    def _on_close(self, wsapp, *args):
        self.connected = False
        logger.warning("Angel One WebSocket closed")

    def connect(self):
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, args=(self.sws,), name="angel-ws", daemon=True)
        self._thread.start()

    def _run(self, sws: SmartWebSocketV2):
        try:
            sws.connect()
        except Exception as e:                  # the SDK re-raises; keep the thread quiet
            logger.error(f"Angel One WebSocket stopped: {e}")
            if sws is self.sws:
                self.connected = False

    def reconnect(self, auth_token: str, feed_token: str):
        """Drop the current socket and open a new one with fresh tokens. Every known token is
        subscribed again when it opens."""
        with self._lock:
            old = self.sws
            self._gen += 1
            self.connected = False
            self.gave_up = False
            self.opened_at = None
            self.sws = self._socket(auth_token, feed_token)
            self.reconnects += 1
        try:
            old.close_connection()
        except Exception:
            pass
        self.connect()

    def silent_for(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the socket last showed life (started, opened, or sent a tick)."""
        last = max(self.last_message_at or 0.0, self.opened_at or 0.0, self.started_at or 0.0)
        return (now or time.time()) - last if last else None

    def subscribe(self, symbol_tokens: Dict[str, str]):
        """symbol -> token. Adds to the subscription; already subscribed tokens are skipped."""
        new = [t for s, t in symbol_tokens.items() if t and t not in self.token_to_symbol]
        for s, t in symbol_tokens.items():
            if t:
                self.token_to_symbol[t] = s
        if self.connected and new:
            self._subscribe(new)

    def _subscribe(self, tokens: Iterable[str]):
        tokens = list(tokens)
        if not tokens:
            return
        try:
            self.sws.subscribe("moneyv2", QUOTE_MODE, [{"exchangeType": NSE_CM, "tokens": tokens}])
            logger.info(f"Subscribed {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")

    def close(self):
        try:
            self.sws.close_connection()
        except Exception:
            pass
