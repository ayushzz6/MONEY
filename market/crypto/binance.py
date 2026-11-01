"""Binance public market data: REST candles and live websocket candles. No account, no API key, no orders.

Crypto pairs are spot, from Binance's market-data-only hosts (REST data-api.binance.vision, websocket
data-stream.binance.vision). XAUUSD is the USD-M futures XAUUSDT perpetual, which tracks spot gold in US dollars an
ounce (REST fapi.binance.com, websocket fstream.binance.com/market: the old /ws and /stream futures paths still accept
a connection but no longer stream). Futures open interest and funding for every pair come from the futures REST host.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests
import websocket

from market.candles.bars import Bar, DailyBar

logger = logging.getLogger("Binance")

SPOT_REST = "https://data-api.binance.vision"
FUT_REST = "https://fapi.binance.com"
SPOT_WS = "wss://data-stream.binance.vision/stream?streams="
FUT_WS = "wss://fstream.binance.com/market/stream?streams="
SILENT_S = 90                    # kline streams push every 2 s even without trades: this long silent means dead


@dataclass
class CBar(Bar):
    tb: float = 0.0              # taker buy volume: bought by aggressive buyers, in the base asset


_http = requests.Session()                 # kept-alive connections: a fresh TLS handshake per call costs about 1 s
_http.headers["User-Agent"] = "MONEYV2/1.0"
_http.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16))


def get(url: str, params: Optional[dict] = None, timeout: float = 10):
    r = _http.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_status(e: Exception) -> Optional[int]:
    return getattr(getattr(e, "response", None), "status_code", None)


def klines(symbol: str, futures: bool, interval: str = "5m", limit: int = 500) -> List[list]:
    url = FUT_REST + "/fapi/v1/klines" if futures else SPOT_REST + "/api/v3/klines"
    return get(url, {"symbol": symbol, "interval": interval, "limit": limit})


def bar(k: list) -> CBar:
    """A REST kline row: [open time ms, o, h, l, c, volume, close time, quote volume, trades, taker buy base, ...]."""
    return CBar(int(k[0]) // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), tb=float(k[9]))


def ws_bar(k: dict) -> CBar:
    """The "k" object of a kline stream message."""
    return CBar(int(k["t"]) // 1000, float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"]),
                tb=float(k["V"]))


def daily_bar(k: list) -> DailyBar:
    return DailyBar(time.strftime("%Y-%m-%d", time.gmtime(int(k[0]) // 1000)), float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]))


def _decimals(info: dict) -> Dict[str, int]:
    """{symbol: decimals of its price tick} for every trading symbol in an exchangeInfo reply (0.01 -> 2)."""
    out = {}
    for s in info["symbols"]:
        f = next((x for x in s.get("filters", []) if x.get("filterType") == "PRICE_FILTER"), None)
        if s.get("status") == "TRADING" and f:
            t = f["tickSize"].rstrip("0")
            out[s["symbol"]] = len(t.split(".")[1]) if "." in t else 0
    return out


def futures_symbols() -> Dict[str, int]:
    """Every trading USD-M futures symbol with its price decimals."""
    return _decimals(get(FUT_REST + "/fapi/v1/exchangeInfo", timeout=20))


def spot_decimals(symbols: List[str]) -> Dict[str, int]:
    return _decimals(get(SPOT_REST + "/api/v3/exchangeInfo", {"symbols": json.dumps(symbols, separators=(",", ":"))}, timeout=20))


def open_interest(symbol: str, limit: int = 300) -> List[Tuple[int, float]]:
    """[(epoch s, open interest in the base asset)] every 5 minutes, oldest first (300 = the last 25 hours)."""
    rows = get(FUT_REST + "/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "limit": limit})
    return [(int(r["timestamp"]) // 1000, float(r["sumOpenInterest"])) for r in rows]


def funding(symbol: str) -> Optional[float]:
    """The last funding rate, a fraction per 8 hours (0.0001 = 0.01%)."""
    v = get(FUT_REST + "/fapi/v1/premiumIndex", {"symbol": symbol}).get("lastFundingRate")
    return float(v) if v not in (None, "") else None


class Stream:
    """One combined websocket: a 5-minute kline stream and a 24-hour mini ticker per pair. Reconnects with backoff
    (Binance closes every connection after 24 hours); on_open runs after every (re)connect so the caller can fill gaps."""

    def __init__(self, name: str, base: str, symbols: List[str], on_message: Callable[[str, str, dict], None],
                 on_open: Optional[Callable[[], None]] = None):
        self.name = name
        self.url = base + "/".join(f"{s.lower()}@{k}" for s in symbols for k in ("kline_5m", "miniTicker"))
        self.on_message, self.on_open = on_message, on_open
        self.connected = False
        self.last_msg = 0.0
        self.reconnects = 0
        self.error = ""
        self._app: Optional[websocket.WebSocketApp] = None

    def start(self):
        threading.Thread(target=self._run, name=f"binance-{self.name}", daemon=True).start()

    def status(self) -> Dict:
        return {"name": self.name, "connected": self.connected, "last_msg": self.last_msg or None,
                "reconnects": self.reconnects, "error": self.error}

    def kick(self):
        """Close a socket that went silent; _run opens a new one."""
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass

    def _run(self):
        tries = 0
        while True:
            t0 = time.time()
            self._app = websocket.WebSocketApp(self.url, on_open=self._opened, on_message=self._msg,
                                               on_error=self._err, on_close=lambda *a: None)
            try:
                self._app.run_forever(ping_interval=60, ping_timeout=20)
            except Exception as e:
                self.error = str(e)
            self.connected = False
            tries = 0 if time.time() - t0 > 120 else tries + 1
            self.reconnects += 1
            time.sleep(min(60, 2 ** min(tries, 6)))

    def _opened(self, _ws):
        self.connected, self.error, self.last_msg = True, "", time.time()
        logger.info("Binance %s stream connected", self.name)
        if self.on_open:
            threading.Thread(target=self.on_open, daemon=True).start()

    def _err(self, _ws, e):
        self.error = str(e) or type(e).__name__
        logger.warning("Binance %s stream: %s", self.name, self.error)

    def _msg(self, _ws, raw):
        self.last_msg = time.time()
        try:
            m = json.loads(raw)
            stream, data = m.get("stream", ""), m.get("data") or {}
            sym, _, kind = stream.partition("@")
            self.on_message(sym.upper(), "kline" if kind.startswith("kline") else "ticker", data)
        except Exception:
            logger.exception("Binance %s message failed", self.name)
