"""Free Yahoo Finance data for NSE stocks (SYMBOL.NS) and the NIFTY 50 (^NSEI).

Unofficial endpoints: they can lag, throttle or change. Bars are 5-minute bars at best,
never ticks. Yahoo keeps about 60 days of 5-minute bars and about 7 days of 1-minute bars.
"""
import logging
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from core.utilities.http import FetchError, get_json
from market.calendar import SESSION_BARS, to_ist
from market.candles.bars import Bar, DailyBar

logger = logging.getLogger("yahoo")

BASE = "https://query2.finance.yahoo.com"
ALT_HOSTS = ("query1.finance.yahoo.com",)


def yahoo_symbol(symbol: str) -> str:
    if symbol.startswith("^"):
        return symbol
    return f"{symbol}.NS"


def _chart(symbol: str, rng: str, interval: str) -> dict:
    ys = urllib.parse.quote(yahoo_symbol(symbol), safe="")
    url = f"{BASE}/v8/finance/chart/{ys}?range={rng}&interval={interval}&includePrePost=false"
    data = get_json(url, hosts=ALT_HOSTS)
    res = ((data or {}).get("chart") or {}).get("result") or []
    if not res:
        err = ((data or {}).get("chart") or {}).get("error")
        raise FetchError(f"no chart for {symbol}: {err}")
    return res[0]


def parse_intraday(result: dict, now_ts: Optional[float] = None, interval_s: int = 300) -> List[Bar]:
    """Completed regular-session bars only, oldest first. Drops bars with no price, bars off
    the 5-minute grid (the live tick), and bars that have not closed yet."""
    now_ts = now_ts if now_ts is not None else time.time()
    meta = result.get("meta") or {}
    rmt = meta.get("regularMarketTime")
    stamps = result.get("timestamp") or []
    q = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    cols = [q.get(k) or [] for k in ("open", "high", "low", "close", "volume")]
    out = {}
    for i, ts in enumerate(stamps):
        vals = [col[i] if i < len(col) else None for col in cols]
        o, h, l, c, v = vals
        if ts is None or None in (o, h, l, c):
            continue
        ts = int(ts)
        if ts % interval_s:
            continue
        end = ts + interval_s
        if end > now_ts:
            continue
        # Yahoo has not caught up to this bar's end yet (a minute's grace for quiet names)
        if isinstance(rmt, (int, float)) and end > rmt + 60 and end > now_ts - 120:
            continue
        b = Bar(ts, float(o), float(h), float(l), float(c), float(v or 0))
        if 0 <= b.slot < SESSION_BARS:
            out[ts] = b
    return [out[t] for t in sorted(out)]


def intraday(symbol: str, rng: str = "1d", interval: str = "5m", now_ts: Optional[float] = None) -> List[Bar]:
    return parse_intraday(_chart(symbol, rng, interval), now_ts=now_ts)


def daily(symbol: str, rng: str = "1y") -> List[DailyBar]:
    res = _chart(symbol, rng, "1d")
    stamps = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0] or {}
    out: Dict[str, DailyBar] = {}
    for i, ts in enumerate(stamps):
        try:
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            v = (q.get("volume") or [0] * len(stamps))[i] or 0
        except (KeyError, IndexError):
            continue
        if ts is None or None in (o, h, l, c):
            continue
        d = to_ist(int(ts)).date().isoformat()
        out[d] = DailyBar(d, float(o), float(h), float(l), float(c), float(v))
    return [out[d] for d in sorted(out)]


def quotes(symbols: List[str]) -> Dict[str, dict]:
    """Last price, its time and the previous close for up to hundreds of symbols, 20 per
    request through the spark endpoint."""
    out: Dict[str, dict] = {}
    for i in range(0, len(symbols), 20):
        chunk = symbols[i:i + 20]
        ys = ",".join(urllib.parse.quote(yahoo_symbol(s), safe="^") for s in chunk)
        url = f"{BASE}/v7/finance/spark?symbols={ys}&range=1d&interval=5m"
        try:
            data = get_json(url, hosts=ALT_HOSTS)
        except FetchError as e:
            logger.warning("spark quote failed for %d symbols: %s", len(chunk), e)
            continue
        back = {yahoo_symbol(s): s for s in chunk}
        for row in ((data or {}).get("spark") or {}).get("result") or []:
            sym = back.get(row.get("symbol"))
            resp = (row.get("response") or [{}])[0]
            meta = resp.get("meta") or {}
            price = meta.get("regularMarketPrice")
            if sym is None or price is None:
                continue
            prev = meta.get("previousClose") or meta.get("chartPreviousClose")
            out[sym] = {
                "price": float(price),
                "time": meta.get("regularMarketTime"),
                "prev_close": float(prev) if prev else None,
                "day_high": meta.get("regularMarketDayHigh"),
                "day_low": meta.get("regularMarketDayLow"),
                "volume": meta.get("regularMarketVolume"),
            }
        time.sleep(0.2)
    return out


def fetch_many(fn, symbols: List[str], workers: int = 6, **kw) -> Dict[str, object]:
    """Run fn(symbol, **kw) for each symbol in a small thread pool; failures are skipped."""
    out: Dict[str, object] = {}

    def one(sym):
        try:
            return sym, fn(sym, **kw)
        except Exception as e:  # one bad chart must not stop a scan
            logger.debug("fetch failed %s: %s", sym, e)
            return sym, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, val in ex.map(one, symbols):
            if val:
                out[sym] = val
    return out
