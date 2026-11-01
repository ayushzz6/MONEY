"""Crypto & XAUUSD board: live Binance prices, each pair's strength, order-block zone, alignment and confirmations.

Analytics only: nothing here places or simulates a trade. Crypto pairs are ranked against each other with the NSE
board's rules (market/orderblocks.py, finished 5-minute candles); XAUUSD (Binance's XAUUSDT perpetual) gets the same
zone, levels and checks but is not ranked with coins. Crypto trades all day, so "today" is the UTC day, the day
Binance's daily candles use (05:30 to 05:30 IST): previous close, high, low, ATR14, CPR and the 20-day trend come from
those daily candles. Prices arrive on Binance's websockets (market/crypto/binance.py); open interest, funding, daily
and hourly candles are re-read every few minutes over REST.

Alignment (the board's 4 checks, crypto version): top 5 on its side, BTC moving the same way today (not counted for
BTC itself), most pairs moving the same way, price on the right side of today's VWAP. Gold only has the VWAP check.
Confirmations: market/crypto/confirmations.py.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from market import confirmations as CF
from market import orderblocks as OB
from market.candles.bars import DailyBar, vwap_series
from market.crypto import binance as B
from market.crypto import confirmations as CC

logger = logging.getLogger("CryptoBoard")

DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT",
                 "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "UNIUSDT", "AAVEUSDT"]
GOLD = "XAUUSDT"
NAMES = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "BNBUSDT": "BNB", "SOLUSDT": "Solana", "XRPUSDT": "XRP",
         "DOGEUSDT": "Dogecoin", "ADAUSDT": "Cardano", "TRXUSDT": "TRON", "AVAXUSDT": "Avalanche", "LINKUSDT": "Chainlink",
         "DOTUSDT": "Polkadot", "LTCUSDT": "Litecoin", "BCHUSDT": "Bitcoin Cash", "NEARUSDT": "NEAR", "SUIUSDT": "Sui",
         "UNIUSDT": "Uniswap", "AAVEUSDT": "Aave", "XAUUSDT": "Gold"}
CATEGORY = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Smart-contract chains", "SOLUSDT": "Smart-contract chains",
            "ADAUSDT": "Smart-contract chains", "TRXUSDT": "Smart-contract chains", "AVAXUSDT": "Smart-contract chains",
            "DOTUSDT": "Smart-contract chains", "NEARUSDT": "Smart-contract chains", "SUIUSDT": "Smart-contract chains",
            "BNBUSDT": "Exchange tokens", "XRPUSDT": "Payments", "LTCUSDT": "Payments", "BCHUSDT": "Payments",
            "DOGEUSDT": "Meme", "LINKUSDT": "DeFi", "UNIUSDT": "DeFi", "AAVEUSDT": "DeFi"}
KEEP = 600                      # finished 5-minute candles kept per pair (about two days)
TOP_ALIGN = 5
SLOW_S = 300                    # open interest, funding, daily and hourly candles


def day0(now: Optional[float] = None) -> int:
    """00:00 UTC today, epoch seconds."""
    return int((now or time.time()) // 86400 * 86400)


def daily_context(daily: List[DailyBar], today: str) -> Optional[Dict]:
    """Yesterday's close, high and low, ATR14, today's CPR and the 20-day average close, from finished daily candles."""
    before = [b for b in daily if b.date < today]
    if len(before) < 15:
        return None
    trs = [max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c)) for a, b in zip(before[-15:-1], before[-14:])]
    return {"prev_close": before[-1].c, "prev_high": before[-1].h, "prev_low": before[-1].l, "atr": sum(trs) / len(trs),
            "cpr": CF.cpr(before[-1].h, before[-1].l, before[-1].c), "cpr_rank": CF.cpr_rank(before),
            "sma": sum(b.c for b in before[-CF.TREND_DAYS:]) / min(len(before), CF.TREND_DAYS)}


def oi_change(rows, start: int) -> Optional[Dict]:
    """Open interest at 00:00 UTC (the last reading at or before it) and now, from 5-minute readings."""
    if not rows:
        return None
    base = next((v for t, v in reversed(rows) if t <= start), None) or next((v for t, v in rows if t >= start), None)
    t, now = rows[-1]
    if not base:
        return None
    return {"open": base, "now": now, "chg": 100 * (now / base - 1), "at": t}


def why(e: Exception) -> str:
    code = B.http_status(e)
    if code in (403, 451):
        return f"Binance refused requests from this location (HTTP {code}): its data is not available from here"
    return f"Binance: {e}"


class CryptoBoard:
    def __init__(self, settings):
        cfg = settings.get("crypto", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.coins = [s.upper() for s in (cfg.get("symbols") or DEFAULT_COINS)]
        self.gold = str(cfg.get("gold") or GOLD).upper()
        self._lock = threading.RLock()
        self.closed: Dict[str, List[B.CBar]] = {}
        self.forming: Dict[str, Optional[B.CBar]] = {}
        self.ticker: Dict[str, Dict] = {}
        self.daily: Dict[str, List[DailyBar]] = {}
        self.hourly: Dict[str, List[B.CBar]] = {}
        self.oi: Dict[str, Dict] = {}
        self.funding: Dict[str, Optional[float]] = {}
        self.fut: Dict[str, int] = {}             # futures symbol -> price decimals
        self.dec: Dict[str, int] = {}             # every pair's price decimals (its Binance tick size)
        self.streams: List[B.Stream] = []
        self.board: Optional[Dict] = None
        self.details: Dict[str, Dict] = {}
        self.prev_close: Dict[str, float] = {}
        self.error = ""
        self.slow_at = 0.0
        self._dirty = False
        self._last = 0.0

    def symbols(self) -> List[str]:
        return self.coins + [self.gold]

    def is_fut(self, s: str) -> bool:
        return s == self.gold

    # ------------------------------------------------------------ data in
    def start(self):
        if self.enabled:
            threading.Thread(target=self._boot, name="crypto", daemon=True).start()

    def _boot(self):
        while not self.closed:
            try:
                self.fut = B.futures_symbols()
            except Exception as e:
                self.error = why(e)
            syms = list(self.symbols())
            with ThreadPoolExecutor(8) as ex:
                failed = [(s, e) for s, e in zip(syms, ex.map(self._try_history, syms)) if e is not None]
            for s, e in failed:
                if B.http_status(e) == 400 and s in self.coins:      # not a Binance spot pair: drop it
                    self.coins.remove(s)
                    logger.warning("%s is not a Binance spot pair: left out", s)
                else:
                    self.error = why(e)
            if not self.closed:
                time.sleep(60)
        coins = [s for s in self.coins if s in self.closed]
        self.coins = coins
        try:
            self.dec = dict(B.spot_decimals(coins), **{s: d for s, d in self.fut.items() if s == self.gold})
        except Exception as e:
            logger.warning("tick sizes: %s", e)
        if coins:
            self.streams.append(B.Stream("crypto (spot)", B.SPOT_WS, coins, self._on_msg, lambda: self._backfill(coins)))
        if self.gold in self.closed:
            self.streams.append(B.Stream("XAUUSD (futures)", B.FUT_WS, [self.gold], self._on_msg,
                                         lambda: self._backfill([self.gold])))
        for st in self.streams:
            st.start()
        self.error = ""
        threading.Thread(target=self._slow_loop, name="crypto-slow", daemon=True).start()
        self._loop()

    def _try_history(self, s: str) -> Optional[Exception]:
        try:
            self._history(s)
            return None
        except Exception as e:
            return e

    def _history(self, s: str):
        fut = self.is_fut(s)
        rows = B.klines(s, fut, "5m", KEEP + 1)
        now = time.time()
        bars = [B.bar(k) for k in rows]
        done = [b for b, k in zip(bars, rows) if int(k[6]) / 1000 < now]
        d = B.klines(s, fut, "1d", 40)
        h = B.klines(s, fut, "1h", 24 * (CF.VOL_DAYS + 2))
        with self._lock:
            self.closed[s] = done[-KEEP:]
            self.forming[s] = bars[-1] if len(done) < len(bars) else None
            self.daily[s] = [B.daily_bar(k) for k in d]
            self.hourly[s] = [B.bar(k) for k in h]
            self._dirty = True

    def _backfill(self, syms: List[str]):
        """After every (re)connect: the candles that finished while the socket was down."""
        for s in syms:
            try:
                rows = B.klines(s, self.is_fut(s), "5m", 60)
            except Exception as e:
                logger.warning("backfill %s: %s", s, e)
                continue
            now = time.time()
            new = [B.bar(k) for k in rows if int(k[6]) / 1000 < now]
            with self._lock:
                merged = {b.start: b for b in self.closed.get(s, [])}
                merged.update({b.start: b for b in new})
                self.closed[s] = sorted(merged.values(), key=lambda b: b.start)[-KEEP:]
                self._dirty = True

    def _on_msg(self, sym: str, kind: str, d: Dict):
        if kind == "kline":
            k = d.get("k") or {}
            b = B.ws_bar(k)
            with self._lock:
                cur = self.closed.setdefault(sym, [])
                prev = self.forming.get(sym)
                if prev is not None and prev.start < b.start and (not cur or cur[-1].start < prev.start):
                    cur.append(prev)                 # its closing message was missed: its last update stands in
                    self._dirty = True
                if k.get("x"):
                    if cur and cur[-1].start == b.start:
                        cur[-1] = b
                    elif not cur or b.start > cur[-1].start:
                        cur.append(b)
                    del cur[:-KEEP]
                    self.forming[sym] = None
                    self._dirty = True
                else:
                    self.forming[sym] = b
        elif kind == "ticker" and d.get("c"):
            with self._lock:
                self.ticker[sym] = {"last": float(d["c"]), "open24": float(d["o"]), "high24": float(d["h"]),
                                    "low24": float(d["l"]), "qvol24": float(d["q"]), "at": time.time()}

    def _slow(self):
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(self._slow_one, self.symbols()))
        self.slow_at = time.time()
        self._dirty = True

    def _slow_one(self, s: str):
        if s in self.fut:
            try:
                o = oi_change(B.open_interest(s), day0())
                f = B.funding(s)
                with self._lock:
                    if o:
                        self.oi[s] = o
                    self.funding[s] = f
            except Exception as e:
                logger.debug("open interest %s: %s", s, e)
        try:
            d = B.klines(s, self.is_fut(s), "1d", 40)
            h = B.klines(s, self.is_fut(s), "1h", 24 * (CF.VOL_DAYS + 2))
            with self._lock:
                self.daily[s] = [B.daily_bar(k) for k in d]
                self.hourly[s] = [B.bar(k) for k in h]
        except Exception as e:
            logger.debug("daily candles %s: %s", s, e)

    def _slow_loop(self):
        while True:
            try:
                self._slow()
            except Exception:
                logger.exception("crypto slow refresh failed")
            time.sleep(SLOW_S)

    def _loop(self):
        while True:
            try:
                for st in self.streams:
                    if st.connected and time.time() - st.last_msg > B.SILENT_S:
                        logger.warning("Binance %s stream silent for %ds: reconnecting", st.name, B.SILENT_S)
                        st.kick()
                if self._dirty or time.time() - self._last > 60:
                    self._dirty = False
                    self.refresh()
            except Exception as e:
                logger.exception("crypto board failed")
                self.error = str(e)
            time.sleep(1.0)

    # ------------------------------------------------------------ the board
    def refresh(self) -> str:
        start = day0()
        today_iso = time.strftime("%Y-%m-%d", time.gmtime(start))
        with self._lock:
            snap = {s: (list(self.closed.get(s, [])), list(self.daily.get(s, [])), list(self.hourly.get(s, [])))
                    for s in self.symbols()}
            oi, fund = dict(self.oi), dict(self.funding)
        rows, details, prev_close = [], {}, {}
        for s, (closed, daily, hourly) in snap.items():
            ctx = daily_context(daily, today_iso)
            if not ctx or not closed:
                continue
            today = [b for b in closed if b.start >= start]
            st = OB.strength(today, ctx["prev_close"], ctx["atr"]) if today else \
                {"score": 0.0, "move_atr": None, "clean": None, "place": None, "change_pct": None}
            ev = OB.event_risk(st)
            zones = [] if ev else OB.detect_zones(today, ctx["prev_high"], ctx["prev_low"])
            vw = vwap_series(today) if today else []
            last = today[-1].c if today else closed[-1].c
            day_open = today[0].o if today else None
            vol, tb = sum(b.v for b in today), sum(b.tb for b in today)
            upto = (today[-1].start + 300 - start) if today else 0
            o = oi.get(s) or {}
            cf_in = {"last": last, "day_chg": 100 * (last / day_open - 1) if day_open else None, "oi_chg": o.get("chg"),
                     "funding": fund.get(s), "taker": tb / vol if vol > 0 else None,
                     "rvol": CC.rel_volume(vol, hourly, start, upto) if today else None,
                     "upto": time.strftime("%H:%M IST", time.gmtime(start + upto + 19800)), "cpr": ctx["cpr"],
                     "cpr_rank": ctx["cpr_rank"], "prev_high": ctx["prev_high"], "prev_low": ctx["prev_low"], "sma": ctx["sma"]}
            gold = s == self.gold
            rows.append({"symbol": s, "name": NAMES.get(s, s[:-4]), "sector": "Precious metal" if gold else CATEGORY.get(s, "Other"),
                         "gold": gold, "last": last, **st, "event": ev, "above_vwap": bool(vw) and last >= vw[-1],
                         "vwap": vw[-1] if vw else None, "atr": ctx["atr"], "prev_close": ctx["prev_close"],
                         "day_open": day_open, "oi_chg": o.get("chg"), "funding": fund.get(s), "taker": cf_in["taker"],
                         "spark": [b.c for b in today][-96:], "candles_today": len(today)})
            details[s] = {"bars": closed[-288:], "vwap": [(b.start, v) for b, v in zip(today, vw)], "zones": zones,
                          "ctx": ctx, "cf": cf_in}
            prev_close[s] = ctx["prev_close"]
        coins = [r for r in rows if not r["gold"]]
        gold = next((r for r in rows if r["gold"]), None)
        ranked = OB.rank_board(coins)
        up = sum(1 for r in coins if (r["change_pct"] or 0) > 0)
        down = sum(1 for r in coins if (r["change_pct"] or 0) < 0)
        breadth_side = 1 if up > down else -1 if down > up else 0
        btc = next((r for r in coins if r["symbol"] == "BTCUSDT"), None)
        btc_chg = (btc or {}).get("change_pct") or 0
        for r in coins + ([gold] if gold else []):
            if r["gold"]:
                r["side"] = 1 if r["score"] > 0 else -1 if r["score"] < 0 else 0
                r["rule"] = "flat or choppy" if not r["side"] else "event risk" if r["event"] else "gold: not ranked with coins"
            side = r.get("side", 0)
            setup = OB.current_setup(details[r["symbol"]]["zones"], side) if side else None
            r["zone"] = setup.to_dict() if setup else None
            vw_ok = side != 0 and (r["above_vwap"] if side > 0 else not r["above_vwap"])
            if r["gold"]:
                checks = {"vwap": vw_ok}
            else:
                checks = {"rank": side != 0 and r.get("rank", 99) <= TOP_ALIGN,
                          "btc": None if r["symbol"] == "BTCUSDT" else side != 0 and btc_chg * side > 0,
                          "breadth": side != 0 and breadth_side == side, "vwap": vw_ok}
            r["checks"] = checks
            r["aligned"] = sum(1 for v in checks.values() if v is True)
            r["aligned_of"] = sum(1 for v in checks.values() if v is not None)
            r["confirm"] = CC.evaluate(side, details[r["symbol"]]["cf"]) if side else None
        board = {"day": today_iso, "day_start": start, "updated_at": time.time(), "coins": len(coins), "up": up,
                 "down": down, "btc_chg": btc_chg if btc else None, "gold": gold,
                 "bullish": ranked["bullish"], "bearish": ranked["bearish"],
                 "picks": ranked["bullish_picks"] + ranked["bearish_picks"],
                 "rows": sorted(coins, key=lambda r: self.symbols().index(r["symbol"])),
                 "confirm_rules": [{"key": k, "label": lab, "rule": rule} for k, lab, rule in CC.RULES],
                 "slow_at": self.slow_at or None, "dec": self.dec}
        with self._lock:
            self.board, self.details, self.prev_close = board, details, prev_close
            self._last = time.time()
        return f"{len(coins)} pairs and {'gold' if gold else 'no gold'} ranked"

    # ------------------------------------------------------------ out
    def live(self) -> Dict[str, Dict]:
        """The latest websocket price of every pair, with its change today and over 24 hours."""
        out = {}
        with self._lock:
            for s in self.symbols():
                f, t, c = self.forming.get(s), self.ticker.get(s) or {}, self.closed.get(s)
                p = f.c if f is not None else t.get("last") or (c[-1].c if c else None)
                if p is None:
                    continue
                pc = self.prev_close.get(s)
                out[s] = {"price": p, "chg_day": 100 * (p / pc - 1) if pc else None,
                          "chg24": 100 * (p / t["open24"] - 1) if t.get("open24") else None,
                          "high24": t.get("high24"), "low24": t.get("low24"), "qvol24": t.get("qvol24"), "at": t.get("at")}
        return out

    def view(self) -> Dict:
        if not self.enabled:
            return {"ready": False, "error": "crypto is switched off (crypto.enabled in configs/trading.json)"}
        with self._lock:
            b = self.board
        streams = [st.status() for st in self.streams]
        if not b:
            return {"ready": False, "streams": streams,
                    "error": self.error or "loading candles from Binance (a few seconds on first start)"}
        return dict(b, ready=True, error=self.error, streams=streams, live=self.live())

    def stock(self, symbol: str) -> Optional[Dict]:
        symbol = symbol.upper()
        with self._lock:
            d, b, f = self.details.get(symbol), self.board, self.forming.get(symbol)
        if not d or not b:
            return None
        rows = b["rows"] + ([b["gold"]] if b.get("gold") else [])
        row = next((r for r in rows if r["symbol"] == symbol), {"symbol": symbol})
        bars = [x.to_dict() for x in d["bars"]] + ([dict(f.to_dict(), forming=True)] if f is not None else [])
        c = d["ctx"]
        return {"symbol": symbol, "name": NAMES.get(symbol, symbol), "gold": symbol == self.gold, "day_start": b["day_start"],
                "row": row, "bars": bars, "vwap": [{"t": t, "v": v} for t, v in d["vwap"]],
                "zones": [z.to_dict() for z in d["zones"]], "prev_close": c["prev_close"], "prev_high": c["prev_high"],
                "prev_low": c["prev_low"], "cpr": c["cpr"], "live": self.live().get(symbol)}
