"""The NIFTY 500 universe with Angel One tokens and tick sizes.

- Constituents: niftyindices.com's ind_nifty500list.csv (refreshed weekly, cached).
- Tokens and tick sizes: Angel One's public OpenAPIScripMaster.json (about 37 MB,
  refreshed daily, cached as a small NSE-equity extract). Tick sizes there are in paise.
- Stock futures (every expiry) from the same file, for the live futures OI on the order-block board.
"""
import csv
import datetime as dt
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.config.settings import CACHE_DIR, ROOT
from core.models.instrument import Instrument
from core.utilities.http import FetchError, get_bytes

logger = logging.getLogger("Universe")

NIFTY500_URLS = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
)
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/130 Safari/537.36"}

NIFTY500_CSV = CACHE_DIR / "ind_nifty500list.csv"
ANGEL_EXTRACT = CACHE_DIR / "angel_nse_eq.json"
ANGEL_FUTURES = CACHE_DIR / "angel_nfo_fut.json"

INDEX_TOKENS = {"^NSEI": "99926000"}   # NIFTY 50 on Angel One's feed


def _fresh(path: Path, max_age_s: float) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < max_age_s


def _price_band_tick(price: Optional[float]) -> float:
    """Fallback only, when the scrip master is unavailable: NSE price-band tick sizes."""
    if price is None:
        return 0.05
    if price < 250:
        return 0.01
    if price < 1000:
        return 0.05
    if price < 5000:
        return 0.10
    return 1.00


class Universe:
    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}
        self.futures: Dict[str, List[dict]] = {}     # {symbol: [{"token", "symbol", "expiry"}]} by expiry

    # ---------------------------------------------------------------- loading
    def load(self, refresh: bool = False) -> "Universe":
        rows = self._constituents(refresh)
        master = self._angel_master(refresh)
        for r in rows:
            sym = r["Symbol"].strip().upper()
            m = master.get(sym, {})
            self.instruments[sym] = Instrument(
                instrument_id=f"NSE_{sym}",
                symbol=sym,
                exchange="NSE",
                trading_symbol=m.get("trading_symbol", f"{sym}-EQ"),
                angel_token=m.get("token", ""),
                sector=r.get("Industry", "") or "",
                industry=r.get("Industry", "") or "",
                active=True,
                name=(r.get("Company Name") or sym).strip(),
                tick_size=m.get("tick", 0.0) or 0.0,
            )
        logger.info("Universe: %d NIFTY 500 stocks, %d with Angel One tokens",
                    len(self.instruments), sum(1 for i in self.instruments.values() if i.angel_token))
        return self

    def _constituents(self, refresh: bool) -> List[dict]:
        if refresh or not _fresh(NIFTY500_CSV, 7 * 86400):
            for url in NIFTY500_URLS:
                try:
                    body = get_bytes(url, headers=BROWSER_UA, timeout=30)
                    if b"Symbol" in body[:200]:
                        NIFTY500_CSV.write_bytes(body)
                        break
                except FetchError as e:
                    logger.warning("NIFTY 500 list download failed (%s): %s", url, e)
        if NIFTY500_CSV.exists():
            text = NIFTY500_CSV.read_text(encoding="utf-8-sig")
            return [r for r in csv.DictReader(io.StringIO(text)) if r.get("Symbol")]
        raise RuntimeError("No NIFTY 500 constituent list: could not download it and no cached copy "
                           f"at {NIFTY500_CSV}. The repo's '{ROOT / 'nifty 500.csv'}' has names only, no symbols.")

    def _angel_master(self, refresh: bool) -> Dict[str, dict]:
        if refresh or not _fresh(ANGEL_EXTRACT, 20 * 3600) or not _fresh(ANGEL_FUTURES, 20 * 3600):
            try:
                raw = json.loads(get_bytes(SCRIP_MASTER_URL, timeout=90).decode("utf-8"))
                extract, futures = {}, {}
                for x in raw:
                    if x.get("exch_seg") == "NSE" and str(x.get("symbol", "")).endswith("-EQ"):
                        try:
                            tick = float(x.get("tick_size") or 0) / 100.0
                        except ValueError:
                            tick = 0.0
                        extract[x["name"]] = {"token": x["token"], "trading_symbol": x["symbol"], "tick": tick}
                    elif x.get("exch_seg") == "NFO" and x.get("instrumenttype") == "FUTSTK":
                        try:
                            exp = dt.datetime.strptime(str(x.get("expiry")), "%d%b%Y").date().isoformat()
                        except ValueError:
                            continue
                        futures.setdefault(x["name"], []).append({"token": x["token"], "symbol": x["symbol"], "expiry": exp})
                ANGEL_EXTRACT.write_text(json.dumps(extract), encoding="utf-8")
                ANGEL_FUTURES.write_text(json.dumps({k: sorted(v, key=lambda f: f["expiry"]) for k, v in futures.items()}),
                                         encoding="utf-8")
            except (FetchError, ValueError) as e:
                logger.warning("Angel One scrip master download failed: %s (using cache if any)", e)
        if ANGEL_FUTURES.exists():
            self.futures = json.loads(ANGEL_FUTURES.read_text(encoding="utf-8"))
        if ANGEL_EXTRACT.exists():
            return json.loads(ANGEL_EXTRACT.read_text(encoding="utf-8"))
        return {}

    # ---------------------------------------------------------------- lookups
    @property
    def symbols(self) -> List[str]:
        return sorted(self.instruments)

    def get(self, symbol: str) -> Optional[Instrument]:
        return self.instruments.get(symbol)

    def tick(self, symbol: str, price: Optional[float] = None) -> float:
        inst = self.instruments.get(symbol)
        if inst and inst.tick_size > 0:
            return inst.tick_size
        return _price_band_tick(price)

    def name(self, symbol: str) -> str:
        inst = self.instruments.get(symbol)
        return inst.name if inst else symbol

    def token(self, symbol: str) -> str:
        if symbol in INDEX_TOKENS:
            return INDEX_TOKENS[symbol]
        inst = self.instruments.get(symbol)
        return inst.angel_token if inst else ""

    def futures_for(self, symbol: str, on: str) -> List[dict]:
        """The stock's futures contracts that have not expired by `on` (ISO date), nearest first."""
        return [f for f in self.futures.get(symbol, []) if f["expiry"] >= on]

    def symbol_for_token(self) -> Dict[str, str]:
        out = {v: k for k, v in INDEX_TOKENS.items()}
        out.update({i.angel_token: s for s, i in self.instruments.items() if i.angel_token})
        return out
