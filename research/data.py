"""Five years of 5-minute NSE bars from Angel One SmartAPI, cached as one .npz per symbol,
and loaded as day-by-slot matrices (sessions x 75 bars) for vectorized backtests.

Angel One's prices are adjusted for splits and bonuses. The stock universe is the CURRENT
NIFTY 100, so the test carries survivorship bias (stocks that joined the index because
they did well are in it for the whole period).
"""
import csv
import datetime as dt
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.config.settings import DATA_DIR
from core.utilities.http import get_bytes
from market.calendar import IST

logger = logging.getLogger("research.data")

HIST_DIR = DATA_DIR / "research" / "hist5m"
NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/130 Safari/537.36"}
INDEX_TOKENS = {"NIFTY": "99926000", "BANKNIFTY": "99926009", "INDIAVIX": "99926017"}
SLOTS = 75
OPEN_SEC = 9 * 3600 + 15 * 60


def nifty100() -> List[str]:
    path = DATA_DIR / "research" / "ind_nifty100list.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get_bytes(NIFTY100_URL, headers=BROWSER_UA, timeout=30))
    rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
    return [r["Symbol"].strip().upper() for r in rows if r.get("Symbol")]


# ---------------------------------------------------------------- download
class AngelHistory:
    def __init__(self, smart_api, min_interval: float = 0.45):
        self.api = smart_api
        self.min_interval = min_interval
        self._last = 0.0

    def _call(self, token: str, frm: str, to: str):
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.api.getCandleData({"exchange": "NSE", "symboltoken": token, "interval": "FIVE_MINUTE",
                                            "fromdate": frm, "todate": to})
            except Exception as e:
                r = {"status": False, "message": str(e)}
            if r and r.get("status"):
                return r.get("data") or []
            msg = str((r or {}).get("message"))
            logger.debug("retry %s %s %s: %s", token, frm, to, msg)
            time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"getCandleData failed for {token} {frm}..{to}")

    def download(self, symbol: str, token: str, start: dt.date, end: dt.date, force=False) -> Optional[Path]:
        out = HIST_DIR / f"{symbol}.npz"
        if out.exists() and not force:
            return out
        rows = []
        d = start
        while d <= end:
            e = min(end, d + dt.timedelta(days=95))
            rows += self._call(token, f"{d.isoformat()} 09:15", f"{e.isoformat()} 15:30")
            d = e + dt.timedelta(days=1)
        if not rows:
            return None
        seen = {}
        for ts, o, h, l, c, v in rows:
            t = int(dt.datetime.fromisoformat(ts).timestamp())
            seen[t] = (o, h, l, c, v)
        t = np.array(sorted(seen), dtype=np.int64)
        arr = np.array([seen[x] for x in t], dtype=np.float64)
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, t=t, o=arr[:, 0], h=arr[:, 1], l=arr[:, 2], c=arr[:, 3], v=arr[:, 4])
        return out


def download_all(symbols: List[str], universe, start: dt.date, end: dt.date, log=print):
    from brokers.angelone.auth import AngelOneAuth
    from core.config.settings import Settings
    creds = Settings().angel_credentials()
    if not creds:
        raise RuntimeError("Angel One credentials are needed in .env for 5 years of intraday history")
    auth = AngelOneAuth(*creds)
    auth.login()
    h = AngelHistory(auth.smart_api)
    todo = [(s, INDEX_TOKENS.get(s) or universe.token(s)) for s in symbols]
    t0 = time.time()
    for i, (s, tok) in enumerate(todo, 1):
        if not tok:
            log(f"  {s}: no Angel One token, skipped")
            continue
        if (HIST_DIR / f"{s}.npz").exists():
            continue
        try:
            h.download(s, tok, start, end)
        except Exception as e:
            log(f"  {s}: download failed: {e}")
            continue
        if i % 10 == 0:
            log(f"  {i}/{len(todo)} symbols, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- loading
@dataclass
class DayMatrix:
    symbol: str
    dates: np.ndarray       # (D,) datetime64[D]
    o: np.ndarray           # (D, 75)
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    start: np.ndarray       # (D,) epoch seconds of the 09:15 bar

    @property
    def n(self):
        return len(self.dates)

    def daily(self):
        """Daily open, high, low, close, volume (D,)."""
        return self.o[:, 0], np.nanmax(self.h, axis=1), np.nanmin(self.l, axis=1), self.c[:, -1], np.nansum(self.v, axis=1)


def load(symbol: str, max_missing: int = 6) -> Optional[DayMatrix]:
    p = HIST_DIR / f"{symbol}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    t, o, h, l, c, v = z["t"], z["o"], z["h"], z["l"], z["c"], z["v"]
    ist = t + 19800
    day = ist // 86400
    sec = ist % 86400
    slot = (sec - OPEN_SEC) // 300
    ok = (slot >= 0) & (slot < SLOTS)
    day, slot, o, h, l, c, v = day[ok], slot[ok], o[ok], h[ok], l[ok], c[ok], v[ok]
    udays, inv = np.unique(day, return_inverse=True)
    D = len(udays)
    M = {k: np.full((D, SLOTS), np.nan) for k in ("o", "h", "l", "c")}
    V = np.zeros((D, SLOTS))
    for k, arr in (("o", o), ("h", h), ("l", l), ("c", c)):
        M[k][inv, slot] = arr
    V[inv, slot] = v
    # forward-fill short gaps with flat bars at the previous close; drop days with too many
    present = ~np.isnan(M["c"])
    keep = (present.sum(axis=1) >= SLOTS - max_missing) & present[:, 0]
    for i in np.where(keep)[0]:
        for j in range(1, SLOTS):
            if not present[i, j]:
                pc = M["c"][i, j - 1]
                for k in ("o", "h", "l", "c"):
                    M[k][i, j] = pc
    dates = (udays * 86400).astype("datetime64[s]").astype("datetime64[D]")
    starts = (udays * 86400 - 19800 + OPEN_SEC).astype(np.int64)
    o_, h_, l_, c_, v_ = (M["o"][keep], M["h"][keep], M["l"][keep], M["c"][keep], V[keep])
    # a split or bonus the feed did not adjust shows up as an overnight jump no NIFTY 100 stock
    # really made in this period (over 35%): rescale everything before it
    for i in range(1, len(o_)):
        r = o_[i, 0] / c_[i - 1, -1]
        if r < 0.65 or r > 1.55:
            for arr in (o_, h_, l_, c_):
                arr[:i] *= r
            v_[:i] /= r
            logger.info("%s: rescaled bars before %s by %.4f (unadjusted split)", symbol, dates[keep][i], r)
    return DayMatrix(symbol, dates[keep], o_, h_, l_, c_, v_, starts[keep])


def align(mats: Dict[str, DayMatrix], dates: np.ndarray) -> Dict[str, np.ndarray]:
    """Row index of each master date in each matrix (-1 when the symbol has no session that day)."""
    out = {}
    for s, m in mats.items():
        pos = {d: i for i, d in enumerate(m.dates)}
        out[s] = np.array([pos.get(d, -1) for d in dates])
    return out


def ist_dt(epoch: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(epoch, IST)


def nifty200():
    """NIFTY 200 symbols and their industry (sector) from the saved constituent list."""
    path = DATA_DIR / "research" / "ind_nifty200list.csv"
    if not path.exists():
        path.write_bytes(get_bytes("https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                                   headers=BROWSER_UA, timeout=30))
    rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
    return {r["Symbol"].strip().upper(): (r.get("Industry") or "").strip() for r in rows if r.get("Symbol")}


def nifty500():
    """NIFTY 500 symbols (equity series) and their industry from the saved constituent list."""
    path = DATA_DIR / "research" / "ind_nifty500list.csv"
    if not path.exists():
        from core.config.settings import CACHE_DIR
        cached = CACHE_DIR / "ind_nifty500list.csv"
        path.write_bytes(cached.read_bytes() if cached.exists() else
                         get_bytes("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                                   headers=BROWSER_UA, timeout=30))
    rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
    return {r["Symbol"].strip().upper(): (r.get("Industry") or "").strip() for r in rows
            if r.get("Symbol") and (r.get("Series") or "EQ").strip() == "EQ" and not r["Symbol"].startswith("DUMMY")}
