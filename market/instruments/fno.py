"""NSE F&O stock list and yesterday's futures open interest, from NSE's public F&O bhavcopy (cached per day).

The latest published session's file gives every stock with futures, its total futures open interest and the
day's change, so each stock's build-up can be read: price up + OI up = long build-up, price down + OI up =
short build-up, price up + OI down = short covering, price down + OI down = long unwinding. The same file gives
each stock's options call and put OI and their change (put / call ratio, put or call writing), in total and by expiry.

Live, from Angel One:
- LiveOI: today's futures OI (every expiry, summed), streamed from the websocket as it changes, and read from the
  quote API every few minutes for every contract;
- LiveOptions: today's call and put OI of the nearest expiry for the board's leaders, every few minutes;
- BuildUpLists: Angel One's own lists of the near-month futures with the biggest build-up, covering and unwinding.
"""
import datetime as dt
import json
import logging
import re
import statistics
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from core.config.settings import CACHE_DIR

logger = logging.getLogger("FnO")
INDEXES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
FOLDER = CACHE_DIR / "fno"
VERSION = 3                     # 2: adds options OI; 3: stock options OI by expiry


def _iso(x) -> Optional[str]:
    for f in ("%Y-%m-%d", "%d-%b-%Y", "%d%b%Y", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(str(x).strip(), f).date().isoformat()
        except ValueError:
            continue
    return None


def latest(today: Optional[dt.date] = None, max_back: int = 7) -> Optional[Dict]:
    """{"date", "symbols": [...], "oi": {sym: {"oi", "chg", "chg_pct", "ce_oi", "pe_oi", "ce_chg", "pe_chg", "pcr",
    "opt_exp": {expiry: [ce_oi, pe_oi]}}}} for the latest session NSE has published."""
    today = today or dt.date.today()
    FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = FOLDER / f"checked_{today.isoformat()}.json"
    if stamp.exists():
        try:
            doc = json.loads((FOLDER / json.loads(stamp.read_text())["file"]).read_text())
            if doc.get("v") == VERSION:
                return doc
        except Exception:
            pass
    from research.lead import nse_daily
    for back in range(0, max_back + 1):
        day = today - dt.timedelta(days=back)
        if day.weekday() >= 5:
            continue
        path = FOLDER / f"{day.isoformat()}.json"
        doc = old = None
        if path.exists():
            doc = json.loads(path.read_text())
            if doc.get("v") != VERSION:
                doc, old = None, doc                # read it again for the newer fields; keep it if NSE refuses
        if doc is None:
            try:
                rows, by_exp = nse_daily.fo(day, by_expiry=True)
            except Exception as e:
                logger.warning("F&O bhavcopy %s: %s", day, e)
                rows, by_exp = None, None
            if not rows and old:
                return old
            if rows:
                oi = {}
                for s, r in rows.items():
                    if s in INDEXES or r[0] is None:
                        continue
                    base = (r[0] or 0) - (r[1] or 0)
                    ce, pe = r[3], r[4]
                    exps = {}
                    for e, v in ((by_exp or {}).get(s) or {}).items():
                        if _iso(e):
                            exps[_iso(e)] = v
                    oi[s] = {"oi": r[0], "chg": r[1], "chg_pct": round(100 * r[1] / base, 2) if base > 0 else None,
                             "ce_oi": ce, "pe_oi": pe, "ce_chg": r[5], "pe_chg": r[6],
                             "pcr": round(pe / ce, 2) if ce and pe is not None else None, "opt_exp": exps}
                doc = {"v": VERSION, "date": day.isoformat(), "symbols": sorted(oi), "oi": oi}
                path.write_text(json.dumps(doc))
        if doc:
            stamp.write_text(json.dumps({"file": path.name}))
            return doc
    return None


def buildup(price_chg: Optional[float], oi_chg_pct: Optional[float]) -> Optional[str]:
    if price_chg is None or oi_chg_pct is None:
        return None
    if oi_chg_pct >= 0:
        return "long build-up" if price_chg >= 0 else "short build-up"
    return "short covering" if price_chg >= 0 else "long unwinding"


def _quotes(api, tokens: List[str], batch: int):
    """Angel One's FULL quotes for NFO contracts, `batch` a request: yields (token, open interest), or raises."""
    for i in range(0, len(tokens), batch):
        r = api.getMarketData("FULL", {"NFO": tokens[i:i + batch]})
        if not r or not r.get("status"):
            raise RuntimeError(f"Angel One quote API: {(r or {}).get('message') or 'no answer'}")
        for q in (r.get("data") or {}).get("fetched") or []:
            if q.get("opnInterest") is not None:
                yield str(q.get("symbolToken")), float(q["opnInterest"])
        time.sleep(0.15)


class LiveOI:
    """Today's futures OI per stock (every unexpired contract, summed) and Angel One's put / call ratio.

    Each contract's OI is read from the quote API every EVERY_S seconds, 50 contracts a request, and streamed from
    the websocket (SNAP_QUOTE) as it changes, for the contracts that fit on the socket. The stream is used once its
    numbers match the quote API's. The total is compared with the bhavcopy's (also every expiry), so a roll from one
    expiry to the next is not read as positions closing. A stock counts once every one of its contracts has a reading.
    """
    EVERY_S = 180
    BATCH = 50

    def __init__(self):
        self.tokens: Dict[str, List[str]] = {}      # stock -> its contract tokens
        self.rest: Dict[str, float] = {}            # contract token -> OI from the quote API
        self.stream: Dict[str, float] = {}          # contract token -> OI from the websocket
        self.stream_ok: Optional[bool] = None       # the websocket's numbers match the quote API's
        self.units_ok = True
        self.pcr: Dict[str, float] = {}
        self.day: Optional[str] = None
        self.at = 0.0
        self.stream_at = 0.0
        self.error = ""
        self._lock = threading.Lock()

    def due(self) -> bool:
        return time.time() - self.at >= self.EVERY_S

    def on_stream(self, token: str, oi: float, ts: float = 0.0):
        with self._lock:
            self.stream[token] = oi
            self.stream_at = time.time()

    @property
    def oi(self) -> Dict[str, float]:
        """{stock: total futures OI now} for the stocks whose every contract has a reading."""
        if not self.units_ok:
            return {}
        with self._lock:
            use_stream = self.stream_ok is True
            out = {}
            for s, toks in self.tokens.items():
                vals = [self.stream[t] if use_stream and t in self.stream else self.rest.get(t) for t in toks]
                if toks and all(v is not None for v in vals):
                    out[s] = sum(vals)
            return out

    def refresh(self, api, contracts: Dict[str, List[dict]], base: Dict[str, float], day: str) -> str:
        """contracts: {sym: [{"token", "symbol"}]}; base: {sym: the bhavcopy's futures OI} (a units check)."""
        self.at = time.time()
        tokens = {s: [str(c["token"]) for c in cs] for s, cs in contracts.items()}
        try:
            rest = dict(_quotes(api, [t for ts in tokens.values() for t in ts], self.BATCH))
        except RuntimeError as e:
            self.error = str(e)
            return self.error
        totals = {s: sum(rest[t] for t in ts) for s, ts in tokens.items() if ts and all(t in rest for t in ts)}
        ratios = [totals[s] / base[s] for s in totals if base.get(s)]
        if len(ratios) >= 10 and not 0.3 <= statistics.median(ratios) <= 3.0:
            self.units_ok = False
            self.error = f"Angel One OI does not match NSE's units (median ratio {statistics.median(ratios):.2f}): not used"
            logger.warning(self.error)
            return self.error
        pcr = {}
        try:
            by_symbol = {c["symbol"]: s for s, cs in contracts.items() for c in cs}
            r = api.putCallRatio()
            for row in (r or {}).get("data") or []:
                s = by_symbol.get(str(row.get("tradingSymbol")))
                if s and row.get("pcr") is not None:
                    pcr[s] = float(row["pcr"])
        except Exception as e:                       # PCR is an extra
            logger.warning("Angel One put / call ratio: %s", e)
        with self._lock:
            if self.day is not None and day != self.day:     # a new day: yesterday's streamed readings are stale
                self.stream, self.stream_ok = {}, None
            self.tokens, self.rest, self.pcr, self.day, self.units_ok = tokens, rest, pcr, day, True
            both = [self.stream[t] / rest[t] for t in self.stream if rest.get(t)]
            if len(both) >= 10:
                self.stream_ok = 0.5 <= statistics.median(both) <= 2.0
        self.error = "" if self.stream_ok is not False else "the websocket's OI does not match the quote API's: using the quote API only"
        msg = f"live futures OI for {len(totals)} stocks, PCR for {len(pcr)}"
        if ratios:
            msg += f" (median OI vs bhavcopy {statistics.median(ratios):.3f})"
        if self.stream:
            msg += f"; websocket {'in use' if self.stream_ok else 'not checked yet' if self.stream_ok is None else 'not used'}"
        logger.info(msg)
        return msg


class LiveOptions:
    """Today's options OI for the board's leaders: every call and put of each stock's nearest expiry, read from the
    quote API every EVERY_S seconds, against the same expiry's OI in the latest bhavcopy (NSE)."""
    EVERY_S = 180
    BATCH = 50
    MIN_ANSWERED = 0.98             # a stock counts when nearly every contract answered (a dead strike holds no OI)

    def __init__(self):
        self.data: Dict[str, Dict] = {}            # stock -> {"expiry", "ce", "pe", "pcr", "ce_chg", "pe_chg"}
        self.day: Optional[str] = None
        self.at = 0.0
        self.error = ""

    def due(self) -> bool:
        return time.time() - self.at >= self.EVERY_S

    def refresh(self, api, chains: Dict[str, Dict], base: Dict[str, Dict[str, List[float]]], day: str) -> str:
        """chains: {sym: {"expiry", "CE": [token], "PE": [token]}}; base: {sym: {expiry: [ce_oi, pe_oi]}} (bhavcopy)."""
        self.at = time.time()
        kind, owner = {}, {}
        for s, c in chains.items():
            for k in ("CE", "PE"):
                for t in c.get(k) or []:
                    kind[str(t)], owner[str(t)] = k, s
        got = defaultdict(lambda: {"CE": 0.0, "PE": 0.0})
        n = defaultdict(int)
        try:
            for t, oi in _quotes(api, list(kind), self.BATCH):
                s = owner.get(t)
                if s:
                    got[s][kind[t]] += oi
                    n[s] += 1
        except RuntimeError as e:
            self.error = str(e)
            return self.error
        out, ratios = {}, []
        for s, c in chains.items():
            total = len(c.get("CE") or []) + len(c.get("PE") or [])
            if not total or n[s] < self.MIN_ANSWERED * total:
                continue
            ce, pe = got[s]["CE"], got[s]["PE"]
            row = {"expiry": c["expiry"], "ce": ce, "pe": pe, "pcr": round(pe / ce, 2) if ce else None}
            b = (base.get(s) or {}).get(c["expiry"])
            if b and (b[0] or b[1]):
                row["ce_chg"], row["pe_chg"] = ce - b[0], pe - b[1]
                ratios.append((ce + pe) / (b[0] + b[1]))
            out[s] = row
        if len(ratios) >= 3 and not 0.3 <= statistics.median(ratios) <= 3.0:
            self.error = f"Angel One options OI does not match NSE's units (median ratio {statistics.median(ratios):.2f}): not used"
            logger.warning(self.error)
            return self.error
        self.data, self.day, self.error = out, day, ""
        msg = f"live options OI for {len(out)} of {len(chains)} stocks ({sum(1 for r in out.values() if 'ce_chg' in r)} against the bhavcopy)"
        logger.info(msg)
        return msg


BUILDUP_TYPES = {"Long Built Up": "long build-up", "Short Built Up": "short build-up",
                 "Short Covering": "short covering", "Long Unwinding": "long unwinding"}
_CONTRACT = re.compile(r"^(.+?)\d{2}[A-Z]{3}\d{2}FUT$")


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class BuildUpLists:
    """Angel One's own lists (its OI build-up API) of the near-month stock futures with the biggest long build-up,
    short build-up, short covering and long unwinding, every EVERY_S seconds."""
    EVERY_S = 180
    TOP = 10

    def __init__(self):
        self.lists: Dict[str, List[Dict]] = {}
        self.at = 0.0
        self.error = ""

    def due(self) -> bool:
        return time.time() - self.at >= self.EVERY_S

    def refresh(self, api, owner: Callable[[str], Optional[str]]) -> str:
        """owner(contract trading symbol) -> the stock, else None."""
        self.at = time.time()
        lists, errs = {}, []
        for dtype, name in BUILDUP_TYPES.items():
            try:
                r = api.oIBuildup({"expirytype": "NEAR", "datatype": dtype})
            except Exception as e:
                errs.append(f"{name}: {e}")
                continue
            if not r or not r.get("status"):
                errs.append(f"{name}: {(r or {}).get('message') or 'no answer'}")
                continue
            rows = []
            for x in (r.get("data") or [])[:self.TOP]:
                ts = str(x.get("tradingSymbol") or "")
                m = _CONTRACT.match(ts)
                rows.append({"symbol": owner(ts) or (m.group(1) if m else ts), "contract": ts, "ltp": _f(x.get("ltp")),
                             "chg_pct": _f(x.get("percentChange")), "oi": _f(x.get("opnInterest")),
                             "oi_chg": _f(x.get("netChangeOpnInterest"))})
            lists[name] = rows
            time.sleep(1.1)
        if lists:
            self.lists = lists
        self.error = "; ".join(errs)
        msg = f"OI build-up lists: {', '.join(f'{k} {len(v)}' for k, v in lists.items()) or 'none'}"
        if errs:
            logger.warning("Angel One OI build-up: %s", self.error)
        return msg
