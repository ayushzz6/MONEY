"""NSE F&O stock list and yesterday's futures open interest, from NSE's public F&O bhavcopy (cached per day).

The latest published session's file gives every stock with futures, its total futures open interest and the
day's change, so each stock's build-up can be read: price up + OI up = long build-up, price down + OI up =
short build-up, price up + OI down = short covering, price down + OI down = long unwinding. The same file gives
each stock's options call and put OI and their change (put / call ratio, put or call writing).

LiveOI reads today's futures OI (every expiry, summed) from Angel One's quote API during the session.
"""
import datetime as dt
import json
import logging
import statistics
import time
from typing import Dict, List, Optional

from core.config.settings import CACHE_DIR

logger = logging.getLogger("FnO")
INDEXES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
FOLDER = CACHE_DIR / "fno"
VERSION = 2                     # 2: adds options OI


def latest(today: Optional[dt.date] = None, max_back: int = 7) -> Optional[Dict]:
    """{"date", "symbols": [...], "oi": {sym: {"oi", "chg", "chg_pct", "ce_oi", "pe_oi", "ce_chg", "pe_chg", "pcr"}}}
    for the latest session NSE has published."""
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
                rows = nse_daily.fo(day)
            except Exception as e:
                logger.warning("F&O bhavcopy %s: %s", day, e)
                rows = None
            if not rows and old:
                return old
            if rows:
                oi = {}
                for s, r in rows.items():
                    if s in INDEXES or r[0] is None:
                        continue
                    base = (r[0] or 0) - (r[1] or 0)
                    ce, pe = r[3], r[4]
                    oi[s] = {"oi": r[0], "chg": r[1], "chg_pct": round(100 * r[1] / base, 2) if base > 0 else None,
                             "ce_oi": ce, "pe_oi": pe, "ce_chg": r[5], "pe_chg": r[6],
                             "pcr": round(pe / ce, 2) if ce and pe is not None else None}
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


class LiveOI:
    """Today's futures OI per stock (every unexpired contract, summed) and Angel One's put / call ratio.

    Read every EVERY_S seconds while the market is open, 50 contracts a request. The total is compared with the
    bhavcopy's (also every expiry), so a roll from one expiry to the next is not read as positions closing.
    """
    EVERY_S = 180
    BATCH = 50

    def __init__(self):
        self.oi: Dict[str, float] = {}
        self.pcr: Dict[str, float] = {}
        self.day: Optional[str] = None
        self.at = 0.0
        self.error = ""

    def due(self) -> bool:
        return time.time() - self.at >= self.EVERY_S

    def refresh(self, api, contracts: Dict[str, List[dict]], base: Dict[str, float], day: str) -> str:
        """contracts: {sym: [{"token", "symbol"}]}; base: {sym: the bhavcopy's futures OI} (a units check)."""
        self.at = time.time()
        owner = {c["token"]: s for s, cs in contracts.items() for c in cs}
        tokens = list(owner)
        got: Dict[str, float] = {}
        seen: Dict[str, int] = {}
        for i in range(0, len(tokens), self.BATCH):
            r = api.getMarketData("FULL", {"NFO": tokens[i:i + self.BATCH]})
            if not r or not r.get("status"):
                self.error = f"Angel One quote API: {(r or {}).get('message') or 'no answer'}"
                return self.error
            for q in (r.get("data") or {}).get("fetched") or []:
                s = owner.get(str(q.get("symbolToken")))
                if s is not None and q.get("opnInterest") is not None:
                    got[s] = got.get(s, 0.0) + float(q["opnInterest"])
                    seen[s] = seen.get(s, 0) + 1
            time.sleep(0.15)
        # only stocks whose every contract answered: a missing expiry would read as OI falling
        got = {s: v for s, v in got.items() if seen.get(s) == len(contracts.get(s) or [])}
        ratios = [got[s] / base[s] for s in got if base.get(s)]
        if len(ratios) >= 10 and not 0.3 <= statistics.median(ratios) <= 3.0:
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
        self.oi, self.pcr, self.day, self.error = got, pcr, day, ""
        msg = f"live futures OI for {len(got)} stocks, PCR for {len(pcr)}"
        if ratios:
            msg += f" (median OI vs bhavcopy {statistics.median(ratios):.3f})"
        logger.info(msg)
        return msg
