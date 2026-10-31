"""Results analyst (trader-analyst, for NSE stocks): who reports results soon, and what changed.

trader-analyst reads US filings on SEC EDGAR. The NSE equivalents here: the exchange's board-meeting
calendar (which companies announce results, and on which day) and each company's last quarters from
Yahoo Finance (revenue, gross / operating / net margin, net income, diluted EPS), each quarter against
the one before and the same quarter a year earlier, plus the trailing twelve months when four
consecutive quarters are there.

Yahoo keeps about five quarters for NSE companies and sometimes skips one. Numbers are leads to
check against the company's own filing, not audited facts.
"""
import datetime as dt
import json
import logging
import threading
import time
import urllib.parse
from typing import Dict, List, Optional

from core.config.settings import CACHE_DIR
from core.utilities.http import FetchError, get_json

logger = logging.getLogger("Results")

CAL_URL = "https://www.nseindia.com/api/event-calendar?index=equities"
NSE_HEADERS = {"Accept": "application/json", "Referer": "https://www.nseindia.com/"}
TS_URL = ("https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{sym}"
          "?type={types}&period1={p1}&period2={p2}")
FIELDS = {"quarterlyTotalRevenue": "revenue", "quarterlyGrossProfit": "gross_profit",
          "quarterlyOperatingIncome": "operating_income", "quarterlyNetIncome": "net_income",
          "quarterlyDilutedEPS": "eps"}
CR = 1e7                      # rupees in a crore
CAL_TTL = 3 * 3600
Q_TTL = 12 * 3600


def _pct(new, old):
    if new is None or old in (None, 0):
        return None
    return round(100 * (new - old) / abs(old), 1)


def _margin(part, whole):
    if part is None or not whole:
        return None
    return round(100 * part / whole, 1)


class ResultsAnalyst:
    def __init__(self, settings, universe):
        self.settings = settings
        self.universe = universe
        self._lock = threading.Lock()
        self._cal: List[dict] = []
        self._cal_at = 0.0
        self._q: Dict[str, dict] = {}
        self.error: Optional[str] = None
        self.dir = CACHE_DIR / "results"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ calendar
    def calendar(self, refresh: bool = False) -> List[dict]:
        """Upcoming results board meetings on NSE, soonest first."""
        if not refresh and self._cal and time.time() - self._cal_at < CAL_TTL:
            return self._cal
        try:
            rows = get_json(CAL_URL, headers=NSE_HEADERS, timeout=20) or []
        except FetchError as e:
            self.error = f"NSE calendar unavailable: {e}"
            logger.warning(self.error)
            return self._cal
        today = dt.date.today()
        out = []
        for r in rows:
            purpose = str(r.get("purpose") or "")
            if "result" not in purpose.lower():
                continue
            try:
                day = dt.datetime.strptime(str(r.get("date")), "%d-%b-%Y").date()
            except ValueError:
                continue
            if day < today:
                continue
            sym = str(r.get("symbol") or "").upper()
            out.append({"symbol": sym, "company": r.get("company") or sym, "date": day.isoformat(),
                        "purpose": purpose, "note": str(r.get("bm_desc") or "")[:300],
                        "nifty500": sym in self.universe.instruments})
        out.sort(key=lambda x: (x["date"], x["symbol"]))
        with self._lock:
            self._cal, self._cal_at = out, time.time()
        self.error = None
        return out

    def upcoming(self, symbols: List[str], days: int = 3) -> List[dict]:
        """Results of these stocks due within `days` calendar days (today included)."""
        want = set(symbols)
        last = (dt.date.today() + dt.timedelta(days=days)).isoformat()
        return [r for r in self.calendar() if r["symbol"] in want and r["date"] <= last]

    def due_on(self, symbol: str, day: str) -> bool:
        return any(r["symbol"] == symbol and r["date"] == day for r in self.calendar())

    # ------------------------------------------------------------------ quarters
    def quarterly(self, symbol: str, refresh: bool = False) -> dict:
        symbol = symbol.upper().strip()
        hit = self._q.get(symbol)
        if hit and not refresh and time.time() - hit["fetched_at"] < Q_TTL:
            return hit
        path = self.dir / f"{symbol}.json"
        if not refresh and path.exists() and time.time() - path.stat().st_mtime < Q_TTL:
            try:
                doc = json.loads(path.read_text())
                self._q[symbol] = doc
                return doc
            except ValueError:
                pass
        doc = self._fetch(symbol)
        self._q[symbol] = doc
        if doc.get("quarters"):
            path.write_text(json.dumps(doc))
        return doc

    def _fetch(self, symbol: str) -> dict:
        p2 = int(time.time())
        p1 = p2 - 800 * 86400
        ys = urllib.parse.quote(symbol + ".NS", safe="")
        url = TS_URL.format(sym=ys, types=",".join(FIELDS), p1=p1, p2=p2)
        base = {"symbol": symbol, "name": self.universe.name(symbol), "source": "Yahoo Finance",
                "fetched_at": time.time(), "quarters": [], "summary": [], "ttm": None}
        try:
            data = get_json(url, hosts=("query1.finance.yahoo.com",), timeout=20)
        except FetchError as e:
            return dict(base, error=f"Yahoo Finance did not answer: {e}")
        by_end: Dict[str, dict] = {}
        for res in ((data or {}).get("timeseries") or {}).get("result") or []:
            typ = ((res.get("meta") or {}).get("type") or [None])[0]
            key = FIELDS.get(typ)
            if not key:
                continue
            for v in res.get(typ) or []:
                if not v or not v.get("asOfDate"):
                    continue
                raw = (v.get("reportedValue") or {}).get("raw")
                if raw is None:
                    continue
                by_end.setdefault(v["asOfDate"], {"end": v["asOfDate"]})[key] = float(raw)
        quarters = []
        for end in sorted(by_end):
            q = by_end[end]
            rev = q.get("revenue")
            quarters.append({
                "end": end,
                "revenue_cr": round(rev / CR, 1) if rev is not None else None,
                "net_income_cr": round(q["net_income"] / CR, 1) if q.get("net_income") is not None else None,
                "operating_income_cr": round(q["operating_income"] / CR, 1) if q.get("operating_income") is not None else None,
                "eps": round(q["eps"], 2) if q.get("eps") is not None else None,
                "gross_margin": _margin(q.get("gross_profit"), rev),
                "operating_margin": _margin(q.get("operating_income"), rev),
                "net_margin": _margin(q.get("net_income"), rev),
            })
        if not quarters:
            return dict(base, error="Yahoo Finance has no quarterly numbers for this stock")
        for i, q in enumerate(quarters):
            prev = quarters[i - 1] if i else None
            end = dt.date.fromisoformat(q["end"])
            year_ago = next((x for x in quarters if abs((end - dt.date.fromisoformat(x["end"])).days - 365) <= 20), None)
            for k in ("revenue_cr", "net_income_cr", "eps"):
                q[f"{k}_qoq"] = _pct(q[k], prev[k]) if prev else None
                q[f"{k}_yoy"] = _pct(q[k], year_ago[k]) if year_ago else None
            q["year_ago"] = year_ago["end"] if year_ago else None
        return dict(base, quarters=quarters, ttm=self._ttm(quarters), summary=self._summary(quarters))

    @staticmethod
    def _ttm(quarters: List[dict]) -> Optional[dict]:
        last4 = quarters[-4:]
        if len(last4) < 4:
            return None
        ends = [dt.date.fromisoformat(q["end"]) for q in last4]
        if (ends[-1] - ends[0]).days > 290:          # a quarter is missing in between
            return None
        out = {"from": last4[0]["end"], "to": last4[-1]["end"]}
        for k in ("revenue_cr", "net_income_cr", "eps"):
            vals = [q[k] for q in last4]
            out[k] = round(sum(vals), 2) if None not in vals else None
        out["net_margin"] = _margin(out["net_income_cr"], out["revenue_cr"])
        return out

    @staticmethod
    def _summary(quarters: List[dict]) -> List[str]:
        q = quarters[-1]
        month = dt.date.fromisoformat(q["end"]).strftime("%b %Y")
        out = []

        def ch(v, what):
            if v is None:
                return None
            return f"{'+' if v >= 0 else '−'}{abs(v):.1f}% {what}"

        def line(label, val, unit, qoq, yoy):
            if val is None:
                return
            parts = [x for x in (ch(qoq, "on the previous quarter"), ch(yoy, "on a year earlier")) if x]
            out.append(f"{label} {unit}{val:,.{2 if unit == '₹' and label == 'EPS' else 0}f}"
                       f"{' crore' if label != 'EPS' else ''} in the quarter to {month}"
                       + (f": {', '.join(parts)}." if parts else "."))

        line("Revenue", q["revenue_cr"], "₹", q["revenue_cr_qoq"], q["revenue_cr_yoy"])
        line("Net profit", q["net_income_cr"], "₹", q["net_income_cr_qoq"], q["net_income_cr_yoy"])
        line("EPS", q["eps"], "₹", q["eps_qoq"], q["eps_yoy"])
        prev = quarters[-2] if len(quarters) > 1 else None
        if q["operating_margin"] is not None and prev and prev["operating_margin"] is not None:
            d = q["operating_margin"] - prev["operating_margin"]
            out.append(f"Operating margin {q['operating_margin']:.1f}% "
                       f"({'up' if d >= 0 else 'down'} {abs(d):.1f} points on the previous quarter).")
        if q["net_income_cr"] is not None and q["net_income_cr"] < 0:
            out.append("The company made a loss in the latest quarter.")
        return out
