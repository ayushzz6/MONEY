"""Big-money tracker (trader-pelosi, for NSE stocks): bulk and block deals as NSE posts them.

trader-pelosi reads the trades US Congress members must disclose. India has no such filings; the
closest public record of who traded big, by name, is NSE's daily deal reports:
  - bulk deals: one client trading more than 0.5% of a company's shares in a day, any window;
  - block deals: a single trade of Rs 10 crore or more in the exchange's block-deal windows.
NSE posts them after the close. Each new deal is reported once. Kept in data/state/deals.json
for 60 days.
"""
import datetime as dt
import hashlib
import json
import logging
import threading
import time
from typing import Dict, List, Optional

from core.config.settings import STATE_DIR
from core.utilities.http import FetchError, get_json

logger = logging.getLogger("Deals")

SNAPSHOT_URL = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
HIST_URL = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?optionType={kind}&from={f}&to={t}"
NSE_HEADERS = {"Accept": "application/json", "Referer": "https://www.nseindia.com/"}
PATH = STATE_DIR / "deals.json"
KEEP_DAYS = 60
CR = 1e7


def _num(x) -> Optional[float]:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _day(s: str) -> Optional[str]:
    for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
        try:
            return dt.datetime.strptime(str(s).title(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


class DealsTracker:
    def __init__(self, settings, universe):
        self.settings = settings
        self.universe = universe
        self._lock = threading.Lock()
        self.deals: Dict[str, dict] = {}
        self.as_of: Optional[str] = None
        self.error: Optional[str] = None
        self.checked_at = 0.0
        self._history_days: set = set()          # sessions whose full deal list has been fetched
        self._load()

    def _load(self):
        try:
            doc = json.loads(PATH.read_text())
            self.deals = {d["id"]: d for d in doc.get("deals", [])}
            self.as_of = doc.get("as_of")
            self._history_days = set(doc.get("history_days") or [])
        except (OSError, ValueError, KeyError):
            self.deals = {}

    def _save(self):
        cutoff = (dt.date.today() - dt.timedelta(days=KEEP_DAYS)).isoformat()
        with self._lock:
            self.deals = {k: v for k, v in self.deals.items() if v["date"] >= cutoff}
            rows = sorted(self.deals.values(), key=lambda d: (d["date"], d["value_cr"] or 0), reverse=True)
            self._history_days = {d for d in self._history_days if d >= cutoff}
        PATH.write_text(json.dumps({"as_of": self.as_of, "history_days": sorted(self._history_days), "deals": rows}))

    @staticmethod
    def _make(kind, date, symbol, name, client, side, qty, price) -> Optional[dict]:
        if not date or not symbol or qty is None:
            return None
        key = "|".join(str(x) for x in (kind, date, symbol, client, side, qty, price))
        value = round(qty * price / CR, 2) if price else None
        return {"id": hashlib.sha1(key.encode()).hexdigest()[:12], "kind": kind, "date": date, "symbol": symbol,
                "name": name or symbol, "client": (client or "").strip(), "side": (side or "").upper(),
                "qty": int(qty), "price": price, "value_cr": value}

    def _snapshot(self) -> List[dict]:
        doc = get_json(SNAPSHOT_URL, headers=NSE_HEADERS, timeout=20) or {}
        self.as_of = _day(doc.get("as_on_date")) or self.as_of
        out = []
        for kind, key in (("bulk", "BULK_DEALS_DATA"), ("block", "BLOCK_DEALS_DATA")):
            for r in doc.get(key) or []:
                d = self._make(kind, _day(r.get("date")), str(r.get("symbol") or "").upper(), r.get("name"),
                               r.get("clientName"), r.get("buySell"), _num(r.get("qty")), _num(r.get("watp")))
                if d:
                    out.append(d)
        return out

    def _history(self, days: int = 30) -> List[dict]:
        """NSE's history call answers with the first session of a date range only, so ask one session at a
        time, for weekdays not fetched yet (today's list comes from the snapshot)."""
        today = dt.date.today()
        out = []
        for back in range(days, 0, -1):
            day = today - dt.timedelta(days=back)
            iso = day.isoformat()
            if day.weekday() >= 5 or iso in self._history_days:
                continue
            got = 0
            for kind, opt in (("bulk", "bulk_deals"), ("block", "block_deals")):
                q = day.strftime("%d-%m-%Y")
                doc = get_json(HIST_URL.format(kind=opt, f=q, t=q), headers=NSE_HEADERS, timeout=30) or {}
                for r in doc.get("data") or []:
                    d = self._make(kind, _day(r.get("BD_DT_DATE")), str(r.get("BD_SYMBOL") or "").upper(),
                                   r.get("BD_SCRIP_NAME"), r.get("BD_CLIENT_NAME"), r.get("BD_BUY_SELL"),
                                   _num(r.get("BD_QTY_TRD")), _num(r.get("BD_TP_WATP")))
                    if d and d["date"] == iso:
                        out.append(d)
                        got += 1
                time.sleep(0.4)
            if back > 1 or got:          # yesterday's list may not be posted yet: try again next time
                self._history_days.add(iso)
        return out

    def refresh(self) -> List[dict]:
        """Fetch the latest deals; returns the ones not seen before (oldest first)."""
        rows: List[dict] = []
        first_run = not self.deals
        try:
            rows += self._history()
            rows += self._snapshot()
            self.error = None
        except FetchError as e:
            self.error = f"NSE deal reports unavailable: {e}"
            logger.warning(self.error)
        self.checked_at = time.time()
        new = []
        with self._lock:
            for d in rows:
                if d["id"] in self.deals:
                    continue
                d["seen_at"] = time.time()
                d["nifty500"] = d["symbol"] in self.universe.instruments
                self.deals[d["id"]] = d
                new.append(d)
        if rows:
            self._save()
        # the first fill of an empty store is history, not news
        return [] if first_run else sorted(new, key=lambda d: d["date"])

    def latest(self, limit: int = 300, symbols: Optional[List[str]] = None, nifty500_only: bool = False) -> List[dict]:
        with self._lock:
            rows = list(self.deals.values())
        if symbols is not None:
            want = set(symbols)
            rows = [d for d in rows if d["symbol"] in want]
        if nifty500_only:
            rows = [d for d in rows if d.get("nifty500")]
        rows.sort(key=lambda d: (d["date"], d["value_cr"] or 0), reverse=True)
        return rows[:limit]

    def describe(self, d: dict) -> str:
        qty = d["qty"]
        q = f"{qty / 1e5:,.1f} lakh" if qty >= 1e5 else f"{qty:,}"
        val = f" (₹{d['value_cr']:,.1f} cr)" if d.get("value_cr") else ""
        px = f" at ₹{d['price']:,.2f}" if d.get("price") else ""
        return f"{d['kind'].title()} deal: {d['client'] or 'a client'} {d['side'].lower() or 'traded'} {q} shares of {d['symbol']}{px}{val}"
