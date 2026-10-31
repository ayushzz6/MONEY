"""News monitor (trader-news, for NSE stocks): Google News headlines per company, tagged, deduped.

Headlines are leads, not facts: the dashboard links every one to its source. Tags come
from keywords in the headline. A tag listed in news.block_tags (default: legal_risk)
blocks new entries in that stock for news.lookback_hours.
"""
import hashlib
import html
import logging
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List

from core.utilities.http import FetchError, get_bytes

logger = logging.getLogger("News")

TAGS = [
    ("earnings", r"\b(q[1-4]|quarter(ly)?|results?|earnings|profit|net income|revenue|ebitda|margin)\b"),
    ("analyst", r"\b(target price|upgrade[sd]?|downgrade[sd]?|buy rating|sell rating|outperform|underperform|"
                r"overweight|underweight|brokerage|initiates coverage)\b"),
    ("deal", r"\b(acquir\w*|acquisition|merger|merge[sd]?|stake|takeover|buyout|joint venture|jv)\b"),
    ("order_win", r"\b(order (win|worth|from)|bags? (an? )?order|wins? (an? )?(order|contract)|contract worth|"
                  r"orders? worth)\b"),
    ("capital", r"\b(qip|rights issue|ofs|offer for sale|fund ?rais\w*|block deal|bulk deal|ipo|ncd|preferential)\b"),
    ("corporate_action", r"\b(dividend|bonus|split|buyback|record date)\b"),
    ("people", r"\b(ceo|cfo|md|managing director|chairman|resign\w*|appoint\w*|steps down)\b"),
    ("regulatory", r"\b(sebi|rbi|cci|nclt|nclat|government|ministry|license|approval|usfda|fda|tariff|ban)\b"),
    ("legal_risk", r"\b(fraud|raid\w*|probe|investigation|enforcement directorate|income tax search|default\w*|"
                   r"insolvency|bankrupt\w*|penalty|show cause|forensic|whistleblower|pledge)\b"),
    ("macro", r"\b(nifty|sensex|fii|dii|inflation|repo rate|gdp|monsoon|crude)\b"),
]
TAG_RE = [(t, re.compile(p, re.I)) for t, p in TAGS]
# quote pages, option chains and price "predictions" are not news
NOISE = re.compile(r"(stock price, news, quote|option chain|share price (prediction|target|today|live)|"
                   r"price prediction|stock price today|live share price|technical analysis)", re.I)


def tag(title: str) -> List[str]:
    return [t for t, rx in TAG_RE if rx.search(title)]


def clean_name(name: str) -> str:
    n = re.sub(r"\b(Ltd|Limited|Ltd\.)\.?$", "", name.strip(), flags=re.I).strip(" .,")
    return n


def fetch_company(symbol: str, company: str, hours: float = 24) -> List[dict]:
    q = f'"{clean_name(company)}" (share OR stock OR shares)'
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": q + f" when:{max(1, int(hours // 24) or 1)}d", "hl": "en-IN", "gl": "IN",
                                   "ceid": "IN:en"}))
    body = get_bytes(url, timeout=15)
    root = ET.fromstring(body)
    out = []
    cutoff = time.time() - hours * 3600
    for item in root.iter("item"):
        title = html.unescape(item.findtext("title") or "").strip()
        link = item.findtext("link") or ""
        src = item.find("source")
        source = src.text if src is not None else ""
        try:
            ts = parsedate_to_datetime(item.findtext("pubDate") or "").timestamp()
        except (TypeError, ValueError):
            continue
        if ts < cutoff or not title or NOISE.search(title):
            continue
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        out.append({"id": hashlib.sha1(link.encode()).hexdigest()[:12], "symbol": symbol, "title": title,
                    "link": link, "source": source, "ts": ts, "tags": tag(title)})
    return out


class NewsMonitor:
    def __init__(self, settings, universe):
        self.settings = settings
        self.universe = universe
        self._lock = threading.Lock()
        self.items: Dict[str, dict] = {}
        self.checked_at: Dict[str, float] = {}

    def refresh(self, symbols: List[str]):
        cfg = self.settings.get("news", {}) or {}
        if not cfg.get("enabled", True):
            return 0
        hours = float(cfg.get("lookback_hours", 24))
        every = float(cfg.get("poll_minutes", 20)) * 60
        new = 0
        for sym in symbols:
            if time.time() - self.checked_at.get(sym, 0) < every:
                continue
            try:
                rows = fetch_company(sym, self.universe.name(sym), hours)
            except (FetchError, ET.ParseError) as e:
                logger.debug("news failed for %s: %s", sym, e)
                continue
            finally:
                self.checked_at[sym] = time.time()
            with self._lock:
                for r in rows:
                    if r["id"] not in self.items:
                        self.items[r["id"]] = r
                        new += 1
            time.sleep(0.8)
        with self._lock:
            cutoff = time.time() - 48 * 3600
            self.items = {k: v for k, v in self.items.items() if v["ts"] >= cutoff}
        return new

    def for_symbol(self, symbol: str, hours: float = 24) -> List[dict]:
        cutoff = time.time() - hours * 3600
        with self._lock:
            rows = [r for r in self.items.values() if r["symbol"] == symbol and r["ts"] >= cutoff]
        return sorted(rows, key=lambda r: -r["ts"])

    def latest(self, limit: int = 100) -> List[dict]:
        with self._lock:
            rows = sorted(self.items.values(), key=lambda r: -r["ts"])
        return rows[:limit]

    def block_reason(self, symbol: str):
        cfg = self.settings.get("news", {}) or {}
        block = set(cfg.get("block_tags") or [])
        if not block:
            return None
        for r in self.for_symbol(symbol, float(cfg.get("lookback_hours", 24))):
            hit = block.intersection(r["tags"])
            if hit:
                return f"news ({', '.join(sorted(hit))}): {r['title'][:90]}"
        return None
