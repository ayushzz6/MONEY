"""Small polite HTTP client for the free data feeds (Yahoo, NSE index files, Google News).

One retry on a 429 after a pause, a fallback host when given, and a fixed identifying
User-Agent. Yahoo's edge answers 429 to generic browser and library user agents.
"""
import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional

logger = logging.getLogger("http")

USER_AGENT = "moneyv2/1.0"


class FetchError(Exception):
    pass


def get_bytes(url: str, timeout: float = 20, headers: Optional[dict] = None, retries: int = 1) -> bytes:
    hdrs = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if headers:
        hdrs.update(headers)
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 and attempt < retries:
                time.sleep(5)
                continue
            if e.code >= 500 and attempt < retries:
                time.sleep(1.5)
                continue
            raise FetchError(f"HTTP {e.code} for {url}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(1.5)
                continue
            raise FetchError(f"{type(e).__name__}: {e} for {url}") from e
    raise FetchError(f"failed: {url}: {last}")


def get_json(url: str, timeout: float = 20, hosts: Iterable[str] = (), **kw):
    """GET JSON. `hosts` lists alternative hosts (e.g. query1 for query2) tried in order on failure."""
    urls = [url]
    for alt in hosts:
        head, sep, rest = url.partition("://")
        host, _, path = rest.partition("/")
        urls.append(f"{head}{sep}{alt}/{path}")
    err = None
    for u in urls:
        try:
            return json.loads(get_bytes(u, timeout=timeout, **kw).decode("utf-8"))
        except (FetchError, ValueError) as e:
            err = e
    raise FetchError(str(err))
