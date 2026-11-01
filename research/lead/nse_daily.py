"""Daily positioning data from NSE's public archives (no login, no browser session).

For every session in the 5-minute history:
- delivery: sec_bhavdata_full (per stock: volume, delivered quantity, delivery %)
- F&O bhavcopy: per stock, futures open interest (all expiries) and its change, near-month future close;
  options call / put open interest and their change; the same for the NIFTY index
- participant OI: index futures / options long and short contracts of FII, DII, Pro and Client

Each day is reduced to one small JSON in data/research/lead/daily/ (resumable). The raw files are
not kept. NSE switched the F&O bhavcopy to the UDiFF format in July 2024; both are read.
"""
import csv
import datetime as dt
import io
import json
import logging
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from core.config.settings import DATA_DIR

logger = logging.getLogger(__name__)

OUT = DATA_DIR / "research" / "lead" / "daily"
BASE = "https://nsearchives.nseindia.com"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "*/*"}
UDIFF_FROM = dt.date(2024, 7, 8)
_local = threading.local()
_gate = threading.Lock()
_last = [0.0]
MIN_GAP_S = 0.25            # at most ~4 requests a second across all threads


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers.update(UA)
    return s


def _get(url, tries=4):
    """The file's bytes, None when NSE has no file for that day (404). Raises after repeated refusals."""
    for k in range(tries):
        with _gate:
            wait = _last[0] + MIN_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = _session().get(url, timeout=30)
        except requests.RequestException as e:
            logger.warning("%s: %s", url, e)
            time.sleep(2 * (k + 1))
            continue
        if r.status_code == 200 and r.content:
            return r.content
        if r.status_code == 404:
            return None
        logger.warning("%s: HTTP %s", url, r.status_code)
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"NSE refused {url} {tries} times")


def _num(x):
    try:
        return float(str(x).strip().replace(",", ""))
    except ValueError:
        return None


def delivery(day: dt.date):
    raw = _get(f"{BASE}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv")
    if raw is None:
        return None
    out = {}
    text = raw.decode("utf-8", "replace").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("SERIES") != "EQ":
            continue
        out[r["SYMBOL"]] = [_num(r["CLOSE_PRICE"]), _num(r["PREV_CLOSE"]), _num(r["TTL_TRD_QNTY"]),
                            _num(r["DELIV_QTY"]), _num(r["DELIV_PER"])]
    return out


def fo_text(day: dt.date):
    """The day's F&O bhavcopy as text and whether it is the UDiFF format; (None, None) if NSE has none."""
    mon = day.strftime("%b").upper()
    udiff = f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    old = f"{BASE}/content/historical/DERIVATIVES/{day:%Y}/{mon}/fo{day:%d}{mon}{day:%Y}bhav.csv.zip"
    urls = [(udiff, True), (old, False)] if day >= UDIFF_FROM else [(old, False), (udiff, True)]
    for u, is_udiff in urls:
        raw = _get(u)
        if raw:
            z = zipfile.ZipFile(io.BytesIO(raw))
            return z.read(z.namelist()[0]).decode("utf-8", "replace").replace("\x00", ""), is_udiff
    return None, None


def _fo_rows(day: dt.date):
    """(kind, symbol, expiry, option_type, close, oi, chg_oi) for every F&O contract that day."""
    mon = day.strftime("%b").upper()
    udiff = f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    old = f"{BASE}/content/historical/DERIVATIVES/{day:%Y}/{mon}/fo{day:%d}{mon}{day:%Y}bhav.csv.zip"
    urls = [(udiff, True), (old, False)] if day >= UDIFF_FROM else [(old, False), (udiff, True)]
    raw, is_udiff = None, False
    for u, is_udiff in urls:
        raw = _get(u)
        if raw:
            break
    if raw is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("utf-8", "replace").replace("\x00", "")
    rows = []
    if is_udiff:
        kinds = {"STF": "FUTSTK", "IDF": "FUTIDX", "STO": "OPTSTK", "IDO": "OPTIDX"}
        for r in csv.DictReader(io.StringIO(text)):
            k = kinds.get(r.get("FinInstrmTp", ""))
            if k:
                rows.append((k, r["TckrSymb"], r["XpryDt"], r.get("OptnTp") or "XX", _num(r["ClsPric"]),
                             _num(r["OpnIntrst"]) or 0.0, _num(r["ChngInOpnIntrst"]) or 0.0))
    else:
        for r in csv.DictReader(io.StringIO(text)):
            k = (r.get("INSTRUMENT") or "").strip()
            if k in ("FUTSTK", "FUTIDX", "OPTSTK", "OPTIDX"):
                exp = dt.datetime.strptime(r["EXPIRY_DT"].strip(), "%d-%b-%Y").date().isoformat()
                rows.append((k, r["SYMBOL"].strip(), exp, r["OPTION_TYP"].strip(), _num(r["CLOSE"]),
                             _num(r["OPEN_INT"]) or 0.0, _num(r["CHG_IN_OI"]) or 0.0))
    return rows


def fo(day: dt.date):
    rows = _fo_rows(day)
    if rows is None:
        return None
    fut = defaultdict(lambda: {"oi": 0.0, "chg": 0.0, "near": None})
    opt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])          # ce_oi, pe_oi, ce_chg, pe_chg
    for kind, sym, exp, ot, close, oi, chg in rows:
        if kind in ("FUTSTK", "FUTIDX"):
            f = fut[sym]
            f["oi"] += oi
            f["chg"] += chg
            if f["near"] is None or exp < f["near"][0]:
                f["near"] = (exp, close)
        else:
            o = opt[sym]
            if ot == "CE":
                o[0] += oi
                o[2] += chg
            elif ot == "PE":
                o[1] += oi
                o[3] += chg
    syms = set(fut) | set(opt)
    return {s: [fut[s]["oi"] if s in fut else None, fut[s]["chg"] if s in fut else None,
                fut[s]["near"][1] if s in fut and fut[s]["near"] else None] + (opt[s] if s in opt else [None] * 4)
            for s in syms}


def participants(day: dt.date):
    raw = _get(f"{BASE}/content/nsccl/fao_participant_oi_{day:%d%m%Y}.csv")
    if raw is None:
        return None
    lines = raw.decode("utf-8", "replace").replace("\x00", "").splitlines()
    head = None
    out = {}
    for row in csv.reader(lines):
        row = [c.strip() for c in row]
        if row and row[0] == "Client Type":
            head = row
            continue
        if head and row and row[0] in ("Client", "DII", "FII", "Pro"):
            out[row[0]] = [_num(x) for x in row[1:13]]
    return out or None


def one_day(day: dt.date):
    path = OUT / f"{day.isoformat()}.json"
    if path.exists():
        return day, "cached"
    doc = {"date": day.isoformat()}
    for name, fn in (("delivery", delivery), ("fo", fo), ("participants", participants)):
        try:
            doc[name] = fn(day)
        except RuntimeError:
            raise                                   # NSE refusing us: stop, do not hammer it
        except Exception as e:                      # one damaged file: record the gap, keep going
            logger.warning("%s %s: %s", day, name, e)
            doc[name] = None
    if not any(doc[k] for k in ("delivery", "fo", "participants")):
        return day, "missing"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    tmp.replace(path)
    return day, "ok"


def download(days, workers=3, log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(days)
    t0 = time.time()
    done = {"ok": 0, "cached": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (day, status) in enumerate(ex.map(one_day, days), 1):
            done[status] += 1
            if i % 50 == 0 or i == len(days):
                log(f"  {i}/{len(days)} days ({done}) {time.time() - t0:.0f}s")
    return done


if __name__ == "__main__":
    import sys
    from research import data
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    days = [d.astype(object) for d in data.load("NIFTY").dates]
    print(download(days, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 3), flush=True)
