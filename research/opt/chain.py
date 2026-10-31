"""Real NIFTY / BANKNIFTY option chains, one file per session, from NSE's F&O bhavcopy.

Every index option row is kept when it expires within 100 days and its strike is within 20% of the
index: expiry, strike, call/put, open, high, low, close, settlement price, quantity traded, open
interest and its change (both in units, not lots), and the lot size. The old bhavcopy format (to
July 2024) has no lot size column; it is recovered from turnover / ((strike + price) x contracts),
which is exact to the unit.

Files: data/research/opt/chain/YYYY-MM-DD.npz
"""
import csv
import datetime as dt
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.config.settings import DATA_DIR
from research import data
from research.lead import nse_daily as N

logger = logging.getLogger(__name__)
OUT = DATA_DIR / "research" / "opt" / "chain"
UNDERLYINGS = ("NIFTY", "BANKNIFTY")
FIELDS = ("und", "expiry", "strike", "cp", "open", "high", "low", "close", "settle", "qty", "oi", "chg_oi", "lot")


def _f(x):
    try:
        return float(str(x).strip().replace(",", ""))
    except ValueError:
        return np.nan


def parse(text, is_udiff, day, spot):
    rows = []
    if is_udiff:
        for r in csv.DictReader(io.StringIO(text)):
            if r.get("FinInstrmTp") != "IDO" or r.get("TckrSymb") not in UNDERLYINGS:
                continue
            rows.append((UNDERLYINGS.index(r["TckrSymb"]), r["XpryDt"], _f(r["StrkPric"]), 1 if r["OptnTp"] == "CE" else -1,
                         _f(r["OpnPric"]), _f(r["HghPric"]), _f(r["LwPric"]), _f(r["ClsPric"]), _f(r["SttlmPric"]),
                         _f(r["TtlTradgVol"]), _f(r["OpnIntrst"]), _f(r["ChngInOpnIntrst"]), _f(r["NewBrdLotQty"])))
    else:
        tmp = []
        for r in csv.DictReader(io.StringIO(text)):
            if (r.get("INSTRUMENT") or "").strip() != "OPTIDX" or r["SYMBOL"].strip() not in UNDERLYINGS:
                continue
            exp = dt.datetime.strptime(r["EXPIRY_DT"].strip(), "%d-%b-%Y").date().isoformat()
            k, c, con, val = _f(r["STRIKE_PR"]), _f(r["CLOSE"]), _f(r["CONTRACTS"]), _f(r["VAL_INLAKH"])
            tmp.append((UNDERLYINGS.index(r["SYMBOL"].strip()), exp, k, 1 if r["OPTION_TYP"].strip() == "CE" else -1,
                        _f(r["OPEN"]), _f(r["HIGH"]), _f(r["LOW"]), c, _f(r["SETTLE_PR"]), con, _f(r["OPEN_INT"]),
                        _f(r["CHG_IN_OI"]), val))
        lots = {}
        for u in (0, 1):                         # lot = turnover / ((strike + price) x contracts), median of liquid rows
            est = [t[12] * 1e5 / ((t[2] + t[7]) * t[9]) for t in tmp if t[0] == u and t[9] >= 100 and t[7] > 0]
            lots[u] = float(np.round(np.median(est))) if est else np.nan
        for t in tmp:
            lot = lots[t[0]]
            rows.append(t[:9] + (t[9] * lot, t[10], t[11], lot))
    if not rows:
        return None
    a = {k: np.array([r[i] for r in rows]) for i, k in enumerate(FIELDS)}
    a["expiry"] = a["expiry"].astype("datetime64[D]")
    keep = (a["expiry"] >= np.datetime64(day)) & (a["expiry"] <= np.datetime64(day) + np.timedelta64(100, "D"))
    for u in (0, 1):
        s = spot.get(UNDERLYINGS[u])
        if s:
            m = a["und"] == u
            keep &= ~m | (np.abs(a["strike"] / s - 1) <= 0.20)
    return {k: v[keep] for k, v in a.items()}


def one_day(args):
    day, spot = args
    path = OUT / f"{day.isoformat()}.npz"
    if path.exists():
        return day, "cached"
    text, is_udiff = N.fo_text(day)
    if text is None:
        return day, "missing"
    a = parse(text, is_udiff, day, spot)
    if a is None:
        return day, "empty"
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: v.astype(np.int8 if k in ("und", "cp") else v.dtype if k == "expiry" else np.float64)
                     for k, v in a.items()})
    tmp.replace(path)
    return day, "ok"


def download(log=print, workers=3):
    OUT.mkdir(parents=True, exist_ok=True)
    n, b = data.load("NIFTY"), data.load("BANKNIFTY")
    bclose = dict(zip(b.dates, b.c[:, -1]))
    jobs = [(d.astype(object), {"NIFTY": float(c), "BANKNIFTY": float(bclose.get(d, np.nan))})
            for d, c in zip(n.dates, n.c[:, -1])]
    t0 = time.time()
    done = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (day, st) in enumerate(ex.map(one_day, jobs), 1):
            done[st] = done.get(st, 0) + 1
            if i % 100 == 0 or i == len(jobs):
                log(f"  option chains: {i}/{len(jobs)} days {done} ({time.time() - t0:.0f}s)")
    return done


def load_all():
    """Every saved session concatenated, with a `day` column (datetime64[D])."""
    parts = []
    for p in sorted(OUT.glob("*.npz")):
        if ".tmp" in p.name:
            continue
        z = np.load(p)
        a = {k: z[k] for k in FIELDS}
        a["day"] = np.full(len(a["strike"]), np.datetime64(p.stem), "datetime64[D]")
        parts.append(a)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    print(download(log=lambda m: print(m, flush=True)), flush=True)
