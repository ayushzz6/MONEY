"""MCX commodity and NSE currency futures data for the research backtests.

- MCX: Angel One serves a continuous (rolled) 5-minute series per commodity from June 2021,
  on MCX prices and hours (09:00 to 23:30, 23:55 in US summer time).
- Currencies: Angel One only has the live contract, so USDINR / EURINR / GBPINR / JPYINR use
  Yahoo's spot series (hourly from Jan 2024, daily for 10 years); NSE currency futures track spot
  plus a small forward premium, so intraday moves are the same.

Contract specs and margins are Angel One's, read on 2026-10-24 (margin API, per unit, scaled to a
lot), expressed as a share of notional so they can be applied to older prices.
"""
import datetime as dt
import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.config.settings import DATA_DIR
from core.utilities.http import get_json

DIR = DATA_DIR / "research" / "futures"


@dataclass(frozen=True)
class Spec:
    name: str            # the contract traded with a Rs 1 lakh account
    series: str          # the price series it trades on
    exchange: str
    multiplier: float    # rupees per 1.0 move in the quoted price, per lot
    tick: float
    margin_pct: float    # initial margin as a share of notional (Angel One, 2026-10-24)
    session: tuple       # (open HH:MM, last bar close HH:MM) IST
    flat_by: str
    price_div: float = 1.0   # quoted series / this = the traded contract's price unit
    angel_token: str = ""
    yahoo: str = ""


SPECS: Dict[str, Spec] = {
    # MCX: 1 lot of each fits a Rs 1 lakh account
    "CRUDEOILM": Spec("CRUDEOILM", "CRUDEOIL", "MCX", 10, 1.0, 0.3127, ("09:00", "23:55"), "23:00", angel_token="crude"),
    "NATGASMINI": Spec("NATGASMINI", "NATURALGAS", "MCX", 250, 0.10, 0.3661, ("09:00", "23:55"), "23:00", angel_token="natgas"),
    "SILVERMIC": Spec("SILVERMIC", "SILVER", "MCX", 1, 1.0, 0.1368, ("09:00", "23:55"), "23:00", angel_token="silver"),
    "GOLDPETAL": Spec("GOLDPETAL", "GOLD", "MCX", 1, 1.0, 0.0934, ("09:00", "23:55"), "23:00", price_div=10.0,
                      angel_token="gold"),
    # NSE currency futures
    "USDINR": Spec("USDINR", "USDINR", "CDS", 1000, 0.0025, 0.0240, ("09:00", "17:00"), "16:45", yahoo="USDINR=X"),
    "EURINR": Spec("EURINR", "EURINR", "CDS", 1000, 0.0025, 0.0253, ("09:00", "17:00"), "16:45", yahoo="EURINR=X"),
    "GBPINR": Spec("GBPINR", "GBPINR", "CDS", 1000, 0.0025, 0.0287, ("09:00", "17:00"), "16:45", yahoo="GBPINR=X"),
    "JPYINR": Spec("JPYINR", "JPYINR", "CDS", 1000, 0.0025, 0.0389, ("09:00", "17:00"), "16:45", price_div=0.01,
                   yahoo="JPYINR=X"),
}


def mcx_tokens() -> Dict[str, str]:
    """Near-month MCX futures tokens (the series is continuous whichever contract token is asked)."""
    from market.instruments.universe import SCRIP_MASTER_URL
    from core.utilities.http import get_bytes
    path = DIR / "mcx_tokens.json"
    if path.exists() and time.time() - path.stat().st_mtime < 86400 * 3:
        return json.loads(path.read_text())
    raw = json.loads(get_bytes(SCRIP_MASTER_URL, timeout=90).decode())
    best = {}
    for x in raw:
        if x.get("exch_seg") == "MCX" and x.get("instrumenttype") == "FUTCOM" and x.get("name") in ("CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"):
            e = dt.datetime.strptime(x["expiry"], "%d%b%Y")
            if x["name"] not in best or e < best[x["name"]][0]:
                best[x["name"]] = (e, x["token"])
    out = {k: v[1] for k, v in best.items()}
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def download_mcx(start=dt.date(2021, 6, 1), end=None, log=print):
    import logging
    import logzero
    logzero.loglevel(logging.CRITICAL)
    from brokers.angelone.auth import AngelOneAuth
    from core.config.settings import Settings
    end = end or dt.date.today()
    toks = mcx_tokens()
    a = AngelOneAuth(*Settings().angel_credentials())
    a.login()
    api = a.smart_api
    DIR.mkdir(parents=True, exist_ok=True)
    for series, tok in toks.items():
        out = DIR / f"{series}_5m.npz"
        if out.exists():
            continue
        seen = {}
        d = start
        while d <= end:
            e = min(end, d + dt.timedelta(days=60))
            for attempt in range(5):
                time.sleep(0.5)
                try:
                    r = api.getCandleData({"exchange": "MCX", "symboltoken": tok, "interval": "FIVE_MINUTE",
                                           "fromdate": f"{d} 09:00", "todate": f"{e} 23:59"})
                except Exception as ex:
                    r = {"status": False, "message": str(ex)}
                if r and r.get("status"):
                    for ts, o, h, l, c, v in r.get("data") or []:
                        seen[int(dt.datetime.fromisoformat(ts).timestamp())] = (o, h, l, c, v)
                    break
                time.sleep(2 + 2 * attempt)
            d = e + dt.timedelta(days=1)
        t = np.array(sorted(seen), np.int64)
        a_ = np.array([seen[x] for x in t], float)
        np.savez_compressed(out, t=t, o=a_[:, 0], h=a_[:, 1], l=a_[:, 2], c=a_[:, 3], v=a_[:, 4])
        log(f"  {series}: {len(t):,} five-minute bars")


def download_yahoo(symbol: str, rng: str, interval: str) -> Optional[dict]:
    ys = urllib.parse.quote(symbol, safe="")
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ys}?range={rng}&interval={interval}"
    d = get_json(url, hosts=("query1.finance.yahoo.com",))
    r = ((d or {}).get("chart") or {}).get("result") or []
    return r[0] if r else None


def fetch_currency(log=print):
    DIR.mkdir(parents=True, exist_ok=True)
    for name, sp in SPECS.items():
        if sp.exchange != "CDS":
            continue
        for rng, iv, tag in (("730d", "1h", "1h"), ("10y", "1d", "1d")):
            out = DIR / f"{name}_{tag}.npz"
            if out.exists():
                continue
            r = download_yahoo(sp.yahoo, rng, iv)
            q = r["indicators"]["quote"][0]
            t = np.array(r["timestamp"], np.int64)
            arr = np.array([[q[k][i] if q[k][i] is not None else np.nan for k in ("open", "high", "low", "close")]
                            for i in range(len(t))])
            np.savez_compressed(out, t=t, o=arr[:, 0], h=arr[:, 1], l=arr[:, 2], c=arr[:, 3], v=np.zeros(len(t)))
            log(f"  {name} {tag}: {len(t):,} bars")


# ---------------------------------------------------------------- session matrices
@dataclass
class Sessions:
    name: str
    dates: np.ndarray
    o: np.ndarray       # (days, slots)
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    bar_min: int
    open_min: int       # session start, minutes after midnight IST

    @property
    def n(self):
        return len(self.dates)

    def slot_of_close(self, hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return (h * 60 + m - self.open_min) // self.bar_min - 1

    def daily(self):
        c_last = np.array([row[~np.isnan(row)][-1] if (~np.isnan(row)).any() else np.nan for row in self.c])
        return self.o[:, 0], np.nanmax(self.h, 1), np.nanmin(self.l, 1), c_last


def load_sessions(path, name, bar_min, open_hhmm, close_hhmm, min_bars) -> Sessions:
    z = np.load(path)
    t = z["t"] + 19800
    oh, om = map(int, open_hhmm.split(":"))
    ch, cm = map(int, close_hhmm.split(":"))
    open_min, close_min = oh * 60 + om, ch * 60 + cm
    slots = (close_min - open_min) // bar_min
    day = t // 86400
    mins = (t % 86400) // 60
    slot = (mins - open_min) // bar_min
    ok = (slot >= 0) & (slot < slots) & ~np.isnan(z["c"])
    day, slot = day[ok], slot[ok]
    vals = {k: z[k][ok] for k in ("o", "h", "l", "c", "v")}
    udays, inv = np.unique(day, return_inverse=True)
    M = {k: np.full((len(udays), slots), np.nan) for k in ("o", "h", "l", "c", "v")}
    for k in M:
        M[k][inv, slot] = vals[k]
    # forward-fill gaps inside the session (not after its last bar)
    for i in range(len(udays)):
        row = M["c"][i]
        last = np.flatnonzero(~np.isnan(row))
        if len(last) == 0:
            continue
        for j in range(1, last[-1] + 1):
            if np.isnan(row[j]) and not np.isnan(row[j - 1]):
                for k in ("o", "h", "l", "c"):
                    M[k][i, j] = M["c"][i, j - 1]
                M["v"][i, j] = 0
    first_ok = ~np.isnan(M["c"][:, 0])
    count = (~np.isnan(M["c"])).sum(1)
    keep = first_ok & (count >= min_bars) & (np.array([dt.date(1970, 1, 1) + dt.timedelta(days=int(d)) for d in udays])
                                             .astype("datetime64[D]").astype(object) != None)  # noqa: E711
    dates = (udays * 86400).astype("datetime64[s]").astype("datetime64[D]")
    wd = np.array([d.astype(object).weekday() for d in dates])
    keep &= wd < 5
    return Sessions(name, dates[keep], M["o"][keep], M["h"][keep], M["l"][keep], M["c"][keep], M["v"][keep],
                    bar_min, open_min)


def load(spec_name: str, bar="intraday") -> Sessions:
    sp = SPECS[spec_name]
    if sp.exchange == "MCX":
        s = load_sessions(DIR / f"{sp.series}_5m.npz", spec_name, 5, "09:00", "23:55", min_bars=120)
    else:
        s = load_sessions(DIR / f"{spec_name}_1h.npz", spec_name, 60, "09:00", "17:00", min_bars=7)
    if sp.price_div != 1.0:
        for k in ("o", "h", "l", "c"):
            setattr(s, k, getattr(s, k) / sp.price_div)
    return s


def load_daily(spec_name: str):
    """(dates, o, h, l, c) daily: MCX from the 5-minute sessions, currencies from Yahoo daily."""
    sp = SPECS[spec_name]
    if sp.exchange == "MCX":
        s = load(spec_name)
        o, h, l, c = s.daily()
        return s.dates, o, h, l, c
    z = np.load(DIR / f"{spec_name}_1d.npz")
    dates = ((z["t"] + 19800) // 86400 * 86400).astype("datetime64[s]").astype("datetime64[D]")
    keep = ~np.isnan(z["c"])
    div = sp.price_div
    return dates[keep], z["o"][keep] / div, z["h"][keep] / div, z["l"][keep] / div, z["c"][keep] / div
