"""XAUUSD (spot gold) research on a Rs 1 lakh account.

Price data: MCX gold, real 5-minute bars, Jun 2021 - Oct 2026 (rupees a gram: XAUUSD converted at USDINR
plus duty, so its intraday moves track XAUUSD through the London and New York sessions, 09:00-23:55 IST);
COMEX gold futures (GC=F) daily for 10 years for positional systems.

Two sizings:
- legal: MCX gold petal (1 gram a lot, margin 9.34% of notional): Rs 1 lakh carries about 75 g (2.4 oz);
- offshore: an XAUUSD CFD at 1:100 leverage, about 26 oz on Rs 1 lakh. Indian residents may not trade
  this under FEMA / RBI rules; it is shown only to size what Rs 20,000 a month would demand.
Selection on data to 2024-12-31, judged after. Gold rose from about Rs 7,100 to Rs 14,200 a gram
($2,300 to $4,000 an ounce) in the test period, so long and short results are reported apart.
"""
import datetime as dt
import itertools
import json
import time

import numpy as np

from core.config.settings import DATA_DIR
from research.futures import factory as fx
from research.futures.data import SPECS, Spec, load

OUT = DATA_DIR / "research" / "results"

LEGAL = SPECS["GOLDPETAL"]
OFFSHORE = Spec("XAUUSD CFD 1:100 (offshore)", "GOLD", "CFD", 1, 0.01, 0.01, ("09:00", "23:55"), "23:00", price_div=10.0)
SIZINGS = {"MCX gold petal (legal)": (LEGAL, [1000.0, 2000.0]), "XAUUSD CFD 1:100 (offshore)": (OFFSHORE, [2000.0, 5000.0, 10000.0])}


def _last_sunday(y, m):
    d = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def uk_dst(d):
    return _last_sunday(d.year, 3) <= d < _last_sunday(d.year, 10)


def _slot(s, hhmm, shift=0):
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m + shift - s.open_min) // s.bar_min - 1


def _range_break(s, rows_shift, r_from, r_to, until):
    """For each session: range of bars r_from..r_to (IST, shifted by the session's DST offset), then the first
    close beyond it until `until`. rows_shift: array of minute shifts per session."""
    side = np.zeros(s.n, int)
    slot = np.full(s.n, -1)
    mid = np.full(s.n, np.nan)
    I = np.arange(s.c.shape[1])
    for sh in np.unique(rows_shift):
        rows = np.flatnonzero(rows_shift == sh)
        a, b, e = _slot(s, r_from, sh) + 1, _slot(s, r_to, sh), _slot(s, until, sh)
        if b < a or e <= b:
            continue
        oh, ol = np.nanmax(s.h[rows, a:b + 1], 1), np.nanmin(s.l[rows, a:b + 1], 1)
        win = (I > b) & (I <= e)
        sd, sl = fx._pick(fx.first_true(win & (s.c[rows] > oh[:, None])), fx.first_true(win & (s.c[rows] < ol[:, None])))
        side[rows], slot[rows], mid[rows] = sd, sl, (oh + ol) / 2
    return fx._done(s, side, slot, mid)


def _shifts(s, which):
    days = [d.astype(object) for d in s.dates]
    if which == "uk":
        return np.array([0 if uk_dst(d) else 60 for d in days])
    return np.array([0 if fx._dst(d) else 60 for d in days])


def gold_setups():
    def asian_london(s, f, sp):        # MCX open to the London open (08:00 London = 12:30 IST in UK summer)
        return _range_break(s, _shifts(s, "uk"), "09:00", "12:30", "16:30")

    def london_hour(s, f, sp):
        return _range_break(s, _shifts(s, "uk"), "12:30", "13:30", "17:30")

    def comex_open(s, f, sp):          # COMEX 08:20 New York = 17:50 IST in US summer
        return _range_break(s, _shifts(s, "us"), "17:50", "18:20", "21:30")

    def us_data(s, f, sp):
        """The 08:30 New York bar (US data) moves more than 0.15 ATR: follow it; stop at its other end."""
        sh = _shifts(s, "us")
        rows, slots, sides, stops = [], [], [], []
        for i in range(s.n):
            j = _slot(s, "18:05", sh[i])
            if j < 0 or np.isnan(s.c[i, j]) or not np.isfinite(f.atr[i]):
                continue
            body = s.c[i, j] - s.o[i, j]
            if abs(body) < 0.15 * f.atr[i]:
                continue
            sd = 1 if body > 0 else -1
            rows.append(i); slots.append(j); sides.append(sd); stops.append(s.l[i, j] if sd > 0 else s.h[i, j])
        rows = np.array(rows, int)
        slots = np.array(slots, int)
        return rows, slots, np.array(sides, int), (s.c[rows, slots] if len(rows) else np.zeros(0)), np.array(stops, float), False

    return {"Asian range -> London breakout": asian_london, "London first hour breakout": london_hour,
            "COMEX open 30-min breakout": comex_open, "US data 08:30 NY bar follow": us_data}


def run(log=print):
    s = load("GOLDPETAL")
    f = fx.features(s)
    setups = dict(fx.setups_for(LEGAL))
    setups.update(gold_setups())
    rows = []
    t0 = time.time()
    for sname, fn in setups.items():
        cand = fn(s, f, LEGAL)
        for size_name, (sp, risks) in SIZINGS.items():
            for side_mode in ("both", "long only", "short only"):
                c = cand
                if side_mode != "both" and len(cand[0]):
                    want = 1 if side_mode == "long only" else -1
                    k = cand[2] == want
                    c = tuple(x[k] if isinstance(x, np.ndarray) else x for x in cand)
                for filt, ex, rb in itertools.product(fx.FILTERS, fx.EXITS, risks):
                    res = fx.run_variant(s, f, sp, c, filt, ex, rb)
                    if res is None:
                        continue
                    daily, ntr, lots = res
                    rows.append(fx.stats(daily, ntr, s.dates, {
                        "sizing": size_name, "setup": sname, "sides": side_mode, "filter": filt, "exit": ex.label,
                        "risk": rb, "avg_grams": round(float(lots[lots >= 1].mean()), 1) if (lots >= 1).any() else 0}))
        log(f"  {sname}: {len(rows):,} variants ({time.time() - t0:.0f}s)")
    (OUT / "grid_xau.json").write_text(json.dumps(rows))
    return rows
