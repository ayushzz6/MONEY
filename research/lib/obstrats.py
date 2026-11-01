"""Order-block strategy family: 14 named strategies plus a grid, on 5 years of NSE F&O 5-minute bars.

No train / test split and no tuning: every rule and number below is fixed before the run, and each strategy's
signals are traded over the whole period (Oct 2021 - Oct 2026) and scored after costs. Per-year results are shown
only to see whether a result holds up every year; nothing is chosen on them.

Zones come from market/orderblocks.detect_zones (the live board's own code). For each zone the pass records one
candidate trade per entry method, if the stock's board strength points the trade's way at the decision bar:

  Z        limit order at the zone's entry level (top of a bullish zone) for its first retest (the live rule)
  MID      limit order at the zone's midpoint (the "50% of the block" entry)
  REJ      price action: after the first retest, a 5-minute candle within REJ_BARS bars that closes back out of
           the zone the trade's way, in the best 40% of its range (a rejection); market order at the next open,
           stop beyond the retest's extreme or the zone's invalidation, whichever is further
  BRK      price action: the retest candle closes without breaking the zone; a stop order at its high (low for a
           sell), valid BRK_BARS bars; stop beyond its low or the invalidation, whichever is further
  BREAKER  a zone that failed (a 5-minute close beyond its far edge) is traded the other way on its first return:
           a limit order at the broken edge, stop beyond the opposite edge
Exits: a 2R target (the headline for every strategy), "hold" to 15:15, and for Z the board's liquidity target T2.

Confirmations, point in time (the F&O file is the previous session's, published after its close):
  fut     futures OI build-up on the trade's side the previous session (long build-up / short build-up)
  opt     more put OI added than call OI (buys), or the reverse (sells), the previous session
  narrow  today's CPR narrower than 80% of the last 20
  cpr     price beyond the CPR on the trade's side
  vol     volume so far today at least 1.5x the 20-session average to the same time
  pdl     price beyond the previous session's high (buys) or low (sells)
  trend   price on the trade's side of the 20-session average close
  index   NIFTY 50 on the trade's side of its open
The dashboard's ninth check, today's live futures OI, has no history (Angel One gives it live only).

Trades: Rs 1,000 risk, NSE charges, 3 bps slippage, a stop touched in the entry bar is a loss, a target is never
counted in the entry bar, out at 15:15, one trade per stock a day per strategy (its first signal), stops between
MIN_RISK and MAX_RISK % of price (the live-rule reference has no such limit); account: Rs 1 lakh, 5x intraday
leverage, 5 open at most, 5 trades and Rs 3,000 loss a day (research/lib/analyze.py).
"""
import glob
import json
import time
import warnings
from collections import defaultdict
from itertools import product

import numpy as np

from core.config.settings import DATA_DIR, Settings
from market import orderblocks as OB
from research.lead import intraday as I
from research.lib import analyze as A
from research.lib import engine as E
from research.lib import obtest as OT
from research.lib import round2 as R2

OUT = DATA_DIR / "research" / "obstrats"
RES = DATA_DIR / "research" / "results"
FLAT, LAST_ENTRY = OT.FLAT, OT.LAST_ENTRY
METHODS = ["Z", "MID", "REJ", "BRK", "BREAKER"]
EXITS = ["2R", "hold", "T2"]
CHECKS = ["fut", "opt", "narrow", "cpr", "vol", "pdl", "trend", "index"]
REJ_BARS, REJ_CLOSE, BRK_BARS = 4, 0.6, 3
NARROW, VOL_MIN, CPR_DAYS, VOL_DAYS, TREND_DAYS = 0.8, 1.5, 20, 20, 20
MIN_RISK, MAX_RISK = 0.2, 2.5                    # stop distance, % of price
MORNING = 14                                     # zones given by 10:30 (formed by the 10:25 candle)
START = 25                                       # first session with 20 sessions of CPR, volume and trend history


# ---------------------------------------------------------------- confirmations, every (stock, day[, bar])
def _shift(a):
    out = np.full_like(a, np.nan)
    out[:, 1:] = a[:, :-1]
    return out


def confirm_arrays(P, dates, syms, nifty):
    H, L, C, V = (P[k] for k in "hlcv")
    S, D, _ = C.shape
    valid = np.isfinite(C).all(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dH = np.where(valid, np.nanmax(H, 2), np.nan).astype(np.float64)
        dL = np.where(valid, np.nanmin(L, 2), np.nan).astype(np.float64)
    dC = np.where(valid, C[:, :, -1], np.nan).astype(np.float64)
    pH, pL, pC = _shift(dH), _shift(dL), _shift(dC)
    piv = (pH + pL + pC) / 3
    a = (pH + pL) / 2
    b = 2 * piv - a
    bc, tc = np.minimum(a, b), np.maximum(a, b)
    width = (tc - bc) / piv
    narrow = np.full((S, D), np.nan)
    sma = np.full((S, D), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(CPR_DAYS, D):
            w = width[:, d - CPR_DAYS:d]
            n = np.isfinite(w).sum(1)
            narrow[:, d] = np.where((n >= 15) & np.isfinite(width[:, d]), (w > width[:, d:d + 1]).sum(1) / np.maximum(n, 1), np.nan)
            c = dC[:, d - TREND_DAYS:d]
            sma[:, d] = np.where(np.isfinite(c).sum(1) >= 15, np.nanmean(c, 1), np.nan)
    cv = np.cumsum(np.nan_to_num(V), axis=2)
    cs = np.cumsum(np.where(valid[..., None], cv, 0).astype(np.float64), axis=1)
    cnt = np.cumsum(valid, axis=1)
    rv = np.full(cv.shape, np.nan, np.float32)
    for d in range(VOL_DAYS + 1, D):
        n = cnt[:, d - 1] - cnt[:, d - 1 - VOL_DAYS]
        avg = (cs[:, d - 1] - cs[:, d - 1 - VOL_DAYS]) / np.maximum(n, 1)[:, None]
        ok = (n >= 5)[:, None] & (avg > 0) & valid[:, d, None]
        rv[:, d] = np.where(ok, cv[:, d] / np.where(avg > 0, avg, 1), np.nan)
    pos = {str(x): j for j, x in enumerate(dates)}
    sid = {s: i for i, s in enumerate(syms)}
    fut = np.zeros((S, D), np.int8)
    opt = np.zeros((S, D), np.int8)
    for p in glob.glob(str(DATA_DIR / "research" / "lead" / "daily" / "*.json")):
        doc = json.loads(open(p).read())
        j = pos.get(doc["date"])
        if j is None or j < 1 or j + 1 >= D:
            continue
        for s, row in (doc.get("fo") or {}).items():
            i = sid.get(s)
            if i is None or not row or len(row) < 7:
                continue
            pc = dC[i, j] / dC[i, j - 1] - 1
            if row[1] is not None and row[1] > 0 and np.isfinite(pc) and pc != 0:
                fut[i, j + 1] = 1 if pc > 0 else -1           # fresh positions: long or short build-up
            ce, pe = row[5], row[6]
            if ce is not None and pe is not None and (ce or pe):
                opt[i, j + 1] = int(np.sign(pe - ce))         # put writing (bullish) or call writing (bearish)
    ix = nifty["c"].astype(np.float64) / nifty["o"][:, :1].astype(np.float64) - 1
    return {"tc": tc, "bc": bc, "narrow": narrow, "sma": sma, "rv": rv, "fut": fut, "opt": opt, "nifty": ix}


# ---------------------------------------------------------------- pass: zones, entries and exits per stock
_G = {}


def _exit(Od, Hd, Ld, Cd, e, stop, sd, tp):
    if (Ld[e] <= stop) if sd > 0 else (Hd[e] >= stop):
        return -1, e, stop
    for q in range(e + 1, FLAT + 1):
        if (Ld[q] <= stop) if sd > 0 else (Hd[q] >= stop):
            return -1, q, (min(stop, Od[q]) if sd > 0 else max(stop, Od[q]))
        if tp is not None and ((Hd[q] >= tp) if sd > 0 else (Ld[q] <= tp)):
            return 1, q, (max(tp, Od[q]) if sd > 0 else min(tp, Od[q]))
    return 0, FLAT, Cd[FLAT]


def _stock(s):
    P, Bd, Cf, fno, starts = (_G[k] for k in ("P", "board", "conf", "fno", "starts"))
    O, H, L, C = (P[k][s].astype(np.float64) for k in "ohlc")
    R = defaultdict(list)
    for d in range(START, C.shape[0]):
        if not fno[s, d] or not np.isfinite(C[d]).all():
            continue
        ph, pl = Bd["prev_high"][s, d], Bd["prev_low"][s, d]
        if not (np.isfinite(ph) and np.isfinite(pl) and np.isfinite(Bd["atr"][s, d])):
            continue
        Od, Hd, Ld, Cd = O[d], H[d], L[d], C[d]
        bars = [OT.B_(int(starts[d] + 300 * k), Od[k], Hd[k], Ld[k], Cd[k]) for k in range(75)]
        zones = OB.detect_zones(bars, ph, pl)
        if not zones:
            continue
        t_of = {b.start: k for k, b in enumerate(bars)}
        sh, sl = OB._swings(list(Hd), list(Ld))
        info = []
        for zz in zones:
            t = t_of[zz.formed_at]
            bk = next((q for q in range(t + 1, 75) if ((Cd[q] < zz.bottom) if zz.side > 0 else (Cd[q] > zz.top))), None)
            info.append((t, zz.side, bk))

        def current(t, sd, k):
            """The zone is the stock's latest unbroken one of its side at bar k (the board's current setup)."""
            return not any(o_sd == sd and t < o_t <= k and (o_bk is None or o_bk > k) for o_t, o_sd, o_bk in info)

        def add(m, zz, t, sweep, e, k, fill, stop, ts):
            risk = abs(fill - stop)
            if risk <= 0 or e > LAST_ENTRY or k < 0 or Bd["side"][s, d, k] != ts:
                return
            tps = {"2R": fill + 2 * ts * risk, "hold": None}
            if m == 0:
                known = OB.detect_zones(bars[:e], ph, pl)
                mz = next((o for o in known if o.formed_at == zz.formed_at and o.side == zz.side), None)
                tg = mz.targets if mz else []
                tps["T2"] = tg[-1] if tg else fill + 2 * ts * risk
            for name in EXITS:
                if name in tps:
                    o_, x_, p_ = _exit(Od, Hd, Ld, Cd, e, stop, ts, tps[name])
                else:
                    o_, x_, p_ = 0, FLAT, np.nan
                R[f"out_{name}"].append(o_)
                R[f"x_{name}"].append(x_)
                R[f"px_{name}"].append(p_)
            c = Cd[k]
            checks = {"fut": Cf["fut"][s, d] == ts, "opt": Cf["opt"][s, d] == ts,
                      "narrow": bool(Cf["narrow"][s, d] >= NARROW),
                      "cpr": bool(c > Cf["tc"][s, d]) if ts > 0 else bool(c < Cf["bc"][s, d]),
                      "vol": bool(Cf["rv"][s, d, k] >= VOL_MIN),
                      "pdl": bool(c > ph) if ts > 0 else bool(c < pl),
                      "trend": bool((c - Cf["sma"][s, d]) * ts > 0),
                      "index": bool(Cf["nifty"][d, k] * ts > 0)}
            for n_, v_ in checks.items():
                R[f"c_{n_}"].append(bool(v_))
            for n_, v_ in (("method", m), ("day", d), ("stock", s), ("side", ts), ("slot", e), ("k", k), ("formed", t),
                           ("entry", fill), ("stop", stop), ("risk_pct", 100 * risk / fill), ("strong", bool(zz.strong)),
                           ("sweep", sweep), ("pick", bool(Bd["pick"][s, d, k])), ("rank", int(Bd["rank"][s, d, k])),
                           ("aligned", int(Bd["aligned"][s, d, k])), ("strength", float(Bd["score"][s, d, k])),
                           ("conf", sum(checks.values()))):
                R[n_].append(v_)

        for zi, zz in enumerate(zones):
            t, sd, bk = info[zi]
            ob = t_of[zz.ob_start]
            top, bot, inv = zz.top, zz.bottom, zz.invalid
            buf = (bot - inv) if sd > 0 else (inv - top)
            prior = [i for i in (sl if sd > 0 else sh) if i < ob and i + OB.SWING <= t - 1]
            lo2, hi2 = min(Ld[max(0, ob - 1):ob + 1]), max(Hd[max(0, ob - 1):ob + 1])
            sweep = bool(prior) and ((lo2 < Ld[prior[-1]]) if sd > 0 else (hi2 > Hd[prior[-1]]))

            def first_touch(level):
                for i in range(t + 1, FLAT + 1):
                    if bk is not None and bk <= i - 1:
                        return None
                    if (Ld[i] <= level) if sd > 0 else (Hd[i] >= level):
                        return i
                return None

            # Z: limit at the zone's entry level
            j = first_touch(zz.z)
            if j is not None and 1 <= j <= LAST_ENTRY and current(t, sd, j - 1):
                if (Od[j] > inv) if sd > 0 else (Od[j] < inv):
                    fill = min(zz.z, Od[j]) if sd > 0 else max(zz.z, Od[j])
                    add(0, zz, t, sweep, j, j - 1, fill, inv, sd)
            # MID: limit at the zone's midpoint
            mid = (top + bot) / 2
            jm = first_touch(mid)
            if jm is not None and 1 <= jm <= LAST_ENTRY and current(t, sd, jm - 1):
                if (Od[jm] > inv) if sd > 0 else (Od[jm] < inv):
                    fill = min(mid, Od[jm]) if sd > 0 else max(mid, Od[jm])
                    add(1, zz, t, sweep, jm, jm - 1, fill, inv, sd)
            if j is not None and j < LAST_ENTRY:
                # REJ: a rejection candle closes back out of the zone; enter at the next open
                for b in range(j, min(j + REJ_BARS, LAST_ENTRY)):
                    if (Cd[b] < bot) if sd > 0 else (Cd[b] > top):
                        break
                    rng = Hd[b] - Ld[b]
                    if rng <= 0:
                        continue
                    if sd > 0:
                        ok = Cd[b] > top and Cd[b] > Od[b] and (Cd[b] - Ld[b]) / rng >= REJ_CLOSE
                    else:
                        ok = Cd[b] < bot and Cd[b] < Od[b] and (Hd[b] - Cd[b]) / rng >= REJ_CLOSE
                    if ok:
                        if current(t, sd, b):
                            stop = min(inv, min(Ld[j:b + 1]) - buf) if sd > 0 else max(inv, max(Hd[j:b + 1]) + buf)
                            fill = Od[b + 1]
                            if (fill - stop) * sd > 0:
                                add(2, zz, t, sweep, b + 1, b, fill, stop, sd)
                        break
                # BRK: a stop order beyond the retest candle, valid BRK_BARS bars
                if not ((Cd[j] < bot) if sd > 0 else (Cd[j] > top)) and current(t, sd, j):
                    trig = Hd[j] if sd > 0 else Ld[j]
                    stop = min(inv, Ld[j] - buf) if sd > 0 else max(inv, Hd[j] + buf)
                    for q in range(j + 1, min(j + 1 + BRK_BARS, LAST_ENTRY + 1)):
                        if q - 1 > j and ((Cd[q - 1] < bot) if sd > 0 else (Cd[q - 1] > top)):
                            break
                        if (Hd[q] > trig) if sd > 0 else (Ld[q] < trig):
                            fill = max(trig, Od[q]) if sd > 0 else min(trig, Od[q])
                            if (fill - stop) * sd > 0:
                                add(3, zz, t, sweep, q, j, fill, stop, sd)
                            break
            # BREAKER: the failed zone, traded the other way on its first return to the broken edge
            if bk is not None and bk < LAST_ENTRY:
                ts = -sd
                level = bot if sd > 0 else top
                stop = (top + buf) if sd > 0 else (bot - buf)
                for i in range(bk + 1, LAST_ENTRY + 1):
                    if i - 1 > bk and ((Cd[i - 1] > top) if sd > 0 else (Cd[i - 1] < bot)):
                        break
                    if (Hd[i] >= level) if ts < 0 else (Ld[i] <= level):
                        if (Od[i] < stop) if ts < 0 else (Od[i] > stop):
                            fill = max(level, Od[i]) if ts < 0 else min(level, Od[i])
                            add(4, zz, t, sweep, i, i - 1, fill, stop, ts)
                        break
    if not R:
        return None
    return {k: np.array(v) for k, v in R.items()}


def build(log=print, workers=8, limit=None):
    """limit: only the first n F&O stocks (a quick check of the pass; the result is not saved)."""
    import multiprocessing as mp
    t0 = time.time()
    syms, inds, dates, P, idx = I.panel(log)
    fno = OT.fno_mask(syms, dates)
    keep = np.flatnonzero(fno.any(1))
    syms, inds, fno = [syms[i] for i in keep], [inds[i] for i in keep], fno[keep]
    P = {k: P[k][keep] for k in "ohlcv"}
    log(f"  F&O stocks: {len(syms)} ever, {int(fno.sum(0).mean())} a day on average, {len(dates)} sessions "
        f"{dates[0]} to {dates[-1]}")
    Bd = OT.board_arrays(P, inds, fno)
    log(f"  boards rebuilt for every bar ({time.time() - t0:.0f}s)")
    Cf = confirm_arrays(P, dates, syms, idx["NIFTY"])
    log(f"  confirmations rebuilt ({time.time() - t0:.0f}s)")
    starts = dates.astype("datetime64[s]").astype(np.int64) - 19800 + 9 * 3600 + 15 * 60
    _G.update(P=P, board=Bd, conf=Cf, fno=fno, starts=starts)
    stocks = [s for s in range(len(syms)) if fno[s].any()][:limit]
    parts = []
    with mp.get_context("fork").Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_stock, stocks, chunksize=2), 1):
            if r is not None:
                parts.append(r)
            if n % 25 == 0 or n == len(stocks):
                log(f"  {n}/{len(stocks)} stocks, {sum(len(p['day']) for p in parts):,} candidate trades ({time.time() - t0:.0f}s)")
    T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    costs = Settings().get("costs")
    for ex in EXITS:
        T[f"r_{ex}"] = R2._rnet(T["entry"], T["stop"], T[f"px_{ex}"], T[f"out_{ex}"], T["side"], costs)
        T[f"g_{ex}"] = (T[f"px_{ex}"] - T["entry"]) * T["side"] / np.abs(T["entry"] - T["stop"])
    if limit:
        return T
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "candidates.npz", **T)
    (OUT / "meta.json").write_text(json.dumps({"dates": [str(d) for d in dates], "symbols": list(syms),
                                               "built_at": time.strftime("%Y-%m-%d %H:%M")}))
    log(f"  saved {len(T['day']):,} candidate trades ({time.time() - t0:.0f}s)")
    return T


# ---------------------------------------------------------------- strategies and scoring
def _m(T, method, board="side", zone="all", conf="none", morning=False, risk=True):
    m = T["method"] == METHODS.index(method)
    if board == "pick4":
        m &= T["pick"] & (T["aligned"] >= 4)
    if zone == "strong":
        m &= T["strong"]
    elif zone == "sweep":
        m &= T["sweep"]
    if conf == "5of8":
        m &= T["conf"] >= 5
    elif conf == "oi":
        m &= T["c_fut"] & T["c_opt"]
    elif conf == "vol_trend":
        m &= T["c_vol"] & T["c_trend"]
    elif conf == "fut_vol":
        m &= T["c_fut"] & T["c_vol"]
    elif conf == "trend_index":
        m &= T["c_trend"] & T["c_index"]
    if morning:
        m &= T["formed"] <= MORNING
    if risk:
        m &= (T["risk_pct"] >= MIN_RISK) & (T["risk_pct"] <= MAX_RISK)
    return m


NAMED = [
    ("N0", "Live board rule (reference): pick, 4/4 aligned, limit at Z, liquidity target", dict(method="Z", board="pick4", risk=False), "T2"),
    ("N1", "Order-block retest: limit at Z", dict(method="Z"), "2R"),
    ("N2", "50% of the block: limit at the zone's midpoint", dict(method="MID"), "2R"),
    ("N3", "Order block that left a fair-value gap", dict(method="Z", zone="strong"), "2R"),
    ("N4", "Liquidity sweep, then order block", dict(method="Z", zone="sweep"), "2R"),
    ("N5", "Breaker block: the failed zone traded the other way", dict(method="BREAKER"), "2R"),
    ("N6", "Order block + rejection candle (price action)", dict(method="REJ"), "2R"),
    ("N7", "Order block + break of the retest candle (price action)", dict(method="BRK"), "2R"),
    ("N8", "Order block + at least 5 of 8 confirmations", dict(method="Z", conf="5of8"), "2R"),
    ("N9", "Order block + F&O OI: futures build-up and options writing", dict(method="Z", conf="oi"), "2R"),
    ("N10", "Rejection candle + at least 5 of 8 confirmations", dict(method="REJ", conf="5of8"), "2R"),
    ("N11", "Rejection candle + futures build-up + volume 1.5x", dict(method="REJ", conf="fut_vol"), "2R"),
    ("N12", "Full stack: board pick 4/4 + rejection candle + 5 of 8 confirmations", dict(method="REJ", board="pick4", conf="5of8"), "2R"),
    ("N13", "Morning order block + rejection candle, with the 20-day trend and NIFTY", dict(method="REJ", conf="trend_index", morning=True), "2R"),
]


def _one_per_day(T, m):
    """Each stock's first signal of the day (ties: the stronger board score)."""
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return idx
    idx = idx[np.lexsort((-np.abs(T["strength"][idx]), T["slot"][idx], T["day"][idx], T["stock"][idx]))]
    key = T["stock"][idx].astype(np.int64) * 100000 + T["day"][idx]
    return idx[np.r_[True, key[1:] != key[:-1]]]


def _recs(T, idx, ex):
    rec = np.empty(len(idx), dtype=E.REC)
    rec["v"], rec["day"], rec["stock"] = 0, T["day"][idx], T["stock"][idx]
    rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = T["slot"][idx], T[f"x_{ex}"][idx], T["side"][idx], T[f"out_{ex}"][idx]
    rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"] = T["entry"][idx], T["stop"][idx], T[f"px_{ex}"][idx], T[f"r_{ex}"][idx]
    rec["score"] = np.abs(T["strength"][idx])
    return rec


def evaluate(T, m, ex, dates, curve=False):
    idx = _one_per_day(T, m & np.isfinite(T[f"r_{ex}"]))
    r, g = T[f"r_{ex}"][idx], T[f"g_{ex}"][idx]
    out = {"signals": int(idx.size), "avg_r": float(r.mean()) if r.size else None,
           "gross_r": float(np.nanmean(g)) if r.size else None, "win": float((r > 0).mean() * 100) if r.size else None}
    for nm, sd in (("long", 1), ("short", -1)):
        rr = r[T["side"][idx] == sd]
        out[nm] = {"signals": int(rr.size), "avg_r": float(rr.mean()) if rr.size else None}
    daily, n = A.account(_recs(T, idx, ex), len(dates))
    sel = np.arange(len(dates)) >= START
    month = np.array([str(x)[:7] for x in dates])
    year = np.array([str(x)[:4] for x in dates])
    mm = sorted(set(month[sel]))
    sums = np.array([daily[sel][month[sel] == x].sum() for x in mm])
    cum = np.cumsum(daily[sel])
    out.update({"trades": int(n), "net": float(daily[sel].sum()), "per_month": float(sums.mean()), "months": len(mm),
                "losing_months": int((sums < 0).sum()),
                "max_dd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()),
                "by_year": {y: round(float(daily[sel][year[sel] == y].sum())) for y in sorted(set(year[sel]))}})
    if curve:
        out["curve"] = np.round(cum).astype(int).tolist()
    return out


def analyze(log=print):
    t0 = time.time()
    z = np.load(OUT / "candidates.npz")
    T = {k: z[k] for k in z.files}
    meta = json.loads((OUT / "meta.json").read_text())
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    named = []
    for code, name, spec, ex in NAMED:
        res = evaluate(T, _m(T, **spec), ex, dates, curve=True)
        res.update(code=code, name=name, exit=ex, spec=spec)
        named.append(res)
        log(f"  {code:<4}{name[:62]:<63} {res['signals']:>6} signals {res['avg_r'] if res['avg_r'] is not None else 0:+.3f}R "
            f"(before costs {res['gross_r'] or 0:+.3f}R)  Rs {res['per_month']:>8,.0f}/month")
    grid = []
    for method, board, zone, conf, ex in product(METHODS, ("side", "pick4"), ("all", "strong", "sweep"),
                                                 ("none", "5of8", "oi", "vol_trend"), ("2R", "hold", "T2")):
        if (ex == "T2" and method != "Z") or (method == "BREAKER" and zone != "all"):
            continue
        res = evaluate(T, _m(T, method, board, zone, conf), ex, dates)
        res.update(method=method, board=board, zone=zone, conf=conf, exit=ex)
        grid.append(res)
    ok = [g for g in grid if g["signals"] >= 200]
    every_year = [g for g in ok if g["per_month"] > 0 and all(v > 0 for v in g["by_year"].values())]
    ok.sort(key=lambda g: -g["per_month"])
    doc = {"built_at": meta["built_at"], "first_day": str(dates[START]), "last_day": str(dates[-1]),
           "dates_axis": [str(d) for d in dates[START:]], "candidates": int(len(T["day"])),
           "methods": {m: int((T["method"] == i).sum()) for i, m in enumerate(METHODS)},
           "named": named, "grid_size": len(grid), "grid_ok": len(ok),
           "grid_positive": sum(1 for g in ok if g["per_month"] > 0),
           "grid_positive_before_costs": sum(1 for g in ok if (g["gross_r"] or 0) > 0),
           "grid_every_year": len(every_year), "grid_top": ok[:10], "grid_bottom": ok[-5:],
           "checks_lift": _checks_lift(T)}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "obstrats_summary.json").write_text(json.dumps(doc, indent=1, default=float))
    log(f"  grid: {len(ok)} variants with 200+ signals, {doc['grid_positive']} profitable after costs, "
        f"{doc['grid_positive_before_costs']} positive before costs, {len(every_year)} profitable every year")
    log(f"  saved results/obstrats_summary.json ({time.time() - t0:.0f}s)")
    return doc


def _checks_lift(T):
    """For the plain retest (Z, 2R): average R with and without each confirmation."""
    base = _m(T, "Z")
    idx = _one_per_day(T, base & np.isfinite(T["r_2R"]))
    r = T["r_2R"][idx]
    out = {}
    for c in CHECKS:
        on = T[f"c_{c}"][idx]
        out[c] = {"with": float(r[on].mean()) if on.any() else None, "n_with": int(on.sum()),
                  "without": float(r[~on].mean()) if (~on).any() else None, "n_without": int((~on).sum())}
    return out
