"""Backtest of the order-block board (market/orderblocks.py) on NSE F&O stocks, 5-minute bars.

Point in time, bar by bar: the F&O list of each day (NSE bhavcopy), each stock's strength score from the bars
closed so far, the bullish / bearish boards and picks (top 3 a side, max 2 a sector, no event risk), sector,
breadth and VWAP alignment, and the zones exactly as detect_zones finds them. Zone formation uses only bars up
to the breaking candle; targets are recomputed from the bars known when the entry order rests.

The trade (the site's playbook): when the stock is on the board on the zone's side, rest a limit order at the
zone level Z for its FIRST retest; stop at the invalidation level; target the second liquidity level (the one
its R:R uses), else the first, else 2R; anything open closes at 15:15. A stop touched in the entry bar counts as
a loss, and a target is never counted in the entry bar. Rs 1,000 risk a trade, NSE charges, 3 bps slippage.
"""
import glob
import json
import time
from collections import namedtuple
from itertools import product

import numpy as np

from core.config.settings import DATA_DIR, Settings
from market import orderblocks as OB
from research.lead import intraday as I
from research.lib import analyze as A
from research.lib import engine as E
from research.lib import round2 as R2

OUT = DATA_DIR / "research" / "obtest"
RES = DATA_DIR / "research" / "results"
LEARN_START = np.datetime64("2022-10-25")
TEST_START = np.datetime64("2024-10-24")
FLAT, LAST_ENTRY = 71, 68                         # out at 15:15; no new entry in a bar starting after 14:55
EXITS = ["T2", "T1", "2R", "hold"]
EXIT_LABELS = {"T2": "second liquidity level (the board's R:R target)", "T1": "first liquidity level",
               "2R": "twice the risk", "hold": "no target, out at 15:15"}
B_ = namedtuple("B_", "start o h l c")


# ---------------------------------------------------------------- pass 1: the board, every bar of every day
def fno_mask(syms, dates):
    pos = {str(d): j for j, d in enumerate(dates)}
    sid = {s: i for i, s in enumerate(syms)}
    m = np.zeros((len(syms), len(dates)), bool)
    for p in glob.glob(str(DATA_DIR / "research" / "lead" / "daily" / "*.json")):
        doc = json.loads(open(p).read())
        j = pos.get(doc["date"])
        if j is None:
            continue
        for s, row in (doc.get("fo") or {}).items():
            i = sid.get(s)
            if i is not None and row and row[0] is not None:
                m[i, j] = True
    return m


def board_arrays(P, inds, fno):
    """Strength, side, rank, contender, pick and alignment for every (stock, day, bar)."""
    O, H, L, C, V = (P[k].astype(np.float32) for k in "ohlcv")
    S, D, _ = C.shape
    dH, dL, dC = np.nanmax(H, 2), np.nanmin(L, 2), C[:, :, -1]
    prevC = np.full((S, D), np.nan)
    prevH = np.full((S, D), np.nan)
    prevL = np.full((S, D), np.nan)
    prevC[:, 1:], prevH[:, 1:], prevL[:, 1:] = dC[:, :-1], dH[:, :-1], dL[:, :-1]
    tr = np.maximum(dH - dL, np.maximum(np.abs(dH - prevC), np.abs(dL - prevC)))
    atr = np.full((S, D), np.nan)
    for d in range(15, D):
        atr[:, d] = np.nanmean(tr[:, d - 14:d], axis=1)
    pc, at = prevC[..., None], atr[..., None]
    O0 = O[:, :, :1]
    dC_ = np.abs(np.diff(C, axis=2))
    path = np.abs(C[:, :, :1] - O0) + np.concatenate([np.zeros((S, D, 1)), np.cumsum(dC_, axis=2)], axis=2)
    clean = np.where(path > 0, np.abs(C - O0) / np.where(path > 0, path, 1), 0.0)
    hi, lo = np.maximum.accumulate(H, axis=2), np.minimum.accumulate(L, axis=2)
    move = (C - pc) / at
    place = np.where(hi > lo, np.where(move >= 0, C - lo, hi - C) / np.where(hi > lo, hi - lo, 1), 0.5)
    score = np.where(clean < OB.CHOPPY, 0.0, move * clean * (0.5 + 0.5 * place)).astype(np.float32)
    del path, clean, hi, lo, place, dC_
    chg = 100 * (C / pc - 1)
    event = (np.abs(move) > OB.EVENT_ATR) | (np.abs(chg) > OB.EVENT_PCT)
    tp = (H + L + C) / 3
    cv = np.cumsum(V, axis=2)
    vw = np.where(cv > 0, np.cumsum(tp * V, axis=2) / np.where(cv > 0, cv, 1), np.cumsum(tp, axis=2) / np.arange(1, 76))
    secs = sorted(set(inds))
    sec_id = np.array([secs.index(x) for x in inds])
    side = np.sign(np.nan_to_num(score)).astype(np.int8)
    rank = np.zeros((S, D, 75), np.int16)
    pick = np.zeros((S, D, 75), bool)
    aligned = np.zeros((S, D, 75), np.int8)
    for d in range(D):
        el_d = fno[:, d] & np.isfinite(at[:, d, 0]) & np.isfinite(pc[:, d, 0])
        if el_d.sum() < 20:
            continue
        for k in range(75):
            el = el_d & np.isfinite(score[:, d, k])
            idx = np.flatnonzero(el)
            if idx.size < 20:
                continue
            sc = score[idx, d, k]
            ch = chg[idx, d, k]
            up, dn = (ch > 0).sum(), (ch < 0).sum()
            bside = 1 if up > dn else -1 if dn > up else 0
            sums = np.bincount(sec_id[idx], weights=ch, minlength=len(secs))
            cnts = np.bincount(sec_id[idx], minlength=len(secs))
            savg = sums / np.maximum(cnts, 1)
            for sd in (1, -1):
                g = idx[(sc * sd) > 0]
                if g.size == 0:
                    continue
                g = g[np.argsort(-np.abs(score[g, d, k]), kind="stable")]
                rank[g, d, k] = np.arange(1, g.size + 1)
                n_pick, per = 0, {}
                for s in g:
                    if n_pick >= 3:
                        break
                    if event[s, d, k]:
                        continue
                    if per.get(sec_id[s], 0) >= 2:
                        continue
                    per[sec_id[s]] = per.get(sec_id[s], 0) + 1
                    pick[s, d, k] = True
                    n_pick += 1
                contender = rank[g, d, k] <= 20
                sector_ok = (savg[sec_id[g]] * sd) > 0
                vw_ok = (C[g, d, k] - vw[g, d, k]) * sd >= 0
                aligned[g, d, k] = (contender.astype(int) + sector_ok + (bside == sd) + vw_ok).astype(np.int8)
    return {"side": side, "rank": rank, "pick": pick, "aligned": aligned, "event": event, "score": score.astype(np.float32),
            "prev_high": prevH, "prev_low": prevL, "atr": atr}


# ---------------------------------------------------------------- pass 2: zones and their first retests
_G = {}


def _stock(s):
    P, Bd, fno, dates, costs = (_G[k] for k in ("P", "board", "fno", "dates", "costs"))
    O, H, L, C = (P[k][s].astype(np.float64) for k in "ohlc")
    out = []
    starts = (dates.astype("datetime64[s]").astype(np.int64) - 19800 + 9 * 3600 + 15 * 60)
    for d in range(len(dates)):
        if not fno[s, d] or not np.isfinite(C[d]).all() or dates[d] < LEARN_START:
            continue
        ph, pl = Bd["prev_high"][s, d], Bd["prev_low"][s, d]
        if not (np.isfinite(ph) and np.isfinite(pl)):
            continue
        bars = [B_(int(starts[d] + 300 * k), O[d, k], H[d, k], L[d, k], C[d, k]) for k in range(75)]
        zones = OB.detect_zones(bars, ph, pl)
        if not zones:
            continue
        t_of = {b.start: k for k, b in enumerate(bars)}
        cands = []
        for zz in zones:
            t = t_of[zz.formed_at]
            sd = zz.side
            # first return into the zone, as long as no 5-minute close went through its far edge before
            j = None
            for i in range(t + 1, FLAT + 1):
                if (sd > 0 and C[d, i - 1] < zz.bottom and i - 1 > t) or (sd < 0 and C[d, i - 1] > zz.top and i - 1 > t):
                    break
                if (sd > 0 and L[d, i] <= zz.top) or (sd < 0 and H[d, i] >= zz.bottom):
                    j = i
                    break
            if j is None or j > LAST_ENTRY or j < 1:
                continue
            k = j - 1                                              # the order rests from this bar's close
            if Bd["side"][s, d, k] != sd:
                continue
            # it must be the stock's current setup: the latest unbroken zone of its side at that bar
            later = [o for o in zones if o.side == sd and t_of[o.formed_at] > t and t_of[o.formed_at] <= k
                     and not any((C[d, q] < o.bottom) if sd > 0 else (C[d, q] > o.top) for q in range(t_of[o.formed_at] + 1, k + 1))]
            if later:
                continue
            z, inv = zz.z, zz.invalid
            if sd > 0:
                if O[d, j] <= inv:
                    continue
                fill = min(z, O[d, j])
            else:
                if O[d, j] >= inv:
                    continue
                fill = max(z, O[d, j])
            risk = abs(fill - inv)
            if risk <= 0:
                continue
            known = OB.detect_zones(bars[:j], ph, pl)
            mz = next((o for o in known if o.formed_at == zz.formed_at and o.side == sd), None)
            tg = mz.targets if mz else []
            tgt = {"T2": tg[-1] if tg else fill + 2 * sd * risk, "T1": tg[0] if tg else fill + 2 * sd * risk,
                   "2R": fill + 2 * sd * risk, "hold": None}
            rec = {"day": d, "stock": s, "side": sd, "slot": j, "entry": fill, "stop": inv, "pick": bool(Bd["pick"][s, d, k]),
                   "rank": int(Bd["rank"][s, d, k]), "aligned": int(Bd["aligned"][s, d, k]), "strong": zz.strong,
                   "risk_pct": 100 * risk / fill, "age": j - t, "has_targets": bool(tg)}
            stopped_in_entry = (L[d, j] <= inv) if sd > 0 else (H[d, j] >= inv)
            for name, tp in tgt.items():
                if stopped_in_entry:
                    rec[f"out_{name}"], rec[f"x_{name}"], rec[f"px_{name}"] = -1, j, inv
                    continue
                o_, x_, p_ = 0, FLAT, C[d, FLAT]
                for q in range(j + 1, FLAT + 1):
                    hit_s = (L[d, q] <= inv) if sd > 0 else (H[d, q] >= inv)
                    hit_t = tp is not None and ((H[d, q] >= tp) if sd > 0 else (L[d, q] <= tp))
                    if hit_s:
                        o_, x_, p_ = -1, q, (min(inv, O[d, q]) if sd > 0 else max(inv, O[d, q]))
                        break
                    if hit_t:
                        o_, x_, p_ = 1, q, (max(tp, O[d, q]) if sd > 0 else min(tp, O[d, q]))
                        break
                rec[f"out_{name}"], rec[f"x_{name}"], rec[f"px_{name}"] = o_, x_, p_
            cands.append(rec)
        if cands:
            out.append(min(cands, key=lambda r: r["slot"]))        # one trade per stock per day: the first retest in time
    if not out:
        return None
    T = {k: np.array([r[k] for r in out]) for k in out[0]}
    for name in EXITS:
        T[f"r_{name}"] = R2._rnet(T["entry"], T["stop"], T[f"px_{name}"], T[f"out_{name}"], T["side"], costs)
    return T


def build(log=print, workers=8):
    import multiprocessing as mp
    t0 = time.time()
    syms, inds, dates, P, idx = I.panel(log)
    fno = fno_mask(syms, dates)
    keep = np.flatnonzero(fno.any(1))                          # only stocks that were ever in F&O
    syms, inds, fno = [syms[i] for i in keep], [inds[i] for i in keep], fno[keep]
    P = {k: P[k][keep] for k in "ohlcv"}
    log(f"  F&O stocks in the 5-minute data: {int(fno.any(1).sum())} (point in time, {int(fno.sum(0).mean())} a day on average)")
    Bd = board_arrays(P, inds, fno)
    log(f"  boards rebuilt for every bar of {len(dates)} sessions ({time.time() - t0:.0f}s)")
    _G.update(P=P, board=Bd, fno=fno, dates=dates, costs=Settings().get("costs"))
    stocks = [s for s in range(len(syms)) if fno[s].any()]
    parts = []
    with mp.get_context("fork").Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_stock, stocks, chunksize=2), 1):
            if r is not None:
                parts.append(r)
            if n % 40 == 0 or n == len(stocks):
                log(f"  {n}/{len(stocks)} stocks, {sum(len(p['day']) for p in parts):,} first retests ({time.time() - t0:.0f}s)")
    T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "retests.npz", **T)
    (OUT / "meta.json").write_text(json.dumps({"dates": [str(d) for d in dates], "symbols": list(syms),
                                               "built_at": time.strftime("%Y-%m-%d %H:%M")}))
    log(f"  saved {len(T['day']):,} first retests ({time.time() - t0:.0f}s)")
    return T


# ---------------------------------------------------------------- analysis
def _recs(T, m, ex):
    r = T[f"r_{ex}"]
    m = m & np.isfinite(r)
    rec = np.empty(int(m.sum()), dtype=E.REC)
    rec["v"], rec["day"], rec["stock"] = 0, T["day"][m], T["stock"][m]
    rec["slot"], rec["exit_slot"], rec["side"], rec["outcome"] = T["slot"][m], T[f"x_{ex}"][m], T["side"][m], T[f"out_{ex}"][m]
    rec["entry"], rec["stop"], rec["exit_px"], rec["rnet"] = T["entry"][m], T["stop"][m], T[f"px_{ex}"][m], r[m]
    rec["score"] = -T["rank"][m].astype(float)
    return rec


def _account(T, m, ex, dates):
    daily, n = A.account(_recs(T, m, ex), len(dates))
    month = np.array([str(d)[:7] for d in dates])
    out = {"trades": int(n)}
    for name, pm in (("learn", (dates >= LEARN_START) & (dates < TEST_START)), ("test", dates >= TEST_START)):
        mm = sorted(set(month[pm]))
        sums = np.array([daily[pm][month[pm] == x].sum() for x in mm])
        cum = np.cumsum(daily[pm])
        out[name] = {"per_month": float(sums.mean()), "months": len(mm), "losing_months": int((sums < 0).sum()),
                     "max_dd": float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()),
                     "monthly": {a: round(float(b)) for a, b in zip(mm, sums)}}
    sel = dates >= LEARN_START
    out["curve"] = np.round(np.cumsum(daily[sel])).astype(int).tolist()
    return out


def _stats(T, m, ex):
    r = T[f"r_{ex}"][m]
    r = r[np.isfinite(r)]
    gross = None
    if r.size:
        sd, e, st, px = T["side"][m], T["entry"][m], T["stop"][m], T[f"px_{ex}"][m]
        g = (px - e) * sd / np.maximum(np.abs(e - st), 1e-9)
        gross = float(np.nanmean(g))
    return {"n": int(r.size), "avg_r": float(r.mean()) if r.size else None, "win": float((r > 0).mean() * 100) if r.size else None,
            "gross_r": gross, "total_r": float(r.sum()) if r.size else 0.0,
            "stopped_entry_bar_pct": float(((T[f"out_{ex}"][m] == -1) & (T[f"x_{ex}"][m] == T["slot"][m])).mean() * 100) if m.any() else None}


def analyze(log=print):
    t0 = time.time()
    z = np.load(OUT / "retests.npz")
    T = {k: z[k] for k in z.files}
    meta = json.loads((OUT / "meta.json").read_text())
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    day = dates[T["day"]]
    learn, test = (day >= LEARN_START) & (day < TEST_START), day >= TEST_START
    designed = T["pick"] & (T["aligned"] >= 4)
    versions = {
        "As the site describes: a pick, 4/4 aligned, first retest, target the 2nd liquidity level": (designed, "T2"),
        "Same, 3 of 4 aligned is enough": (T["pick"] & (T["aligned"] >= 3), "T2"),
        "Any top-20 stock on the board, 4/4 aligned": ((T["rank"] <= 20) & (T["aligned"] >= 4), "T2"),
        "Every first retest on the stock's side (no board filter)": (np.ones(len(T["day"]), bool), "T2"),
    }
    main = {}
    for name, (m, ex) in versions.items():
        main[name] = {"learn": _stats(T, m & learn, ex), "test": _stats(T, m & test, ex), "account": _account(T, m, ex, dates)}
        log(f"  {name[:60]}: last 2 years Rs {main[name]['account']['test']['per_month']:,.0f}/month, "
            f"{main[name]['test']['n']} retests, {main[name]['test']['avg_r'] or 0:+.3f}R")
    # every combination: board filter x alignment x exit x zone strength x minimum risk; chosen on the 2 years before
    filters = {"picks": T["pick"], "top 20": T["rank"] <= 20, "any": np.ones(len(T["day"]), bool)}
    grid = []
    for (fname, fm), al, ex, strong, mr in product(filters.items(), (0, 3, 4), EXITS, (False, True), (0.0, 0.15, 0.3)):
        m = fm & (T["aligned"] >= al) & (T["risk_pct"] >= mr)
        if strong:
            m &= T["strong"]
        grid.append({"filter": fname, "aligned": al, "exit": ex, "strong_only": strong, "min_risk_pct": mr,
                     "learn": _stats(T, m & learn, ex), "test": _stats(T, m & test, ex)})
    ok = [g for g in grid if g["learn"]["n"] >= 100]
    ok.sort(key=lambda g: -g["learn"]["total_r"])
    for g in ok[:8]:
        m = filters[g["filter"]] & (T["aligned"] >= g["aligned"]) & (T["risk_pct"] >= g["min_risk_pct"])
        if g["strong_only"]:
            m &= T["strong"]
        g["account"] = _account(T, m, g["exit"], dates)
    sides = {nm: {p: _stats(T, designed & (T["side"] == s) & pm, "T2") for p, pm in (("learn", learn), ("test", test))}
             for s, nm in ((1, "bullish zones (buy)"), (-1, "bearish zones (sell)"))}
    risk_mix = {"median_risk_pct": float(np.median(T["risk_pct"][designed])) if designed.any() else None,
                "share_under_0_15": float((T["risk_pct"][designed] < 0.15).mean() * 100) if designed.any() else None}
    ex = np.flatnonzero(designed & test & np.isfinite(T["r_T2"]))
    ex = ex[np.argsort(T["day"][ex])][-12:]
    syms = meta["symbols"]
    examples = [{"date": str(dates[T["day"][i]]), "stock": syms[int(T["stock"][i])], "side": "buy" if T["side"][i] > 0 else "sell",
                 "time": f"{9 + (15 + 5 * int(T['slot'][i])) // 60:02d}:{(15 + 5 * int(T['slot'][i])) % 60:02d}",
                 "entry": round(float(T["entry"][i]), 2), "stop": round(float(T["stop"][i]), 2), "exit": round(float(T["px_T2"][i]), 2),
                 "risk_pct": round(float(T["risk_pct"][i]), 2), "r": round(float(T["r_T2"][i]), 2)} for i in ex]
    doc = {"built_at": meta["built_at"], "test_start": str(TEST_START), "learn_start": str(LEARN_START), "dates_axis": [str(d) for d in dates[dates >= LEARN_START]],
           "retests": {"learn": int(learn.sum()), "test": int(test.sum())}, "main": main, "sides": sides, "risk_mix": risk_mix,
           "variants": len(grid), "variants_ok": len(ok), "variants_positive_test": sum(1 for g in ok if (g["test"]["avg_r"] or -1) > 0),
           "top": ok[:8], "examples": examples}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "obtest_summary.json").write_text(json.dumps(doc, indent=1, default=float))
    log(f"  saved results/obtest_summary.json ({time.time() - t0:.0f}s)")
    return doc
