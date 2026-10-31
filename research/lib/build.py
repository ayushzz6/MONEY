"""Run the whole library over the NIFTY 500 (5-minute bars, Oct 2021 - Oct 2026) and save every trade.

    ./research.sh library            # build, then research/lib/analyze.py for the router and the report

Saved in data/research/library/: trades.npz (per variant and day, the 5 earliest trades), agg.npy (per variant
and period: trades, winners, sum of R, sum of R^2, all trades), breadth.npz (per day: stocks up at 09:30 and at
the close, for the router), meta.json.
"""
import json
import time
from collections import defaultdict

import numpy as np

from core.config.settings import DATA_DIR, Settings
from research import data
from research.lib import engine as E
from research.lib import features as F
from research.lib.sectors import SECTOR_TOKENS, sector_index
from research.sim import features as sim_features

OUT = DATA_DIR / "research" / "library"
TRAIN_END = np.datetime64("2024-12-31")
CORE_SECTORS = ["SEC_AUTO", "SEC_IT", "SEC_PHARMA", "SEC_FMCG", "SEC_METAL", "SEC_REALTY", "SEC_ENERGY", "SEC_FINSRV",
                "SEC_PSUBANK", "SEC_PVTBANK", "SEC_MEDIA", "SEC_INFRA", "SEC_COMMOD", "SEC_CONSUMP", "SEC_SERVICES"]
MIN_TURNOVER = 2e8          # Rs 20 crore a day on average over the 20 sessions before
MIN_PRICE = 50.0

_G = {}


def _rows_for(index_m, dates):
    pos = {d: i for i, d in enumerate(index_m.dates)}
    idx = np.array([pos.get(d, -1) for d in dates])
    out = []
    for a in (index_m.o, index_m.c, index_m.h, index_m.l):
        arr = np.full((len(dates), data.SLOTS), np.nan)
        arr[idx >= 0] = a[idx[idx >= 0]]
        out.append(arr)
    return tuple(out)


def context():
    N = data.load("NIFTY")
    master = N.dates
    sectors = {s: data.load(s) for s in SECTOR_TOKENS}
    sectors = {k: v for k, v in sectors.items() if v is not None}
    # rank of each core sector by its return since the open, at every bar (1 = strongest)
    R = np.full((len(CORE_SECTORS), len(master), data.SLOTS), np.nan)
    for k, s in enumerate(CORE_SECTORS):
        if s in sectors:
            o, c, h, l = _rows_for(sectors[s], master)
            R[k] = c / o[:, :1] - 1
    order = np.argsort(np.argsort(-np.nan_to_num(R, nan=-9), axis=0), axis=0) + 1
    rank = np.where(np.isnan(R), 99, order).astype(np.int16)
    return N, master, sectors, {s: rank[k] for k, s in enumerate(CORE_SECTORS)}


def _beta(m, N):
    pos = {d: i for i, d in enumerate(N.dates)}
    nc = N.c[:, -1]
    sc = m.c[:, -1]
    idx = np.array([pos.get(d, -1) for d in m.dates])
    rn = np.full(m.n, np.nan)
    ok = idx > 0
    rn[ok] = nc[idx[ok]] / nc[idx[ok] - 1] - 1
    rs = np.r_[np.nan, sc[1:] / sc[:-1] - 1]
    beta = np.full(m.n, np.nan)
    for i in range(61, m.n):
        a, b = rs[i - 60:i], rn[i - 60:i]
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() > 40 and np.var(b[g]) > 0:
            beta[i] = np.cov(a[g], b[g])[0, 1] / np.var(b[g])
    return beta


def _chunk(args):
    syms = args
    N, master, sectors, ranks, industries, costs = (_G[k] for k in ("N", "master", "sectors", "ranks", "industries", "costs"))
    mpos = {d: i for i, d in enumerate(master)}
    kept = []
    agg = np.zeros((E.NV, 2, 5))
    up930 = np.zeros(len(master))
    upclose = np.zeros(len(master))
    cnt = np.zeros(len(master))
    done = []
    for sid, sym in syms:
        m = data.load(sym)
        if m is None or m.n < 260:
            continue
        keep = np.array([d in mpos for d in m.dates])
        if not keep.all():
            m = data.DayMatrix(m.symbol, m.dates[keep], m.o[keep], m.h[keep], m.l[keep], m.c[keep], m.v[keep], m.start[keep])
        f = sim_features(m)
        sec = sector_index(sym, industries.get(sym, ""))
        ctx = {"nifty": _rows_for(N, m.dates)}
        if sec in sectors:
            ctx["sector"] = _rows_for(sectors[sec], m.dates)
        if sec in ranks:
            day_rows = np.array([mpos[d] for d in m.dates])
            ctx["sector_rank"] = ranks[sec][day_rows]
        ctx["beta"] = _beta(m, N)
        X = F.build(m, f, ctx)
        liquid = (np.nan_to_num(X.lv.turn20) >= MIN_TURNOVER) & (np.nan_to_num(f.prev_close) >= MIN_PRICE) & f.valid
        day_index = np.array([mpos[d] for d in m.dates], np.int16)
        train = m.dates <= TRAIN_END
        recs, a = E.process_stock(m, f, X, sid, day_index, train, costs, liquid)
        agg += a
        kept.append(recs)
        if len(kept) >= 4:                       # reduce as we go to keep memory flat
            kept = [E.keep_earliest(np.concatenate(kept))]
        up930[day_index] += (m.c[:, 2] > m.o[:, 0])
        upclose[day_index] += (m.c[:, -1] > np.nan_to_num(f.prev_close, nan=np.inf))
        cnt[day_index] += 1
        done.append(sym)
    recs = E.keep_earliest(np.concatenate(kept)) if kept else np.empty(0, dtype=E.REC)
    return recs, agg, up930, upclose, cnt, done


def build(log=print, workers=8, limit=None):
    import multiprocessing as mp
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    N, master, sectors, ranks = context()
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    if limit:
        syms = syms[:limit]
    _G.update(N=N, master=master, sectors=sectors, ranks=ranks, industries=industries, costs=Settings().get("costs"))
    ids = list(enumerate(syms))
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    log(f"  library: {len(E.LIBRARY)} strategies x {len(E.FILTERS)} filters x {len(E.EXITS)} exits = {E.NV:,} variants; "
        f"{len(syms)} stocks in {len(chunks)} chunks; {len(sectors)} sector indices")
    kept = np.empty(0, dtype=E.REC)
    agg = np.zeros((E.NV, 2, 5))
    up930, upclose, cnt = (np.zeros(len(master)) for _ in range(3))
    done = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (r, a, u9, uc, c, d) in enumerate(pool.imap_unordered(_chunk, chunks), 1):
            kept = E.keep_earliest(np.concatenate([kept, r]))
            agg += a
            up930 += u9
            upclose += uc
            cnt += c
            done += d
            if k % 6 == 0 or k == len(chunks):
                log(f"  {k}/{len(chunks)} chunks, {len(done)} stocks, {len(kept):,} kept trades ({time.time() - t0:.0f}s)")
    np.savez_compressed(OUT / "trades.npz", recs=kept)
    np.save(OUT / "agg.npy", agg)
    np.savez_compressed(OUT / "breadth.npz", up930=up930, upclose=upclose, cnt=cnt)
    meta = {"symbols": syms, "dates": [str(d) for d in master], "train_end": str(TRAIN_END),
            "strategies": [{"key": s.key, "name": s.name, "family": s.family, "source": s.source, "rule": s.rule}
                           for s in E.LIBRARY],
            "filters": E.FILTERS, "exits": E.EXITS, "stocks_done": len(done), "seconds": round(time.time() - t0),
            "built_at": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / "meta.json").write_text(json.dumps(meta))
    log(f"  saved {len(kept):,} trades for {E.NV:,} variants ({time.time() - t0:.0f}s)")
    return meta
