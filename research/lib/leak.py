"""No-lookahead test for the strategy library.

For random (stock, day, bar) cut points: every bar after the cut is replaced by a random walk (the stock, the
NIFTY and its sector alike) and later days are dropped. All features, all strategies and all filters are
rebuilt. Every signal and stop up to the cut, on that day and the day before, must come out the same; any rule
that peeks at a later bar fails.
"""
import numpy as np

from research import data
from research.lib import build as B
from research.lib import engine as E
from research.lib import features as F
from research.lib.sectors import sector_index
from research.sim import features as sim_features


def _cut_matrix(m, d, c, rng):
    keep = slice(0, d + 1)
    o, h, l, cl, v = (a[keep].copy() for a in (m.o, m.h, m.l, m.c, m.v))
    last = cl[d, c]
    n = data.SLOTS - c - 1
    if n > 0:
        walk = last * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        cl[d, c + 1:] = walk
        o[d, c + 1:] = np.r_[last, walk[:-1]]
        h[d, c + 1:] = np.maximum(o[d, c + 1:], walk) * (1 + rng.uniform(0, 0.003, n))
        l[d, c + 1:] = np.minimum(o[d, c + 1:], walk) * (1 - rng.uniform(0, 0.003, n))
        v[d, c + 1:] = v[d, :c + 1].mean() * rng.uniform(0.2, 8, n)
    return data.DayMatrix(m.symbol, m.dates[keep], o, h, l, cl, v, m.start[keep])


def _ctx(m, sym, N, sectors, ranks, industries, mpos, cut=None, rng=None):
    sec = sector_index(sym, industries.get(sym, ""))
    Nm = N
    Sm = sectors.get(sec)
    if cut is not None:
        d_master = mpos[m.dates[-1]]
        Nm = _cut_matrix(N, d_master, cut, rng)
        if Sm is not None:
            spos = {dd: i for i, dd in enumerate(Sm.dates)}
            if m.dates[-1] in spos:
                Sm = _cut_matrix(Sm, spos[m.dates[-1]], cut, rng)
    ctx = {"nifty": B._rows_for(Nm, m.dates)}
    if Sm is not None:
        ctx["sector"] = B._rows_for(Sm, m.dates)
    if sec in ranks:
        rk = ranks[sec][np.array([mpos[dd] for dd in m.dates])].copy()
        if cut is not None:
            rk[-1, cut + 1:] = 99            # the ranks after the cut are not known yet either
        ctx["sector_rank"] = rk
    ctx["beta"] = B._beta(m, N)
    return ctx


def _signals(X, library=None):
    out = {}
    fm = E.filter_masks(X)
    for s in (library or E.LIBRARY):
        L, Sx, stL, stS = s.fn(X)
        out[s.key] = (np.asarray(L, bool), np.asarray(Sx, bool), np.asarray(stL, float) * np.ones(X.C.shape),
                      np.asarray(stS, float) * np.ones(X.C.shape))
    for k in ("volume", "sector", "market", "htf", "confluence"):
        out["filter:" + k] = (fm[1][k], fm[-1][k], np.zeros(X.C.shape), np.zeros(X.C.shape))
    return out


def run(samples=12, seed=7, log=print, library=None):
    rng = np.random.default_rng(seed)
    N, master, sectors, ranks = B.context()
    mpos = {d: i for i, d in enumerate(master)}
    industries = data.nifty500()
    syms = [s for s in industries if (data.HIST_DIR / f"{s}.npz").exists()]
    bad = {}
    checked = 0
    for k in range(samples):
        sym = syms[rng.integers(len(syms))]
        m = data.load(sym)
        if m is None or m.n < 300:
            continue
        keep = np.array([d in mpos for d in m.dates])
        m = data.DayMatrix(m.symbol, m.dates[keep], m.o[keep], m.h[keep], m.l[keep], m.c[keep], m.v[keep], m.start[keep])
        d = int(rng.integers(260, m.n))
        c = int(rng.integers(1, 66))
        full_m = data.DayMatrix(m.symbol, m.dates[:d + 1], m.o[:d + 1], m.h[:d + 1], m.l[:d + 1], m.c[:d + 1],
                                m.v[:d + 1], m.start[:d + 1])
        Xf = F.build(full_m, sim_features(full_m), _ctx(full_m, sym, N, sectors, ranks, industries, mpos))
        cut_m = _cut_matrix(m, d, c, rng)
        Xc = F.build(cut_m, sim_features(cut_m), _ctx(cut_m, sym, N, sectors, ranks, industries, mpos, cut=c, rng=rng))
        a, b = _signals(Xf, library), _signals(Xc, library)
        for key in a:
            for j, (u, w) in enumerate(zip(a[key], b[key])):
                if j < 2:
                    same = np.array_equal(u[d, :c + 1], w[d, :c + 1]) and np.array_equal(u[d - 1], w[d - 1])
                else:
                    sig = a[key][j - 2][d, :c + 1]
                    uu, ww = u[d, :c + 1][sig], w[d, :c + 1][sig]
                    same = np.allclose(uu, ww, equal_nan=True, rtol=1e-9, atol=1e-6)
                if not same:
                    bad.setdefault(key, []).append(f"{sym} {m.dates[d]} cut at bar {c}")
        checked += 1
        log(f"  leak test {checked}: {sym} {m.dates[d]} cut after bar {c}: {len(bad)} rules failing so far")
    return {"cuts": checked, "rules": len(library or E.LIBRARY) + 5, "failing": bad}
