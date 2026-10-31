"""Option-chain features and volatility smiles from the real daily chains (known after each close).

Per session and underlying (NIFTY 0, BANKNIFTY 1):
- front expiry (first with at least one session left), sessions to it, lot size
- ATM IV of the front expiry and of the expiry nearest 20 sessions out ("IV30"), its rank against the
  previous 250 sessions, 20-day realized volatility of the index, IV minus realized
- skew: IV 2% below the forward minus IV 2% above
- implied move: ATM straddle / index
- put/call open-interest ratio (front expiry and all), and its change
- max pain, the largest call-OI strike above the index ("call wall") and put-OI strike below ("put wall")
- where today's new call / put open interest went

Smiles (for pricing intraday): per session, underlying and expiry: forward / index, strikes, OTM IV,
whether the strike traded, and the lot size.
"""
import pickle
import time

import numpy as np

from core.config.settings import DATA_DIR
from research import data
from research.opt import chain as C
from research.opt import model as M

OUT = DATA_DIR / "research" / "opt"


def _interp_iv(k, iv, x):
    ok = np.isfinite(iv)
    if ok.sum() < 2:
        return np.nan
    k, iv = k[ok], iv[ok]
    o = np.argsort(k)
    return float(np.interp(x, k[o], iv[o], left=np.nan, right=np.nan))


def build(log=print):
    t0 = time.time()
    a = C.load_all()
    n = data.load("NIFTY")
    b = data.load("BANKNIFTY")
    dates = n.dates
    D = len(dates)
    spot = np.full((D, 2), np.nan)
    spot[:, 0] = n.c[:, -1]
    bpos = np.searchsorted(dates, b.dates)
    ok = (bpos < D) & (dates[np.minimum(bpos, D - 1)] == b.dates)
    spot[bpos[ok], 1] = b.c[ok, -1]
    di = np.searchsorted(dates, a["day"])
    ei = np.searchsorted(dates, a["expiry"])
    beyond = a["expiry"] > dates[-1]
    ei = np.where(beyond, D - 1 + np.busday_count(dates[-1], a["expiry"]), ei)
    left = ei - di                                            # sessions from this close to expiry day
    T = M.t_years(left, M.MIN_PER_DAY)                        # at the close
    S = spot[di, a["und"]]
    log(f"  chain rows: {len(di):,}; sessions {D}")

    feats = {k: np.full((D, 2), np.nan) for k in (
        "front_left", "lot", "atm_iv", "iv30", "skew", "move", "pcr_front", "pcr_all", "max_pain", "call_wall", "put_wall",
        "call_add", "put_add", "fwd_ratio", "straddle")}
    smiles = {}
    order = np.lexsort((a["strike"], a["expiry"], a["und"], di))
    di_o = di[order]
    starts = np.searchsorted(di_o, np.arange(D))
    ends = np.searchsorted(di_o, np.arange(D), side="right")
    for d in range(D):
        rows = order[starts[d]:ends[d]]
        if rows.size == 0:
            continue
        for u in (0, 1):
            ru = rows[a["und"][rows] == u]
            s = spot[d, u]
            if ru.size == 0 or not np.isfinite(s):
                continue
            exps = np.unique(ei[ru])
            info = []
            for e in exps:
                re_ = ru[ei[ru] == e]
                lft = int(left[re_[0]])
                if lft < 1:
                    continue                                   # expires today: no IV at the close
                k = a["strike"][re_]
                cp = a["cp"][re_]
                px = a["close"][re_]
                traded = a["qty"][re_] > 0
                tt = float(T[re_[0]])
                # forward from put-call parity at the 3 traded strikes nearest the index
                calls = {kk: p for kk, p, c_, tr in zip(k, px, cp, traded) if c_ > 0 and tr}
                puts = {kk: p for kk, p, c_, tr in zip(k, px, cp, traded) if c_ < 0 and tr}
                both = np.array(sorted(set(calls) & set(puts), key=lambda kk: abs(kk - s))[:3])
                if both.size == 0:
                    continue
                F = float(np.median([kk + np.exp(M.R * tt) * (calls[kk] - puts[kk]) for kk in both]))
                iv = M.implied_vol(px, F, k, tt, cp)
                iv = np.where(traded, iv, np.nan)
                otm = ((cp > 0) & (k >= F)) | ((cp < 0) & (k < F))
                ks = np.unique(k)
                smile = np.array([np.nanmean(np.where((k == kk) & otm, iv, np.nan)) if ((k == kk) & otm).any() else np.nan for kk in ks])
                info.append((int(e), lft, F, ks, smile, re_))
                lot = np.nanmedian(a["lot"][re_])
                smiles[(d, u, int(e))] = {"left": lft, "fwd_ratio": F / s, "strikes": ks, "iv": smile, "lot": lot,
                                          "traded": np.array([bool((traded & (k == kk)).any()) for kk in ks])}
            if not info:
                continue
            e, lft, F, ks, smile, re_ = info[0]                # front expiry
            f = feats
            f["front_left"][d, u] = lft
            f["lot"][d, u] = np.nanmedian(a["lot"][re_])
            f["fwd_ratio"][d, u] = F / s
            f["atm_iv"][d, u] = _interp_iv(ks, smile, F)
            f["skew"][d, u] = _interp_iv(ks, smile, F * 0.98) - _interp_iv(ks, smile, F * 1.02)
            near = min(info, key=lambda x: abs(x[1] - 20) if x[1] >= 5 else 999)
            f["iv30"][d, u] = _interp_iv(near[3], near[4], near[2])
            k, cp, px, oi, chg = (a[x][re_] for x in ("strike", "cp", "close", "oi", "chg_oi"))
            katm = ks[np.argmin(np.abs(ks - F))]
            cpx, ppx = px[(k == katm) & (cp > 0)], px[(k == katm) & (cp < 0)]
            if cpx.size and ppx.size:
                f["straddle"][d, u] = cpx[0] + ppx[0]
                f["move"][d, u] = (cpx[0] + ppx[0]) / s
            coi, poi = oi[cp > 0].sum(), oi[cp < 0].sum()
            f["pcr_front"][d, u] = poi / coi if coi > 0 else np.nan
            allr = ru
            coa, poa = a["oi"][allr][a["cp"][allr] > 0].sum(), a["oi"][allr][a["cp"][allr] < 0].sum()
            f["pcr_all"][d, u] = poa / coa if coa > 0 else np.nan
            strikes = np.unique(k)
            coi_k = np.array([oi[(k == kk) & (cp > 0)].sum() for kk in strikes])
            poi_k = np.array([oi[(k == kk) & (cp < 0)].sum() for kk in strikes])
            pain = [np.sum(coi_k * np.maximum(x - strikes, 0)) + np.sum(poi_k * np.maximum(strikes - x, 0)) for x in strikes]
            f["max_pain"][d, u] = strikes[int(np.argmin(pain))]
            above, below = strikes > s, strikes < s
            if above.any():
                f["call_wall"][d, u] = strikes[above][np.argmax(coi_k[above])]
            if below.any():
                f["put_wall"][d, u] = strikes[below][np.argmax(poi_k[below])]
            cadd = np.array([chg[(k == kk) & (cp > 0)].sum() for kk in strikes])
            padd = np.array([chg[(k == kk) & (cp < 0)].sum() for kk in strikes])
            f["call_add"][d, u] = strikes[int(np.argmax(cadd))]
            f["put_add"][d, u] = strikes[int(np.argmax(padd))]
        if d % 250 == 0:
            log(f"  features: {d}/{D} sessions ({time.time() - t0:.0f}s)")
    # rank, realized vol, changes (all from the past only)
    f = feats
    f["spot"] = spot
    ret = np.vstack([np.full((1, 2), np.nan), np.diff(np.log(spot), axis=0)])
    rv = np.full((D, 2), np.nan)
    rank = np.full((D, 2), np.nan)
    for d in range(D):
        lo = max(0, d - 19)
        if d >= 10:
            rv[d] = np.nanstd(ret[lo:d + 1], axis=0) * np.sqrt(252)
        past = f["iv30"][max(0, d - 250):d]
        for u in (0, 1):
            p = past[:, u][np.isfinite(past[:, u])]
            if p.size >= 60 and np.isfinite(f["iv30"][d, u]):
                rank[d, u] = (p < f["iv30"][d, u]).mean()
    f["rv20"] = rv
    f["iv_rank"] = rank
    f["iv_minus_rv"] = f["iv30"] - rv
    f["pcr_change"] = np.vstack([np.full((1, 2), np.nan), np.diff(f["pcr_front"], axis=0)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "features.npz", dates=dates, **f)
    with open(OUT / "smiles.pkl", "wb") as fh:
        pickle.dump(smiles, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  saved option features and {len(smiles):,} smiles ({time.time() - t0:.0f}s)")
    return f, smiles


def load():
    z = np.load(OUT / "features.npz")
    f = {k: z[k] for k in z.files}
    with open(OUT / "smiles.pkl", "rb") as fh:
        smiles = pickle.load(fh)
    return f, smiles
