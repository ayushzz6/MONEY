"""Intraday option prices from the index's 5-minute bars and the previous close's real IV smile.

On trading day d a contract is priced with what was known: the strikes, forward / index ratio and
implied volatility at each strike from day d-1's closing chain. Intraday:
- "fixed": IV stays at the previous close's value for that strike
- "vix": that IV x (India VIX now / India VIX at the previous close)
- "vix_sticky_delta": as "vix", but the strike's IV is read at the same moneyness as yesterday
the forward moves with the index, and time runs out on the trading clock. `validate()` measures how
far each mode lands from the real open, high, low and close of each contract that day.
"""
import json
import time

import numpy as np

from core.config.settings import DATA_DIR
from research import data
from research.opt import chain as C
from research.opt import features as FT
from research.opt import model as M

RES = DATA_DIR / "research" / "results"
IV_MODES = ("fixed", "vix", "vix_sticky_delta")
SLOTS = np.arange(75)


class Market:
    def __init__(self):
        n = data.load("NIFTY")
        self.dates = n.dates
        D = len(self.dates)
        self.idx = {}
        for u, name in enumerate(C.UNDERLYINGS):
            m = data.load(name)
            arr = {k: np.full((D, 75), np.nan) for k in "ohlc"}
            pos = np.searchsorted(self.dates, m.dates)
            ok = (pos < D) & (self.dates[np.minimum(pos, D - 1)] == m.dates)
            for k in "ohlc":
                arr[k][pos[ok]] = getattr(m, k)[ok]
            self.idx[u] = arr
        v = data.load("INDIAVIX")
        vix = np.full((D, 75), np.nan)
        pos = np.searchsorted(self.dates, v.dates)
        ok = (pos < D) & (self.dates[np.minimum(pos, D - 1)] == v.dates)
        vix[pos[ok]] = v.c[ok]
        prev_close = np.concatenate([[np.nan], vix[:-1, -1]])
        ratio = vix / prev_close[:, None]
        ratio[~np.isfinite(ratio)] = 1.0
        self.vix_ratio_c = ratio                                  # at each bar's close
        self.vix_ratio_o = np.concatenate([np.ones((D, 1)), ratio[:, :-1]], axis=1)   # known at each bar's open
        self.f, self.smiles = FT.load()
        self.expiries = {}
        for (d, u, e) in self.smiles:
            self.expiries.setdefault((d, u), []).append(e)
        for k in self.expiries:
            self.expiries[k].sort()

    def expiry_for(self, d, u, skip_expiry_day=False):
        """Nearest expiry still alive on trading day d, from the previous close's chain."""
        for e in self.expiries.get((d - 1, u), []):
            if e > d or (e == d and not skip_expiry_day):
                return e
        return None

    def smile(self, d, u, e):
        return self.smiles.get((d - 1, u, e))

    def base_iv(self, d, u, e, K):
        sm = self.smiles.get((d - 1, u, e))
        if sm is None:
            return np.nan
        ok = np.isfinite(sm["iv"])
        if ok.sum() < 2:
            return np.nan
        return float(np.interp(K, sm["strikes"][ok], sm["iv"][ok]))

    def pick_strike(self, d, u, e, S, cp, target_delta, slot):
        """Strike (traded at the previous close) whose delta at the signal is nearest the target."""
        sm = self.smiles.get((d - 1, u, e))
        if sm is None:
            return None
        ok = sm["traded"] & np.isfinite(sm["iv"])
        ks = sm["strikes"][ok]
        if ks.size < 3:
            return None
        F = S * sm["fwd_ratio"]
        T = M.t_years(e - d, 5 * (slot + 1))
        sig = sm["iv"][ok] * self.vix_ratio_c[d, slot]
        dl = np.abs(M.delta(F, ks, T, sig, cp))
        return float(ks[np.argmin(np.abs(dl - target_delta))])

    def paths(self, d, u, e, K, cp, mode="vix"):
        """(N, 75) option open / high / low / close for N contracts (arrays d, e, K, cp; one underlying)."""
        d, e, K, cp = (np.asarray(x) for x in (d, e, K, cp))
        N = len(d)
        ix = self.idx[u]
        o, h, l, c = (ix[k][d] for k in "ohlc")
        fr = np.array([self.smiles[(dd - 1, u, ee)]["fwd_ratio"] for dd, ee in zip(d, e)])[:, None]
        left = (e - d)[:, None]
        T_o = M.t_years(left, 5 * SLOTS[None, :])
        T_c = M.t_years(left, 5 * (SLOTS[None, :] + 1))
        cpc = cp[:, None].astype(float)
        if mode == "vix_sticky_delta":
            S0 = self.f["spot"][d - 1, u][:, None]
            sig_o = np.empty((N, 75))
            sig_c = np.empty((N, 75))
            for i in range(N):
                sm = self.smiles[(d[i] - 1, u, e[i])]
                ok = np.isfinite(sm["iv"])
                ks, iv = sm["strikes"][ok], sm["iv"][ok]
                sig_o[i] = np.interp(K[i] * S0[i] / o[i], ks, iv)
                sig_c[i] = np.interp(K[i] * S0[i] / c[i], ks, iv)
            sig_o *= self.vix_ratio_o[d]
            sig_c *= self.vix_ratio_c[d]
        else:
            base = np.array([self.base_iv(dd, u, ee, kk) for dd, ee, kk in zip(d, e, K)])[:, None]
            if mode == "fixed":
                sig_o = np.broadcast_to(base, (N, 75))
                sig_c = sig_o
            else:
                sig_o = base * self.vix_ratio_o[d]
                sig_c = base * self.vix_ratio_c[d]
        Kc = K[:, None].astype(float)
        hi_S = np.where(cpc > 0, h, l)
        lo_S = np.where(cpc > 0, l, h)
        out = {"o": M.black(o * fr, Kc, T_o, sig_o, cpc), "h": M.black(hi_S * fr, Kc, T_c, sig_c, cpc),
               "l": M.black(lo_S * fr, Kc, T_c, sig_c, cpc), "c": M.black(c * fr, Kc, T_c, sig_c, cpc),
               "sig_c": np.asarray(sig_c), "T_c": T_c, "fr": fr}
        return out


def _real_rows(day):
    p = C.OUT / f"{day}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in C.FIELDS}


def validate(mk=None, log=print):
    """Simulated vs real open / high / low / close of front-expiry options at 5 strikes around the
    index, calls and puts, every session; by IV mode, days to expiry, moneyness and year."""
    t0 = time.time()
    mk = mk or Market()
    D = len(mk.dates)
    rows = []
    for d in range(1, D):
        real = _real_rows(str(mk.dates[d]))
        if real is None:
            continue
        for u in (0, 1):
            e = mk.expiry_for(d, u)
            if e is None or e >= D:
                continue
            sm = mk.smile(d, u, e)
            F0 = mk.f["spot"][d - 1, u] * sm["fwd_ratio"]
            ks = sm["strikes"]
            exp_date = mk.dates[e]
            for mny in (-0.02, -0.01, 0.0, 0.01, 0.02):
                K = float(ks[np.argmin(np.abs(ks - F0 * (1 + mny)))])
                for cp in (1, -1):
                    m = (real["und"] == u) & (real["expiry"] == exp_date) & (real["strike"] == K) & (real["cp"] == cp)
                    if not m.any() or real["qty"][m][0] <= 0 or real["close"][m][0] < 5 or not np.isfinite(mk.base_iv(d, u, e, K)):
                        continue
                    rows.append((d, u, e, K, cp, e - d, mny * cp, int(str(mk.dates[d])[:4]),
                                 *(float(real[x][m][0]) for x in ("open", "high", "low", "close"))))
        if d % 300 == 0:
            log(f"  validation sample: {d}/{D} sessions, {len(rows):,} contract-days ({time.time() - t0:.0f}s)")
    R_ = np.array(rows, dtype=float)
    d, u, e, K, cp = (R_[:, i] for i in range(5))
    left, mny, year = R_[:, 5], R_[:, 6], R_[:, 7]
    real = R_[:, 8:12]
    out = {"contract_days": len(rows), "modes": {}}
    for mode in IV_MODES:
        sim = np.full((len(rows), 4), np.nan)
        for uu in (0, 1):
            sel = np.flatnonzero(u == uu)
            for i in range(0, sel.size, 4000):
                part = sel[i:i + 4000]
                p = mk.paths(d[part].astype(int), uu, e[part].astype(int), K[part], cp[part].astype(int), mode)
                sim[part, 0] = p["o"][:, 0]
                sim[part, 1] = np.nanmax(p["h"], axis=1)
                sim[part, 2] = np.nanmin(p["l"], axis=1)
                sim[part, 3] = p["c"][:, -1]
        err = np.log(np.maximum(sim, 0.05) / np.maximum(real, 0.05))
        res = {}
        groups = (("all", np.ones(len(rows), bool)), ("NIFTY", u == 0), ("BANKNIFTY", u == 1),
                  ("expiry day", left == 0), ("1-2 days left", (left >= 1) & (left <= 2)), ("3+ days left", left >= 3),
                  ("at the money", mny == 0), ("out of the money", mny > 0), ("in the money", mny < 0),
                  ("2021-24", year <= 2024), ("2025-26", year >= 2025))
        for name, sel in groups:
            if not sel.any():
                continue
            res[name] = {"n": int(sel.sum())}
            for j, k in enumerate(("open", "high", "low", "close")):
                x = err[sel, j]
                x = x[np.isfinite(x)]
                res[name][k] = {"median_abs_pct": round(float(np.median(np.abs(np.expm1(x))) * 100), 1),
                                "bias_pct": round(float(np.expm1(np.median(x)) * 100), 1)}
        out["modes"][mode] = res
        log(f"  {mode}: close error median {res['all']['close']['median_abs_pct']}% (bias {res['all']['close']['bias_pct']}%), "
            f"high {res['all']['high']['median_abs_pct']}%, low {res['all']['low']['median_abs_pct']}%")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "opt_validation.json").write_text(json.dumps(out, indent=1))
    log(f"  validation saved ({time.time() - t0:.0f}s)")
    return out
