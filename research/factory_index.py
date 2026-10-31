"""Strategy factory: the same price setups on NIFTY and BANKNIFTY, traded two ways:

- index futures, 1 lot (NIFTY 65 / BANKNIFTY 30): needs about Rs 1.7-2 lakh (NIFTY) margin,
  more than a Rs 1 lakh account, so these are shown per lot for reference;
- buying 1 lot of the ATM option (next weekly for NIFTY, monthly for BANKNIFTY, never same-day
  expiry), priced with the research option model: fits a Rs 1 lakh account.
Volume setups and filters are skipped (an index has no volume). Selection on TRAIN only.
"""
import itertools
import time

import numpy as np

from research import options as op
from research.costs import futures_order_charges, option_order_charges, option_slippage
from research.factory_equity import (DEFAULT_EXITS, EXIT_GRID, SETUPS, TRAIN_END, Exit, levels, resolve_exits,
                                     stats)
from research.sim import features
from research.strategies_fno import BANKNIFTY_IV_MULT, FnoContext

INDEX_SETUPS = [k for k in SETUPS if not k.startswith("Volume spike")]
INDEX_FILTERS = ["none", "trend", "narrow CPR", "NR7", "VIX high", "VIX low", "not expiry", "expiry day"]
FUT_SLIP_PTS = {"NIFTY": 1.0, "BANKNIFTY": 3.0}


def _mask(name, f, lv, vx_all, rows, sides, vix_t, is_exp):
    if name == "none":
        return np.ones(len(rows), bool)
    if name == "trend":
        return np.where(sides > 0, f.trend_up[rows], f.trend_down[rows])
    if name == "narrow CPR":
        return lv.cpr_w[rows] < 0.12
    if name == "NR7":
        return lv.nr7[rows]
    vx = vx_all[rows]
    if name == "VIX high":
        return vx >= vix_t[1]
    if name == "VIX low":
        return vx < vix_t[0]
    if name == "not expiry":
        return ~is_exp[rows]
    if name == "expiry day":
        return is_exp[rows]
    raise ValueError(name)


class IndexSearch:
    def __init__(self, nifty, banknifty, vix):
        self.ctx = FnoContext(nifty, banknifty, vix)
        vpos = {d: i for i, d in enumerate(vix.dates)}
        vix_open = {d: vix.o[i, 0] for d, i in vpos.items()}
        tr = [v for d, v in vix_open.items() if d <= TRAIN_END]
        self.vix_t = np.nanpercentile(tr, [33.3, 66.7])
        exp = set(self.ctx.weekly.expiries)
        self.inst = {}
        for name, M in (("NIFTY", nifty), ("BANKNIFTY", banknifty)):
            f = features(M)
            f.valid = ~np.isnan(f.sma50)
            vx = np.array([vix_open.get(d, np.nan) for d in M.dates])
            isx = np.array([d.astype(object) in exp for d in M.dates])
            self.inst[name] = (M, f, levels(M, f), vx, isx)
        self.dates = nifty.dates

    def evaluate(self, inst, setup, filt, ex: Exit):
        M, f, lv, vx, isx = self.inst[inst]
        rows, slots, sides, entry, sstop, touch = SETUPS[setup](M, f, lv)
        if len(rows) == 0:
            return None
        keep = f.valid[rows] & _mask(filt, f, lv, vx, rows, sides, self.vix_t, isx)
        rows, slots, sides, entry, sstop = rows[keep], slots[keep], sides[keep], entry[keep], sstop[keep]
        ok, rows, slots, sides, entry, stop, risk, xpx, xs, out = resolve_exits(
            M, f, rows, slots, sides, entry, sstop, ex, min_risk_pct=0.08, max_risk_pct=2.0)
        if len(rows) == 0:
            return None
        lot = self.ctx.lots[inst]
        slip = FUT_SLIP_PTS[inst]
        fut = (xpx - entry) * sides * lot - 2 * slip * lot - np.array(
            [futures_order_charges(e * lot, "buy") + futures_order_charges(x * lot, "sell") for e, x in zip(entry, xpx)])
        clock = self.ctx.weekly if inst == "NIFTY" else self.ctx.monthly
        step = 50 if inst == "NIFTY" else 100
        ivm = 1.0 if inst == "NIFTY" else BANKNIFTY_IV_MULT
        opt = np.empty(len(rows))
        for i, (r, s, sd, e, x, xsl) in enumerate(zip(rows, slots, sides, entry, xpx, xs)):
            day = M.dates[r].astype(object)
            expiry = clock.expiry_for(day, 1)
            K = float(op.round_strike(e, step))
            p_in = self.ctx.prem(e, K, day, s, expiry, sd > 0, clock, ivm)
            p_out = self.ctx.prem(x, K, day, xsl, expiry, sd > 0, clock, ivm)
            buy, sell = p_in + option_slippage(p_in), max(0.05, p_out - option_slippage(p_out))
            opt[i] = (sell - buy) * lot - option_order_charges(buy * lot, "buy") - option_order_charges(sell * lot, "sell")
        pos = {d: i for i, d in enumerate(self.dates)}
        di = np.array([pos.get(M.dates[r], -1) for r in rows])
        good = di >= 0
        di, fut, opt = di[good], fut[good], opt[good]
        D = len(self.dates)
        return (np.bincount(di, weights=fut, minlength=D), np.bincount(di, weights=opt, minlength=D),
                np.bincount(di, minlength=D))


def run_index_search(nifty, banknifty, vix, log=print, stage2_top=12):
    t0 = time.time()
    S = IndexSearch(nifty, banknifty, vix)
    rows = []
    for inst, setup, filt, ex in itertools.product(("NIFTY", "BANKNIFTY"), INDEX_SETUPS, INDEX_FILTERS, DEFAULT_EXITS):
        res = S.evaluate(inst, setup, filt, ex)
        if res is None:
            continue
        fut, opt, n = res
        for kind, daily in (("futures (1 lot, ~Rs 1.7-2 L margin)", fut), ("ATM option buy (1 lot)", opt)):
            rows.append(stats(daily, n, None, S.dates, {"family": f"index {kind}", "instrument": inst, "setup": setup,
                                                        "filter": filt, "exit": ex.label}))
    log(f"  index stage 1: {len(rows):,} variants ({time.time() - t0:.0f}s)")
    best = {}
    for r in sorted(rows, key=lambda r: -r["train_per_month"]):
        key = (r["family"], r["instrument"], r["setup"], r["filter"])
        if r["train_trades"] >= 120 and key not in best:
            best[key] = r
        if len(best) >= stage2_top:
            break
    stage2 = []
    for (family, inst, setup, filt) in best:
        for ex in EXIT_GRID:
            res = S.evaluate(inst, setup, filt, ex)
            if res is None:
                continue
            fut, opt, n = res
            daily = fut if family.startswith("index futures") else opt
            stage2.append(stats(daily, n, None, S.dates, {"family": family, "instrument": inst, "setup": setup,
                                                          "filter": filt, "exit": ex.label}))
    log(f"  index stage 2: {len(stage2):,} variants ({time.time() - t0:.0f}s)")
    return rows, stage2
