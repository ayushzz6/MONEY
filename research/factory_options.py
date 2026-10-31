"""Strategy factory: NIFTY weekly option-selling variants, evaluated over 5 years.

Every combination of entry time x strikes x wings x expiry choice x stop x profit target x
exit time is priced with the research option model (IV 0.9 x India VIX, trading time,
skew), then split by day filters (weekday, expiry day, days to expiry, VIX regime, gap).
Lots are sized to a Rs 1 lakh account from Angel One's real margin for these hedged structures
(about 5.9% to 6.5% of notional a lot), plus the 2% extra exposure margin SEBI added on
expiry-day short options from 2024-11-20: Rs 1 lakh carries 1 lot, and none on an expiry day.

Selection is done on the TRAIN period only (to 2024-12-31); the TEST period (2025-01-01 on)
is reported for whatever the train period picked.
"""
import datetime as dt
import itertools
import time
from typing import Dict, List

import numpy as np

from research import options as op
from research.data import SLOTS, DayMatrix
from research.sim import FLAT_SLOT, slot_of_close
from research.strategies_fno import BASE_IV_SCALE, SL_SLIP_MULT

LOT = 65
CAPITAL = 100000.0
MARGIN_USE = 1.0
MAX_LOTS = 10
# Angel One margin API, 2026-10-24, 1 lot of 65, NIFTY 22,600: straddle + 1.5% wings Rs 85,916 (5.85% of
# notional), + 2.5% wings Rs 95,829 (6.52%); strangles a little less. Exposure margin dominates, so margin is
# modeled as a share of notional: 4.85% + 0.67 x wing distance.
MARGIN_BASE, MARGIN_PER_WING = 0.0485, 0.67
TRAIN_END = np.datetime64("2024-12-31")
ELM_START = np.datetime64("2024-11-20")

ENTRY_TIMES = ["09:20", "09:30", "09:45", "10:00", "10:30", "11:00", "12:00", "13:00"]
STRUCTS = {"straddle": 0.0, "strangle0.5": 0.005, "strangle1.0": 0.010, "strangle1.5": 0.015}
WINGS = {"wing1.5": 0.015, "wing2.5": 0.025}
EXPIRY_MODES = ["current", "skip_expiry_day"]
STOPS = [("none", 0)] + [("leg", x) for x in (0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)] + \
        [("combined", x) for x in (0.2, 0.3, 0.5, 0.75)]
TARGETS = [None, 0.3, 0.5]
EXITS = ["14:30", "15:15"]


def _slip(p):
    return np.maximum(0.5, 0.005 * p)


def _charges(prem_value, side, lots):
    """Per-order option charges with the flat Rs 20 brokerage once per order (lots in one order)."""
    v = prem_value
    txn, sebi = v * 0.03503 / 100, v * 0.0001 / 100
    stt = v * 0.1 / 100 if side == "sell" else 0.0
    stamp = v * 0.003 / 100 if side == "buy" else 0.0
    return 20.0 + txn + sebi + stt + stamp + 0.18 * (20.0 + txn + sebi)


class OptionFactory:
    def __init__(self, nifty: DayMatrix, vix: DayMatrix, iv_scale: float = BASE_IV_SCALE):
        self.N, self.V = nifty, vix
        self.iv_scale = iv_scale
        self.days = [d.astype(object) for d in nifty.dates]
        self.D = len(self.days)
        self.clock = op.ExpiryClock(self.days, op.nifty_weekly_expiries(self.days))
        vpos = {d: i for i, d in enumerate(vix.dates)}
        iv = np.full((self.D, SLOTS), np.nan)
        for i, d in enumerate(nifty.dates):
            j = vpos.get(d)
            if j is not None:
                iv[i] = vix.c[j]
        # fill gaps with the day's mean, then the previous day's
        for i in range(self.D):
            row = iv[i]
            if np.isnan(row).all():
                iv[i] = iv[i - 1] if i else 15.0
            else:
                row[np.isnan(row)] = np.nanmean(row)
        self.atm_iv = iv / 100.0
        self.expiry_set = set(self.clock.expiries)
        self.is_expiry = np.array([d in self.expiry_set for d in self.days])
        self.minutes = op.slot_minute(np.arange(SLOTS)).astype(float)
        self.weekday = np.array([d.weekday() for d in self.days])
        self.vix_open = self.atm_iv[:, 0] * 100
        prev = np.concatenate([[np.nan], nifty.c[:-1, -1]])
        self.gap = 100 * (nifty.o[:, 0] / prev - 1)
        self.train = nifty.dates <= TRAIN_END

    def _T(self, expiries):
        full = np.array([self.clock.sessions_between(d, e) for d, e in zip(self.days, expiries)], float)
        left = np.maximum(0.0, op.CLOSE_MIN - self.minutes)[None, :]
        return np.maximum(5.0, left + op.MIN_PER_DAY * full[:, None]) / op.YEAR_MIN

    def _price(self, S, K, T, call):
        iv = op.iv_at(K, S, self.atm_iv, self.iv_scale)
        return op.bs(S, K, T, iv, call)

    def precompute(self, entry: str, mode: str):
        e = slot_of_close(entry)
        if mode == "current":
            exps = [self.clock.expiry_for(d, 0) for d in self.days]
        else:
            exps = [self.clock.expiry_for(d, 1) for d in self.days]
        T = self._T(exps)
        N = self.N
        S0 = N.c[:, e]
        out = {"e": e, "S0": S0, "dte": np.array([self.clock.sessions_between(d, x) for d, x in zip(self.days, exps)])}
        for sname, d in STRUCTS.items():
            if d == 0:
                kc = kp = op.round_strike(S0, 50)
            else:
                kc = np.ceil(S0 * (1 + d) / 50) * 50
                kp = np.floor(S0 * (1 - d) / 50) * 50
            KC, KP = kc[:, None], kp[:, None]
            leg = {
                "kc": kc, "kp": kp,
                "ce0": self._price(S0[:, None], KC, T[:, e:e + 1], True)[:, 0],
                "pe0": self._price(S0[:, None], KP, T[:, e:e + 1], False)[:, 0],
                "ce_hi": self._price(N.h, KC, T, True), "ce_cl": self._price(N.c, KC, T, True),
                "pe_lo": self._price(N.l, KP, T, False), "pe_cl": self._price(N.c, KP, T, False),
            }
            for wname, w in WINGS.items():
                step = np.round(S0 * w / 50) * 50
                WC, WP = (kc + step)[:, None], (kp - step)[:, None]
                leg[wname] = {
                    "width": step,
                    "wc0": self._price(S0[:, None], WC, T[:, e:e + 1], True)[:, 0],
                    "wp0": self._price(S0[:, None], WP, T[:, e:e + 1], False)[:, 0],
                    "wc_cl": self._price(N.c, WC, T, True), "wp_cl": self._price(N.c, WP, T, False),
                }
            out[sname] = leg
        return out

    def evaluate(self, pre, sname, wname, stop, target, exit_t):
        """Per-day net P&L for ONE lot (and the lots a Rs 1 L account can carry)."""
        e = pre["e"]
        X = slot_of_close(exit_t) if exit_t != "15:15" else FLAT_SLOT
        L = pre[sname]
        W = L[wname]
        D = self.D
        idx = np.arange(SLOTS)[None, :]
        live = (idx > e) & (idx <= X)
        ce0, pe0 = L["ce0"], L["pe0"]
        ce_hi, ce_cl, pe_lo, pe_cl = L["ce_hi"], L["ce_cl"], L["pe_lo"], L["pe_cl"]
        r = np.arange(D)
        big = SLOTS + 5

        def first(mask):
            a = mask.any(1)
            return np.where(a, mask.argmax(1), big)

        kind, x = stop
        ce_stop_slot = np.full(D, big)
        pe_stop_slot = np.full(D, big)
        ce_exit_px = np.zeros(D)
        pe_exit_px = np.zeros(D)
        comb_sl = np.full(D, big)
        tp = np.full(D, big)
        if kind == "leg":
            ce_stop_slot = first(live & (ce_hi >= ((1 + x) * ce0)[:, None]))
            pe_stop_slot = first(live & (pe_lo >= ((1 + x) * pe0)[:, None]))
            ce_exit_px = (1 + x) * ce0
            pe_exit_px = (1 + x) * pe0
        elif kind == "combined":
            comb_sl = first(live & ((ce_cl + pe_cl) >= ((1 + x) * (ce0 + pe0))[:, None]))
        if target is not None:
            ce_val = np.where(idx < ce_stop_slot[:, None], ce_cl, ce_exit_px[:, None])
            pe_val = np.where(idx < pe_stop_slot[:, None], pe_cl, pe_exit_px[:, None])
            tp = first(live & ((ce_val + pe_val) <= ((1 - target) * (ce0 + pe0))[:, None]))
        both = np.maximum(ce_stop_slot, pe_stop_slot)          # big unless both legs stopped
        end = np.minimum.reduce([comb_sl, tp, both, np.full(D, X)])
        # legs stopped on their own: the stop price with a worse fill; the rest close at the end slot
        ce_stopped = ce_stop_slot <= end
        pe_stopped = pe_stop_slot <= end
        sl_all = comb_sl == end                                 # a combined stop closed everything
        ce_end, pe_end = ce_cl[r, end], pe_cl[r, end]
        k_end = np.where(sl_all, SL_SLIP_MULT, 1.0)
        ce_out = np.where(ce_stopped, ce_exit_px + SL_SLIP_MULT * _slip(ce_exit_px), ce_end + k_end * _slip(ce_end))
        pe_out = np.where(pe_stopped, pe_exit_px + SL_SLIP_MULT * _slip(pe_exit_px), pe_end + k_end * _slip(pe_end))
        wc_end, wp_end = W["wc_cl"][r, end], W["wp_cl"][r, end]
        ce_in, pe_in = ce0 - _slip(ce0), pe0 - _slip(pe0)
        wc_in, wp_in = W["wc0"] + _slip(W["wc0"]), W["wp0"] + _slip(W["wp0"])
        wc_out, wp_out = np.maximum(0.05, wc_end - _slip(wc_end)), np.maximum(0.05, wp_end - _slip(wp_end))
        per_unit = (ce_in - ce_out) + (pe_in - pe_out) + (wc_out - wc_in) + (wp_out - wp_in)
        credit = ce_in + pe_in - wc_in - wp_in
        wing_pct = WINGS[wname]
        margin = pre["S0"] * LOT * (MARGIN_BASE + MARGIN_PER_WING * wing_pct)
        elm = np.where(self.is_expiry & (self.N.dates >= ELM_START) & (pre["dte"] == 0), 0.02 * pre["S0"] * LOT, 0.0)
        lots = np.clip(np.floor(MARGIN_USE * CAPITAL / (margin + elm)), 0, MAX_LOTS)
        def ch(v, side):
            return _charges(v * LOT * lots, side, lots)
        charges = (ch(ce_in, "sell") + ch(pe_in, "sell") + ch(wc_in, "buy") + ch(wp_in, "buy") +
                   ch(ce_out, "buy") + ch(pe_out, "buy") + ch(wc_out, "sell") + ch(wp_out, "sell"))
        pnl = per_unit * LOT * lots - np.where(lots > 0, charges, 0.0)
        return pnl, lots, credit * LOT

    # ---------------------------------------------------------------- day filters
    def filters(self):
        vix_t = np.nanpercentile(self.vix_open[self.train], [33.3, 66.7])
        f = {"all days": np.ones(self.D, bool)}
        f["expiry day only"] = self.is_expiry
        f["not expiry day"] = ~self.is_expiry
        for k, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
            f[nm] = self.weekday == k
        f[f"VIX low (<{vix_t[0]:.1f})"] = self.vix_open < vix_t[0]
        f[f"VIX mid"] = (self.vix_open >= vix_t[0]) & (self.vix_open < vix_t[1])
        f[f"VIX high (>={vix_t[1]:.1f})"] = self.vix_open >= vix_t[1]
        f["VIX not high"] = self.vix_open < vix_t[1]
        f["small gap (<0.5%)"] = np.abs(np.nan_to_num(self.gap)) < 0.5
        f["small gap & not expiry"] = (np.abs(np.nan_to_num(self.gap)) < 0.5) & ~self.is_expiry
        return f


def run_grid(nifty, vix, log=print) -> List[dict]:
    fac = OptionFactory(nifty, vix)
    filters = fac.filters()
    dates = nifty.dates
    month = np.array([str(d)[:7] for d in dates])
    train, test = fac.train, ~fac.train
    months_train = len(set(month[train]))
    months_test = len(set(month[test]))
    rows = []
    t0 = time.time()
    for entry in ENTRY_TIMES:
        for mode in EXPIRY_MODES:
            pre = fac.precompute(entry, mode)
            for sname, wname, stop, target, exit_t in itertools.product(STRUCTS, WINGS, STOPS, TARGETS, EXITS):
                if slot_of_close(exit_t) <= pre["e"] + 2:
                    continue
                pnl, lots, credit = fac.evaluate(pre, sname, wname, stop, target, exit_t)
                for fname, fm in filters.items():
                    if mode == "skip_expiry_day" and fname == "expiry day only":
                        continue
                    p = np.where(fm, pnl, 0.0)
                    rows.append(_stats(p, fm & (lots > 0), train, test, months_train, months_test, {
                        "family": "options_selling", "entry": entry, "expiry": mode, "structure": sname,
                        "wings": wname, "stop": f"{stop[0]} {stop[1]:g}" if stop[0] != "none" else "none",
                        "target": f"{int(target * 100)}% of credit" if target else "none", "exit": exit_t,
                        "filter": fname, "avg_lots": round(float(lots[fm].mean()), 2) if fm.any() else 0}))
        log(f"  options grid: entry {entry} done, {len(rows):,} variants ({time.time() - t0:.0f}s)")
    return rows


def _equity_stats(p):
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((peak - eq).max()) if len(p) else 0.0


def _stats(p, traded, train, test, mtr, mte, meta):
    out = dict(meta)
    for name, m, months in (("train", train, mtr), ("test", test, mte)):
        q = p[m]
        n = int((traded & m).sum())
        tq = q[(traded & m)[m]] if n else q[:0]
        out[f"{name}_net"] = round(float(q.sum()), 0)
        out[f"{name}_per_month"] = round(float(q.sum()) / months, 0)
        out[f"{name}_trades"] = n
        out[f"{name}_win"] = round(100 * float((tq > 0).mean()), 1) if n else None
        out[f"{name}_maxdd"] = round(_equity_stats(q), 0)
        out[f"{name}_worst_day"] = round(float(q.min()), 0) if len(q) else 0
    return out
