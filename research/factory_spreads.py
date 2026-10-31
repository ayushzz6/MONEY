"""Strategy factory: directional NIFTY credit spreads (theta with a view, defined risk).

At the entry time a direction signal picks the side: bullish -> sell a put spread, bearish ->
sell a call spread (short strike at the money or 0.5% / 1% out, hedge 1% or 1.5% beyond it).
Margin from Angel One (Oct 2026, 1 lot of 65 at NIFTY 22,600): Rs 47,220 for a 250-point spread,
Rs 52,000-54,000 for 350 points -> modeled as 2.2% + 0.9 x hedge width, of notional; plus SEBI's
2% expiry-day margin on a same-day-expiry short leg. Rs 1 lakh carries 1 or 2 lots.
"""
import itertools
import time

import numpy as np

import research.factory_options as fo
from research.factory_options import CAPITAL, ELM_START, LOT, MAX_LOTS, OptionFactory, _charges, _slip, _stats
from research.sim import FLAT_SLOT, slot_of_close
from research.strategies_fno import SL_SLIP_MULT

SPREAD_WINGS = {"wing1.0": 0.010, "wing1.5": 0.015}
SHORT_STRIKES = {"ATM": "straddle", "0.5% OTM": "strangle0.5", "1% OTM": "strangle1.0"}
SIGNALS = ["move since open", "side of TWAP", "gap direction", "yesterday's candle", "fade the move"]
STOPS = [("none", 0)] + [("leg", x) for x in (0.5, 1.0, 2.0)]
TARGETS = [None, 0.5]
EXITS = ["14:30", "15:15"]
MARGIN_BASE, MARGIN_PER_WING = 0.022, 0.9


def direction(N, e, signal):
    o0 = N.o[:, 0]
    ce = N.c[:, e]
    prev_c = np.r_[np.nan, N.c[:-1, -1]]
    prev_o = np.r_[np.nan, N.o[:-1, 0]]
    if signal == "move since open":
        mv = ce / o0 - 1
        return np.where(mv > 0.001, 1, np.where(mv < -0.001, -1, 0))
    if signal == "fade the move":
        mv = ce / o0 - 1
        return np.where(mv > 0.001, -1, np.where(mv < -0.001, 1, 0))
    if signal == "side of TWAP":
        tp = (N.h[:, :e + 1] + N.l[:, :e + 1] + N.c[:, :e + 1]) / 3
        tw = tp.mean(1)
        return np.where(ce > tw, 1, np.where(ce < tw, -1, 0))
    if signal == "gap direction":
        g = o0 / prev_c - 1
        return np.where(g > 0.001, 1, np.where(g < -0.001, -1, 0))
    if signal == "yesterday's candle":
        return np.where(prev_c > prev_o, 1, np.where(prev_c < prev_o, -1, 0))
    raise ValueError(signal)


def spread_side(fac: OptionFactory, pre, sname, wname, put: bool, stop, target, exit_t):
    """Per-day P&L for ONE lot of a put (put=True) or call credit spread, and the lots the account carries."""
    e = pre["e"]
    X = FLAT_SLOT if exit_t == "15:15" else slot_of_close(exit_t)
    L, W = pre[sname], pre[sname][wname]
    D = fac.D
    idx = np.arange(fo.SLOTS)[None, :]
    live = (idx > e) & (idx <= X)
    r = np.arange(D)
    big = fo.SLOTS + 5
    if put:
        s0, s_bad, s_cl, w0, w_cl = L["pe0"], L["pe_lo"], L["pe_cl"], W["wp0"], W["wp_cl"]
    else:
        s0, s_bad, s_cl, w0, w_cl = L["ce0"], L["ce_hi"], L["ce_cl"], W["wc0"], W["wc_cl"]

    def first(mask):
        a = mask.any(1)
        return np.where(a, mask.argmax(1), big)

    kind, x = stop
    sl = first(live & (s_bad >= ((1 + x) * s0)[:, None])) if kind == "leg" else np.full(D, big)
    tp = np.full(D, big)
    if target is not None:
        val = s_cl - w_cl
        tp = first(live & (val <= ((1 - target) * (s0 - w0))[:, None]))
    end = np.minimum.reduce([sl, tp, np.full(D, X)])
    stopped = sl == end
    s_in, w_in = s0 - _slip(s0), w0 + _slip(w0)
    s_end, w_end = s_cl[r, end], w_cl[r, end]
    stop_px = (1 + x) * s0
    s_out = np.where(stopped, stop_px + SL_SLIP_MULT * _slip(stop_px), s_end + _slip(s_end))
    w_out = np.maximum(0.05, w_end - _slip(w_end))
    per_unit = (s_in - s_out) + (w_out - w_in)
    margin = pre["S0"] * LOT * (MARGIN_BASE + MARGIN_PER_WING * SPREAD_WINGS[wname])
    elm = np.where(fac.is_expiry & (fac.N.dates >= ELM_START) & (pre["dte"] == 0), 0.02 * pre["S0"] * LOT, 0.0)
    lots = np.clip(np.floor(CAPITAL / (margin + elm)), 0, MAX_LOTS)
    ch = (_charges(s_in * LOT * lots, "sell", lots) + _charges(w_in * LOT * lots, "buy", lots) +
          _charges(s_out * LOT * lots, "buy", lots) + _charges(w_out * LOT * lots, "sell", lots))
    return per_unit * LOT * lots - np.where(lots > 0, ch, 0.0), lots


def run_spread_grid(nifty, vix, log=print):
    keep, fo.WINGS = fo.WINGS, SPREAD_WINGS      # precompute the hedge legs at the spread widths
    try:
        return _spread_grid(nifty, vix, log)
    finally:
        fo.WINGS = keep


def _spread_grid(nifty, vix, log):
    fac = OptionFactory(nifty, vix)
    filters = fac.filters()
    month = np.array([str(d)[:7] for d in nifty.dates])
    train, test = fac.train, ~fac.train
    mtr, mte = len(set(month[train])), len(set(month[test]))
    rows, t0 = [], time.time()
    for entry in fo.ENTRY_TIMES:
        for mode in fo.EXPIRY_MODES:
            pre = fac.precompute(entry, mode)
            sigs = {s: direction(nifty, pre["e"], s) for s in SIGNALS}
            for (kname, sname), wname, stop, target, exit_t in itertools.product(
                    SHORT_STRIKES.items(), SPREAD_WINGS, STOPS, TARGETS, EXITS):
                if slot_of_close(exit_t) <= pre["e"] + 2:
                    continue
                put_pnl, put_lots = spread_side(fac, pre, sname, wname, True, stop, target, exit_t)
                call_pnl, call_lots = spread_side(fac, pre, sname, wname, False, stop, target, exit_t)
                for sig, dirn in sigs.items():
                    pnl = np.where(dirn > 0, put_pnl, np.where(dirn < 0, call_pnl, 0.0))
                    lots = np.where(dirn > 0, put_lots, np.where(dirn < 0, call_lots, 0))
                    for fname, fm in filters.items():
                        if mode == "skip_expiry_day" and fname == "expiry day only":
                            continue
                        p = np.where(fm, pnl, 0.0)
                        rows.append(_stats(p, fm & (lots > 0) & (dirn != 0), train, test, mtr, mte, {
                            "family": "credit_spread", "entry": entry, "expiry": mode, "signal": sig,
                            "structure": sname, "short_strike": kname, "wings": wname,
                            "stop": f"{stop[0]} {stop[1]:g}" if stop[0] != "none" else "none",
                            "target": f"{int(target * 100)}% of credit" if target else "none", "exit": exit_t,
                            "filter": fname, "avg_lots": round(float(lots[fm & (dirn != 0)].mean()), 2) if (fm & (dirn != 0)).any() else 0}))
        log(f"  spreads grid: entry {entry} done, {len(rows):,} variants ({time.time() - t0:.0f}s)")
    return rows


def spread_daily(fac, pre, r):
    """Recompute one variant's daily P&L (for the report)."""
    stop = ("none", 0) if r["stop"] == "none" else (r["stop"].split()[0], float(r["stop"].split()[1]))
    tgt = None if r["target"] == "none" else int(r["target"].split("%")[0]) / 100
    put_pnl, put_lots = spread_side(fac, pre, r["structure"], r["wings"], True, stop, tgt, r["exit"])
    call_pnl, call_lots = spread_side(fac, pre, r["structure"], r["wings"], False, stop, tgt, r["exit"])
    dirn = direction(fac.N, pre["e"], r["signal"])
    pnl = np.where(dirn > 0, put_pnl, np.where(dirn < 0, call_pnl, 0.0))
    lots = np.where(dirn > 0, put_lots, np.where(dirn < 0, call_lots, 0))
    fm = fac.filters()[r["filter"]]
    return np.where(fm, pnl, 0.0), (fm & (lots > 0) & (dirn != 0)).astype(int)
