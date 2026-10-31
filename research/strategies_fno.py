"""Six index F&O strategies (NIFTY / BANKNIFTY), 1 lot each, with modeled option premiums
(see research.options). Index futures need about Rs 1.7 to 2 lakh margin a NIFTY lot, more
than the Rs 1 lakh account, so directional index views are traded with ATM options; the
futures-equivalent result per lot is reported alongside for reference."""
import datetime as dt
from typing import Dict, List

import numpy as np

from research import options as op
from research.costs import futures_order_charges, option_order_charges, option_slippage
from research.data import SLOTS, DayMatrix
from research.sim import FLAT_SLOT, bracket_exits, first_true, slot_of_close

S_IDX = np.arange(SLOTS)


BASE_IV_SCALE = 0.9      # real NIFTY ATM IV averaged 0.91 x India VIX in a check against live option candles
BANKNIFTY_IV_MULT = 1.25  # BANKNIFTY has no VIX of its own; its IV usually runs well above NIFTY's
SL_SLIP_MULT = 3.0        # stop-loss fills on short options come in fast markets


class FnoContext:
    def __init__(self, nifty: DayMatrix, banknifty: DayMatrix, vix: DayMatrix, iv_scale: float = BASE_IV_SCALE):
        self.N, self.B, self.V = nifty, banknifty, vix
        self.iv_scale = iv_scale
        days = [d.astype(object) for d in nifty.dates]
        self.days = days
        self.weekly = op.ExpiryClock(days, op.nifty_weekly_expiries(days))
        self.monthly = op.ExpiryClock(days, op.banknifty_monthly_expiries(days))
        vpos = {d: i for i, d in enumerate(vix.dates)}
        self.vix_rows = {d: vpos.get(d) for d in nifty.dates}
        self.bpos = {d: i for i, d in enumerate(banknifty.dates)}
        self.lots = op.lot_sizes()

    def iv(self, date, slot):
        r = self.vix_rows.get(date)
        if r is None:
            return 0.15
        v = self.V.c[r, slot]
        return (v if not np.isnan(v) else np.nanmean(self.V.c[r])) / 100.0

    def prem(self, S, K, day, slot, expiry, call, clock, iv_mult: float = 1.0):
        T = clock.T(day, float(op.slot_minute(slot)), expiry)
        return float(op.price(S, K, T, self.iv(np.datetime64(day), slot) * iv_mult, call, self.iv_scale))


def _option_trade(ctx, clock, day, side, S_in, slot_in, S_out, slot_out, lot, step, expiry, extra=None, iv_mult=1.0):
    call = side > 0
    K = float(op.round_strike(S_in, step))
    p_in = ctx.prem(S_in, K, day, slot_in, expiry, call, clock, iv_mult)
    p_out = ctx.prem(S_out, K, day, slot_out, expiry, call, clock, iv_mult)
    buy = p_in + option_slippage(p_in)
    sell = max(0.05, p_out - option_slippage(p_out))
    gross = (sell - buy) * lot
    ch = option_order_charges(buy * lot, "buy") + option_order_charges(sell * lot, "sell")
    fut_gross = (S_out - S_in) * side * lot
    fut_ch = futures_order_charges(S_in * lot, "buy" if side > 0 else "sell") + \
        futures_order_charges(S_out * lot, "sell" if side > 0 else "buy")
    t = {"day": np.datetime64(day), "side": side, "strike": K, "expiry": str(expiry), "premium_in": round(p_in, 2),
         "premium_out": round(p_out, 2), "index_in": round(S_in, 2), "index_out": round(S_out, 2),
         "slot": int(slot_in), "exit_slot": int(slot_out), "gross": gross, "charges": ch, "net": gross - ch,
         "risk_inr": buy * lot, "fut_net": fut_gross - fut_ch, "R": None}
    if extra:
        t.update(extra)
    return t


def _breakout_options(ctx, M: DayMatrix, n_range, until, wmin, wmax, target_r, lot, step, clock, market=None,
                      iv_mult=1.0):
    oh, ol = M.h[:, :n_range].max(1), M.l[:, :n_range].min(1)
    mid = (oh + ol) / 2
    width = 100 * (oh - ol) / mid
    win = (S_IDX >= n_range) & (S_IDX <= until)
    ku, kd = first_true(win & (M.c > oh[:, None])), first_true(win & (M.c < ol[:, None]))
    side = np.where((ku >= 0) & ((kd < 0) | (ku < kd)), 1, np.where(kd >= 0, -1, 0))
    slot = np.where(side == 1, ku, kd)
    ok = (width >= wmin) & (width <= wmax) & (side != 0)
    rows = np.where(ok)[0]
    if market is not None:
        keep = []
        for r in rows:
            j = market.pos.get(M.dates[r])
            s = slot[r]
            if j is None:
                continue
            d = np.sign(market.n.c[j, s] - market.n.o[j, 0])
            keep.append(d * side[r] > 0)
        rows = rows[np.array(keep, bool)] if len(keep) else rows[:0]
    slot, side = slot[rows], side[rows]
    entry = M.c[rows, slot]
    stop = mid[rows]
    target = entry + target_r * (entry - stop)
    es, px, oc = bracket_exits(M, rows, slot, side, stop, target, FLAT_SLOT)
    out = []
    for i, r in enumerate(rows):
        day = M.dates[r].astype(object)
        exp = clock.expiry_for(day, min_sessions_left=1)
        out.append(_option_trade(ctx, clock, day, side[i], entry[i], slot[i], px[i], es[i], lot, step, exp,
                                 {"outcome": int(oc[i])}, iv_mult))
    return out


# ---------------------------------------------------------------- I1
def i1_nifty_orb_options(ctx: FnoContext):
    """NIFTY 15-minute opening-range breakout (range 0.2% to 1.2%), first close beyond it
    09:30-12:00; buy 1 lot ATM call (long) or put (short) of the next weekly expiry (never
    same-day expiry). Exit on the index: range middle (stop), 2R (target), or 15:15."""
    return _breakout_options(ctx, ctx.N, 3, slot_of_close("12:00"), 0.2, 1.2, 2.0, ctx.lots["NIFTY"], 50, ctx.weekly)


# ---------------------------------------------------------------- I2
def i2_nifty_late_momentum(ctx: FnoContext):
    """Intraday momentum (Gao, Han, Li and Zhou, 2018): the move from yesterday's close to
    09:45 predicts the last half hour. If it is more than 0.25%, buy 1 lot ATM call (up) or
    put (down) at 15:00, sell at 15:25. Next weekly expiry."""
    N = ctx.N
    out = []
    s_sig, s_in, s_out = slot_of_close("09:45"), slot_of_close("15:00"), slot_of_close("15:25")
    for r in range(1, N.n):
        pc = N.c[r - 1, -1]
        r1 = N.c[r, s_sig] / pc - 1
        if abs(r1) < 0.0025:
            continue
        day = N.dates[r].astype(object)
        side = 1 if r1 > 0 else -1
        exp = ctx.weekly.expiry_for(day, min_sessions_left=1)
        out.append(_option_trade(ctx, ctx.weekly, day, side, N.c[r, s_in], s_in, N.c[r, s_out], s_out,
                                 ctx.lots["NIFTY"], 50, exp, {"signal_pct": round(100 * r1, 2)}))
    return out


# ---------------------------------------------------------------- I3
class _Mk:
    def __init__(self, n: DayMatrix):
        self.n = n
        self.pos = {d: i for i, d in enumerate(n.dates)}


def i3_banknifty_hour_breakout(ctx: FnoContext):
    """BANKNIFTY first-hour breakout: range 09:15-10:15 (0.3% to 1.5%), first close beyond it
    10:20-13:30 with the NIFTY on the same side of its open; buy 1 lot ATM monthly call/put.
    Exit on the index: range middle, 1.5R, or 15:15."""
    return _breakout_options(ctx, ctx.B, 12, slot_of_close("13:30"), 0.3, 1.5, 1.5, ctx.lots["BANKNIFTY"], 100,
                             ctx.monthly, market=_Mk(ctx.N), iv_mult=BANKNIFTY_IV_MULT)


# ---------------------------------------------------------------- short-premium helpers
def _legs_value(ctx, day, slot, S, legs, expiry):
    """legs: [(K, call, qty_sign)] -> value per unit of the position (short = negative)."""
    return sum(q * ctx.prem(S, K, day, slot, expiry, c, ctx.weekly) for K, c, q in legs)


def _short_premium_day(ctx, r, entry_slot, kc, kp, wc, wp, sl_mult, expiry, lot):
    """Sell call kc and put kp at entry_slot's close, buy wings wc / wp; each short leg has its
    own stop at sl_mult x its entry premium (checked against the bar's high for the call and
    low for the put); everything else closes at 15:15."""
    N = ctx.N
    day = N.dates[r].astype(object)
    S0 = N.c[r, entry_slot]
    pr = lambda S, K, s, call: ctx.prem(S, K, day, s, expiry, call, ctx.weekly)
    ce0, pe0, wc0, wp0 = pr(S0, kc, entry_slot, True), pr(S0, kp, entry_slot, False), \
        pr(S0, wc, entry_slot, True), pr(S0, wp, entry_slot, False)
    fills = {"ce": ce0 - option_slippage(ce0), "pe": pe0 - option_slippage(pe0),
             "wc": wc0 + option_slippage(wc0), "wp": wp0 + option_slippage(wp0)}
    exits = {}
    for s in range(entry_slot + 1, FLAT_SLOT + 1):
        for leg, K, call, Sx in (("ce", kc, True, N.h[r, s]), ("pe", kp, False, N.l[r, s])):
            if leg in exits:
                continue
            sl = sl_mult * (ce0 if leg == "ce" else pe0)
            worst = pr(Sx, K, s, call)
            if worst >= sl:
                at_open = pr(N.o[r, s], K, s, call)
                px = max(sl, at_open)
                exits[leg] = (px + SL_SLIP_MULT * option_slippage(px), s)
        if len(exits) == 2:
            break
    end = max([FLAT_SLOT] if len(exits) < 2 else [s for _, s in exits.values()])
    S_end = N.c[r, end]
    for leg, K, call in (("ce", kc, True), ("pe", kp, False)):
        if leg not in exits:
            p = pr(S_end, K, end, call)
            exits[leg] = (p + option_slippage(p), end)
    for leg, K, call in (("wc", wc, True), ("wp", wp, False)):
        p = pr(S_end, K, end, call)
        exits[leg] = (max(0.05, p - option_slippage(p)), end)
    gross = ((fills["ce"] - exits["ce"][0]) + (fills["pe"] - exits["pe"][0]) +
             (exits["wc"][0] - fills["wc"]) + (exits["wp"][0] - fills["wp"])) * lot
    ch = (option_order_charges(fills["ce"] * lot, "sell") + option_order_charges(fills["pe"] * lot, "sell") +
          option_order_charges(fills["wc"] * lot, "buy") + option_order_charges(fills["wp"] * lot, "buy") +
          option_order_charges(exits["ce"][0] * lot, "buy") + option_order_charges(exits["pe"][0] * lot, "buy") +
          option_order_charges(exits["wc"][0] * lot, "sell") + option_order_charges(exits["wp"][0] * lot, "sell"))
    credit = (fills["ce"] + fills["pe"] - fills["wc"] - fills["wp"]) * lot
    width = max(wc - kc, kp - wp)
    return {"day": N.dates[r], "index_in": round(S0, 2), "strikes": [kp, kc], "wings": [wp, wc],
            "expiry": str(expiry), "credit": round(credit, 2), "stopped": sorted(k for k, (_, s) in exits.items()
                                                                             if k in ("ce", "pe") and s < FLAT_SLOT),
            "slot": entry_slot, "exit_slot": end, "gross": gross, "charges": ch, "net": gross - ch,
            "risk_inr": width * lot - credit, "R": None}


# ---------------------------------------------------------------- O1
def o1_920_straddle(ctx: FnoContext):
    """The 09:20 short straddle: sell the ATM call and put of the current weekly expiry at
    09:20 (expiry days included), buy 2.5% out-of-the-money wings for margin, stop each
    short leg at +25% of its premium, close everything at 15:15. 1 lot."""
    N, out = ctx.N, []
    lot = ctx.lots["NIFTY"]
    for r in range(N.n):
        day = N.dates[r].astype(object)
        exp = ctx.weekly.expiry_for(day, 0)
        S = N.c[r, 0]
        K = float(op.round_strike(S, 50))
        out.append(_short_premium_day(ctx, r, 0, K, K, float(op.round_strike(S * 1.025, 50)),
                                      float(op.round_strike(S * 0.975, 50)), 1.25, exp, lot))
    return out


# ---------------------------------------------------------------- O2
def o2_expiry_day_strangle(ctx: FnoContext):
    """Expiry-day short strangle: on weekly expiry days only, at 09:45 sell the call and put
    about 0.6% out of the money, buy 2% wings, stop each short leg at +50%, close at 15:15
    (before settlement). 1 lot."""
    N, out = ctx.N, []
    lot = ctx.lots["NIFTY"]
    s_in = slot_of_close("09:45")
    exps = set(ctx.weekly.expiries)
    for r in range(N.n):
        day = N.dates[r].astype(object)
        if day not in exps:
            continue
        S = N.c[r, s_in]
        kc = float(np.ceil(S * 1.006 / 50) * 50)
        kp = float(np.floor(S * 0.994 / 50) * 50)
        out.append(_short_premium_day(ctx, r, s_in, kc, kp, float(op.round_strike(S * 1.02, 50)),
                                      float(op.round_strike(S * 0.98, 50)), 1.5, day, lot))
    return out


# ---------------------------------------------------------------- O3
def o3_weekly_iron_condor(ctx: FnoContext):
    """Weekly iron condor, held overnight: at 15:15 on the first session after an expiry, sell
    the call and put one expected move away (S x VIX x sqrt(time to expiry), rounded outward to
    50), buy wings 200 points further. Exit at 15:15 on any day the NIFTY closes beyond a short
    strike, else at 15:15 on expiry day. Marked to market every day at 15:15. 1 lot."""
    N, W = ctx.N, ctx.weekly
    lot = ctx.lots["NIFTY"]
    days = ctx.days
    out, mtm = [], {}
    exps = sorted(W.expiries)
    for k in range(1, len(exps)):
        prev_e, e = exps[k - 1], exps[k]
        if prev_e not in W.index or e not in W.index:
            continue
        i0 = W.index[prev_e] + 1
        if i0 >= W.index[e]:
            continue
        r0 = i0
        d0 = days[r0]
        S0 = N.c[r0, FLAT_SLOT]
        T = W.T(d0, float(op.slot_minute(FLAT_SLOT)), e)
        move = S0 * ctx.iv(N.dates[r0], FLAT_SLOT) * np.sqrt(T)
        kc, kp = float(np.ceil((S0 + move) / 50) * 50), float(np.floor((S0 - move) / 50) * 50)
        wc, wp = kc + 200, kp - 200
        legs = [(kc, True, -1), (kp, False, -1), (wc, True, 1), (wp, False, 1)]
        pr = lambda S, K, d, call: ctx.prem(S, K, d, FLAT_SLOT, e, call, W)
        prem0 = {K: pr(S0, K, d0, c) for K, c, _ in legs}
        fill0 = {K: (prem0[K] - option_slippage(prem0[K])) if q < 0 else (prem0[K] + option_slippage(prem0[K]))
                 for K, c, q in legs}
        value_prev = sum(q * fill0[K] for K, c, q in legs) * lot      # negative: a credit
        ch_in = sum(option_order_charges(fill0[K] * lot, "sell" if q < 0 else "buy") for K, c, q in legs)
        exit_r, reason = W.index[e], "expiry"
        for r in range(r0 + 1, W.index[e] + 1):
            S = N.c[r, FLAT_SLOT]
            if S > kc or S < kp:
                exit_r, reason = r, "short strike breached"
                break
        total_net, ch_total = -ch_in, ch_in
        mtm[N.dates[r0]] = mtm.get(N.dates[r0], 0.0) - ch_in
        for r in range(r0 + 1, exit_r + 1):
            d = days[r]
            S = N.c[r, FLAT_SLOT]
            mark = {K: pr(S, K, d, c) for K, c, _ in legs}
            if r == exit_r:
                mark = {K: (mark[K] + option_slippage(mark[K])) if q < 0 else max(0.05, mark[K] - option_slippage(mark[K]))
                        for K, c, q in legs}
            value = sum(q * mark[K] for K, c, q in legs) * lot
            pnl = value - value_prev
            if r == exit_r:
                ch_out = sum(option_order_charges(mark[K] * lot, "buy" if q < 0 else "sell") for K, c, q in legs)
                pnl -= ch_out
                ch_total += ch_out
            mtm[N.dates[r]] = mtm.get(N.dates[r], 0.0) + pnl
            total_net += pnl
            value_prev = value
        credit = -sum(q * fill0[K] for K, c, q in legs) * lot
        out.append({"day": N.dates[r0], "exit_day": str(N.dates[exit_r]), "index_in": round(S0, 2),
                    "strikes": [kp, kc], "wings": [wp, wc], "expiry": str(e), "credit": round(credit, 2),
                    "reason": reason, "net": total_net, "gross": total_net + ch_total, "charges": ch_total,
                    "risk_inr": 200 * lot - credit, "R": None, "slot": FLAT_SLOT, "exit_slot": FLAT_SLOT})
    return out, mtm


FNO = [
    ("I1", "NIFTY ORB with ATM option buying", i1_nifty_orb_options),
    ("I2", "NIFTY last-half-hour momentum (options)", i2_nifty_late_momentum),
    ("I3", "BANKNIFTY first-hour breakout (options)", i3_banknifty_hour_breakout),
    ("O1", "09:20 short straddle, 25% leg stop (hedged)", o1_920_straddle),
    ("O2", "Expiry-day short strangle (hedged)", o2_expiry_day_strangle),
    ("O3", "Weekly iron condor, positional", o3_weekly_iron_condor),
]
