"""Vectorized bracket exits, per-stock daily features, and the equity portfolio simulator."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from research.costs import equity_order_charges
from research.data import SLOTS, DayMatrix

FLAT_SLOT = 71          # the 15:10-15:15 bar: everything intraday is flat by 15:15
SLOT_IDX = np.arange(SLOTS)


def slot_of_close(hhmm: str) -> int:
    """Last bar whose close is at or before HH:MM."""
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m - (9 * 60 + 15)) // 5 - 1


# ---------------------------------------------------------------- features
@dataclass
class Features:
    d_open: np.ndarray
    d_high: np.ndarray
    d_low: np.ndarray
    d_close: np.ndarray
    prev_close: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray
    atr14: np.ndarray        # Wilder ATR of the sessions BEFORE each day
    sma20: np.ndarray
    sma50: np.ndarray
    high20: np.ndarray       # highest high of the 20 sessions before
    low20: np.ndarray
    trend_up: np.ndarray
    trend_down: np.ndarray
    or_vol: np.ndarray       # volume of the 09:20 and 09:25 bars
    or_vol_avg: np.ndarray   # its 20-session average before the day
    v0: np.ndarray           # volume of the 09:15 bar
    v0_avg: np.ndarray       # its 14-session average before the day
    vwap: np.ndarray         # (D, 75)
    vsd: np.ndarray          # (D, 75) volume-weighted std of typical price around VWAP
    valid: np.ndarray = field(default=None)


def _shift(x, k=1, fill=np.nan):
    out = np.full_like(x, fill, dtype=float)
    out[k:] = x[:-k]
    return out


def _rolling(x, n, fn):
    out = np.full(len(x), np.nan)
    for i in range(n, len(x) + 1):
        out[i - 1] = fn(x[i - n:i])
    return out


def features(m: DayMatrix) -> Features:
    o, h, l, c, v = m.daily()
    prev_close = _shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    tr[0] = h[0] - l[0]
    atr = np.full(len(c), np.nan)
    if len(c) > 15:
        a = np.mean(tr[1:15])
        atr[14] = a
        for i in range(15, len(c)):
            a = (a * 13 + tr[i]) / 14
            atr[i] = a
    atr14 = _shift(atr)                      # known before the day starts
    sma20 = _shift(_rolling(c, 20, np.mean))
    sma50 = _shift(_rolling(c, 50, np.mean))
    high20 = _shift(_rolling(h, 20, np.max))
    low20 = _shift(_rolling(l, 20, np.min))
    trend_up = (prev_close > sma20) & (sma20 > sma50)
    trend_down = (prev_close < sma20) & (sma20 < sma50)
    or_vol = m.v[:, 1:3].sum(axis=1)
    or_vol_avg = _shift(_rolling(or_vol, 20, np.mean))
    v0 = m.v[:, 0]
    v0_avg = _shift(_rolling(v0, 14, np.mean))
    tp = (m.h + m.l + m.c) / 3.0
    vol = np.where(m.v > 0, m.v, 1e-9)
    cv = np.cumsum(vol, axis=1)
    vwap = np.cumsum(tp * vol, axis=1) / cv
    var = np.cumsum(vol * tp * tp, axis=1) / cv - vwap * vwap
    vsd = np.sqrt(np.maximum(var, 0))
    return Features(o, h, l, c, prev_close, _shift(h), _shift(l), atr14, sma20, sma50, high20, low20,
                    trend_up, trend_down, or_vol, or_vol_avg, v0, v0_avg, vwap, vsd,
                    valid=~np.isnan(sma50))


def first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or -1."""
    any_ = mask.any(axis=1)
    idx = mask.argmax(axis=1)
    return np.where(any_, idx, -1)


# ---------------------------------------------------------------- exits
def bracket_exits(m: DayMatrix, rows, entry_slot, side, stop, target, flat_slot=FLAT_SLOT,
                  include_entry_bar=False):
    """Stop first, then target, bar by bar after the entry; flat at flat_slot's close.
    side +1 long / -1 short. target may be NaN (no target). Returns exit_slot, exit_px, outcome
    (0 flat, 1 target, -1 stop)."""
    rows = np.asarray(rows, int)
    n = len(rows)
    if n == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    H, L, O, C = m.h[rows], m.l[rows], m.o[rows], m.c[rows]
    es = np.asarray(entry_slot)[:, None]
    after = (SLOT_IDX[None, :] >= es) if include_entry_bar else (SLOT_IDX[None, :] > es)
    after &= SLOT_IDX[None, :] <= flat_slot
    s = np.asarray(side)[:, None]
    st = np.asarray(stop, float)[:, None]
    tg = np.asarray(target, float)[:, None]
    stop_hit = after & np.where(s > 0, L <= st, H >= st)
    if include_entry_bar:
        # inside the bar a stop-entry filled in, assume the usual path: a green bar goes low
        # then high, a red bar high then low. The stop counts in that bar only if the price
        # reached it after the fill (or the bar closed beyond it).
        eb = SLOT_IDX[None, :] == es
        Oe, Ce = O, C
        after_fill = np.where(s > 0, (Ce <= st) | ((Oe > Ce) & (L <= st)),
                              (Ce >= st) | ((Oe < Ce) & (H >= st)))
        stop_hit = np.where(eb, stop_hit & after_fill, stop_hit)
    tgt_hit = after & ~np.isnan(tg) & np.where(s > 0, H >= tg, L <= tg)
    if include_entry_bar:
        tgt_hit &= ~(SLOT_IDX[None, :] == es)       # never credit a target inside the entry bar
    fs, ft = first_true(stop_hit), first_true(tgt_hit)
    fs_ = np.where(fs < 0, 999, fs)
    ft_ = np.where(ft < 0, 999, ft)
    outcome = np.where((fs_ <= ft_) & (fs_ < 999), -1, np.where(ft_ < 999, 1, 0))
    exit_slot = np.where(outcome == -1, fs_, np.where(outcome == 1, ft_, flat_slot))
    r = np.arange(n)
    o_at = O[r, np.minimum(exit_slot, SLOTS - 1)]
    entry_bar = exit_slot == np.asarray(entry_slot)
    sd = np.asarray(side)
    stp, tgv = np.asarray(stop, float), np.asarray(target, float)
    stop_px = np.where(entry_bar, stp, np.where(sd > 0, np.minimum(stp, o_at), np.maximum(stp, o_at)))
    tgt_px = np.where(entry_bar, tgv, np.where(sd > 0, np.maximum(tgv, o_at), np.minimum(tgv, o_at)))
    flat_px = C[r, flat_slot]
    px = np.where(outcome == -1, stop_px, np.where(outcome == 1, tgt_px, flat_px))
    return exit_slot, px, outcome


def stop_entries(m: DayMatrix, rows, level, side, from_slot, until_slot):
    """A buy-stop (sell-stop) order at `level`, live from from_slot to until_slot: the first
    bar that trades through it, and the fill (the level, or the bar's open on a gap)."""
    rows = np.asarray(rows, int)
    H, L, O = m.h[rows], m.l[rows], m.o[rows]
    lv = np.asarray(level, float)[:, None]
    s = np.asarray(side)[:, None]
    fs = np.broadcast_to(np.asarray(from_slot), (len(rows),))
    win = (SLOT_IDX[None, :] >= fs[:, None]) & (SLOT_IDX[None, :] <= until_slot)
    hit = win & np.where(s > 0, H >= lv, L <= lv)
    k = first_true(hit)
    r = np.arange(len(rows))
    ok = k >= 0
    kk = np.where(ok, k, 0)
    o_at = O[r, kk]
    fill = np.where(np.asarray(side) > 0, np.maximum(lv[:, 0], o_at), np.minimum(lv[:, 0], o_at))
    return k, fill, ok


# ---------------------------------------------------------------- equity portfolio
@dataclass
class EquityRules:
    capital: float = 100000.0
    risk_per_trade: float = 1000.0
    leverage: float = 5.0             # MIS: up to 5x on NIFTY 100 stocks
    max_position_value: float = 250000.0
    max_positions: int = 5
    max_trades_per_day: int = 8
    daily_loss_stop: float = 3000.0
    slippage_bps: float = 3.0         # NIFTY 100 large caps: spreads of 1 to 3 bps
    min_risk_pct: float = 0.15
    max_risk_pct: float = 2.5
    top_n_per_day: Optional[int] = None


def simulate_equity(cands: List[dict], rules: EquityRules, costs: dict) -> List[dict]:
    """cands: {day, symbol, slot, side, entry, stop, target, exit_slot, exit_px, outcome, score}.
    Takes them day by day in time order under the account's limits; returns filled trades."""
    by_day = defaultdict(list)
    for c in cands:
        by_day[c["day"]].append(c)
    trades = []
    k = rules.slippage_bps / 10000.0
    for day in sorted(by_day):
        rows = by_day[day]
        if rules.top_n_per_day:
            rows = sorted(rows, key=lambda r: -r["score"])[:rules.top_n_per_day]
        rows.sort(key=lambda r: (r["slot"], -r["score"]))
        open_, closed, used = [], [], set()
        for c in rows:
            if c["symbol"] in used:
                continue
            still = [t for t in open_ if t["exit_slot"] > c["slot"]]
            closed += [t for t in open_ if t["exit_slot"] <= c["slot"]]
            open_ = still
            if len(open_) >= rules.max_positions or len(open_) + len(closed) >= rules.max_trades_per_day:
                continue
            if sum(t["net"] for t in closed) <= -rules.daily_loss_stop:
                continue
            side = c["side"]
            entry = c["entry"] * (1 + k * side)
            risk = (entry - c["stop"]) * side
            if risk <= 0:
                continue
            rp = 100.0 * risk / entry
            if rp < rules.min_risk_pct or rp > rules.max_risk_pct:
                continue
            exposure = sum(t["value"] for t in open_)
            room = rules.capital * rules.leverage - exposure
            qty = int(min(rules.risk_per_trade / risk, rules.max_position_value / entry, room / entry))
            if qty < 1:
                continue
            exit_px = c["exit_px"] if c["outcome"] == 1 else c["exit_px"] * (1 - k * side)
            gross = (exit_px - entry) * qty * side
            s_open, s_close = ("buy", "sell") if side > 0 else ("sell", "buy")
            ch = equity_order_charges(entry * qty, s_open, costs) + equity_order_charges(exit_px * qty, s_close, costs)
            t = dict(c, qty=qty, entry_fill=entry, exit_fill=exit_px, value=entry * qty, gross=gross, charges=ch,
                     net=gross - ch, risk_inr=risk * qty, R=(gross - ch) / (risk * qty))
            open_.append(t)
            used.add(c["symbol"])
            trades.append(t)
    return trades


def candidates(m: DayMatrix, rows, slots, sides, entries, stops, targets, scores=None, include_entry_bar=False,
               flat_slot=FLAT_SLOT) -> List[dict]:
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return []
    es, px, oc = bracket_exits(m, rows, slots, sides, stops, targets, flat_slot, include_entry_bar)
    sc = np.zeros(len(rows)) if scores is None else np.asarray(scores, float)
    out = []
    for i, r in enumerate(rows):
        out.append({"day": m.dates[r], "symbol": m.symbol, "slot": int(slots[i]), "side": int(sides[i]),
                    "entry": float(entries[i]), "stop": float(stops[i]),
                    "target": float(targets[i]) if not np.isnan(targets[i]) else None,
                    "exit_slot": int(es[i]), "exit_px": float(px[i]), "outcome": int(oc[i]), "score": float(sc[i])})
    return out
