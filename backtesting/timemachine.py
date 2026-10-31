"""Time Machine (trader-timemachine, on NSE sessions).

Before a plan is sized it is replayed:
1. The pool: every FULL past session of the stock (75 five-minute bars, 09:15 to 15:30;
   up to 3 empty bars forward-filled). Every bar becomes four ratios to the bar before
   it: open, high, low and close over the previous close.
2. One replay starts at the entry and, for each bar left until the flat-by bar, draws a
   past bar from the same time of day (plus or minus `window` bars) and scales it to the
   current price. Stop first, then target, bar by bar. A bar touching both is the stop.
3. N replays (seeded, so the same plan and data give the same numbers) give the odds of
   target first, stop first, neither (still open at the flat-by time), the price there,
   and the average R. R is +target_r at the target, -1 at the stop, otherwise the
   distance from the entry at the flat-by bar over the risk.
4. The history: the same strategy, the same stop rule and target R run on each of those
   past sessions (backtesting.engine.run_day). That is what really happened.

The verdict follows trader-timemachine: of the replays that hit a side, how often the
target came first, against the break-even share 1 / (1 + R).

It is a resampling of the stock's own recent bars, not a prediction.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from market.calendar import SESSION_BARS
from market.candles.bars import Bar

COIN_FLIP, STRONG, MANY_NEITHER = 0.05, 0.15, 0.40


def build_pool(sessions: List[List[Bar]]) -> Optional[np.ndarray]:
    """(sessions, 75, 4) ratios; slot 0 is unused (it has no previous bar in the session)."""
    if not sessions:
        return None
    pool = np.ones((len(sessions), SESSION_BARS, 4), dtype=np.float64)
    for s, D in enumerate(sessions):
        for i in range(1, SESSION_BARS):
            pc = D[i - 1].c
            if pc <= 0:
                continue
            b = D[i]
            pool[s, i] = (b.o / pc, b.h / pc, b.l / pc, b.c / pc)
    return pool


def simulate(pool: np.ndarray, entry: float, stop: float, target: float, k: int, last: int,
             replays: int = 5000, seed: int = 19, window: int = 2, long: bool = True) -> Dict:
    risk = (entry - stop) if long else (stop - entry)
    r_target = ((target - entry) if long else (entry - target)) / risk
    rng = np.random.default_rng(seed)
    n_sess = pool.shape[0]
    p = np.full(replays, float(entry))
    outcome = np.zeros(replays, dtype=np.int8)      # 1 target, -1 stop, 0 neither
    for i in range(k + 1, last + 1):
        if window:
            slots = np.clip(i + rng.integers(-window, window + 1, replays), 1, SESSION_BARS - 1)
        else:
            slots = np.full(replays, i)
        sess = rng.integers(0, n_sess, replays)
        r = pool[sess, slots]
        hi, lo, cl = p * r[:, 1], p * r[:, 2], p * r[:, 3]
        live = outcome == 0
        if long:
            hit_stop = live & (lo <= stop)
            hit_tgt = live & ~hit_stop & (hi >= target)
        else:
            hit_stop = live & (hi >= stop)
            hit_tgt = live & ~hit_stop & (lo <= target)
        outcome[hit_stop] = -1
        outcome[hit_tgt] = 1
        p = cl
    move = (p - entry) if long else (entry - p)
    R = np.where(outcome == 1, r_target, np.where(outcome == -1, -1.0, move / risk))
    green = (outcome == 1) | ((outcome == 0) & (move > 0))
    q = np.percentile(p, [5, 50, 95])
    return {
        "replays": replays, "seed": seed, "window": window, "bars_left": last - k,
        "sessions": n_sess, "r_at_target": round(r_target, 2),
        "p_target": round(float(np.mean(outcome == 1)), 4),
        "p_stop": round(float(np.mean(outcome == -1)), 4),
        "p_neither": round(float(np.mean(outcome == 0)), 4),
        "p_green": round(float(np.mean(green)), 4),
        "avg_R": round(float(np.mean(R)), 3),
        "p5_close": round(float(q[0]), 2), "median_close": round(float(q[1]), 2), "p95_close": round(float(q[2]), 2),
    }


def lean(mc: Dict):
    pt, ps = mc["p_target"], mc["p_stop"]
    if pt + ps <= 0:
        return None
    r = mc.get("r_at_target") or 1.0
    share, even = pt / (pt + ps), 1.0 / (1.0 + r)
    return share - even, share, even


def verdict(mc: Dict, until: str) -> str:
    lv = lean(mc)
    if lv is None:
        return f"No replay reached the target or the stop by {until}: the price at {until} decides every one."
    d, share, even = lv
    r = mc.get("r_at_target") or 1.0
    one_r = abs(r - 1.0) < 0.005
    if abs(d) < COIN_FLIP:
        head = "Close to a coin flip" if one_r else "Close to break-even"
    elif abs(d) < STRONG:
        head = "Leans toward the target" if d > 0 else "Leans toward the stop"
    else:
        head = "Strongly toward the target" if d > 0 else "Strongly toward the stop"
    odds = f"target first {100 * mc['p_target']:.1f}% vs stop first {100 * mc['p_stop']:.1f}%"
    if one_r:
        return f"{head}: {odds}."
    return (f"{head}: at {r:g}R the target has to come first in {100 * even:.0f}% of the replays that hit "
            f"one side; it did in {100 * share:.0f}% ({odds}).")


def notes(mc: Dict, until: str) -> List[str]:
    out = []
    if mc.get("p_neither", 0) > MANY_NEITHER and lean(mc) is not None:
        out.append(f"{100 * mc['p_neither']:.1f}% of the replays hit neither by {until}; the price at {until} "
                   f"decides those.")
    lv = lean(mc)
    if lv is not None and mc["avg_R"] != 0 and (mc["avg_R"] > 0) != (lv[0] > 0) and abs(lv[0]) >= COIN_FLIP:
        out.append(f"Average R is {mc['avg_R']:+.3f}, against the lean.")
    return out


def history_summary(results: List[dict]) -> Dict:
    """results: run_day outcomes for past sessions ({'outcome', 'R'})."""
    if not results:
        return {"triggered": 0, "target": 0, "stop": 0, "flat": 0, "avg_R": None}
    return {
        "triggered": len(results),
        "target": sum(1 for r in results if r["outcome"] == "target"),
        "stop": sum(1 for r in results if r["outcome"] == "stop"),
        "flat": sum(1 for r in results if r["outcome"] == "flat"),
        "avg_R": round(sum(r["R"] for r in results) / len(results), 3),
    }


def gate(mc: Optional[Dict], hist: Dict, cfg) -> Tuple[bool, str]:
    """The engine's use of the Time Machine. Lenient by default: it refuses plans whose
    target is out of reach by the flat-by time, a replay average far below zero, or a
    setup that has been losing on this stock in the recent past."""
    tm = cfg.get("timemachine", {}) or {}
    if mc is None:
        return False, "time machine: not enough full sessions to replay"
    if mc["p_neither"] > float(tm.get("max_neither", 0.75)):
        return False, (f"time machine: {100 * mc['p_neither']:.0f}% of replays reach neither the stop nor the target "
                       f"by the flat-by time")
    if mc["avg_R"] < float(tm.get("min_mc_avg_r", -0.15)):
        return False, f"time machine: replay average {mc['avg_R']:+.3f}R"
    if hist["triggered"] >= int(tm.get("min_history_trades", 5)) and hist["avg_R"] is not None \
            and hist["avg_R"] < float(tm.get("min_history_avg_r", -0.05)):
        return False, (f"time machine: this setup fired {hist['triggered']} times on this stock in the replay window "
                       f"and averaged {hist['avg_R']:+.3f}R")
    return True, "time machine: pass"
