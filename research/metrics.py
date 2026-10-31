"""Strategy statistics from a daily P&L series (every session of the test, zero when flat)."""
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

OOS_START = np.datetime64("2025-01-01")


def daily_series(days: np.ndarray, trades: List[dict], mtm: Optional[Dict] = None) -> np.ndarray:
    pos = {d: i for i, d in enumerate(days)}
    out = np.zeros(len(days))
    if mtm is not None:
        for d, v in mtm.items():
            if d in pos:
                out[pos[d]] += v
        return out
    for t in trades:
        i = pos.get(np.datetime64(t["day"], "D"))
        if i is not None:
            out[i] += t["net"]
    return out


def _dd(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    i = int(np.argmax(dd)) if len(dd) else 0
    longest, cur = 0, 0
    for x in dd:
        cur = cur + 1 if x > 0 else 0
        longest = max(longest, cur)
    return float(dd.max()) if len(dd) else 0.0, longest, i


def _part(days, pnl, trades, mask, capital):
    p = pnl[mask]
    n_days = int(mask.sum())
    if n_days == 0:
        return {}
    years = n_days / 250.0
    tr = [t for t in trades if mask[days == np.datetime64(t["day"], "D")].any()] if trades else []
    return {"sessions": n_days, "net": round(float(p.sum()), 0), "per_day": round(float(p.mean()), 1),
            "per_month": round(float(p.sum()) / max(1e-9, n_days / 21.0), 0),
            "annual_return_pct": round(100 * float(p.sum()) / capital / years, 1) if years else None,
            "trades": len(tr)}


def summarize(code: str, name: str, days: np.ndarray, pnl: np.ndarray, trades: List[dict], capital: float = 1e5,
              target: float = 2000.0, extra: Optional[dict] = None) -> dict:
    equity = capital + np.cumsum(pnl)
    max_dd, dd_len, _ = _dd(equity)
    ruin = np.where(equity <= 0)[0]
    nets = np.array([t["net"] for t in trades]) if trades else np.zeros(0)
    wins, losses = nets[nets > 0], nets[nets <= 0]
    months = defaultdict(float)
    years = defaultdict(lambda: {"net": 0.0, "trades": 0, "wins": 0})
    for d, v in zip(days, pnl):
        months[str(d)[:7]] += v
        years[str(d)[:4]]["net"] += v
    for t in trades:
        y = str(t["day"])[:4]
        years[y]["trades"] += 1
        years[y]["wins"] += 1 if t["net"] > 0 else 0
    mvals = np.array(list(months.values()))
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0
    traded_days = int((pnl != 0).sum())
    out = {
        "code": code, "name": name, "from": str(days[0]), "to": str(days[-1]), "sessions": len(days),
        "net_pnl": round(float(pnl.sum()), 0),
        "annual_return_pct": round(100 * float(pnl.sum()) / capital / (len(days) / 250.0), 1),
        "avg_per_session": round(float(pnl.mean()), 1),
        "avg_per_traded_day": round(float(pnl[pnl != 0].mean()), 1) if traded_days else 0.0,
        "avg_per_month": round(float(mvals.mean()), 0) if len(mvals) else 0.0,
        "median_month": round(float(np.median(mvals)), 0) if len(mvals) else 0.0,
        "months": len(mvals), "positive_months_pct": round(100 * float((mvals > 0).mean()), 0) if len(mvals) else 0,
        "months_at_target_pct": round(100 * float((mvals >= target).mean()), 0) if len(mvals) else 0,
        "days_at_target_pct": round(100 * float((pnl >= target).mean()), 1),
        "trades": int(len(nets)), "traded_days": traded_days,
        "win_rate_pct": round(100 * len(wins) / len(nets), 1) if len(nets) else None,
        "avg_win": round(float(wins.mean()), 0) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 0) if len(losses) else None,
        "expectancy_per_trade": round(float(nets.mean()), 0) if len(nets) else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
        "max_drawdown": round(max_dd, 0), "max_drawdown_pct": round(100 * max_dd / capital, 1),
        "longest_drawdown_sessions": dd_len,
        "worst_day": round(float(pnl.min()), 0), "best_day": round(float(pnl.max()), 0),
        "worst_month": round(float(mvals.min()), 0) if len(mvals) else 0,
        "best_month": round(float(mvals.max()), 0) if len(mvals) else 0,
        "sharpe": round(float(pnl.mean() / sd * math.sqrt(250)), 2) if sd > 0 else None,
        "charges": round(float(sum(t.get("charges", 0) for t in trades)), 0),
        "gross_pnl": round(float(sum(t.get("gross", 0) for t in trades)), 0),
        "yearly": {y: {"net": round(v["net"], 0), "trades": v["trades"],
                       "win_rate_pct": round(100 * v["wins"] / v["trades"], 0) if v["trades"] else None}
                   for y, v in sorted(years.items())},
        "monthly": {k: round(v, 0) for k, v in sorted(months.items())},
        "in_sample": _part(days, pnl, trades, days < OOS_START, capital),
        "out_of_sample": _part(days, pnl, trades, days >= OOS_START, capital),
        "equity": [[str(d), round(float(e), 0)] for d, e in zip(days[::5], equity[::5])],
        "ruined_on": str(days[ruin[0]]) if len(ruin) else None,
        "target_daily_met": bool(pnl.mean() >= target),
        "target_monthly_met": bool(len(mvals) and mvals.mean() >= target),
    }
    if extra:
        out.update(extra)
    return out
