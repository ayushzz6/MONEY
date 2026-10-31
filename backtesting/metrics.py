"""Backtest statistics. Every rupee figure is after slippage and charges."""
import math
from collections import defaultdict
from typing import Dict, List


def summarize(trades: List[dict], capital: float) -> Dict:
    n = len(trades)
    if not n:
        return {"trades": 0}
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    gross_win = sum(t["net_pnl"] for t in wins)
    gross_loss = -sum(t["net_pnl"] for t in losses)
    net = sum(t["net_pnl"] for t in trades)
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["date"]] += t["net_pnl"]
    equity, peak, max_dd, curve = capital, capital, 0.0, []
    for d in sorted(by_day):
        equity += by_day[d]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        curve.append({"date": d, "pnl": round(by_day[d], 2), "equity": round(equity, 2)})
    daily = list(by_day.values())
    mean = sum(daily) / len(daily)
    sd = math.sqrt(sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)) if len(daily) > 1 else 0.0
    return {
        "trades": n,
        "wins": len(wins),
        "win_rate": round(len(wins) / n, 3),
        "avg_R": round(sum(t["R"] for t in trades) / n, 3),
        "avg_net_R": round(sum(t["net_R"] for t in trades) / n, 3),
        "net_pnl": round(net, 2),
        "gross_pnl": round(sum(t["gross_pnl"] for t in trades), 2),
        "charges": round(sum(t["charges"] for t in trades), 2),
        "expectancy_inr": round(net / n, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown_inr": round(max_dd, 2),
        "days_traded": len(by_day),
        "best_day": round(max(daily), 2),
        "worst_day": round(min(daily), 2),
        "sharpe_daily_annualized": round(mean / sd * math.sqrt(250), 2) if sd > 0 else None,
        "outcomes": {k: sum(1 for t in trades if t["outcome"] == k) for k in ("target", "stop", "flat")},
        "equity_curve": curve,
    }


def by_strategy(trades: List[dict], capital: float) -> Dict[str, Dict]:
    groups = defaultdict(list)
    for t in trades:
        groups[t["strategy"]].append(t)
    return {k: summarize(v, capital) for k, v in sorted(groups.items())}
