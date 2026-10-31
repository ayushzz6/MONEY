"""Auto-select: which strategies may trade, from the latest walk-forward backtest.

A strategy trades when it is enabled and, with auto_select on, its recent backtest is
positive after costs (avg_net_R >= min_expectancy_r over at least min_trades trades).
In PAPER, a strategy with too few trades to judge runs on "probation" so it can build a
forward record; LIVE needs the full proof. A strategy marked paper_only trades in PAPER
whatever its backtest says (a paper trial you asked for) and never in LIVE.
"""
import json
from typing import Optional, Tuple

from core.config.settings import DATA_DIR

LATEST = DATA_DIR / "backtests" / "latest.json"


def load_latest() -> Optional[dict]:
    try:
        return json.loads(LATEST.read_text())
    except (OSError, ValueError):
        return None


def strategy_allowed(settings, backtest: Optional[dict], name: str) -> Tuple[bool, str]:
    p = settings.strategy(name)
    if not p.get("enabled", True):
        return False, "disabled in configs/trading.json"
    if p.get("paper_only"):
        if settings.live:
            return False, f"paper only: it never trades real money (strategies.{name}.paper_only)"
        m = ((backtest or {}).get("by_strategy") or {}).get(name) or {}
        note = (f"; the re-test made {m['avg_net_R']:+.2f}R per trade after costs over {m['trades']} trades"
                if m.get("trades") and m.get("avg_net_R") is not None else "")
        return True, "paper trial: trades on paper whatever the re-test says, never live" + note
    a = settings.get("auto_select", {}) or {}
    if not a.get("enabled"):
        return True, "auto-select off: trades whenever enabled"
    if not backtest or not backtest.get("overall"):
        return (False, "benched: no backtest yet (LIVE needs one)") if settings.live else \
            (True, "probation: no backtest yet (paper)")
    m = (backtest.get("by_strategy") or {}).get(name)
    n = (m or {}).get("trades", 0)
    need = int(a.get("min_trades", 15))
    if not m or n < need:
        if settings.live:
            return False, f"benched: {n} backtest trades, LIVE needs {need}"
        return True, f"probation: {n} backtest trades (needs {need} to be proven)"
    er = m.get("avg_net_R", 0.0)
    if er < float(a.get("min_expectancy_r", 0.0)):
        return False, f"benched: backtest {er:+.2f}R per trade after costs over {n} trades"
    return True, f"active: backtest {er:+.2f}R per trade after costs over {n} trades"
