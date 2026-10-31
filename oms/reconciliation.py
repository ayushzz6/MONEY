"""LIVE reconciliation (plan.md section 25): the broker's intraday positions must equal the
open trades. A mismatch trips the kill switch: no new entries until a human looks."""
import logging
from typing import Dict, List

logger = logging.getLogger("Reconciliation")


def expected_positions(open_trades) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in open_trades:
        if t.status in ("OPEN", "EXITING"):
            out[t.symbol] = out.get(t.symbol, 0) + (t.qty if t.long else -t.qty)
    return out


def reconcile(broker_positions: Dict[str, int], open_trades) -> List[str]:
    exp = expected_positions(open_trades)
    problems = []
    for sym in sorted(set(exp) | set(broker_positions)):
        a, b = exp.get(sym, 0), broker_positions.get(sym, 0)
        if a != b:
            problems.append(f"{sym}: the engine expects {a:+d} shares, the broker shows {b:+d}")
    return problems
