"""Broker-side positions for reconciliation (see oms/reconciliation.py)."""
from typing import Dict


def broker_positions(broker) -> Dict[str, int]:
    """Net intraday quantity per symbol at the broker: positive long, negative short."""
    return broker.positions()
