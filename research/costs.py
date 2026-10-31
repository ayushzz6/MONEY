"""Charges per executed order, current NSE schedules (Oct 2024 onward, applied to the whole test).

Equity intraday: configs/trading.json costs (brokerage min(Rs 20, 0.03%), STT 0.025% sell, ...).
Index options: brokerage Rs 20, STT 0.1% of premium on the sell side, NSE transaction 0.03503%
of premium, SEBI Rs 10/crore, stamp 0.003% on buys, GST 18% on brokerage+transaction+SEBI.
Index futures: brokerage Rs 20, STT 0.02% sell, NSE transaction 0.00173%, stamp 0.002% buys.
"""
from portfolio.pnl import order_charges as equity_order_charges  # noqa: F401  (re-exported)

GST = 0.18
SEBI = 0.0001 / 100


def option_order_charges(premium_value: float, side: str) -> float:
    if premium_value <= 0:
        return 20.0 * (1 + GST)
    brokerage = 20.0
    txn = premium_value * 0.03503 / 100
    sebi = premium_value * SEBI
    stt = premium_value * 0.1 / 100 if side == "sell" else 0.0
    stamp = premium_value * 0.003 / 100 if side == "buy" else 0.0
    return brokerage + txn + sebi + stt + stamp + GST * (brokerage + txn + sebi)


def futures_order_charges(value: float, side: str) -> float:
    brokerage = 20.0
    txn = value * 0.00173 / 100
    sebi = value * SEBI
    stt = value * 0.02 / 100 if side == "sell" else 0.0
    stamp = value * 0.002 / 100 if side == "buy" else 0.0
    return brokerage + txn + sebi + stt + stamp + GST * (brokerage + txn + sebi)


def option_slippage(premium: float) -> float:
    """Per side, per unit: half a typical NIFTY spread, wider in rupees for richer options."""
    return max(0.5, 0.005 * premium)
