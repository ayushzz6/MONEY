"""Indian intraday equity charges and trade P&L.

Default rates (configs/trading.json -> costs) are the usual NSE intraday schedule:
brokerage min(Rs 20, 0.03%) per executed order, STT 0.025% on the sell side, NSE
transaction charges 0.00297%, SEBI fee Rs 10 per crore, stamp duty 0.003% on the buy side,
GST 18% on brokerage + transaction + SEBI charges. Check them against your broker's
current charge sheet: costs decide whether a small intraday edge survives.
"""
from typing import Dict


def order_charges(value: float, side: str, costs: Dict) -> float:
    """Charges for ONE executed order of `value` rupees. side: buy or sell."""
    if value <= 0:
        return 0.0
    brokerage = min(float(costs.get("brokerage_per_order_inr", 20)),
                    value * float(costs.get("brokerage_pct", 0.03)) / 100.0)
    txn = value * float(costs.get("exchange_txn_pct", 0.00297)) / 100.0
    sebi = value * float(costs.get("sebi_pct", 0.0001)) / 100.0
    stt = value * float(costs.get("stt_sell_pct", 0.025)) / 100.0 if side == "sell" else 0.0
    stamp = value * float(costs.get("stamp_buy_pct", 0.003)) / 100.0 if side == "buy" else 0.0
    gst = (brokerage + txn + sebi) * float(costs.get("gst_pct", 18)) / 100.0
    return brokerage + txn + sebi + stt + stamp + gst


def round_trip_charges(entry: float, exit_price: float, qty: int, long: bool, costs: Dict) -> float:
    open_side, close_side = ("buy", "sell") if long else ("sell", "buy")
    return order_charges(entry * qty, open_side, costs) + order_charges(exit_price * qty, close_side, costs)


def gross_pnl(entry: float, exit_price: float, qty: int, long: bool) -> float:
    return (exit_price - entry) * qty if long else (entry - exit_price) * qty


def slip(price: float, bps: float, adverse_for_buy: bool) -> float:
    """Price after slippage: a buy pays up, a sell gets less."""
    k = bps / 10000.0
    return price * (1 + k) if adverse_for_buy else price * (1 - k)
