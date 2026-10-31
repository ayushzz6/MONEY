"""Risk gate (trader-risk, in rupees, NSE hours). Every entry passes through it before it can
reach a broker or the approval queue. A BLOCK is final for that order.

Entries (open or add to a position) get every check. Exits only get a format check:
getting out is never blocked by a day limit.

Daily loss: realized losses today + the loss at the stop of every open position and
pending entry + this order's loss at the stop must fit max_daily_loss_inr. Profits do
not extend the limit. If every stop fills at its price, the day cannot lose more than
the limit; a gap or a fast market can still fill worse.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from market.calendar import minute_of, parse_hhmm, session_state
from portfolio.pnl import round_trip_charges


@dataclass
class AccountView:
    capital: float
    cash_available: float
    realized_today: float = 0.0
    open_risk: float = 0.0
    entries_today: int = 0
    orders_today: int = 0
    open_positions: int = 0
    entries_by_symbol: Dict[str, int] = field(default_factory=dict)
    active_symbols: Set[str] = field(default_factory=set)   # open or pending


@dataclass
class Verdict:
    passed: bool
    blocks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        return "PASS" if self.passed else "BLOCK: " + "; ".join(self.blocks)


def fingerprint(order: dict) -> str:
    keys = ("symbol", "side", "qty", "order_type", "stop", "target", "date", "strategy")
    blob = json.dumps({k: order.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class RiskGate:
    def __init__(self, settings, kill_switch, universe=None):
        self.settings = settings
        self.kill = kill_switch
        self.universe = universe

    def check_entry(self, order: dict, account: AccountView, now: datetime,
                    fresh_price: Optional[float] = None, price_age_s: Optional[float] = None) -> Verdict:
        """order: {symbol, side, qty, order_type, entry, stop, target, strategy, date}"""
        cfg = self.settings
        blocks, warns = [], []
        sym, side, strategy = order.get("symbol"), order.get("side"), order.get("strategy")
        qty = order.get("qty")
        long = side == "buy"

        why = self.kill.blocked(strategy, sym)
        if why:
            blocks.append(why)

        # format
        if not sym:
            blocks.append("format: no symbol")
        if side not in ("buy", "sell"):
            blocks.append(f"format: side must be buy or sell, got {side!r}")
        if not isinstance(qty, int) or qty <= 0:
            blocks.append(f"format: quantity must be a whole number above zero, got {qty!r}")
            qty = 0
        entry, stop, target = order.get("entry"), order.get("stop"), order.get("target")
        if not all(isinstance(x, (int, float)) and x > 0 for x in (entry, stop or 0.01, target or 0.01)):
            blocks.append("format: entry, stop and target must be positive prices")
            return Verdict(False, blocks, warns, fingerprint=fingerprint(order))

        # asset
        if self.universe is not None and sym not in self.universe.instruments:
            blocks.append(f"asset: {sym} is not an NSE equity in the NIFTY 500 universe")
        # short sales
        if not long and not cfg.get("allow_short", False):
            blocks.append("short: allow_short is false")
        if sym in (cfg.get("blocked") or []):
            blocks.append(f"blocked: {sym} is on your blocked list")

        # the entry the risk is measured from: never better than a fresh price
        measured = float(entry)
        if fresh_price:
            measured = max(entry, fresh_price) if long else min(entry, fresh_price)
            slip_pct = 100.0 * ((fresh_price - entry) / entry if long else (entry - fresh_price) / entry)
            max_slip = float(cfg.get("max_entry_slip_pct", 0.3))
            if slip_pct > max_slip:
                blocks.append(f"chase: the price {fresh_price:.2f} is {slip_pct:.2f}% past the planned entry "
                              f"{entry:.2f} (limit {max_slip}%): the move already happened")
        else:
            warns.append("no fresh price: risk measured from the plan's entry")
        stale = float(cfg.get("data.stale_seconds", 180))
        if price_age_s is not None and price_age_s > stale:
            blocks.append(f"stale data: the last price is {price_age_s:.0f}s old (limit {stale:.0f}s)")

        # stop and loss
        if cfg.get("require_stop", True) and not stop:
            blocks.append("stop: a stop is required")
        risk = (measured - stop) if long else (stop - measured)
        if stop and risk <= 0:
            blocks.append(f"stop: {stop:.2f} is on the wrong side of the entry {measured:.2f}")
        costs = cfg.get("costs", {}) or {}
        charges = round_trip_charges(measured, stop, qty, long, costs) if qty else 0.0
        loss = qty * max(risk, 0) + charges
        max_loss = float(cfg.get("max_loss_per_trade_inr", 0))
        if loss > max_loss + 1e-6:
            blocks.append(f"loss: {qty} x {risk:.2f} + Rs {charges:.2f} charges = Rs {loss:.2f} at the stop, "
                          f"over your Rs {max_loss:.0f} max loss per trade")
        rr = (((target - measured) if long else (measured - target)) / risk) if risk > 0 else 0.0
        min_rr = float(cfg.get("min_reward_risk", 1.0))
        if rr + 1e-9 < min_rr:
            blocks.append(f"reward:risk {rr:.2f} from {measured:.2f} is under your minimum {min_rr:g}")

        # size, position, cash
        value = qty * measured
        if value < float(cfg.get("min_order_inr", 0)):
            blocks.append(f"size: Rs {value:,.0f} is under the Rs {cfg.get('min_order_inr'):,} minimum order")
        if value > float(cfg.get("max_order_inr", 1e18)):
            blocks.append(f"size: Rs {value:,.0f} is over the Rs {cfg.get('max_order_inr'):,} max order")
        cap = account.capital * float(cfg.get("max_position_pct", 100)) / 100.0
        if value > cap + 1e-6:
            blocks.append(f"position: Rs {value:,.0f} is over {cfg.get('max_position_pct')}% of capital (Rs {cap:,.0f})")
        buffer = account.capital * float(cfg.get("cash_buffer_pct", 0)) / 100.0
        lev = max(1.0, float(cfg.get("intraday_leverage", 1) or 1))
        room = (account.cash_available - buffer) * lev
        if value > room + 1e-6:
            blocks.append(f"cash: Rs {value:,.0f} needs more than the Rs {max(room, 0):,.0f} available after the "
                          f"Rs {buffer:,.0f} cash buffer")

        # day limits
        realized_loss = max(0.0, -account.realized_today)
        max_daily = float(cfg.get("max_daily_loss_inr", 0))
        committed = realized_loss + account.open_risk + loss
        if committed > max_daily + 1e-6:
            blocks.append(f"daily loss: Rs {realized_loss:,.0f} lost today + Rs {account.open_risk:,.0f} open risk + "
                          f"Rs {loss:,.0f} for this trade = Rs {committed:,.0f}, over your Rs {max_daily:,.0f} limit")
        if account.entries_today >= int(cfg.get("max_trades_per_day", 99)):
            blocks.append(f"trades: {account.entries_today} of {cfg.get('max_trades_per_day')} entries already today")
        per_sym = int(cfg.get("max_trades_per_symbol", 99))
        if account.entries_by_symbol.get(sym, 0) >= per_sym:
            blocks.append(f"duplicate: {sym} already traded {account.entries_by_symbol[sym]} time(s) today "
                          f"(limit {per_sym})")
        if sym in account.active_symbols:
            blocks.append(f"duplicate: {sym} already has an open position or a pending entry")
        if account.open_positions >= int(cfg.get("max_open_positions", 99)):
            blocks.append(f"positions: {account.open_positions} open, the limit is {cfg.get('max_open_positions')}")
        if account.orders_today >= int(cfg.get("max_orders_per_day", 999)):
            blocks.append(f"daily limit: {account.orders_today} orders already today")

        # window
        state = session_state(now)
        m = minute_of(now)
        if state != "OPEN":
            blocks.append(f"window: the market is {state.lower().replace('_', ' ')}")
        else:
            nb, na = cfg.get("no_entries_before"), cfg.get("no_entries_after")
            if nb and m < parse_hhmm(nb):
                blocks.append(f"window: no new entries before {nb}")
            if na and m > parse_hhmm(na):
                blocks.append(f"window: no new entries after {na}")

        if cfg.live:
            warns.append("LIVE account: real money")
        metrics = {
            "measured_entry": round(measured, 2), "risk_per_share": round(risk, 4), "qty": qty,
            "value": round(value, 2), "charges_at_stop": round(charges, 2), "max_loss": round(loss, 2),
            "reward_risk": round(rr, 2), "daily_committed": round(committed, 2), "daily_limit": max_daily,
            "entries_today": account.entries_today, "max_trades_per_day": cfg.get("max_trades_per_day"),
        }
        return Verdict(not blocks, blocks, warns, metrics, fingerprint(order))

    @staticmethod
    def check_exit(symbol: str, qty: int, held_qty: int) -> Verdict:
        blocks = []
        if not isinstance(qty, int) or qty <= 0:
            blocks.append(f"format: quantity must be a whole number above zero, got {qty!r}")
        elif qty > held_qty:
            blocks.append(f"exit of {qty} is more than the {held_qty} held: it would open a new position")
        return Verdict(not blocks, blocks)
