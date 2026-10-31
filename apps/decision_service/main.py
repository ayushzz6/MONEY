"""Decision pipeline: from a closed bar to a trade, or to a recorded reason why not.

  strategy trigger (fresh: its bar closed in the last 6 minutes; or the order-block board on live prices)
  -> market filter (NIFTY 50 with the trade)
  -> exit plan (stop and target set first)
  -> news (no entries on a stock with a blocking headline)
  -> Time Machine (replays and the setup's own recent record on this stock)
  -> optional LLM veto (it can only say no)
  -> sizing (four limits) -> risk gate (with a fresh price) -> order manager

Every signal is journaled with its decision, so the dashboard shows what was taken,
what was skipped, and why.
"""
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from backtesting import timemachine as tm
from backtesting.engine import make_context, market_ok, or_volume_avg, symbol_history
from market.candles.bars import full_session, split_sessions
from portfolio.positions import OPEN_STATES, PENDING_STATES, Trade
from risk.limits import AccountView
from risk.position_sizing import size_position
from strategies.exits import flat_by_slot, plan_exit

logger = logging.getLogger("DecisionEngine")

FRESH_SECONDS = 360


class DecisionPipeline:
    def __init__(self, settings, universe, strategies, market_state, history, scanner, news, gate, kill,
                 oms, book, journal, strategy_gate, llm=None):
        self.settings = settings
        self.universe = universe
        self.strategies = strategies
        self.state = market_state
        self.history = history
        self.scanner = scanner
        self.news = news
        self.gate = gate
        self.kill = kill
        self.oms = oms
        self.book = book
        self.journal = journal
        self.strategy_gate = strategy_gate      # name -> (allowed, why)
        self.llm = llm
        self.fired = set()                      # (date, symbol, strategy)
        self.signals: List[dict] = []
        self._tm_cache: Dict[tuple, dict] = {}

    def new_day(self):
        self.fired.clear()
        self.signals = []
        self._tm_cache.clear()

    # ---------------------------------------------------------------- account
    def account_view(self, date: str) -> AccountView:
        trades = [t for t in self.book.trades.values() if t.date == date]
        live_like = [t for t in self.book.trades.values() if t.status in OPEN_STATES + PENDING_STATES]
        capital = float(self.settings.get("capital_inr", 100000))
        realized = sum(t.net_pnl or 0.0 for t in trades if t.status == "CLOSED")
        lev = max(1.0, float(self.settings.get("intraday_leverage", 1) or 1))
        committed = sum((t.entry_price or t.plan["entry"]) * t.qty for t in live_like) / lev
        counted = [t for t in trades if t.entry_order_id or t.status in PENDING_STATES]
        return AccountView(
            capital=capital,
            cash_available=capital + realized - committed,
            realized_today=realized,
            open_risk=sum(t.open_risk() for t in live_like),
            entries_today=len(counted),
            orders_today=sum(1 for t in trades for o in (t.entry_order_id, t.stop_order_id,
                                                         t.target_order_id, t.exit_order_id) if o),
            open_positions=len(live_like),
            entries_by_symbol=dict(Counter(t.symbol for t in counted)),
            active_symbols={t.symbol for t in live_like},
        )

    # ---------------------------------------------------------------- per bar
    def on_bar(self, symbol: str, now: datetime, date: str):
        bars = self.state.bars(symbol)
        if not bars:
            return
        daily = self.scanner.stats_for(symbol)
        hist = None
        for strat in self.strategies:
            key = (date, symbol, strat.name)
            if key in self.fired or not strat.enabled or strat.live_trigger != "bars" or not strat.wants(daily):
                continue
            if hist is None:
                hist = self.history.intraday60([symbol]).get(symbol, [])
            prior = [b for d, b in split_sessions(hist).items() if d < date]
            ctx = make_context(self.settings, symbol, date,
                               or_volume_avg(prior, int(strat.params.get("or_minutes", 15))))
            sig = strat.first_trigger(bars, daily, ctx)
            if sig is None:
                continue
            self.fired.add(key)
            row = self._row(sig, strat)
            close_ts = sig_close_ts(bars[sig.bar_index])
            if time.time() - close_ts > FRESH_SECONDS:
                self._record(date, row, "STALE", f"triggered at {sig.time}, {int((time.time() - close_ts) // 60)} "
                                                 f"min before the engine saw it: a late entry is a different trade")
                continue
            self._process(sig, bars, daily, hist, strat, now, date, row)

    def on_signal(self, sig, strat, now: datetime, date: str, skip: Optional[str] = None):
        """A trigger the engine found itself (the order-block board on live prices): the same checks from here."""
        key = (date, sig.symbol, strat.name)
        if key in self.fired:
            return None
        self.fired.add(key)
        row = self._row(sig, strat)
        if skip:
            return self._record(date, row, "SKIPPED", skip)
        hist = self.history.intraday60([sig.symbol]).get(sig.symbol, [])
        return self._process(sig, self.state.bars(sig.symbol), self.scanner.stats_for(sig.symbol), hist, strat,
                             now, date, row)

    @staticmethod
    def _row(sig, strat) -> dict:
        return {"ts": time.time(), "time": sig.time, "symbol": sig.symbol, "strategy": strat.name,
                "label": strat.label, "side": sig.side, "entry": sig.entry, "level": sig.level,
                "reason": sig.reason, "context": sig.context}

    def _process(self, sig, bars, daily, hist, strat, now, date, row):
        S = self.settings
        allowed, why = self.strategy_gate(strat.name)
        if not allowed:
            return self._record(date, row, "BENCHED", why)
        why = self.kill.blocked(strat.name, sig.symbol)
        if why:
            return self._record(date, row, "BLOCKED", why)
        if S.get("market_filter.enabled", True):
            idx = self.state.bars(S.get("market_filter.symbol", "^NSEI"))
            if not market_ok(sig.side, idx, sig.slot):
                return self._record(date, row, "SKIPPED", "market filter: the NIFTY 50 is against this trade")
        tick = self.universe.tick(sig.symbol, sig.entry)
        plan, why = plan_exit(sig, bars, tick, S, date)
        if plan is None:
            return self._record(date, row, "REFUSED", f"exit plan: {why}")
        row["plan"] = plan.to_dict()
        why = self.news.block_reason(sig.symbol) if self.news else None
        if why:
            return self._record(date, row, "BLOCKED", why)
        tmc = S.get("timemachine", {}) or {}
        if tmc.get("enabled", True):
            res = self.time_machine(plan, strat, hist)
            row["tm"] = res
            if not res["pass"]:
                return self._record(date, row, "SKIPPED", res["why"])
        if self.llm is not None and (S.get("llm_veto", {}) or {}).get("enabled"):
            ev = self.llm.evaluate_trade_candidate({
                "symbol": sig.symbol, "strategy": strat.label, "side": sig.side, "plan": plan.to_dict(),
                "time_machine": row.get("tm"), "headlines": [n["title"] for n in self.news.for_symbol(sig.symbol)[:8]],
            })
            row["llm"] = ev
            if ev.get("recommendation") == "AVOID":
                return self._record(date, row, "SKIPPED", f"LLM veto: {', '.join(ev.get('reason_codes') or [])}")
        price, age = self.state.price(sig.symbol)
        acct = self.account_view(date)
        entry_for_size = max(plan.entry, price) if (price and plan.long) else (
            min(plan.entry, price) if price else plan.entry)
        size = size_position(entry_for_size, plan.stop, plan.long, S, acct.cash_available)
        row["size"] = size.to_dict()
        if size.refusal:
            return self._record(date, row, "REFUSED", f"sizing: {size.refusal}")
        order = self.order_dict(plan, size.qty)
        v = self.gate.check_entry(order, acct, now, fresh_price=price, price_age_s=age)
        row["verdict"] = v.to_dict()
        if not v.passed:
            return self._record(date, row, "BLOCKED", "risk: " + "; ".join(v.blocks))
        t = Trade(symbol=sig.symbol, strategy=strat.name, side=sig.side, qty=size.qty, date=date, mode=S.mode,
                  plan=plan.to_dict(), stop=plan.stop, target=plan.target, max_loss=v.metrics["max_loss"],
                  verdict=v.to_dict(), size=size.to_dict(), tm=row.get("tm") or {},
                  news=[n["title"] for n in self.news.for_symbol(sig.symbol)[:5]] if self.news else [])
        t.log(f"{strat.label}: {sig.reason}")
        row["trade_id"] = t.id
        self.oms.submit(t)
        decision = "AWAITING_APPROVAL" if t.status == "PENDING_APPROVAL" else (
            "TRADED" if t.status in ("ENTRY_PLACED", "OPEN") else t.status)
        self._record(date, row, decision, f"{size.qty} shares, bound by {size.binding}; max loss Rs {t.max_loss:,.0f}")

    def order_dict(self, plan, qty) -> dict:
        return {"symbol": plan.symbol, "side": plan.side, "qty": int(qty), "order_type": "market",
                "entry": plan.entry, "stop": plan.stop, "target": plan.target,
                "strategy": plan.strategy, "date": plan.date}

    def recheck(self, t: Trade, now: datetime) -> Optional[str]:
        """The risk gate again, at the moment a LIVE trade is approved."""
        price, age = self.state.price(t.symbol)
        acct = self.account_view(t.date)
        # this trade is still pending: do not count it against itself
        acct.entries_today = max(0, acct.entries_today - 1)
        acct.open_positions = max(0, acct.open_positions - 1)
        acct.open_risk = max(0.0, acct.open_risk - t.open_risk())
        acct.active_symbols.discard(t.symbol)
        acct.entries_by_symbol[t.symbol] = max(0, acct.entries_by_symbol.get(t.symbol, 1) - 1)
        acct.cash_available += t.plan["entry"] * t.qty / max(1.0, float(self.settings.get("intraday_leverage", 1) or 1))
        v = self.gate.check_entry(self.order_dict(_PlanView(t), t.qty), acct, now, fresh_price=price, price_age_s=age)
        t.verdict = v.to_dict()
        return None if v.passed else "; ".join(v.blocks)

    # ---------------------------------------------------------------- the Time Machine
    def time_machine(self, plan, strat, hist) -> dict:
        S = self.settings
        tmc = S.get("timemachine", {}) or {}
        key = (plan.symbol, plan.strategy, plan.date, plan.entry, plan.stop, plan.target)
        if key in self._tm_cache:
            return self._tm_cache[key]
        days = split_sessions(hist)
        sessions = [fs for fs in (full_session(b) for d, b in days.items() if d < plan.date) if fs][-60:]
        out = {"sessions": len(sessions), "seed": int(tmc.get("seed", 19))}
        if len(sessions) < int(tmc.get("min_sessions", 20)):
            out.update({"pass": False, "why": f"time machine: only {len(sessions)} full sessions of history"})
            return out
        last = flat_by_slot(plan.flat_by)
        mc = tm.simulate(tm.build_pool(sessions), plan.entry, plan.stop, plan.target, plan.entry_slot, last,
                         replays=int(tmc.get("replays", 5000)), seed=int(tmc.get("seed", 19)),
                         window=int(tmc.get("window", 2)), long=plan.long)
        daily = self.history.daily([plan.symbol]).get(plan.symbol, [])
        idx = self.history.intraday60([S.get("market_filter.symbol", "^NSEI")])
        idx_bars = next(iter(idx.values()), None) if S.get("market_filter.enabled", True) else None
        past = symbol_history(strat, plan.symbol, hist, daily, S, plan.tick, idx_bars, before=plan.date)
        hs = tm.history_summary(past)
        ok, why = tm.gate(mc, hs, S)
        out.update({"pass": ok, "why": why, "mc": mc, "history": hs,
                    "verdict": tm.verdict(mc, plan.flat_by), "notes": tm.notes(mc, plan.flat_by)})
        self._tm_cache[key] = out
        return out

    # ---------------------------------------------------------------- journal
    def _record(self, date, row, decision, reason):
        row["decision"], row["why"] = decision, reason
        self.signals.insert(0, row)
        del self.signals[300:]
        try:
            self.journal.save_signal(date, row)
        except Exception as e:
            logger.error("journal failed: %s", e)
        logger.info("%s %s %s %s: %s", row["time"], row["symbol"], row["strategy"], decision, reason)
        return row


class _PlanView:
    """Adapts a Trade back to the plan fields order_dict() reads."""

    def __init__(self, t: Trade):
        self.symbol, self.side, self.entry = t.symbol, t.side, t.plan["entry"]
        self.stop, self.target, self.strategy, self.date = t.stop, t.target, t.strategy, t.date


def sig_close_ts(bar) -> float:
    return bar.start + 300
