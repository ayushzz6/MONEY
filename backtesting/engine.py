"""Walk-forward backtest of the live pipeline, and the per-stock history the Time Machine uses.

run_day() is one strategy on one stock for one session: trigger, market filter, exit plan,
and how the bracket ended. Backtester.run() replays whole days the way the engine trades
them: the watchlist picked from the daily context known that morning, every strategy,
the Time Machine gate (from sessions before that day only), sizing, the same RiskGate as
live (trades per day, daily loss, open positions, cash), slippage and charges.

With Yahoo's 60 days of 5-minute bars and a 20-session warm-up for the Time Machine,
about 40 sessions are tested. That is a small sample: treat the numbers as a sanity
check, not as proof of an edge.
"""
import datetime as dt
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from backtesting import timemachine as tm
from backtesting.broker_simulator import simulate_bracket
from backtesting.metrics import by_strategy, summarize
from market.calendar import IST, parse_hhmm
from market.candles.bars import Bar, DailyBar, full_session, opening_range, split_sessions
from portfolio.pnl import gross_pnl, round_trip_charges, slip
from risk.limits import AccountView, RiskGate
from risk.position_sizing import size_position
from scanner.features.daily import compute_daily_stats
from scanner.ranking.watchlist import rank_watchlist, score
from strategies.base import BaseStrategy, DayContext
from strategies.exits import flat_by_slot, plan_exit

logger = logging.getLogger("Backtest")


def or_volume_avg(prior_sessions: List[List[Bar]], or_minutes: int, last_n: int = 20) -> Optional[float]:
    n = or_minutes // 5
    vols = []
    for D in prior_sessions[-last_n:]:
        orr = opening_range(D, n)
        if orr and orr[2] > 0:
            vols.append(orr[2])
    return sum(vols) / len(vols) if len(vols) >= 5 else None


def market_ok(side: str, index_bars: Optional[List[Bar]], slot: int) -> bool:
    """Long only when the NIFTY 50 is above its open at the trigger bar; short the opposite."""
    if not index_bars:
        return True
    upto = [b for b in index_bars if b.slot <= slot]
    if not upto:
        return True
    up = upto[-1].c >= upto[0].o
    return up if side == "buy" else not up


def make_context(settings, symbol: str, date: str, or_avg: Optional[float]) -> DayContext:
    return DayContext(
        symbol=symbol, date=date,
        entry_from=parse_hhmm(settings.get("no_entries_before", "09:30")),
        entry_until=parse_hhmm(settings.get("no_entries_after", "14:30")),
        allow_short=bool(settings.get("allow_short", False)),
        or_volume_avg=or_avg,
    )


def run_day(strategy: BaseStrategy, ctx: DayContext, bars: List[Bar], daily, settings, tick: float,
            index_bars: Optional[List[Bar]] = None) -> Optional[dict]:
    sig = strategy.first_trigger(bars, daily, ctx)
    if sig is None:
        return None
    base = {"date": ctx.date, "symbol": ctx.symbol, "strategy": strategy.name, "side": sig.side,
            "time": sig.time, "slot": sig.slot, "signal": sig}
    if settings.get("market_filter.enabled", True) and not market_ok(sig.side, index_bars, sig.slot):
        return dict(base, skipped="market filter: NIFTY 50 against the trade")
    plan, why = plan_exit(sig, bars, tick, settings, ctx.date)
    if plan is None:
        return dict(base, skipped=f"exit plan: {why}")
    flat = flat_by_slot(plan.flat_by)
    b = bars[sig.bar_index]
    if sig.context.get("fill") == "limit" and ((b.l <= plan.stop) if plan.long else (b.h >= plan.stop)):
        # filled at the entry level inside the trigger bar: a stop touched in that bar counts as a loss
        res = {"outcome": "stop", "exit": plan.stop, "exit_slot": b.slot}
    else:
        res = simulate_bracket(plan.long, plan.stop, plan.target, bars[sig.bar_index + 1:], flat)
    if res["exit"] is None:
        return dict(base, skipped="no bars after the entry")
    move = (res["exit"] - plan.entry) if plan.long else (plan.entry - res["exit"])
    R = plan.target_r if res["outcome"] == "target" else (-1.0 if res["outcome"] == "stop"
                                                           else move / plan.risk_per_share)
    return dict(base, plan=plan, outcome=res["outcome"], exit=res["exit"], exit_slot=res["exit_slot"],
                R=round(R, 3))


def symbol_history(strategy: BaseStrategy, symbol: str, bars: List[Bar], daily: List[DailyBar], settings,
                   tick: float, index_bars: Optional[List[Bar]] = None, before: Optional[str] = None) -> List[dict]:
    """The strategy's rule on each past session of one stock (the Time Machine's history)."""
    days = split_sessions(bars)
    idx = split_sessions(index_bars) if index_bars else {}
    ordered = [(d, b) for d, b in days.items() if before is None or d < before]
    out = []
    near = float(settings.get("strategies.level_breakout.near_pct", 3.0))
    for k, (d, day_bars) in enumerate(ordered):
        st = compute_daily_stats(symbol, daily, before=d, near_pct=near)
        if not strategy.wants(st):
            continue
        prior = [b for _, b in ordered[max(0, k - 20):k]]
        ctx = make_context(settings, symbol, d, or_volume_avg(prior, int(strategy.params.get("or_minutes", 15))))
        r = run_day(strategy, ctx, day_bars, st, settings, tick, idx.get(d))
        if r and not r.get("skipped"):
            out.append(r)
    return out


class _NoKill:
    @staticmethod
    def blocked(strategy=None, symbol=None):
        return None


class Backtester:
    def __init__(self, settings, universe, strategies: List[BaseStrategy]):
        self.settings = settings
        self.universe = universe
        self.strategies = [s for s in strategies if s.enabled]
        self.gate = RiskGate(settings, _NoKill(), universe)

    def run(self, intraday: Dict[str, List[Bar]], daily: Dict[str, List[DailyBar]],
            index_bars: Optional[List[Bar]] = None, use_tm: bool = True,
            progress: Optional[Callable[[str], None]] = None) -> Dict:
        S = self.settings
        tmc = S.get("timemachine", {}) or {}
        warmup = int(tmc.get("min_sessions", 20))
        sessions = {s: split_sessions(b) for s, b in intraday.items()}
        index_days = split_sessions(index_bars) if index_bars else {}
        dates = sorted({d for days in sessions.values() for d in days})
        if len(dates) <= warmup:
            return {"error": f"only {len(dates)} sessions of data, need more than {warmup}"}
        test_dates = dates[warmup:]
        if progress:
            progress(f"{len(intraday)} stocks, {len(dates)} sessions, testing {len(test_dates)} after a {warmup}-session warm-up")

        # 1. every strategy on every stock-day: the raw record (also the Time Machine's history)
        raw: Dict[tuple, dict] = {}
        stats_cache: Dict[tuple, object] = {}
        full: Dict[str, Dict[str, List[Bar]]] = defaultdict(dict)
        for sym, days in sessions.items():
            tick = self.universe.tick(sym) if self.universe else 0.05
            ordered = list(days.items())
            for k, (d, bars) in enumerate(ordered):
                fs = full_session(bars)
                if fs:
                    full[sym][d] = fs
                st = compute_daily_stats(sym, daily.get(sym, []), before=d,
                                         near_pct=float(S.get("strategies.level_breakout.near_pct", 3.0)))
                stats_cache[(sym, d)] = st
                prior = [b for _, b in ordered[max(0, k - 20):k]]
                for strat in self.strategies:
                    if not strat.wants(st):
                        continue
                    ctx = make_context(S, sym, d, or_volume_avg(prior, int(strat.params.get("or_minutes", 15))))
                    r = run_day(strat, ctx, bars, st, S, tick, index_days.get(d))
                    if r:
                        raw[(sym, strat.name, d)] = r

        by_pair: Dict[tuple, List[dict]] = defaultdict(list)
        for (sym, name, d), r in sorted(raw.items(), key=lambda kv: kv[0][2]):
            if not r.get("skipped"):
                by_pair[(sym, name)].append(r)

        # 2. day by day, as the engine would have traded it
        capital = float(S.get("capital_inr", 100000))
        costs = S.get("costs", {}) or {}
        bps = float(S.get("slippage_bps", 5))
        size_n = int(S.get("universe.watchlist_size", 50))
        trades, skipped = [], defaultdict(int)
        for d in test_dates:
            day_stats = {s: stats_cache.get((s, d)) for s in sessions if stats_cache.get((s, d))}
            watch = rank_watchlist(day_stats, size_n, float(S.get("universe.min_price", 50)),
                                   float(S.get("universe.min_turnover_cr", 15)))
            wl = {s for s, _ in watch}
            cands = [raw[(s, st.name, d)] for s in wl for st in self.strategies if (s, st.name, d) in raw]
            cands.sort(key=lambda r: (r["slot"], -score(day_stats[r["symbol"]])))
            open_trades: List[dict] = []
            closed_today: List[dict] = []
            entries_by_symbol = defaultdict(int)
            for c in cands:
                if c.get("skipped"):
                    skipped[c["skipped"].split(":")[0]] += 1
                    continue
                plan = c["plan"]
                slot = c["slot"]
                still_open = [t for t in open_trades if t["exit_slot"] > slot]
                done = [t for t in open_trades if t["exit_slot"] <= slot]
                closed_today.extend(done)
                open_trades = still_open
                if use_tm and tmc.get("enabled", True):
                    ok, why = self._tm_gate(c, full, by_pair, dates, d, tmc)
                    if not ok:
                        skipped["time machine"] += 1
                        continue
                long = plan.long
                fill_entry = slip(plan.entry, bps, adverse_for_buy=long)
                lev = max(1.0, float(S.get("intraday_leverage", 1) or 1))
                cash = capital + sum(t["net_pnl"] for t in closed_today) - sum(t["value"] for t in open_trades) / lev
                size = size_position(fill_entry, plan.stop, long, S, cash)
                if size.refusal or size.qty <= 0:
                    skipped["sizing"] += 1
                    continue
                account = AccountView(
                    capital=capital, cash_available=cash,
                    realized_today=sum(t["net_pnl"] for t in closed_today),
                    open_risk=sum(t["max_loss"] for t in open_trades),
                    entries_today=len(open_trades) + len(closed_today),
                    orders_today=2 * (len(open_trades) + len(closed_today)),
                    open_positions=len(open_trades),
                    entries_by_symbol=dict(entries_by_symbol),
                    active_symbols={t["symbol"] for t in open_trades},
                )
                now = dt.datetime.combine(dt.date.fromisoformat(d), dt.time(*divmod(parse_hhmm(plan.entry_time), 60)), IST)
                order = {"symbol": plan.symbol, "side": plan.side, "qty": size.qty, "order_type": "market",
                         "entry": fill_entry, "stop": plan.stop, "target": plan.target,
                         "strategy": plan.strategy, "date": d}
                v = self.gate.check_entry(order, account, now)
                if not v.passed:
                    skipped["risk: " + v.blocks[0].split(":")[0]] += 1
                    continue
                exit_px = c["exit"] if c["outcome"] == "target" else slip(c["exit"], bps, adverse_for_buy=not long)
                g = gross_pnl(fill_entry, exit_px, size.qty, long)
                ch = round_trip_charges(fill_entry, exit_px, size.qty, long, costs)
                planned_risk = size.qty * plan.risk_per_share
                t = {"date": d, "symbol": plan.symbol, "strategy": plan.strategy, "side": plan.side,
                     "entry_time": plan.entry_time, "entry": round(fill_entry, 2), "stop": plan.stop,
                     "target": plan.target, "exit": round(exit_px, 2), "exit_slot": c["exit_slot"],
                     "outcome": c["outcome"], "qty": size.qty, "value": round(size.qty * fill_entry, 2),
                     "max_loss": size.max_loss, "R": c["R"], "gross_pnl": round(g, 2), "charges": round(ch, 2),
                     "net_pnl": round(g - ch, 2), "net_R": round((g - ch) / planned_risk, 3) if planned_risk else 0.0}
                open_trades.append(t)
                entries_by_symbol[plan.symbol] += 1
                trades.append(t)
        overall = summarize(trades, capital)
        return {
            "from": test_dates[0], "to": test_dates[-1], "sessions": len(test_dates), "stocks": len(intraday),
            "warmup_sessions": warmup, "time_machine": bool(use_tm and tmc.get("enabled", True)),
            "overall": overall, "by_strategy": by_strategy(trades, capital),
            "skipped": dict(sorted(skipped.items(), key=lambda kv: -kv[1])),
            "trades": trades[-500:],
        }

    def _tm_gate(self, c, full, by_pair, dates, d, tmc):
        sym, plan = c["symbol"], c["plan"]
        prior = [full[sym][x] for x in dates if x < d and x in full[sym]][-60:]
        if len(prior) < int(tmc.get("min_sessions", 20)):
            return False, "time machine: not enough sessions"
        pool = tm.build_pool(prior)
        last = flat_by_slot(plan.flat_by)
        mc = tm.simulate(pool, plan.entry, plan.stop, plan.target, plan.entry_slot, last,
                         replays=int(tmc.get("replays", 5000)), seed=int(tmc.get("seed", 19)),
                         window=int(tmc.get("window", 2)), long=plan.long)
        start = prior[0][0].date
        hist = tm.history_summary([r for r in by_pair.get((sym, plan.strategy), []) if start <= r["date"] < d])
        return tm.gate(mc, hist, self.settings)
