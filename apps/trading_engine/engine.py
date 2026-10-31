"""The trading engine: one process that runs the day by itself.

Before the open (or at start-up): load the NIFTY 500 universe, scan it, build the
watchlist, fetch the Time Machine's history, re-run the walk-forward backtest that decides
which strategies may trade (auto_select), and read the news.

During the session, once a second: new closed bars go to the strategies, prices go to
the open trades (paper fills, live targets), the order manager advances every trade, the
kill switch watches the day's losses, and LIVE positions are reconciled with Angel One.
At the flat-by time everything is closed. After the close: a summary.
"""
import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, Optional

from apps.decision_service.main import DecisionPipeline
from apps.market_data_service.main import FeedManager
from apps.scanner_service.main import ScannerService
from backtesting.run import run_backtest
from backtesting.selection import load_latest, strategy_allowed
from brokers.paper import PaperBroker
from core.config.settings import STATE_DIR, Settings
from core.utilities.notify import Notifier
from core.utilities.store import Journal
from market import calendar as cal
from market.candles.bars import opening_range, split_sessions, vwap_series
from market.history import HistoryStore
from market.instruments.universe import Universe
from market.intel.deals import DealsTracker
from market.intel.results import ResultsAnalyst
from market.news.feed import NewsMonitor
from market.state.manager import MarketState
from oms.order_manager import OrderManager
from oms.reconciliation import reconcile
from portfolio.positions import FINAL_STATES, TradeBook
from risk.kill_switch import KillSwitch
from risk.limits import RiskGate
from strategies.base import DayContext
from strategies.ensemble import build_strategies

logger = logging.getLogger("TradingEngine")

ALERTS_PATH = STATE_DIR / "price_alerts.json"


class Engine:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        S = self.settings
        self.universe = Universe()
        self.state = MarketState()
        self.history = HistoryStore()
        self.kill = KillSwitch()
        self.book = TradeBook()
        self.journal = Journal()
        self.notifier = Notifier(S)
        self.news = NewsMonitor(S, self.universe)
        self.results = ResultsAnalyst(S, self.universe)
        self.deals = DealsTracker(S, self.universe)
        self.scanner = ScannerService(S, self.universe, self.history)
        from apps.trading_engine.obboard import OrderBlockBoard
        self.obboard = OrderBlockBoard(self)
        self.strategies = build_strategies(S)
        self.feed = FeedManager(S, self.universe, self.state)
        self.gate = RiskGate(S, self.kill, self.universe)
        self.broker = PaperBroker(self.state, float(S.get("slippage_bps", 5)))
        self.oms = OrderManager(S, self.broker, self.book, self.journal, self.state, on_event=self._on_event)
        self.llm = None
        self.pipeline = None
        self.adapter = None
        self.alerts = deque(maxlen=300)
        self.price_alerts = self._load_price_alerts()
        self.backtest = load_latest()
        self.jobs: Dict[str, dict] = {}
        self._jobq: "queue.Queue" = queue.Queue()
        self.ready = False
        self.fatal: Optional[str] = None
        self.day: Optional[str] = None
        self.trade_date: Optional[str] = None
        self.prepared_for: Optional[str] = None
        self._prepare_tried = 0.0
        self._last_seen: Dict[str, int] = {}
        self._flags = set()
        self._last_reconcile = 0.0
        self._last_news = 0.0
        self._last_intel = 0.0
        self.started_at = time.time()

    # ================================================================ start-up
    def start(self):
        threading.Thread(target=self._job_worker, name="jobs", daemon=True).start()
        self.submit_job("startup", self._startup)
        threading.Thread(target=self._loop, name="engine", daemon=True).start()

    def _startup(self):
        S = self.settings
        self.universe.load()
        creds = S.angel_credentials()
        provider = str(S.get("data.provider", "auto")).lower()
        if creds and provider in ("auto", "angel"):
            try:
                from brokers.angelone.adapter import AngelOneAdapter
                self.adapter = AngelOneAdapter(*creds).start(connect_feed=True)
                self.alert("info", "Logged in to Angel One: real-time quotes on")
            except Exception as e:
                self.adapter = None
                self.alert("warn", f"Angel One login failed ({e}); using free Yahoo data")
        elif provider == "angel":
            self.alert("warn", "data.provider is angel but the Angel One credentials are not in .env")
        if S.live:
            if self.adapter is None:
                self.fatal = "LIVE mode needs a working Angel One login. No orders will be placed."
                self.kill.trip(self.fatal)
                self.alert("error", self.fatal)
            else:
                from brokers.angelone.orders import AngelOneBroker
                self.broker = AngelOneBroker(self.adapter.smart_api, self.universe)
                self.oms.broker = self.broker
                self.alert("warn", "LIVE mode: real orders through Angel One. Every entry waits for your Approve.")
        self.feed.start(self.adapter)
        self.obboard.start()                          # its own thread: ranks every F&O stock each minute
        if (S.get("llm_veto", {}) or {}).get("enabled"):
            from llm.gemini_client import GeminiClient
            self.llm = GeminiClient()
        self.pipeline = DecisionPipeline(S, self.universe, self.strategies, self.state, self.history, self.scanner,
                                         self.news, self.gate, self.kill, self.oms, self.book, self.journal,
                                         self.strategy_gate, self.llm)
        today = cal.now_ist().date().isoformat()
        self.oms.restore(self.journal.trades(date=today))
        held = {t.symbol for t in self.book.open()}
        if held:
            self.feed.subscribe(held)
        self.ready = True
        self.alert("info", f"Engine ready in {S.mode} mode")

    # ================================================================ jobs
    def submit_job(self, name: str, fn, *args):
        if self.jobs.get(name, {}).get("state") in ("queued", "running"):
            return False
        self.jobs[name] = {"state": "queued", "queued_at": time.time(), "message": ""}
        self._jobq.put((name, fn, args))
        return True

    def _job_worker(self):
        while True:
            name, fn, args = self._jobq.get()
            self.jobs[name] = {"state": "running", "started_at": time.time(), "message": ""}
            try:
                msg = fn(*args)
                self.jobs[name].update(state="done", finished_at=time.time(), message=msg or "")
            except Exception as e:
                logger.exception("job %s failed", name)
                self.jobs[name].update(state="failed", finished_at=time.time(), message=str(e))
                self.alert("error", f"{name} failed: {e}")

    # ================================================================ the day
    def _trade_date(self, now: datetime) -> str:
        st = cal.session_state(now)
        if st in ("PRE_OPEN", "OPEN"):
            return now.date().isoformat()
        return cal.next_open(now).date().isoformat()

    def _new_day(self, now: datetime):
        self.day = now.date().isoformat()
        self._flags = set()
        self._last_seen = {}
        if self.pipeline:
            self.pipeline.new_day()
        for t in list(self.book.trades.values()):
            if t.date != self.day and t.status in FINAL_STATES:
                self.book.trades.pop(t.id, None)

    def prepare_day(self) -> str:
        """Scan, watchlist, history, strategy selection, news. Safe to run again."""
        now = cal.now_ist()
        td = self._trade_date(now)
        res = self.scanner.run_daily(td)
        wl = [r["symbol"] for r in res["watchlist"]]
        idx = self.settings.get("market_filter.symbol", "^NSEI")
        self.feed.subscribe(wl + [idx] + [t.symbol for t in self.book.open()])
        try:
            self.obboard.subscribe()                  # live 5-minute bars for every F&O stock (order-block board)
        except Exception as e:                        # the board is an extra: never fail the scan
            logger.warning("order-block board subscribe failed: %s", e)
        self.history.intraday60(wl + [idx])
        if (self.settings.get("auto_select", {}) or {}).get("enabled") and not self._backtest_fresh():
            self.run_backtest_job()
        self.trade_date = td
        self.prepared_for = td
        top = ", ".join(f"{k} {v}" for k, v in list(res["setups"].items())[:4])
        msg = (f"Watchlist for {td}: {len(wl)} stocks from {res['scanned']} charts ({res['liquid']} liquid). "
               f"Setups: {top}")
        self.alert("info", msg)
        self.notifier.send("Watchlist ready", msg)
        self._last_news = 0.0
        try:
            self.refresh_intel()
        except Exception as e:                       # results / deals are extras: never fail the scan
            logger.warning("intel refresh failed: %s", e)
        return msg

    def run_backtest_job(self) -> str:
        res = run_backtest(self.settings, self.universe, self.history, log=lambda m: logger.info(m))
        self.backtest = res
        o = res.get("overall", {})
        msg = (f"Backtest {res.get('from')} to {res.get('to')}: {o.get('trades', 0)} trades, "
               f"net Rs {o.get('net_pnl', 0):,.0f} after charges")
        self.alert("info", msg)
        return msg

    def refresh_news(self) -> str:
        syms = [r["symbol"] for r in self.scanner.watchlist] + [t.symbol for t in self.book.open()]
        n = self.news.refresh(list(dict.fromkeys(syms)))
        for item in self.news.latest(20):
            if item["ts"] > time.time() - 3600 and set(item["tags"]) & {"legal_risk", "earnings", "deal"} \
                    and ("news", item["id"]) not in self._flags:
                self._flags.add(("news", item["id"]))
                self.alert("news", f"{item['symbol']}: {item['title']} ({item['source']})", item["symbol"])
        return f"{n} new headlines"

    def refresh_intel(self) -> str:
        """Results calendar (trader-analyst) and bulk / block deals (trader-pelosi): alert once per item
        that concerns a watchlist stock or an open trade."""
        self._last_intel = time.time()
        watch = [r["symbol"] for r in self.scanner.watchlist] + [t.symbol for t in self.book.open()]
        self.results.calendar(refresh=True)
        for r in self.results.upcoming(watch, days=3):
            if ("results", r["symbol"], r["date"]) in self._flags:
                continue
            self._flags.add(("results", r["symbol"], r["date"]))
            day = datetime.fromisoformat(r["date"]).strftime("%a %d %b")
            self.alert("results", f"{r['symbol']} announces results on {day}. Prices can jump on results day.",
                       r["symbol"])
        new = self.deals.refresh()
        mine = [d for d in new if d["symbol"] in set(watch)]
        groups: Dict[tuple, list] = {}
        for d in mine:
            groups.setdefault((d["symbol"], d["date"]), []).append(d)
        for (sym, day), ds in groups.items():
            buy = sum(d["value_cr"] or 0 for d in ds if d["side"] == "BUY")
            sell = sum(d["value_cr"] or 0 for d in ds if d["side"] == "SELL")
            self.alert("deal", self.deals.describe(ds[0]) if len(ds) == 1 else
                       f"{sym}: {len(ds)} bulk / block deals on {day} (bought ₹{buy:,.1f} cr, sold ₹{sell:,.1f} cr)", sym)
        return f"{len(self.results.calendar())} results dates, {len(new)} new deals ({len(mine)} on the watchlist)"

    # ================================================================ strategy selection
    def _backtest_fresh(self) -> bool:
        bt = self.backtest or {}
        return bool(bt.get("run_at", "")[:10] == cal.now_ist().date().isoformat() and bt.get("overall"))

    def strategy_gate(self, name: str):
        return strategy_allowed(self.settings, self.backtest, name)

    # ================================================================ the loop
    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception:
                logger.exception("engine loop error")
            time.sleep(1.0)

    def _tick(self):
        now = cal.now_ist()
        if now.date().isoformat() != self.day:
            self._new_day(now)
        if not self.ready:
            return
        S = self.settings
        st = cal.session_state(now)
        trading_day = cal.is_trading_day(now.date())
        td = self._trade_date(now)
        if self.prepared_for != td and time.time() - self._prepare_tried > 120:
            self._prepare_tried = time.time()
            self.submit_job("prepare_day", self.prepare_day)

        if trading_day and st == "OPEN" and "bell" not in self._flags:
            self._flags.add("bell")
            self.alert("bell", f"NSE open ({now.strftime('%H:%M')}). Mode {S.mode}.")

        # new closed bars -> paper fills on the bar's range, then the strategies
        watch = [r["symbol"] for r in self.scanner.watchlist] if self.prepared_for == now.date().isoformat() else []
        held = [t.symbol for t in self.book.open()]
        for sym in dict.fromkeys(watch + held):
            bars = self.state.bars(sym)
            if not bars:
                continue
            seen = self._last_seen.get(sym, 0)
            if bars[-1].start <= seen:
                continue
            fresh = [b for b in bars if b.start > seen]
            self._last_seen[sym] = bars[-1].start
            for b in fresh:
                self.oms.on_price(sym, b.c, bar=b)
            if st == "OPEN" and sym in watch and self.pipeline:
                self.pipeline.on_bar(sym, now, now.date().isoformat())
        if st == "OPEN" and self.pipeline:
            self._board_entries(now)

        # prices -> open trades (paper fills on ticks, live target watch)
        for t in self.book.open():
            p, _ = self.state.price(t.symbol)
            if p:
                self.oms.on_price(t.symbol, p)
        self.oms.step(now)
        self._check_price_alerts()

        if st == "OPEN":
            self._guards(now)
            every = float(S.get("news.poll_minutes", 20)) * 60
            if time.time() - self._last_news > every and self.scanner.watchlist:
                self._last_news = time.time()
                self.submit_job("news", self.refresh_news)
            flat = cal.parse_hhmm(S.get("flat_by", "15:10"))
            if cal.minute_of(now) >= flat and "flat" not in self._flags:
                self._flags.add("flat")
                self.alert("flat", f"Flat-by time {S.get('flat_by')}: closing anything still open.")
        elif st == "PRE_OPEN" and self.scanner.watchlist and time.time() - self._last_news > 1800:
            self._last_news = time.time()
            self.submit_job("news", self.refresh_news)
        if self.prepared_for and time.time() - self._last_intel > (1800 if st in ("OPEN", "PRE_OPEN") else 4 * 3600):
            self._last_intel = time.time()
            self.submit_job("intel", self.refresh_intel)
        if trading_day and st == "CLOSED" and "eod" not in self._flags:
            self._flags.add("eod")
            self._end_of_day(now)

    def _board_entries(self, now):
        """Strategies the engine fires itself: order-block board picks whose live price is back at Z
        (strategies/smc/order_block.py), checked every second against the board's last minute."""
        date = now.date().isoformat()
        strats = [s for s in self.strategies if s.live_trigger == "board" and s.enabled]
        if not strats or self.prepared_for != date:
            return
        b = self.obboard.view()
        if not (b.get("ready") and b.get("live") and b.get("session") == date):
            return
        ctx = DayContext("", date, cal.parse_hhmm(self.settings.get("no_entries_before", "09:30")),
                         cal.parse_hhmm(self.settings.get("no_entries_after", "14:30")),
                         bool(self.settings.get("allow_short", False)))
        if not ctx.entry_from <= cal.minute_of(now) <= ctx.entry_until:
            return
        for strat in strats:
            for row in b.get("bullish", []) + b.get("bearish", []):
                if (date, row["symbol"], strat.name) in self.pipeline.fired or not strat.armed(row, ctx):
                    continue
                price, age = self.state.price(row["symbol"])
                if not price or age is None or age > 60:
                    continue
                sig, why = strat.from_board(row, price, self.state.bars(row["symbol"]), now)
                if sig:
                    self.pipeline.on_signal(sig, strat, now, date, skip=why)

    def board_paper(self) -> dict:
        """What the paper order-block strategy is doing with each board stock today (order-block page)."""
        strat = next((s for s in self.strategies if s.live_trigger == "board"), None)
        b = self.obboard.view()
        if strat is None or not b.get("ready"):
            return {}
        ok, why = self.strategy_gate(strat.name)
        now = cal.now_ist()
        date = now.date().isoformat()
        ctx = DayContext("", date, cal.parse_hhmm(self.settings.get("no_entries_before", "09:30")),
                         cal.parse_hhmm(self.settings.get("no_entries_after", "14:30")),
                         bool(self.settings.get("allow_short", False)))
        live = b.get("live") and b.get("session") == date
        when = None if live and ctx.entry_from <= cal.minute_of(now) <= ctx.entry_until else (
            f"new trades only from {cal.hhmm(ctx.entry_from)} to {cal.hhmm(ctx.entry_until)}" if live else "the board shows the last session")
        sigs = {s["symbol"]: s for s in reversed(self.pipeline.signals if self.pipeline else []) if s["strategy"] == strat.name}
        trades = {t.symbol: t for t in self.book.today(date) if t.strategy == strat.name}
        out = {"strategy": strat.label, "trading": bool(strat.enabled and ok), "why": why, "rows": {}}
        for row in b.get("bullish", []) + b.get("bearish", []):
            sym, s, t = row["symbol"], sigs.get(row["symbol"]), trades.get(row["symbol"])
            if not out["trading"]:
                st = {"state": "off", "text": "off: " + (why if strat.enabled else "switched off in configs/trading.json")}
            elif t is not None:
                st = {"state": "traded", "text": f"{'bought' if t.side == 'buy' else 'sold'} {t.qty} at {s['time'] if s else '–'} · "
                      f"{t.status.lower().replace('_', ' ')}" + (f" · ₹{t.net_pnl:+,.0f}" if t.net_pnl is not None else ""),
                      "trade_id": t.id}
            elif s is not None:
                st = {"state": s["decision"].lower(), "text": f"{s['decision'].lower().replace('_', ' ')} at {s['time']}: {s['why']}"}
            else:
                no = strat.why_not(row, ctx) or when
                st = {"state": "idle", "text": "not armed: " + no} if no else \
                    {"state": "armed", "text": f"armed: {'buys' if row['side'] > 0 else 'sells'} the moment the price comes back to Z"}
            out["rows"][sym] = st
        return out

    def _guards(self, now):
        S = self.settings
        date = now.date().isoformat()
        realized = self.book.realized(date)
        limit = float(S.get("max_daily_loss_inr", 0))
        if limit and realized <= -limit and not self.kill.active:
            self.kill.trip(f"daily loss limit: Rs {realized:,.0f} today (limit Rs {limit:,.0f})")
            self.alert("error", f"Daily loss limit hit (Rs {realized:,.0f}). No new entries today.")
            self.notifier.send("Daily loss limit", f"Rs {realized:,.0f} today. Trading halted.", important=True)
        if self.feed.mode == "yahoo":
            age = self.feed.status().get("last_quotes_age_s")
            if age and age > float(S.get("data.stale_seconds", 180)) and "stale" not in self._flags:
                self._flags.add("stale")
                self.alert("warn", f"Prices are {age:.0f}s old: new entries are blocked until data resumes.")
        elif self.feed.mode == "angel":
            fs = self.feed.status()
            down = (not fs.get("angel_connected")
                    or (fs.get("angel_silent_s") or 0) > float(S.get("data.stale_seconds", 180)))
            if down and "feed_down" not in self._flags:
                self._flags.add("feed_down")
                self.alert("warn", "Angel One price feed is down: reconnecting. Entries are blocked "
                                   "while prices are stale.")
                if "feed_down_sent" not in self._flags:      # one push a day, however often it flaps
                    self._flags.add("feed_down_sent")
                    self.notifier.send("Price feed down", "Angel One feed dropped; the engine is reconnecting.",
                                       important=True)
            elif not down and "feed_down" in self._flags:
                self._flags.discard("feed_down")
                self.alert("info", "Angel One price feed is back.")
        if not self.broker.paper and time.time() - self._last_reconcile > 60:
            self._last_reconcile = time.time()
            problems = reconcile(self.broker.positions(), self.book.open())
            if problems:
                self.kill.trip("reconciliation: " + "; ".join(problems))
                self.alert("error", "Position mismatch with Angel One: " + "; ".join(problems))
                self.notifier.send("Position mismatch", "; ".join(problems), important=True)

    def _end_of_day(self, now):
        date = now.date().isoformat()
        closed = [t for t in self.book.today(date) if t.status == "CLOSED"]
        if not closed and not self.pipeline.signals:
            return
        net = sum(t.net_pnl or 0 for t in closed)
        wins = sum(1 for t in closed if (t.net_pnl or 0) > 0)
        msg = (f"Day over: {len(closed)} trades, {wins} winners, net Rs {net:,.2f} after charges. "
               f"{len(self.pipeline.signals)} signals seen.")
        self.alert("info", msg)
        self.journal.event(date, "eod", msg)
        self.notifier.send("Day summary", msg)

    # ================================================================ events and alerts
    def alert(self, kind: str, text: str, symbol: Optional[str] = None):
        row = {"ts": time.time(), "kind": kind, "text": text, "symbol": symbol}
        self.alerts.appendleft(row)
        logger.info("[%s] %s", kind, text)
        try:
            self.journal.event(cal.now_ist().date().isoformat(), kind, text, symbol)
        except Exception:
            pass

    def _on_event(self, kind, text, trade=None):
        self.alert(kind, text, trade.symbol if trade else None)
        if kind in ("approval", "fill", "closed", "error"):
            self.notifier.send({"approval": "Approve trade?", "fill": "Filled", "closed": "Trade closed",
                                "error": "Problem"}[kind], text, important=kind in ("approval", "error"))

    def _load_price_alerts(self):
        try:
            return json.loads(ALERTS_PATH.read_text())
        except (OSError, ValueError):
            return []

    def _save_price_alerts(self):
        ALERTS_PATH.write_text(json.dumps(self.price_alerts, indent=1))

    def add_price_alert(self, symbol: str, op: str, price: float, note: str = "") -> dict:
        row = {"id": uuid.uuid4().hex[:8], "symbol": symbol.upper(), "op": op, "price": float(price),
               "note": note, "created": time.time(), "fired_at": None}
        self.price_alerts.append(row)
        self._save_price_alerts()
        self.feed.subscribe([row["symbol"]])
        return row

    def delete_price_alert(self, alert_id: str):
        self.price_alerts = [a for a in self.price_alerts if a["id"] != alert_id]
        self._save_price_alerts()

    def _check_price_alerts(self):
        changed = False
        for a in self.price_alerts:
            if a.get("fired_at"):
                continue
            p, _ = self.state.price(a["symbol"])
            if p is None:
                continue
            if (a["op"] == "above" and p >= a["price"]) or (a["op"] == "below" and p <= a["price"]):
                a["fired_at"] = time.time()
                changed = True
                text = f"{a['symbol']} {p:.2f} is {a['op']} {a['price']:.2f}" + (f": {a['note']}" if a["note"] else "")
                self.alert("price", text, a["symbol"])
                self.notifier.send("Price alert", text, important=True)
        if changed:
            self._save_price_alerts()

    # ================================================================ actions (from the dashboard)
    def approve(self, trade_id: str) -> str:
        return self.oms.approve(trade_id, lambda t: self.pipeline.recheck(t, cal.now_ist()))

    def reject(self, trade_id: str) -> str:
        return self.oms.reject(trade_id)

    def exit_trade(self, trade_id: str) -> str:
        return self.oms.exit_now(trade_id, "manual exit from the dashboard")

    def flatten_all(self, reason: str = "flatten all from the dashboard") -> int:
        n = self.oms.flatten_all(reason)
        self.alert("warn", f"Flatten all: {n} trade(s) closing")
        return n

    def halt(self, reason: str = "halted from the dashboard"):
        self.kill.trip(reason, by="user")
        self.alert("error", f"Kill switch ON: {reason}. No new entries.")

    def resume(self):
        if self.fatal:
            return self.fatal
        self.kill.reset()
        self.alert("info", "Kill switch reset: entries allowed again")
        return "resumed"

    # ================================================================ views
    def account(self) -> dict:
        S = self.settings
        date = cal.now_ist().date().isoformat()
        acct = self.pipeline.account_view(date) if self.pipeline else None
        costs = S.get("costs", {}) or {}
        unreal = 0.0
        for t in self.book.open():
            u = t.unrealized(self.state.price(t.symbol)[0], costs)
            unreal += u or 0.0
        realized = self.book.realized(date)
        limit = float(S.get("max_daily_loss_inr", 0))
        committed = (max(0.0, -realized) + (acct.open_risk if acct else 0.0))
        return {
            "capital": float(S.get("capital_inr", 0)), "cash_available": round(acct.cash_available, 2) if acct else None,
            "realized": realized, "unrealized": round(unreal, 2), "net": round(realized + unreal, 2),
            "open_risk": round(acct.open_risk, 2) if acct else 0.0, "daily_limit": limit,
            "risk_used": round(committed, 2), "risk_used_pct": round(100 * committed / limit, 1) if limit else None,
            "entries_today": acct.entries_today if acct else 0, "max_trades": S.get("max_trades_per_day"),
            "open_positions": acct.open_positions if acct else 0, "max_open": S.get("max_open_positions"),
            "max_loss_per_trade": S.get("max_loss_per_trade_inr"),
        }

    def watch_rows(self):
        rows = []
        date = cal.now_ist().date().isoformat()
        snap = self.state.snapshot([r["symbol"] for r in self.scanner.watchlist])
        traded = {t.symbol: t.status for t in self.book.today(date)}
        live_day = self.prepared_for == date      # strategy states only for the session the scan is for
        for r in self.scanner.watchlist:
            sym = r["symbol"]
            bars = self.state.bars(sym) if live_day else []
            daily = self.scanner.stats_for(sym)
            ctx = DayContext(sym, date, cal.parse_hhmm(self.settings.get("no_entries_before", "09:30")),
                             cal.parse_hhmm(self.settings.get("no_entries_after", "14:30")),
                             bool(self.settings.get("allow_short", False)))
            states = {}
            for s in self.strategies:
                if not s.enabled or s.live_trigger != "bars" or not s.wants(daily):
                    continue
                fired = self.pipeline and (date, sym, s.name) in self.pipeline.fired
                ws = None
                if bars:
                    try:
                        ws = s.watch_state(bars, daily, ctx)
                    except Exception:
                        ws = None
                states[s.name] = dict(ws or {"state": "idle"}, fired=bool(fired))
            news = self.news.for_symbol(sym, 24)
            rows.append(dict(r, **(snap.get(sym) or {}), strategies=states, news=len(news),
                             news_tags=sorted({t for n in news for t in n["tags"]}), trade=traded.get(sym)))
        return rows

    def chart(self, symbol: str) -> dict:
        symbol = symbol.upper()
        bars = self.state.bars(symbol)
        forming = self.state.forming_bar(symbol)
        source = "today"
        if not bars:
            hist = self.history.intraday60([symbol]).get(symbol) or []
            days = split_sessions(hist)
            if days:
                last_day = list(days)[-1]
                bars, source = days[last_day], f"last session {last_day}"
        allbars = bars + ([forming] if forming and (not bars or forming.start > bars[-1].start) else [])
        vw = vwap_series(allbars) if allbars else []
        n = int(self.settings.get("strategies.orb.or_minutes", 15)) // 5
        orr = opening_range(bars, n) if bars else None
        trades = [t.to_dict() for t in self.book.trades.values() if t.symbol == symbol]
        sigs = [s for s in (self.pipeline.signals if self.pipeline else []) if s["symbol"] == symbol]
        daily = self.scanner.stats_for(symbol)
        return {
            "symbol": symbol, "name": self.universe.name(symbol), "source": source,
            "bars": [b.to_dict() for b in allbars], "vwap": [round(x, 2) for x in vw],
            "forming": bool(forming), "or_high": orr[0] if orr else None, "or_low": orr[1] if orr else None,
            "prev_close": self.state.prev_close(symbol) or (daily.close if daily else None),
            "high20": daily.high20 if daily else None, "trades": trades, "signals": sigs[:20],
            "news": self.news.for_symbol(symbol, 48)[:15],
        }

    def snapshot(self) -> dict:
        S = self.settings
        now = cal.now_ist()
        date = now.date().isoformat()
        trades = []
        costs = S.get("costs", {}) or {}
        for t in sorted(self.book.today(date), key=lambda x: -x.created_at):
            d = t.to_dict()
            p, age = self.state.price(t.symbol)
            d["ltp"], d["unrealized"] = p, t.unrealized(p, costs)
            trades.append(d)
        strat_rows = []
        bt = (self.backtest or {}).get("by_strategy") or {}
        for s in self.strategies:
            ok, why = self.strategy_gate(s.name)
            m = bt.get(s.name) or {}
            strat_rows.append({"name": s.name, "label": s.label, "enabled": s.enabled, "trading": ok, "why": why,
                               "params": s.params, "killed": self.kill.blocked(s.name),
                               "backtest": {k: m.get(k) for k in ("trades", "win_rate", "avg_R", "avg_net_R",
                                                                 "net_pnl", "profit_factor", "max_drawdown_inr")}})
        scan = {k: v for k, v in (self.scanner.result or {}).items() if k != "watchlist"}
        return {
            "now": now.strftime("%Y-%m-%d %H:%M:%S"), "clock": cal.describe(now), "mode": S.mode,
            "ready": self.ready, "fatal": self.fatal, "trade_date": self.trade_date,
            "prepared_for": self.prepared_for, "feed": self.feed.status(), "broker": self.broker.name,
            "kill": self.kill.to_dict(), "account": self.account() if self.pipeline else {},
            "watchlist": self.watch_rows() if self.pipeline else [],
            "signals": (self.pipeline.signals[:150] if self.pipeline else []),
            "trades": trades, "alerts": list(self.alerts)[:150], "price_alerts": self.price_alerts,
            "news": self.news.latest(80), "strategies": strat_rows, "jobs": self.jobs, "scan": scan,
            "intel": self.intel_view(),
            "backtest": {k: v for k, v in (self.backtest or {}).items() if k != "trades"},
            "rules": {k: S.get(k) for k in ("capital_inr", "max_loss_per_trade_inr", "max_daily_loss_inr",
                                            "max_trades_per_day", "max_open_positions", "min_reward_risk",
                                            "no_entries_before", "no_entries_after", "flat_by", "allow_short",
                                            "cash_buffer_pct", "max_position_pct", "max_order_inr",
                                            "intraday_leverage", "approval_timeout_seconds")},
        }

    def intel_view(self) -> dict:
        watch = [r["symbol"] for r in self.scanner.watchlist] + [t.symbol for t in self.book.open()]
        deals = self.deals.latest(40, symbols=watch)
        return {"results_soon": self.results.upcoming(watch, days=3), "watch_deals": deals[:10],
                "deals_as_of": self.deals.as_of, "deals_total": len(self.deals.deals),
                "errors": [e for e in (self.results.error, self.deals.error) if e]}

    def skills_view(self) -> list:
        from apps.trading_engine.skills import skills_status
        return skills_status(self)

    def journal_view(self, days: int = 30) -> dict:
        import datetime as dt
        since = (cal.now_ist().date() - dt.timedelta(days=days)).isoformat()
        rows = self.journal.trades(since=since)
        closed = [r for r in rows if r.get("status") == "CLOSED"]
        by_day: Dict[str, dict] = {}
        by_strat: Dict[str, dict] = {}
        for r in closed:
            d = by_day.setdefault(r["date"], {"date": r["date"], "trades": 0, "net": 0.0, "wins": 0})
            d["trades"] += 1
            d["net"] = round(d["net"] + (r.get("net_pnl") or 0), 2)
            d["wins"] += 1 if (r.get("net_pnl") or 0) > 0 else 0
            s = by_strat.setdefault(r["strategy"], {"strategy": r["strategy"], "trades": 0, "net": 0.0, "wins": 0,
                                                    "R": 0.0})
            s["trades"] += 1
            s["net"] = round(s["net"] + (r.get("net_pnl") or 0), 2)
            s["wins"] += 1 if (r.get("net_pnl") or 0) > 0 else 0
            s["R"] = round(s["R"] + (r.get("R") or 0), 3)
        return {"since": since, "days": sorted(by_day.values(), key=lambda x: x["date"]),
                "strategies": list(by_strat.values()), "trades": closed[-200:]}
