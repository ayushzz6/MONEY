"""The 11 day-trader skills (github.com/Jakeschincariol/day-trader-skill) and where each one lives in
MONEYV2, with a live status line for the dashboard's Skills page.

The original skills target US stocks (Alpaca, NYSE hours, SEC EDGAR, the US House Clerk); each one
here is the NSE / Angel One version. Order = the order a trade goes through them.
"""
import time

SKILLS = [
    {"id": "trader-scanner", "name": "Chart scanner", "page": "live",
     "what": "Scans the market every morning and picks the stocks worth watching today.",
     "india": "NIFTY 500 instead of Nasdaq's symbol lists: daily bars, liquidity filter, 20-day high, "
              "bull flag, oversold and other setups, ranked into a watchlist.",
     "code": ["scanner/", "apps/scanner_service/main.py"]},
    {"id": "trader-entry", "name": "Entry timer", "page": "strategies",
     "what": "Watches each watchlist stock bar by bar and says the moment a setup triggers.",
     "india": "NSE 5-minute bars from Angel One (or Yahoo): opening-range breakout, 20-day-high breakout "
              "and VWAP reclaim, entries only between the allowed times.",
     "code": ["strategies/breakout/orb.py", "strategies/breakout/level.py", "strategies/vwap/reclaim.py"]},
    {"id": "trader-exit", "name": "Exit planner", "page": "strategies",
     "what": "Sets the stop-loss and the target before any entry, so the way out is known in advance.",
     "india": "Stop at the opening-range middle or the last swing, target at 2R, rounded to the NSE tick; "
              "everything is closed by the flat-by time.",
     "code": ["strategies/exits.py"]},
    {"id": "trader-budget", "name": "Budget manager", "page": "settings",
     "what": "Works out how many shares to buy so a stop-out loses no more than your limit.",
     "india": "Rupee limits: max loss per trade, position cap, max order, cash buffer, NSE charges included.",
     "code": ["risk/position_sizing.py"]},
    {"id": "trader-risk", "name": "Risk manager", "page": "settings",
     "what": "Checks every trade against your rules and blocks anything over them.",
     "india": "Daily loss limit, trades per day, open positions, reward:risk, entry window, news blocks, "
              "and the kill switch.",
     "code": ["risk/limits.py", "risk/kill_switch.py"]},
    {"id": "trader-timemachine", "name": "Time machine", "page": "signals",
     "what": "Before a trade, replays thousands of possible outcomes and every past time the setup fired.",
     "india": "Replays the stock's own NSE 5-minute bars from the last 60 days; weak odds bench the signal.",
     "code": ["backtesting/timemachine.py"]},
    {"id": "trader-broker", "name": "Order executor", "page": "trades",
     "what": "Sends orders to the broker: practice money by default, never real money without your yes.",
     "india": "Paper broker by default; Angel One SmartAPI in LIVE mode, where every entry waits for your "
              "Approve and exits are automatic.",
     "code": ["brokers/paper.py", "brokers/angelone/", "oms/order_manager.py"]},
    {"id": "trader-alerts", "name": "Price alerts", "page": "alerts",
     "what": "Rings when a price, a stop, a target or the opening bell is hit, once each.",
     "india": "NSE hours (09:15 bell, flat-by warning), price alerts on any NIFTY 500 stock, desktop and "
              "Telegram notifications.",
     "code": ["apps/trading_engine/engine.py (price alerts)", "core/utilities/notify.py"]},
    {"id": "trader-news", "name": "News monitor", "page": "intel",
     "what": "Tracks headlines on every watched stock and flags what could move it.",
     "india": "Google News India per company, tagged (results, deals, orders, legal risk...); a legal-risk "
              "headline blocks new entries in that stock.",
     "code": ["market/news/feed.py"]},
    {"id": "trader-analyst", "name": "Results analyst", "page": "intel",
     "what": "Reads a company's reported numbers and sums up what changed.",
     "india": "NSE's board-meeting calendar (who reports results, when) and the last quarters from Yahoo "
              "Finance: revenue, margins, net profit and EPS against the previous quarter and a year earlier. "
              "(The original reads US filings on SEC EDGAR.)",
     "code": ["market/intel/results.py"]},
    {"id": "trader-pelosi", "name": "Big-money tracker", "page": "intel",
     "what": "Reports big, named trades as soon as they are published.",
     "india": "India has no Congress-style trade disclosures, so this tracks NSE's bulk and block deals: "
              "who bought or sold large amounts of a stock, posted after each session. "
              "(The original reads US Congress members' filings.)",
     "code": ["market/intel/deals.py"]},
]


def skills_status(engine) -> list:
    S = engine.settings
    scan = engine.scanner.result or {}
    wl = engine.scanner.watchlist
    strategies = engine.strategies
    allowed = [s for s in strategies if s.enabled and engine.strategy_gate(s.name)[0]]
    tm = S.get("timemachine", {}) or {}
    kill = engine.kill.to_dict().get("global")
    soon = engine.results.upcoming([r["symbol"] for r in wl], days=3) if wl else []
    pending = [t for t in engine.book.pending() if t.status == "PENDING_APPROVAL"]
    status = {
        "trader-scanner": (bool(wl), f"{len(wl)} stocks on the watchlist from {scan.get('scanned', 0)} charts"
                           if wl else "the morning scan has not finished yet"),
        "trader-entry": (bool(allowed), f"{len(allowed)} of {len(strategies)} strategies allowed to trade: "
                         + (", ".join(s.label for s in allowed) or "none")),
        "trader-exit": (True, f"stop and target set before every entry; flat by {S.get('flat_by')}"),
        "trader-budget": (True, f"max loss ₹{S.get('max_loss_per_trade_inr'):,} a trade, max order "
                          f"₹{S.get('max_order_inr'):,}"),
        "trader-risk": (not kill, f"HALTED: {kill['reason']}" if kill else
                        f"armed · daily loss limit ₹{S.get('max_daily_loss_inr'):,}, "
                        f"{S.get('max_trades_per_day')} trades a day"),
        "trader-timemachine": (bool(tm.get("enabled")), f"{tm.get('replays', 0):,} replays per signal"
                               if tm.get("enabled") else "off"),
        "trader-broker": (True, f"{engine.broker.name}" + (f" · {len(pending)} waiting for your approval"
                                                          if pending else "")),
        "trader-alerts": (True, f"{sum(1 for a in engine.price_alerts if not a.get('fired_at'))} price alerts "
                          f"watching · opening bell and flat-by on"),
        "trader-news": (bool((S.get('news', {}) or {}).get("enabled", True)),
                        f"{len(engine.news.items)} headlines in the last 48 hours"),
        "trader-analyst": (not engine.results.error, engine.results.error or
                           f"{len(engine.results.calendar())} results dates ahead · {len(soon)} watchlist stocks "
                           f"report within 3 days"),
        "trader-pelosi": (not engine.deals.error, engine.deals.error or
                          f"{len(engine.deals.deals)} bulk / block deals tracked"
                          + (f" · latest {engine.deals.as_of}" if engine.deals.as_of else "")),
    }
    out = []
    for sk in SKILLS:
        ok, line = status[sk["id"]]
        out.append(dict(sk, active=bool(ok), status=line, checked_at=time.time()))
    return out
