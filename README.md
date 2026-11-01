# MONEYV2: NSE intraday trading engine

An autonomous intraday trading system for the NIFTY 500 on Angel One SmartAPI, with a live
dashboard. It scans every morning, watches the best setups bar by bar, plans every exit
before the entry, replays each trade before risking money, sizes it to your loss limit,
checks it against your risk rules, and places it: automatically on paper, and with your
one-click approval with real money.

It is built from the [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill)
pack, ported from US markets to NSE hours, rupees, Indian intraday charges and Angel One.

> No strategy here is a promise of profit. Read the Strategies and Research pages before trusting any of it:
> it shows, after charges, how each strategy did on the last ~40 sessions. Most intraday
> traders lose money. Run on paper for weeks before going live.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate   # once
pip install -r requirements.txt                    # once
cp .env.example .env                               # once: add your Angel One keys
./run.sh                                           # engine + dashboard
./start_dashboard.sh                               # opens http://127.0.0.1:8050
```

Without Angel One credentials it runs on free, delayed Yahoo data. With them, quotes come
from the Angel One WebSocket in real time. Either way the default mode is **PAPER**.

## What happens during a day

| time (IST) | what the engine does |
| --- | --- |
| start-up / before 09:15 | loads NIFTY 500 + Angel One tokens and tick sizes; scans ~500 daily charts; builds a 50-stock watchlist; fetches 60 days of 5-minute bars for the Time Machine; re-runs the walk-forward backtest that decides which strategies may trade (auto-select); reads the news |
| 09:15 | opening bell alert; bars start |
| every closed 5-min bar | each strategy checks its rule on each watchlist stock; a fresh trigger goes through the pipeline below |
| 09:30 to 14:30 | the entry window (no new entries outside it) |
| continuously | paper fills, stop and target handling, price alerts, kill switch on the daily loss, LIVE reconciliation with Angel One |
| 15:10 | flat by: everything still open is closed at market |
| after 15:30 | day summary (dashboard, desktop notification, Telegram if set up) |

### The pipeline, for every signal

1. **Strategy trigger** on a *closed* bar (a wick through a level is not a trigger); the order-block
   strategy instead triggers the moment the live price reaches its zone's entry level
2. **Market filter**: longs only while the NIFTY 50 is above its open (shorts the reverse)
3. **Exit plan**: stop and target set first, stop rounded away from the entry to the stock's tick
4. **News**: no entries on a stock with a blocking headline (default: fraud, raids, probes, defaults)
5. **Time Machine**: 5,000 seeded replays of the rest of the day from the stock's own last 60
   sessions, plus the same strategy's real record on that stock; refuses a target out of reach
   by 15:10 or a setup that has been losing on that stock
6. **Sizing**: whole shares inside four limits (max loss incl. charges, position cap, max order, cash)
7. **Risk gate**: stop required, loss per trade, reward:risk, daily loss budget, trades per day,
   open positions, duplicates, entry window, stale data, chasing a move that already happened
8. **Order**: PAPER places it at once; LIVE waits for your Approve (risk re-checked at that moment)

Every signal is shown on the dashboard with what was done and why, and journaled.

## Strategies

| name | rule | stop / target |
| --- | --- | --- |
| `orb` opening-range breakout | first 5-min close beyond the 09:15 to 09:30 range, on stocks "in play" (range volume >= 1.5x its average) | range midpoint / 2R |
| `level_breakout` 20-day high | stock closed within 3% of its 20-session high; first close above it (gaps over 1.5% skipped) | low of the last 3 bars / 2R |
| `vwap_reclaim` trend pullback | daily uptrend; dips under VWAP after the range, then closes back above VWAP and the prior high | low of the dip / 2R |
| `order_block` order-block retest (paper only) | a pick on the Order blocks page with 4 of 4 checks aligned; enters the first time the live price comes back to its zone's entry level Z (checked every second) | the zone's invalidation / its 2nd liquidity level (T2), else 2R |

Short sides exist for `orb`, `vwap_reclaim` and `order_block` but are off until `allow_short` is true.

`order_block` has `paper_only: true`: it trades on paper whatever the morning re-test says and never in LIVE.
Its 2-year backtest lost money (Research page, order-block report), so its paper trades are a forward test.
The Order blocks page shows, for every pick, whether it is armed, traded or why not.

**Auto-select** (on by default): each morning a walk-forward backtest (top 100 NIFTY 500 stocks by
turnover, the same strategies, sizing, risk rules, slippage and charges) decides which
strategies trade. A strategy whose recent average R after costs is below zero is benched. In PAPER, one
with too few trades runs on probation; LIVE needs at least `min_trades`.

## Paper vs live

| | PAPER (default) | LIVE |
| --- | --- | --- |
| entries | placed automatically | wait for your Approve on the dashboard (expire after 120 s) |
| fills | simulated: market at the next price + 5 bps, stop at its price or a gap's open, target at its price | Angel One, INTRADAY product |
| stop | resting in the paper broker | SL-M order at Angel One (protects you even if the engine stops) |
| target | resting limit (one-cancels-other) | watched on live quotes, closed at market |
| exits | automatic (target, stop, 15:10 flat-by, kill switch, your Exit button) | same, never blocked |

To go live: set `TRADING_MODE=LIVE` in `.env` and restart. Before you do:
- paper trade for several weeks and look at the Journal, not just the backtest;
- register your static IP with Angel One: SEBI's retail algo rules tie order APIs to it;
- size `capital_inr` and the loss limits in `configs/trading.json` to money you can afford to lose;
- keep the Angel One app open the first days to check every order.

## Dashboard

`http://127.0.0.1:8050` (local only). A sidebar with one page per job; the top bar always shows
the market clock, the data feed and whether new trades are allowed, with **Pause new trades** and
**Exit all positions**. LIVE approvals appear as a yellow bar on every page.

| Page | What it is for |
|---|---|
| Home | one sentence on what the bot is doing, the day's timeline, anything that needs you, P&L, open positions, latest signals, results and big deals coming up |
| Live chart | candlestick chart (VWAP, opening range, stop / target lines, fills) and the watchlist with each stock's nearest trigger |
| Order blocks | every NSE F&O stock ranked each minute by clean one-way strength (bullish and bearish boards, top 3 picks a side, max 2 per sector, event-risk filter), each stock's 5-minute order-block zone with entry, invalidation, liquidity targets and R:R, the time the entry price was given (the break candle's close) and the time price first came back to it (the entry signal), alignment checks (stock, sector, breadth, VWAP), yesterday's futures OI build-up, a market map and sector flow. Rules in `market/orderblocks.py` |
| Crypto & XAUUSD: Live prices | Binance websocket prices for 17 coins and gold (XAUUSD, Binance's XAUUSDT perpetual): today's and 24-hour change, range, volume, and a live 5-minute chart |
| Crypto & XAUUSD: Order blocks | the order-block board for crypto (coins ranked by strength over the UTC day, picks, zones with entry, stop-loss, target and times) and gold on its own; analytics only (`apps/trading_engine/cxboard.py`) |
| Crypto & XAUUSD: Confirmations | futures open interest and funding, taker buying, volume, CPR, previous day and 20-day trend for each pair on a side (`market/crypto/confirmations.py`) |
| Signals | every rule match today and what the bot did (traded, waiting for you, blocked, benched), filterable |
| Trades | today's trades with Exit buttons; 30-day history by day and by strategy |
| Alerts | price alerts and the full activity log |
| Strategies | each live strategy in plain words, whether it may trade today and why, the morning re-test |
| Research | the 5-year backtests: can ₹20k a month be made on ₹1 lakh, the 15 best strategies, stress tests, all reports |
| Market intel | news, quarterly results (calendar + any company's last quarters), bulk and block deals |
| Skills | the 11 day-trader skills, the NSE version of each, and its live status |
| Settings & safety | mode, safety switch, risk rules in plain language, data feed, background jobs |
| Help | how the bot works, a daily routine, a glossary |

## Tools

```bash
./backtest.sh                      # walk-forward backtest, writes data/backtests/latest.json
./backtest.sh --universe-size 200 --no-tm
./replay.sh 2026-10-22             # replay a past session through the live components
venv/bin/python -m pytest tests    # rules, strategies, Time Machine, order lifecycle, research
```

## Configuration: `configs/trading.json`

Money: `capital_inr`, `cash_buffer_pct`, `max_position_pct`, `max_order_inr`, `min_order_inr`,
`intraday_leverage` (1 = no MIS leverage). Risk: `max_loss_per_trade_inr`, `max_daily_loss_inr`,
`max_trades_per_day`, `max_trades_per_symbol`, `max_open_positions`, `min_reward_risk`,
`min_risk_pct` / `max_risk_pct` (stop distance bounds), `max_entry_slip_pct`, `allow_short`,
`blocked`. Time (IST): `no_entries_before`, `no_entries_after`, `flat_by`. Plus `strategies.*`,
`timemachine.*`, `auto_select.*`, `universe.*`, `news.*`, `crypto.*`, `costs.*` (check them against your
broker's charge sheet), `slippage_bps`, `notify.*`, `llm_veto.*` (Gemini can only veto).
Restart `./run.sh` after editing.

## Layout

| path | what (and the skill it came from) |
| --- | --- |
| `apps/trading_engine/` | the engine loop and start-up |
| `apps/decision_service/` | the signal-to-order pipeline |
| `apps/scanner_service/`, `scanner/` | morning scan, setups, watchlist (trader-scanner) |
| `apps/market_data_service/`, `market/` | Angel One WebSocket / Yahoo feed, bars, NSE calendar, universe, history, news (trader-news) |
| `strategies/` | ORB, 20-day breakout, VWAP reclaim (trader-entry); exit planner (trader-exit) |
| `backtesting/` | Time Machine (trader-timemachine), walk-forward backtest, replay, auto-select |
| `risk/` | sizing (trader-budget), risk gate and kill switch (trader-risk) |
| `oms/`, `brokers/`, `portfolio/` | order lifecycle, paper and Angel One brokers (trader-broker), trades and charges |
| `market/intel/results.py` | results analyst (trader-analyst): NSE results calendar, last quarters from Yahoo Finance |
| `market/intel/deals.py` | big-money tracker (trader-pelosi): NSE bulk and block deals, one alert per stock and day |
| `apps/api_service/`, `apps/dashboard/` | local API and dashboard (trader-alerts lives here too) |
| `apps/trading_engine/skills.py` | the 11 skills, where each lives, and its live status (Skills page) |
| `data/` | caches, journal (`moneyv2.db`), backtests, logs; not committed |

trader-analyst reads US filings on SEC EDGAR and trader-pelosi reads US Congress members' trade
disclosures. India has neither, so the NSE versions use the exchange's results calendar plus Yahoo
Finance's quarterly numbers, and NSE's bulk / block deal reports (who traded big, by name).
NSE's insider-trading (SEBI PIT) feed needs a browser session and is not used.

Crypto and gold prices come from Binance's public market data, with no account or key: the spot websocket
(`data-stream.binance.vision`) for coins, the futures market websocket (`wss://fstream.binance.com/market/...`; the old
futures paths no longer stream) for XAUUSDT, and REST for history, open interest and funding (`market/crypto/binance.py`).
"Today" for crypto is the UTC day (05:30 to 05:30 IST). Configure it in `configs/trading.json` → `crypto`
(`enabled`, `symbols`, `gold`). The crypto pages are analytics only: the bot never trades crypto or gold. For Indian
residents, XAUUSD and forex CFDs with offshore brokers are not permitted under FEMA; gold trades in India on MCX or as
gold ETFs and Sovereign Gold Bonds.

`MONEYV2_DATA_PROVIDER=yahoo ./run.sh` starts without logging in to Angel One (`auto` / `angel`
are the other values), for example while a research download is using the Angel session.

## Data sources and limits

Free Yahoo data is unofficial and can lag or throttle; the engine says which feed it is on.
Angel One's feed is real time. Angel One drops idle sockets outside market hours, so from
09:00 to the close on trading days the engine logs in again and opens a new socket whenever
the feed is down, or silent for 90 s while the market is open (Home shows "reconnects before
the open" at other times). SmartApi's failed-request logs are masked so the API key, session
token, PIN and one-time code never reach `logs/`. NSE holidays are listed for 2026; add 2027's in
`market/calendar.py` when NSE publishes them (the dashboard flags an unknown year).
