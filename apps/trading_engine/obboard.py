"""The order-block board: every NSE F&O stock ranked by clean one-way strength each minute, with its
order-block zone, levels, sector, market alignment and yesterday's futures OI build-up (rules: market/orderblocks.py),
plus the confirmation checks: futures and options OI, live OI, CPR, volume, previous day, trend, NIFTY (market/confirmations.py).

Live during the session from the engine's 5-minute bars (the F&O stocks are subscribed to the feed); at other
times it shows the last session, rebuilt from cached 5-minute history. It runs on its own thread so a long
morning scan never delays it.
"""
import datetime as dt
import logging
import threading
import time
from typing import Dict, List, Optional

from market import calendar as cal
from market import confirmations as CF
from market import orderblocks as OB
from market.candles.bars import DailyBar, split_sessions, vwap_series
from market.instruments import fno

logger = logging.getLogger("OrderBlocks")


class OrderBlockBoard:
    def __init__(self, engine):
        self.e = engine
        self.board: Optional[Dict] = None
        self.details: Dict[str, Dict] = {}
        self._prev_ranks: Dict[str, int] = {}
        self._prev_minute = None
        self._lock = threading.Lock()
        self.fno: Optional[Dict] = None
        self._fno_day = None
        self._last = 0.0
        self.error = ""
        self.live_oi = fno.LiveOI()
        self._vol_prof: Dict[str, Optional[List[float]]] = {}
        self._days5m: Dict[str, Dict[str, DailyBar]] = {}
        self._hist_session = None

    # ------------------------------------------------------------ inputs
    def symbols(self) -> List[str]:
        today = cal.now_ist().date()
        if self.fno is None or self._fno_day != today:
            got = fno.latest(today)
            if got:
                self.fno, self._fno_day = got, today
        syms = (self.fno or {}).get("symbols") or []
        return [s for s in syms if self.e.universe.token(s)]

    def subscribe(self):
        """Live bars for every F&O stock (called when the day is prepared)."""
        syms = self.symbols()
        if syms:
            self.e.feed.subscribe(syms)
        return len(syms)

    def _daily_context(self, daily_bars, session: str):
        """Previous close / high / low and ATR14 from daily bars before `session`; the bhavcopy day's price change."""
        before = [b for b in daily_bars if b.date < session]
        if len(before) < 15:
            return None
        trs = []
        for a, b in zip(before[-15:-1], before[-14:]):
            trs.append(max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c)))
        out = {"prev_close": before[-1].c, "prev_high": before[-1].h, "prev_low": before[-1].l, "atr": sum(trs) / len(trs),
               "cpr": CF.cpr(before[-1].h, before[-1].l, before[-1].c), "cpr_rank": CF.cpr_rank(before),
               "sma": sum(b.c for b in before[-CF.TREND_DAYS:]) / min(len(before), CF.TREND_DAYS)}
        oi_day = (self.fno or {}).get("date")
        if oi_day:
            hist = [b for b in daily_bars if b.date <= oi_day]
            if len(hist) >= 2 and hist[-1].date == oi_day:
                out["oi_day_chg"] = 100 * (hist[-1].c / hist[-2].c - 1)
        return out

    def _session_bars(self, syms):
        """{sym: bars} for today (live) when the market has opened, else the last full session from history."""
        now = cal.now_ist()
        live = cal.is_trading_day(now.date()) and cal.session_state(now) in ("OPEN", "CLOSED") and now.hour >= 9
        out, session = {}, None
        if live:
            for s in syms:
                b = self.e.state.bars(s)
                if b:
                    out[s] = b
            if len(out) >= max(10, len(syms) // 3):
                return out, now.date().isoformat(), True
        hist = self.e.history.intraday60(syms)
        last_days = {}
        for s, bars in hist.items():
            days = split_sessions(bars)
            if days:
                d = list(days)[-1]
                last_days[s] = (d, days[d])
        if last_days:
            session = max(d for d, _ in last_days.values())
            out = {s: b for s, (d, b) in last_days.items() if d == session}
        return out, session, False

    def _history(self, syms, session):
        """Once per session, from the 5-minute history before it: each stock's usual volume by time of day, and
        each session as a daily bar (Yahoo's daily bars skip some sessions; _filled puts them back)."""
        if self._hist_session != session:
            hist = self.e.history.intraday60(syms)
            self._vol_prof = {s: CF.volume_profile(hist.get(s) or [], session) for s in syms}
            self._days5m = {s: {d: DailyBar(d, b[0].o, max(x.h for x in b), min(x.l for x in b), b[-1].c, sum(x.v for x in b))
                                for d, b in split_sessions([x for x in hist.get(s) or [] if x.date < session]).items()
                                if len(b) >= 60}                  # most of the 75 candles, or the high / low may be off
                            for s in syms}
            self._hist_session = session
        return self._vol_prof

    def _filled(self, symbol, daily_bars):
        """Daily bars with the sessions Yahoo skipped rebuilt from 5-minute bars (else the 'previous day' can be two days ago)."""
        have = {b.date for b in daily_bars}
        add = [b for d, b in (self._days5m.get(symbol) or {}).items() if d not in have]
        return sorted(daily_bars + add, key=lambda b: b.date) if add else daily_bars

    def _index_change(self, session, live):
        """NIFTY 50's finished candles: % from the session's open to the last close."""
        sym = self.e.settings.get("market_filter.symbol", "^NSEI")
        bars = self.e.state.bars(sym) if live else []
        if not bars:
            bars = [b for b in (self.e.history.intraday60([sym]).get(sym) or []) if b.date == session]
        return 100 * (bars[-1].c / bars[0].o - 1) if bars and bars[0].o else None

    def _confirm_inputs(self, r, ctx, bars, prof, idx_chg, live_oi):
        s = r["symbol"]
        oi = ((self.fno or {}).get("oi") or {}).get(s) or {}
        live = live_oi.get(s)
        last = bars[-1]
        return {"last": r["last"], "change_pct": r["change_pct"], "fut_buildup": r["oi_buildup"], "fut_oi_chg": oi.get("chg_pct"),
                "live_oi_chg": 100 * (live / oi["oi"] - 1) if live and oi.get("oi") else None,
                "pe_chg": oi.get("pe_chg"), "ce_chg": oi.get("ce_chg"), "pcr": oi.get("pcr"), "live_pcr": self.live_oi.pcr.get(s),
                "cpr": ctx["cpr"], "cpr_rank": ctx["cpr_rank"], "rvol": CF.rel_volume(bars, prof.get(s)),
                "upto": cal.hhmm(last.close_minute), "prev_high": ctx["prev_high"], "prev_low": ctx["prev_low"],
                "sma": ctx["sma"], "index_chg": idx_chg}

    def update_live_oi(self) -> Optional[str]:
        """Today's futures OI from Angel One, every few minutes in the session (used by the next refresh)."""
        api = self.e.adapter.smart_api if self.e.adapter else None
        if api is None or not self.fno or not self.live_oi.due():
            return None
        today = cal.now_ist().date().isoformat()
        base = {s: v["oi"] for s, v in (self.fno.get("oi") or {}).items() if v.get("oi")}
        contracts = {s: self.e.universe.futures_for(s, today) for s in self.symbols()}
        contracts = {s: c for s, c in contracts.items() if c}
        if not contracts:
            self.live_oi.error = "no futures contracts in Angel One's scrip master"
            return self.live_oi.error
        return self.live_oi.refresh(api, contracts, base, today)

    # ------------------------------------------------------------ the board
    def refresh(self) -> str:
        t0 = time.time()
        syms = self.symbols()
        if not syms:
            self.error = "the F&O stock list is not available (NSE bhavcopy could not be read)"
            return self.error
        bars_by, session, live = self._session_bars(syms)
        if not bars_by:
            self.error = "no 5-minute bars yet"
            return self.error
        daily = self.e.history.daily(list(bars_by))
        prof = self._history(list(bars_by), session)
        idx_chg = self._index_change(session, live)
        live_oi = self.live_oi.oi if self.live_oi.day == session else {}
        rows, details = [], {}
        for s, bars in bars_by.items():
            ctx = self._daily_context(self._filled(s, daily.get(s) or []), session)
            if not ctx or len(bars) < 2:
                continue
            st = OB.strength(bars, ctx["prev_close"], ctx["atr"])
            inst = self.e.universe.get(s)
            sector = (inst.industry if inst else "") or "Other"
            vw = vwap_series(bars)
            ev = OB.event_risk(st)
            zones = [] if ev else OB.detect_zones(bars, ctx["prev_high"], ctx["prev_low"])
            oi = ((self.fno or {}).get("oi") or {}).get(s) or {}
            row = {"symbol": s, "name": self.e.universe.name(s), "sector": sector, "last": round(bars[-1].c, 2),
                   **st, "event": ev, "above_vwap": bars[-1].c >= vw[-1], "atr": round(ctx["atr"], 2),
                   "oi_buildup": fno.buildup(ctx.get("oi_day_chg"), oi.get("chg_pct")), "oi_chg_pct": oi.get("chg_pct"),
                   "spark": [round(b.c, 2) for b in bars][-75:]}
            rows.append(row)
            details[s] = {"bars": bars, "vwap": vw, "zones": zones, "ctx": ctx}
        ranked = OB.rank_board(rows)
        # market picture
        up = sum(1 for r in rows if (r["change_pct"] or 0) > 0)
        down = sum(1 for r in rows if (r["change_pct"] or 0) < 0)
        sectors = {}
        for r in rows:
            g = sectors.setdefault(r["sector"], {"sector": r["sector"], "n": 0, "up": 0, "chg": 0.0, "score": 0.0})
            g["n"] += 1
            g["up"] += (r["change_pct"] or 0) > 0
            g["chg"] += r["change_pct"] or 0
            g["score"] += r["score"] or 0
        sec_rows = []
        for g in sectors.values():
            sec_rows.append({"sector": g["sector"], "stocks": g["n"], "up": g["up"], "avg_change": round(g["chg"] / g["n"], 2),
                             "avg_score": round(g["score"] / g["n"], 3)})
        sec_rows.sort(key=lambda x: -x["avg_score"])
        sec_chg = {x["sector"]: x["avg_change"] for x in sec_rows}
        breadth_side = 1 if up > down else -1 if down > up else 0
        minute = cal.now_ist().strftime("%H:%M")
        contenders = {r["symbol"] for side in ("bullish", "bearish") for r in ranked[side]}
        for r in rows:
            side = r.get("side", 0)
            setup = OB.current_setup(details[r["symbol"]]["zones"], side) if side else None
            r["zone"] = setup.to_dict() if setup else None
            checks = {"stock": r["symbol"] in contenders,
                      "sector": side != 0 and (sec_chg.get(r["sector"], 0) * side) > 0,
                      "breadth": side != 0 and breadth_side == side,
                      "vwap": side != 0 and (r["above_vwap"] if side > 0 else not r["above_vwap"])}
            r["checks"] = checks
            r["aligned"] = sum(checks.values())
            d = details[r["symbol"]]
            r["confirm"] = CF.evaluate(side, self._confirm_inputs(r, d["ctx"], d["bars"], prof, idx_chg, live_oi)) if side else None
            key = f"{side}:{r['symbol']}"
            prev = self._prev_ranks.get(key)
            r["rank_change"] = (prev - r["rank"]) if (prev and r.get("rank")) else None
            r["new_top10"] = bool(r.get("rank") and r["rank"] <= 10 and (prev is None or prev > 10))
        if minute != self._prev_minute:
            self._prev_ranks = {f"{r.get('side', 0)}:{r['symbol']}": r["rank"] for r in rows if r.get("rank")}
            self._prev_minute = minute
        by_move = sorted(sec_rows, key=lambda x: -x["avg_change"])
        lead = by_move[0] if by_move else None
        lag = by_move[-1] if by_move else None
        board = {"session": session, "live": live, "updated_at": time.time(), "stocks": len(rows),
                 "fno_date": (self.fno or {}).get("date"), "up": up, "down": down,
                 "leading": lead, "lagging": lag, "sectors": sec_rows,
                 "bullish": ranked["bullish"], "bearish": ranked["bearish"],
                 "picks": ranked["bullish_picks"] + ranked["bearish_picks"],
                 "confirm_rules": [{"key": k, "label": lab, "rule": rule} for k, lab, rule in CF.RULES],
                 "index_chg": round(idx_chg, 2) if idx_chg is not None else None,
                 "live_oi": {"stocks": len(live_oi), "at": self.live_oi.at if live_oi else None, "error": self.live_oi.error},
                 "map": [{"symbol": r["symbol"], "sector": r["sector"], "change_pct": r["change_pct"], "side": r.get("side", 0),
                          "contender": r["symbol"] in contenders, "pick": bool(r.get("pick")), "event": r["event"],
                          "score": r["score"]} for r in rows],
                 "seconds": round(time.time() - t0, 2)}
        with self._lock:
            self.board, self.details = board, details
            self._last = time.time()
        self.error = ""
        return f"{len(rows)} F&O stocks ranked ({'live' if live else 'last session ' + str(session)})"

    def view(self) -> Dict:
        with self._lock:
            b = self.board
        if not b:
            return {"ready": False, "error": self.error or "building the board (about a minute on first start)"}
        out = {k: v for k, v in b.items()}
        out["ready"] = True
        out["error"] = self.error
        return out

    def stock(self, symbol: str) -> Optional[Dict]:
        symbol = symbol.upper()
        with self._lock:
            d = self.details.get(symbol)
            b = self.board
        if not d or not b:
            return None
        row = next((r for side in ("bullish", "bearish") for r in b[side] if r["symbol"] == symbol), None)
        if row is None:
            row = next((r for r in b["map"] if r["symbol"] == symbol), {"symbol": symbol})
        return {"symbol": symbol, "name": self.e.universe.name(symbol), "session": b["session"], "live": b["live"],
                "row": row, "bars": [x.to_dict() for x in d["bars"]], "vwap": [round(v, 2) for v in d["vwap"]],
                "zones": [z.to_dict() for z in d["zones"]],
                "prev_close": round(d["ctx"]["prev_close"], 2), "prev_high": round(d["ctx"]["prev_high"], 2),
                "prev_low": round(d["ctx"]["prev_low"], 2), "cpr": {k: round(v, 2) for k, v in d["ctx"]["cpr"].items()}}

    # ------------------------------------------------------------ its own clock
    def start(self):
        threading.Thread(target=self._loop, name="orderblocks", daemon=True).start()

    def _loop(self):
        while True:
            try:
                now = cal.now_ist()
                open_ = cal.session_state(now) == "OPEN"
                age = time.time() - self._last
                # in the session: just after each minute's candle closes; otherwise every 30 minutes
                due = (age >= 55 and now.second >= 2) if open_ else age >= 1800
                if due:
                    self.e.jobs["order_blocks"] = {"state": "running", "started_at": time.time(), "message": ""}
                    msg = self.refresh()
                    self.e.jobs["order_blocks"].update(state="done" if not self.error else "failed", finished_at=time.time(), message=msg)
                    if open_:
                        try:
                            self.update_live_oi()
                        except Exception as e:           # live OI is an extra: never fail the board
                            logger.warning("live futures OI failed: %s", e)
                            self.live_oi.error = str(e)
            except Exception as e:
                logger.exception("order-block board failed")
                self.error = str(e)
                self.e.jobs["order_blocks"] = {"state": "failed", "finished_at": time.time(), "message": str(e)}
                self._last = time.time()
            time.sleep(1.0)
