"""The order-block board: every NSE F&O stock ranked by clean one-way strength each minute, with its
order-block zone, levels, sector, market alignment and yesterday's futures OI build-up (rules: market/orderblocks.py),
plus the confirmation checks: futures and options OI, live OI, CPR, volume, previous day, trend, NIFTY (market/confirmations.py).

Live during the session from the engine's 5-minute bars (the F&O stocks are subscribed to the feed); at other
times it shows the last session, rebuilt from cached 5-minute history. It runs on its own thread so a long
morning scan never delays it. Ranks, zones and alignment move with finished candles (each minute, and right after
each 5-minute close); live_view() re-reads every board stock's price, live OI and price-based confirmations
whenever it is asked, at most once a second. Angel One's OI readings (futures, the leaders' options, its build-up
lists) run on a second thread.

Heads-up, every second in the session, for the picks and the top 20 a side (on each stock's own side): a break
forming (the live price beyond the swing level whose 5-minute close would give a zone, with the zone it would
give), a zone just given (the first-stage plan: the live price, the zone's stop, the targets as given, R:R from
the live price and the risk rules' verdict on it) and the price near Z (within NEAR_ATR daily ATRs, before the
first touch). Each one is logged once (alerts_view) for the dashboard's list, beep and notification.
"""
import datetime as dt
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional

from market import calendar as cal
from market import confirmations as CF
from market import orderblocks as OB
from market.candles.bars import Bar, DailyBar, split_sessions, vwap_series
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
        self.live_opt = fno.LiveOptions()
        self.buildups = fno.BuildUpLists()
        self._live: Optional[Dict] = None
        self._vol_prof: Dict[str, Optional[List[float]]] = {}
        self._days5m: Dict[str, Dict[str, DailyBar]] = {}
        self._hist_session = None
        self.heads: Dict[str, Dict] = {}          # symbol -> its heads-up right now
        self.alerts = deque(maxlen=300)           # today's heads-ups, each logged once
        self.alert_seq = 0
        self._alerted = set()
        self._alert_day = None
        self._alock = threading.Lock()
        self._scan_err = ""

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
        """Live bars for every F&O stock, and their futures' open interest on the websocket, nearest expiry first
        (called when the day is prepared)."""
        syms = self.symbols()
        if syms:
            self.e.feed.subscribe(syms)
            today = cal.now_ist().date().isoformat()
            cs = sorted((c["expiry"], str(c["token"]), c["symbol"]) for s in syms for c in self.e.universe.futures_for(s, today))
            n = self.e.feed.subscribe_fo({t: sym for _, t, sym in cs}, self.live_oi.on_stream)
            if n:
                logger.info("futures OI on the websocket: %d of %d contracts", n, len(cs))
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
                **self._options_inputs(s),
                "cpr": ctx["cpr"], "cpr_rank": ctx["cpr_rank"], "rvol": CF.rel_volume(bars, prof.get(s)),
                "upto": cal.hhmm(last.close_minute), "prev_high": ctx["prev_high"], "prev_low": ctx["prev_low"],
                "sma": ctx["sma"], "index_chg": idx_chg}

    def _options_inputs(self, s) -> Dict:
        o = self.live_opt.data.get(s) if self.live_opt.day == cal.now_ist().date().isoformat() else None
        if not o or "ce_chg" not in o:
            return {}
        return {"opt_pe_chg": o["pe_chg"], "opt_ce_chg": o["ce_chg"], "opt_pcr": o.get("pcr"), "opt_expiry": o["expiry"]}

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

    LEADERS = 10                                  # live options OI for the picks and the top 10 on each side

    def update_live_options(self) -> Optional[str]:
        """Today's options OI of the nearest expiry for the board's leaders, every few minutes in the session."""
        api = self.e.adapter.smart_api if self.e.adapter else None
        with self._lock:
            b = self.board
        if api is None or not self.fno or not b or not b.get("live") or not self.live_opt.due():
            return None
        today = cal.now_ist().date().isoformat()
        leaders = [r["symbol"] for r in b.get("picks") or []] + \
                  [r["symbol"] for side in ("bullish", "bearish") for r in (b.get(side) or [])[:self.LEADERS]]
        chains = {}
        for s in dict.fromkeys(leaders):
            c = self.e.universe.options_for(s, today)
            if c:
                chains[s] = c
        if not chains:
            self.live_opt.error = "no stock options in Angel One's scrip master"
            return self.live_opt.error
        base = {s: (v.get("opt_exp") or {}) for s, v in (self.fno.get("oi") or {}).items()}
        return self.live_opt.refresh(api, chains, base, today)

    def update_buildups(self) -> Optional[str]:
        """Angel One's OI build-up lists (near-month futures), every few minutes in the session."""
        api = self.e.adapter.smart_api if self.e.adapter else None
        if api is None or not self.buildups.due():
            return None
        owner = {c["symbol"]: s for s, cs in self.e.universe.futures.items() for c in cs}
        return self.buildups.refresh(api, owner.get)

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
        live_opt = self.live_opt.data if self.live_opt.day == session else {}
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
            details[s] = {"bars": bars, "vwap": vw, "zones": zones, "ctx": ctx, "nb": OB.next_break(bars) if not ev else {}}
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
                 "live_oi": {"stocks": len(live_oi), "at": self.live_oi.at if live_oi else None, "error": self.live_oi.error,
                             "streamed": len(self.live_oi.stream) if self.live_oi.stream_ok else 0,
                             "stream_at": self.live_oi.stream_at if self.live_oi.stream_ok else None},
                 "live_opt": {"stocks": len(live_opt), "at": self.live_opt.at if live_opt else None, "error": self.live_opt.error,
                              "rows": {s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()} for s, o in live_opt.items()}},
                 "oi_lists": {"lists": self.buildups.lists, "at": self.buildups.at or None, "error": self.buildups.error},
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

    # ------------------------------------------------------------ live, between the minute refreshes
    LIVE_S = 0.8                                  # live_view() rebuilds its rows at most this often

    def _index_live(self) -> Optional[float]:
        """NIFTY 50 now (its last tick) against the session's open; None without a live price."""
        sym = self.e.settings.get("market_filter.symbol", "^NSEI")
        p, _ = self.e.state.price(sym)
        bars = self.e.state.bars(sym)
        first = bars[0] if bars else self.e.state.forming_bar(sym)
        return 100 * (p / first.o - 1) if p and first and first.o else None

    @staticmethod
    def _rvol_now(volume: Optional[float], profile: Optional[List[float]], now) -> Optional[float]:
        """Today's volume so far against the usual volume by this moment (inside the current candle, pro rata)."""
        if not volume or not profile:
            return None
        mins = now.hour * 60 + now.minute + now.second / 60.0 - cal.OPEN_MIN
        if mins <= 0:
            return None
        k = min(int(mins // cal.BAR_MINUTES), len(profile) - 1)
        frac = 1.0 if k == len(profile) - 1 else (mins - k * cal.BAR_MINUTES) / cal.BAR_MINUTES
        before = profile[k - 1] if k > 0 else 0.0
        usual = before + (profile[k] - before) * frac
        return volume / usual if usual > 0 else None

    def _live_rows(self, b: Dict, details: Dict) -> Dict:
        now_t = time.time()
        out = {"at": now_t, "board_at": b["updated_at"], "live": b["live"], "rows": {}, "chg": {}}
        if not b["live"]:
            return out
        now = cal.now_ist()
        session = b["session"]
        live_oi = self.live_oi.oi if self.live_oi.day == session else {}
        idx = self._index_live()
        if idx is None:
            idx = b.get("index_chg")
        syms = [m["symbol"] for m in b["map"]]
        snap = self.e.state.snapshot(syms)
        for s in syms:
            p, d = (snap.get(s) or {}).get("ltp"), details.get(s)
            if p and d:
                out["chg"][s] = round(100 * (p / d["ctx"]["prev_close"] - 1), 2)
        seen = set()
        for r in (b.get("picks") or []) + (b.get("bullish") or []) + (b.get("bearish") or []):
            s, side = r["symbol"], r.get("side", 0)
            if s in seen or s not in details or s not in out["chg"]:
                continue
            seen.add(s)
            d, q = details[s], snap[s]
            row = dict(r, last=q["ltp"], change_pct=out["chg"][s])
            x = self._confirm_inputs(row, d["ctx"], d["bars"], self._vol_prof, idx, live_oi)
            rv = self._rvol_now(q.get("volume"), self._vol_prof.get(s), now)
            if rv is not None:
                x["rvol"], x["upto"] = rv, now.strftime("%H:%M:%S")
            out["rows"][s] = {"last": round(q["ltp"], 2), "chg": out["chg"][s], "age": q.get("age_s"),
                              "confirm": CF.evaluate(side, x) if side else None, "heads": self.heads.get(s)}
        out["index_chg"] = round(idx, 2) if idx is not None else None
        out["live_oi"] = {"stocks": len(live_oi), "streamed": len(self.live_oi.stream) if self.live_oi.stream_ok else 0,
                          "stream_at": self.live_oi.stream_at if self.live_oi.stream_ok else None}
        return out

    def live_view(self, symbol: Optional[str] = None, after: int = 0) -> Dict:
        """Every board stock's price and change, and the board's top 20 a side with their confirmations re-checked
        on live prices; with `symbol`, its candles after `after` (epoch s) and the forming one, for its chart."""
        with self._lock:
            b, details = self.board, self.details
        if not b:
            return {"ready": False}
        cached = self._live
        if cached is None or time.time() - cached["at"] >= self.LIVE_S or cached["board_at"] != b["updated_at"]:
            cached = self._live = self._live_rows(b, details)
        out = dict(cached, ready=True)
        if symbol and b["live"]:
            symbol = symbol.upper()
            bars = self.e.state.bars(symbol)
            tail = [x.to_dict() for x in bars if x.start > after]
            f = self.e.state.forming_bar(symbol)
            if f and (not bars or f.start > bars[-1].start) and f.start > after:
                tail.append(f.to_dict())
            out["tail"] = {"symbol": symbol, "bars": tail, "forming": bool(f and tail and tail[-1]["t"] == f.start)}
        return out

    # ------------------------------------------------------------ heads-up, every second in the session
    NEAR_ATR = 0.1                                # "near Z": within this many daily ATRs of Z, before the first touch

    def _first_rules(self) -> Dict:
        """The first-stage plan's rules: the first-stage strategy's window and target, the risk rules' limits."""
        S = self.e.settings
        p = S.strategy("order_block_first") or {}
        return {"window": int(p.get("window_bars", 1)) * cal.BAR_MINUTES * 60, "t1": str(p.get("target", "T2")).upper() == "T1",
                "target_r": float(p.get("target_r", 2.0)), "min_rr": float(S.get("min_reward_risk", 1.0)),
                "min_pct": float(S.get("min_risk_pct", 0.1)), "max_pct": float(S.get("max_risk_pct", 3.0))}

    def _scan(self, t_now: float):
        """The heads-up of the picks and the top 20 a side, on each stock's own side; each new one logged once."""
        with self._lock:
            b, details = self.board, self.details
        if not b or not b.get("live"):
            self.heads = {}
            return
        if self._alert_day != b["session"]:
            with self._alock:
                self._alert_day, self._alerted = b["session"], set()
                self.alerts.clear()
        rules = self._first_rules()
        heads, seen = {}, set()
        for r in (b.get("picks") or []) + (b.get("bullish") or []) + (b.get("bearish") or []):
            s, side = r["symbol"], r.get("side", 0)
            if s in seen or not side or r.get("event") or s not in details:
                continue
            seen.add(s)
            p, age = self.e.state.price(s)
            if not p or age is None or age > 60:
                continue
            h = self._heads_for(s, side, r.get("zone"), p, details[s], t_now, rules)
            if h:
                heads[s] = h
                self._alert(s, side, h, t_now)
        self.heads = heads

    def _heads_for(self, s: str, side: int, z: Optional[Dict], p: float, d: Dict, t_now: float, rules: Dict) -> Optional[Dict]:
        """One stock's heads-up at the live price p: a break forming, a zone just given (first stage) or near Z."""
        bars = self.e.state.bars(s) or d["bars"]
        nb = d.get("nb") or {}
        if bars and d["bars"] and bars[-1].start != d["bars"][-1].start:
            nb = OB.next_break(bars)                  # a candle closed since the board's last refresh
        x = nb.get(side)
        if x and ((p > x["level"]) if side > 0 else (p < x["level"])):
            last, f = bars[-1], self.e.state.forming_bar(s)
            if f is not None and f.start > last.start:
                cur = Bar(f.start, f.o, max(f.h, p), min(f.l, p), p, f.v)
            else:                                     # Yahoo: no forming candle, the last price stands in for one
                cur = Bar(last.start + cal.BAR_MINUTES * 60, last.c, max(last.c, p), min(last.c, p), p, 0.0)
            wz = next((zz for zz in OB.detect_zones(bars + [cur], d["ctx"]["prev_high"], d["ctx"]["prev_low"])
                       if zz.formed_at == cur.start and zz.side == side), None)
            if wz:
                return {"kind": "break", "price": round(p, 2), "level": OB._r(x["level"]), "z": OB._r(wz.z),
                        "stop": OB._r(wz.invalid), "t": [OB._r(v) for v in wz.given_targets], "closes": cur.start + cal.BAR_MINUTES * 60}
        if not z or z.get("status") == "broken" or z.get("entry_at"):
            return None
        beyond = (p - z["z"]) * side
        if t_now <= z["known_at"] + rules["window"] and beyond > 0:
            risk = (p - z["invalid"]) * side
            ahead = [v for v in z.get("given_targets") or [] if (v - p) * side > 0]
            tgt = (ahead[0] if rules["t1"] else ahead[-1]) if ahead else p + side * rules["target_r"] * risk
            rr = (tgt - p) * side / risk if risk > 0 else 0.0
            pct = 100.0 * risk / p
            gate = (f"reward:risk {rr:.2f} from {p:.2f} is under {rules['min_rr']:g}" if rr + 1e-9 < rules["min_rr"] else
                    f"the stop is {pct:.2f}% away, over {rules['max_pct']:g}%" if pct > rules["max_pct"] else
                    f"the stop is {pct:.2f}% away, under {rules['min_pct']:g}%" if pct < rules["min_pct"] else None)
            return {"kind": "given", "price": round(p, 2), "z": z["z"], "stop": z["invalid"], "t": z.get("given_targets") or [],
                    "target": OB._r(tgt), "rr": round(rr, 2), "risk_pct": round(pct, 2), "gate": gate,
                    "until": z["known_at"] + rules["window"], "formed": z["formed_at"]}
        atr = d["ctx"].get("atr") or 0.0
        if atr > 0 and 0 < beyond <= self.NEAR_ATR * atr:
            return {"kind": "near", "price": round(p, 2), "z": z["z"], "dist": round(beyond, 2), "dist_atr": round(beyond / atr, 3),
                    "formed": z["formed_at"]}
        return None

    def _alert(self, s: str, side: int, h: Dict, t_now: float):
        key = (s, side, h["kind"], h["closes"] if h["kind"] == "break" else h["formed"])
        with self._alock:
            if key in self._alerted:
                return
            self._alerted.add(key)
            self.alert_seq += 1
            self.alerts.append(dict(h, seq=self.alert_seq, at=t_now, sym=s, side=side, text=self._alert_text(s, side, h)))

    @staticmethod
    def _alert_text(s: str, side: int, h: Dict) -> str:
        buy, hm = side > 0, (lambda ts: cal.to_ist(ts).strftime("%H:%M"))
        if h["kind"] == "break":
            return (f"{s} breaking {'out' if buy else 'down'}: {h['price']:.2f} is {'above' if buy else 'below'} the swing "
                    f"{'high' if buy else 'low'} {h['level']:.2f}. If the {hm(h['closes'] - cal.BAR_MINUTES * 60)} candle closes "
                    f"there, a {'buy' if buy else 'sell'} zone: Z {h['z']:.2f}, stop-loss {h['stop']:.2f}")
        if h["kind"] == "given":
            return (f"{s} {'buy' if buy else 'sell'} zone given: first stage at {h['price']:.2f}, stop-loss {h['stop']:.2f}, "
                    f"target {h['target']:.2f} (R:R {h['rr']:.1f}) until {hm(h['until'])}"
                    + (f". The risk check would refuse it: {h['gate']}" if h["gate"] else ""))
        return f"{s} near Z: {h['price']:.2f} is {h['dist']:.2f} from Z {h['z']:.2f}"

    def alerts_status(self) -> Dict:
        return {"seq": self.alert_seq, "active": len(self.heads)}

    def alerts_view(self, after: int = 0) -> Dict:
        with self._alock:
            rows = [a for a in self.alerts if a["seq"] > after]
        return {"seq": self.alert_seq, "day": self._alert_day, "alerts": rows}

    # ------------------------------------------------------------ its own clock
    def start(self):
        threading.Thread(target=self._loop, name="orderblocks", daemon=True).start()
        threading.Thread(target=self._oi_loop, name="angel-oi", daemon=True).start()

    def _oi_loop(self):
        """Angel One's OI readings, each when due (every few minutes) in the session; the build-up lists once at
        other times too, so the page shows the last session's."""
        while True:
            open_ = cal.session_state(cal.now_ist()) == "OPEN"
            for name, fn, always in (("live futures OI", self.update_live_oi, False), ("live options OI", self.update_live_options, False),
                                     ("OI build-up lists", self.update_buildups, True)):
                if not (open_ or (always and not self.buildups.at)):
                    continue
                try:
                    fn()
                except Exception as e:                   # live OI is an extra: never fail the board
                    logger.warning("%s failed: %s", name, e)
            time.sleep(1.0)

    def _bar_lag(self) -> int:
        """Seconds after a 5-minute close before its candle is in the engine's bars (Yahoo re-reads them later)."""
        feed = getattr(self.e, "feed", None)
        return 2 if getattr(feed, "mode", "") == "angel" else int(self.e.settings.get("data.bar_delay_seconds", 20)) + 3

    def _loop(self):
        while True:
            open_ = False
            try:
                now = cal.now_ist()
                open_ = cal.session_state(now) == "OPEN"
                age = time.time() - self._last
                # in the session: just after each minute's candle closes, and just after each 5-minute close (a zone
                # given at that close reaches the board in seconds); otherwise every 30 minutes
                lag, since = self._bar_lag(), (now.minute % cal.BAR_MINUTES) * 60 + now.second
                due = ((age >= 55 and now.second >= 2) or (since >= lag and age > since - lag + 1)) if open_ else age >= 1800
                if due:
                    self.e.jobs["order_blocks"] = {"state": "running", "started_at": time.time(), "message": ""}
                    msg = self.refresh()
                    self.e.jobs["order_blocks"].update(state="done" if not self.error else "failed", finished_at=time.time(), message=msg)
            except Exception as e:
                logger.exception("order-block board failed")
                self.error = str(e)
                self.e.jobs["order_blocks"] = {"state": "failed", "finished_at": time.time(), "message": str(e)}
                self._last = time.time()
            if open_:
                try:
                    self._scan(time.time())
                except Exception as e:                   # a heads-up is an extra: never stop the board
                    if str(e) != self._scan_err:
                        logger.warning("order-block heads-up failed: %s", e)
                    self._scan_err = str(e)
            elif self.heads:
                self.heads = {}
            time.sleep(1.0)
