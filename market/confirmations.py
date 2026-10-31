"""Confirmation checks for the order-block board: does the rest of the evidence agree with a stock's side?

The board's side comes from its strength score (+1 bullish, -1 bearish). Each check below is True (agrees),
False (disagrees) or None (no data: not counted). The score is the number of True checks out of the ones with data.
All rules are published here; none of them is backtested (research/lib/obtest.py tests the board alone).

    futures     NSE's latest F&O bhavcopy: futures OI rose with the price that day, i.e. fresh positions on the
                stock's side: long build-up (bullish) or short build-up (bearish). Covering and unwinding do not count.
    live_oi     futures OI now (every expiry, summed, from Angel One) against that bhavcopy, with today's price change:
                OI up at least LIVE_OI_MIN % and the price moving the stock's way (a build-up today).
    options     the bhavcopy's options OI change: more put OI added than call OI (put writing, bullish), or more call
                OI than put OI (call writing, bearish). PCR = put OI / call OI over every expiry, shown, not scored;
                Angel One's live PCR (near expiry only) is shown beside it.
    narrow_cpr  today's central pivot range from the previous session's high, low and close:
                pivot = (H + L + C) / 3, BC = (H + L) / 2, TC = 2 x pivot - BC, width = |TC - BC| / pivot.
                Narrow = narrower than at least CPR_NARROW of the stock's last CPR_DAYS CPRs (a trending day is likelier).
    cpr_side    price above the CPR's top (bullish) or below its bottom (bearish).
    volume      volume of today's finished candles against the same candles' average over the last VOL_DAYS sessions
                (the same time of day): at least VOL_MIN times.
    prev_day    price beyond the previous session's high (bullish) or low (bearish).
    trend       price above (bullish) or below (bearish) the average close of the last TREND_DAYS sessions.
    index       the NIFTY 50 above (bullish) or below (bearish) its open (finished candles).
"""
from typing import Dict, List, Optional

from market.calendar import SESSION_BARS
from market.candles.bars import Bar, DailyBar, split_sessions

LIVE_OI_MIN = 1.0
CPR_DAYS, CPR_NARROW = 20, 0.8
VOL_DAYS, VOL_MIN = 20, 1.5
TREND_DAYS = 20

# (key, column header, rule) in the order the dashboard shows them
RULES = [
    ("futures", "Futures OI", "futures OI build-up on the stock's side in the latest NSE bhavcopy (long build-up for bulls, short build-up for bears)"),
    ("live_oi", "OI today", f"futures OI up at least {LIVE_OI_MIN:g}% since the bhavcopy, with today's price moving the stock's way (Angel One, every few minutes)"),
    ("options", "Options OI", "more put OI written than call OI (bulls), or more call OI than put OI (bears), in the latest bhavcopy"),
    ("narrow_cpr", "Narrow CPR", f"today's central pivot range narrower than {round(100 * CPR_NARROW)}% of the stock's last {CPR_DAYS}"),
    ("cpr_side", "vs CPR", "price above the CPR's top (bulls) or below its bottom (bears)"),
    ("volume", "Volume", f"today's volume at least {VOL_MIN:g}x the average of the last {VOL_DAYS} sessions by the same time"),
    ("prev_day", "Prev day", "price beyond the previous session's high (bulls) or low (bears)"),
    ("trend", "20D trend", f"price above (bulls) or below (bears) the {TREND_DAYS}-session average close"),
    ("index", "NIFTY 50", "the NIFTY 50 above (bulls) or below (bears) its open"),
]
SHORT_BUILDUP = {"long build-up": "LBU", "short build-up": "SBU", "short covering": "SC", "long unwinding": "LU"}


def cpr(h: float, l: float, c: float) -> Dict:
    pivot = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * pivot - bc
    bc, tc = min(bc, tc), max(bc, tc)
    return {"pivot": pivot, "bc": bc, "tc": tc, "width_pct": 100 * (tc - bc) / pivot if pivot else 0.0}


def cpr_rank(before: List[DailyBar], n: int = CPR_DAYS) -> Optional[float]:
    """Share of the last n sessions' CPRs wider than the one `before[-1]` sets for the next session (1 = narrowest)."""
    if len(before) < n + 1:
        return None
    widths = [cpr(b.h, b.l, b.c)["width_pct"] for b in before[-(n + 1):]]
    return sum(1 for w in widths[:-1] if w > widths[-1]) / n


def volume_profile(bars: List[Bar], session: str, n: int = VOL_DAYS) -> Optional[List[float]]:
    """Average cumulative volume by 5-minute slot over the last n sessions before `session` (None under 5 sessions)."""
    days = list(split_sessions([b for b in bars if b.date < session]).values())[-n:]
    if len(days) < 5:
        return None
    prof = [0.0] * SESSION_BARS
    for day in days:
        by_slot = [0.0] * SESSION_BARS
        for b in day:
            if 0 <= b.slot < SESSION_BARS:
                by_slot[b.slot] += b.v
        run = 0.0
        for k in range(SESSION_BARS):
            run += by_slot[k]
            prof[k] += run / len(days)
    return prof


def rel_volume(today: List[Bar], profile: Optional[List[float]]) -> Optional[float]:
    """Today's volume through its last finished candle, as a multiple of the profile at the same slot."""
    if not today or not profile:
        return None
    k = min(today[-1].slot, SESSION_BARS - 1)
    if k < 0 or profile[k] <= 0:
        return None
    vol = sum(b.v for b in today if b.slot <= k)
    return vol / profile[k] if vol > 0 else None


def _qty(x: float) -> str:
    a = abs(x)
    s = f"{a / 1e7:.1f}Cr" if a >= 1e7 else f"{a / 1e5:.1f}L" if a >= 1e5 else f"{a / 1e3:.0f}k" if a >= 1e3 else f"{a:.0f}"
    return ("+" if x >= 0 else "-") + s


def evaluate(side: int, x: Dict) -> Dict:
    """x: last, change_pct, fut_buildup, fut_oi_chg, live_oi_chg, pe_chg, ce_chg, pcr, live_pcr, cpr, cpr_rank,
    rvol, upto (HH:MM), prev_high, prev_low, sma, index_chg. Missing keys mean no data.
    Returns {"checks": {key: {"ok", "v" (short value), "text"}}, "score", "of"}."""
    out: Dict[str, Dict] = {}
    last = x.get("last")
    bull = side > 0

    def put(key, ok, v="–", text="no data"):
        out[key] = {"ok": ok, "v": v, "text": text}

    b = x.get("fut_buildup")
    if b:
        chg = x.get("fut_oi_chg")
        put("futures", b == ("long build-up" if bull else "short build-up"), SHORT_BUILDUP.get(b, b),
            f"{b}" + (f" (futures OI {chg:+.1f}%)" if chg is not None else ""))
    else:
        put("futures", None)

    oi, pc = x.get("live_oi_chg"), x.get("change_pct")
    if oi is not None and pc is not None:
        what = ("long build-up" if pc >= 0 else "short build-up") if oi >= 0 else ("short covering" if pc >= 0 else "long unwinding")
        put("live_oi", oi >= LIVE_OI_MIN and pc * side > 0, f"{oi:+.1f}%", f"futures OI {oi:+.1f}% with price {pc:+.2f}%: {what}")
    else:
        put("live_oi", None, text="no live OI (needs the Angel One login, during the session)")

    pe, ce = x.get("pe_chg"), x.get("ce_chg")
    if pe is not None and ce is not None and (pe or ce):
        pcr, live_pcr = x.get("pcr"), x.get("live_pcr")
        txt = f"put OI {_qty(pe)}, call OI {_qty(ce)}" + (f" · PCR {pcr:.2f} (every expiry)" if pcr is not None else "") + \
              (f" · near-expiry PCR now {live_pcr:.2f} (Angel One)" if live_pcr is not None else "")
        put("options", (pe > ce) if bull else (ce > pe), f"PCR {pcr:.2f}" if pcr is not None else "–", txt)
    else:
        put("options", None)

    c, rank = x.get("cpr"), x.get("cpr_rank")
    if c and rank is not None:
        put("narrow_cpr", rank >= CPR_NARROW, f"{c['width_pct']:.2f}%",
            f"{c['width_pct']:.2f}% wide, narrower than {round(100 * rank)}% of the last {CPR_DAYS} sessions' CPRs")
    else:
        put("narrow_cpr", None)

    if c and last is not None:
        where = "above" if last > c["tc"] else "below" if last < c["bc"] else "inside"
        put("cpr_side", where == ("above" if bull else "below"), where,
            f"price {last:.2f} {where} the CPR {c['bc']:.2f} – {c['tc']:.2f} (pivot {c['pivot']:.2f})")
    else:
        put("cpr_side", None)

    rv = x.get("rvol")
    if rv is not None:
        put("volume", rv >= VOL_MIN, f"{rv:.1f}x",
            f"{100 * (rv - 1):+.0f}% vs the last {VOL_DAYS} sessions' average by {x.get('upto') or 'this time'}")
    else:
        put("volume", None)

    ph, pl = x.get("prev_high"), x.get("prev_low")
    if last is not None and ph and pl:
        if bull:
            put("prev_day", last > ph, "above PDH" if last > ph else f"{100 * (last / ph - 1):+.1f}%",
                f"price {last:.2f} vs the previous high {ph:.2f}")
        else:
            put("prev_day", last < pl, "below PDL" if last < pl else f"{100 * (last / pl - 1):+.1f}%",
                f"price {last:.2f} vs the previous low {pl:.2f}")
    else:
        put("prev_day", None)

    sma = x.get("sma")
    if last is not None and sma:
        d = 100 * (last / sma - 1)
        put("trend", (d > 0) if bull else (d < 0), f"{d:+.1f}%", f"price {d:+.1f}% from its {TREND_DAYS}-session average close {sma:.2f}")
    else:
        put("trend", None)

    ix = x.get("index_chg")
    if ix is not None:
        put("index", (ix > 0) if bull else (ix < 0), f"{ix:+.2f}%", f"{ix:+.2f}% from its open")
    else:
        put("index", None)

    return {"checks": out, "score": sum(1 for v in out.values() if v["ok"] is True),
            "of": sum(1 for v in out.values() if v["ok"] is not None)}
