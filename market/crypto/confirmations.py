"""Confirmation checks for the crypto & XAUUSD board: does the rest of the evidence agree with a pair's side?

The same idea as market/confirmations.py for NSE stocks: each check is True (agrees), False (disagrees) or None (no
data: not counted), and the score is the True checks out of the ones with data. "Today" is the UTC day (05:30 to 05:30
IST), the day Binance's daily candles use. None of these checks is backtested.

    oi          futures open interest (Binance USD-M perpetual) up at least OI_MIN % since 00:00 UTC, with the price
                moving the pair's way since then: fresh positions on its side (long or short build-up).
    taker       share of today's volume bought by aggressive buyers (taker buys): at least TAKER_BULL for bulls,
                at most TAKER_BEAR for bears.
    volume      today's volume through the last finished candle against the average of the last VOL_DAYS days by the
                same time of day (from hourly candles): at least VOL_MIN times.
    narrow_cpr  today's central pivot range from yesterday's high, low and close (market/confirmations.cpr), narrower
                than at least CPR_NARROW of the last CPR_DAYS days' CPRs.
    cpr_side    price above the CPR's top (bulls) or below its bottom (bears).
    prev_day    price beyond yesterday's high (bulls) or low (bears).
    trend       price above (bulls) or below (bears) the average close of the last TREND_DAYS days.
Funding (what longs pay shorts every 8 hours) is shown with the open interest, not scored.
"""
from typing import Dict, List, Optional

from market.confirmations import CPR_DAYS, CPR_NARROW, TREND_DAYS, VOL_DAYS, VOL_MIN

OI_MIN = 1.0
TAKER_BULL, TAKER_BEAR = 0.52, 0.48

# (key, column header, rule) in the order the dashboard shows them
RULES = [
    ("oi", "Futures OI", f"futures open interest up at least {OI_MIN:g}% since 05:30 IST with the price moving the pair's way (long build-up for bulls, short build-up for bears)"),
    ("taker", "Taker buying", f"aggressive buyers took at least {round(100 * TAKER_BULL)}% of today's volume (bulls), or at most {round(100 * TAKER_BEAR)}% (bears)"),
    ("volume", "Volume", f"today's volume at least {VOL_MIN:g}x the average of the last {VOL_DAYS} days by the same time"),
    ("narrow_cpr", "Narrow CPR", f"today's central pivot range narrower than {round(100 * CPR_NARROW)}% of the last {CPR_DAYS} days'"),
    ("cpr_side", "vs CPR", "price above the CPR's top (bulls) or below its bottom (bears)"),
    ("prev_day", "Prev day", "price beyond yesterday's high (bulls) or low (bears)"),
    ("trend", "20D trend", f"price above (bulls) or below (bears) the {TREND_DAYS}-day average close"),
]


def price(v: float) -> str:
    return f"{v:,.2f}" if v >= 100 else f"{v:.4f}" if v >= 1 else f"{v:.6g}"


def rel_volume(today_vol: float, hourly: List, day0: int, upto: int) -> Optional[float]:
    """Today's volume over the average volume from 00:00 UTC to the same time of day on the last VOL_DAYS full days.
    hourly: 1-hour candles (.start epoch s, .v); upto: seconds after 00:00 UTC through the last finished candle."""
    days: Dict[int, Dict[int, float]] = {}
    for b in hourly:
        if b.start < day0:
            days.setdefault(b.start // 86400, {})[(b.start % 86400) // 3600] = b.v
    keys = [k for k in sorted(days)[-VOL_DAYS:] if len(days[k]) >= 20]
    if len(keys) < 5 or today_vol <= 0:
        return None
    h, frac = upto // 3600, (upto % 3600) / 3600
    avg = sum(sum(days[k].get(i, 0.0) for i in range(h)) + frac * days[k].get(h, 0.0) for k in keys) / len(keys)
    return today_vol / avg if avg > 0 else None


def evaluate(side: int, x: Dict) -> Dict:
    """x: last, day_chg (since 00:00 UTC), oi_chg, funding, taker, rvol, upto (HH:MM IST), cpr, cpr_rank, prev_high,
    prev_low, sma. Missing keys mean no data. Returns {"checks": {key: {"ok", "v", "text"}}, "score", "of"}."""
    out: Dict[str, Dict] = {}
    last = x.get("last")
    bull = side > 0

    def put(key, ok, v="–", text="no data"):
        out[key] = {"ok": ok, "v": v, "text": text}

    oi, pc, fr = x.get("oi_chg"), x.get("day_chg"), x.get("funding")
    fund = f" · funding {100 * fr:+.4f}% every 8 hours" if fr is not None else ""
    if oi is not None and pc is not None:
        what = ("long build-up" if pc >= 0 else "short build-up") if oi >= 0 else ("short covering" if pc >= 0 else "long unwinding")
        put("oi", oi >= OI_MIN and pc * side > 0, f"{oi:+.1f}%",
            f"futures OI {oi:+.1f}% since 05:30 IST with price {pc:+.2f}%: {what}{fund}")
    else:
        put("oi", None, text="no futures open interest for this pair yet")

    tk = x.get("taker")
    if tk is not None:
        put("taker", (tk >= TAKER_BULL) if bull else (tk <= TAKER_BEAR), f"{100 * tk:.0f}%",
            f"aggressive buyers took {100 * tk:.1f}% of today's volume, sellers {100 * (1 - tk):.1f}%")
    else:
        put("taker", None)

    rv = x.get("rvol")
    if rv is not None:
        put("volume", rv >= VOL_MIN, f"{rv:.1f}x",
            f"{100 * (rv - 1):+.0f}% vs the last {VOL_DAYS} days' average by {x.get('upto') or 'this time'}")
    else:
        put("volume", None)

    c, rank = x.get("cpr"), x.get("cpr_rank")
    if c and rank is not None:
        put("narrow_cpr", rank >= CPR_NARROW, f"{c['width_pct']:.2f}%",
            f"{c['width_pct']:.2f}% wide, narrower than {round(100 * rank)}% of the last {CPR_DAYS} days' CPRs")
    else:
        put("narrow_cpr", None)

    if c and last is not None:
        where = "above" if last > c["tc"] else "below" if last < c["bc"] else "inside"
        put("cpr_side", where == ("above" if bull else "below"), where,
            f"price {price(last)} {where} the CPR {price(c['bc'])} – {price(c['tc'])} (pivot {price(c['pivot'])})")
    else:
        put("cpr_side", None)

    ph, pl = x.get("prev_high"), x.get("prev_low")
    if last is not None and ph and pl:
        if bull:
            put("prev_day", last > ph, "above PDH" if last > ph else f"{100 * (last / ph - 1):+.1f}%",
                f"price {price(last)} vs yesterday's high {price(ph)}")
        else:
            put("prev_day", last < pl, "below PDL" if last < pl else f"{100 * (last / pl - 1):+.1f}%",
                f"price {price(last)} vs yesterday's low {price(pl)}")
    else:
        put("prev_day", None)

    sma = x.get("sma")
    if last is not None and sma:
        d = 100 * (last / sma - 1)
        put("trend", (d > 0) if bull else (d < 0), f"{d:+.1f}%", f"price {d:+.1f}% from its {TREND_DAYS}-day average close {price(sma)}")
    else:
        put("trend", None)

    return {"checks": out, "score": sum(1 for v in out.values() if v["ok"] is True),
            "of": sum(1 for v in out.values() if v["ok"] is not None)}
