"""Order-block board: intraday strength ranking and order-block zones for NSE F&O stocks.

All rules are published here (nothing hidden) and use finished 5-minute candles only.

Strength (how cleanly a stock is moving one way today, against its normal daily range):
    move   = (last price - previous close) / ATR14 (daily)
    clean  = |last close - first open| / sum of |close-to-close moves| over today's 5-minute bars (0 = choppy, 1 = straight)
    place  = where the price sits in today's range, from the move's side (1 = at the extreme)
    score  = move x clean x (0.5 + 0.5 x place); stocks with clean < 0.25 score 0 (choppy)
A move over 3 ATR or 8% is flagged "event risk": ranked but never picked and given no levels.

Order block (Smart Money Concepts, as commonly defined):
    a 5-minute candle closes beyond the last confirmed swing high (bullish) or low (bearish): a break of structure.
    The last opposite-coloured candle before that push is the order block; its high-low range is the zone.
    Bullish: Z (entry) = zone top, invalidation S1 = zone bottom - buffer; liquidity R1 / R2 = the next swing highs,
    the day's high, yesterday's high or an opposing zone above. R:R = (R2 - Z) / (Z - S1). Bearish is the mirror.
    A zone is "broken" once a 5-minute candle closes beyond its far edge, "retested" once price comes back into it.

Times (epoch seconds): the entry Z is given when the candle that broke structure closes (known_at); the entry
signal is the first candle after it that trades back to Z (entry_at, that candle's start), before any 5-minute
close beyond the far edge: a limit order at Z fills there. broken_at is the close of the candle that broke it.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from market.calendar import BAR_MINUTES

CHOPPY = 0.25
EVENT_ATR, EVENT_PCT = 3.0, 8.0
SWING = 2                       # bars each side for a swing point (confirmed 2 bars later)


@dataclass
class Zone:
    side: int                    # +1 bullish, -1 bearish
    formed_at: int               # epoch seconds of the candle that broke structure
    ob_start: int                # the order-block candle
    top: float
    bottom: float
    z: float                     # entry level (top for bullish, bottom for bearish)
    invalid: float               # S1 (bullish) / R1 (bearish)
    status: str = "fresh"        # fresh / retested / broken
    visits: int = 0
    strong: bool = False         # left a fair-value gap or a big push away
    targets: List[float] = field(default_factory=list)
    behind: List[float] = field(default_factory=list)
    rr: Optional[float] = None
    known_at: Optional[int] = None       # the break candle's close: when Z, the entry price, is given
    entry_at: Optional[int] = None       # start of the first candle back at Z: the entry signal
    broken_at: Optional[int] = None      # close of the candle that closed beyond the far edge

    def to_dict(self):
        d = asdict(self)
        for k in ("top", "bottom", "z", "invalid"):
            d[k] = _r(d[k])
        if d["rr"] is not None:
            d["rr"] = round(d["rr"], 2)
        d["targets"] = [_r(x) for x in self.targets]
        d["behind"] = [_r(x) for x in self.behind]
        return d


def _r(x: float) -> float:
    """A price rounded for display without losing a cheap coin's digits: 2 decimals from 100 up, 4 from 1, else 6 figures."""
    a = abs(x)
    return round(x, 2) if a >= 100 else round(x, 4) if a >= 1 else float(f"{x:.6g}")


def _swings(H, L):
    """Indices of swing highs / lows (each confirmed SWING bars later)."""
    n = len(H)
    sh, sl = [], []
    for i in range(SWING, n - SWING):
        if H[i] >= max(H[i - SWING:i + SWING + 1]) and H[i] > max(H[i - SWING:i]):
            sh.append(i)
        if L[i] <= min(L[i - SWING:i + SWING + 1]) and L[i] < min(L[i - SWING:i]):
            sl.append(i)
    return sh, sl


def detect_zones(bars, prev_high=None, prev_low=None) -> List[Zone]:
    """Every order block of the session, in the order they formed, with today's status and levels."""
    n = len(bars)
    if n < 2 * SWING + 3:
        return []
    O = [b.o for b in bars]
    H = [b.h for b in bars]
    L = [b.l for b in bars]
    C = [b.c for b in bars]
    rng = [h - l for h, l in zip(H, L)]
    sh, sl = _swings(H, L)
    zones: List[Zone] = []
    used = set()
    for t in range(1, n):
        avg = sum(rng[max(0, t - 20):t]) / max(1, len(rng[max(0, t - 20):t]))
        buf = max(0.1 * avg, 0.0005 * C[t])
        for side in (1, -1):
            pts = [i for i in (sh if side > 0 else sl) if i + SWING <= t - 1]   # confirmed before this bar
            if not pts:
                continue
            k = pts[-1]
            lvl = H[k] if side > 0 else L[k]
            broke = (C[t] > lvl >= C[t - 1]) if side > 0 else (C[t] < lvl <= C[t - 1])
            if not broke or (side, k) in used:
                continue
            used.add((side, k))
            # the last opposite candle before the push, searched back to the start of the push
            ob = None
            for j in range(t - 1, max(-1, t - 16), -1):
                if (side > 0 and C[j] < O[j]) or (side < 0 and C[j] > O[j]):
                    ob = j
                    break
            if ob is None:
                continue
            top, bottom = H[ob], L[ob]
            push = (C[t] - top) if side > 0 else (bottom - C[t])
            fvg = any((L[j] > H[j - 2]) if side > 0 else (H[j] < L[j - 2]) for j in range(ob + 2, t + 1))
            zz = Zone(side=side, formed_at=bars[t].start, ob_start=bars[ob].start, top=top, bottom=bottom,
                      z=top if side > 0 else bottom, invalid=(bottom - buf) if side > 0 else (top + buf),
                      strong=bool(fvg or push >= avg), known_at=bars[t].start + BAR_MINUTES * 60)
            # what happened after it formed: the first touch of Z is the entry signal, even in the candle that breaks it
            for j in range(t + 1, n):
                inside = (L[j] <= top) if side > 0 else (H[j] >= bottom)
                if inside and zz.entry_at is None:
                    zz.entry_at = bars[j].start
                if (side > 0 and C[j] < bottom) or (side < 0 and C[j] > top):
                    zz.status = "broken"
                    zz.broken_at = bars[j].start + BAR_MINUTES * 60
                    break
                was_inside = j - 1 > t and ((L[j - 1] <= top) if side > 0 else (H[j - 1] >= bottom))
                if inside and not was_inside:
                    zz.visits += 1
                    zz.status = "retested"
            zones.append(zz)
    last = C[-1]
    day_hi, day_lo = max(H), min(L)
    for zz in zones:
        if zz.side > 0:
            ahead = [H[i] for i in sh if H[i] > max(zz.z, last)] + [day_hi] + ([prev_high] if prev_high else []) + \
                    [o.bottom for o in zones if o.side < 0 and o.status != "broken" and o.bottom > max(zz.z, last)]
            zz.targets = sorted({_r(x) for x in ahead if x > max(zz.z, last) * 1.0005})[:2]
            zz.behind = sorted({_r(L[i]) for i in sl if L[i] < zz.invalid}, reverse=True)[:3]
            risk = zz.z - zz.invalid
        else:
            ahead = [L[i] for i in sl if L[i] < min(zz.z, last)] + [day_lo] + ([prev_low] if prev_low else []) + \
                    [o.top for o in zones if o.side > 0 and o.status != "broken" and o.top < min(zz.z, last)]
            zz.targets = sorted({_r(x) for x in ahead if x < min(zz.z, last) * 0.9995}, reverse=True)[:2]
            zz.behind = sorted({_r(H[i]) for i in sh if H[i] > zz.invalid})[:3]
            risk = zz.invalid - zz.z
        if zz.targets and risk > 0:
            zz.rr = abs(zz.targets[-1] - zz.z) / risk
    return zones


def current_setup(zones: List[Zone], side: int) -> Optional[Zone]:
    """The latest unbroken zone on the stock's side of the board."""
    for zz in reversed(zones):
        if zz.side == side and zz.status != "broken":
            return zz
    return None


def strength(bars, prev_close: float, atr: float) -> Dict:
    """The one-directional strength score and its parts."""
    if not bars or not prev_close or not atr or atr <= 0:
        return {"score": 0.0, "move_atr": None, "clean": None, "place": None, "change_pct": None}
    last = bars[-1].c
    path = abs(bars[0].c - bars[0].o) + sum(abs(b.c - a.c) for a, b in zip(bars, bars[1:]))
    clean = abs(last - bars[0].o) / path if path > 0 else 0.0
    hi, lo = max(b.h for b in bars), min(b.l for b in bars)
    move = (last - prev_close) / atr
    place = ((last - lo) if move >= 0 else (hi - last)) / (hi - lo) if hi > lo else 0.5
    # only the part of the move that today's session made in the same direction counts as clean
    score = 0.0 if clean < CHOPPY else move * clean * (0.5 + 0.5 * place)
    return {"score": round(score, 3), "move_atr": round(move, 2), "clean": round(clean, 2), "place": round(place, 2),
            "change_pct": round(100 * (last / prev_close - 1), 2)}


def event_risk(s: Dict) -> bool:
    return (s.get("move_atr") is not None and abs(s["move_atr"]) > EVENT_ATR) or \
           (s.get("change_pct") is not None and abs(s["change_pct"]) > EVENT_PCT)


def rank_board(rows: List[Dict], top: int = 20, picks_per_side: int = 3, sector_cap: int = 2) -> Dict:
    """rows: [{symbol, sector, score, event, ...}]. Adds side, rank, percentile and the pick rule."""
    scored = [r for r in rows if r.get("score") is not None]
    allabs = sorted(abs(r["score"]) for r in scored)
    for r in scored:
        r["side"] = 1 if r["score"] > 0 else -1 if r["score"] < 0 else 0
        below = sum(1 for x in allabs if x < abs(r["score"]))
        r["pctile"] = round(100 * below / max(1, len(allabs) - 1), 1) if len(allabs) > 1 else 100.0
    out = {}
    for side, name in ((1, "bullish"), (-1, "bearish")):
        group = sorted([r for r in scored if r["side"] == side], key=lambda r: -abs(r["score"]))
        for i, r in enumerate(group, 1):
            r["rank"], r["of"] = i, len(group)
        picks, per_sector = [], {}
        for r in group:
            if r.get("event"):
                r["rule"] = "not picked · event risk"
                continue
            if len(picks) >= picks_per_side:
                r["rule"] = f"not picked · rank {r['rank']}"
                continue
            sec = r.get("sector") or "Other"
            if per_sector.get(sec, 0) >= sector_cap:
                r["rule"] = "not picked · sector cap"
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
            r["rule"] = "pick"
            r["pick"] = True
            picks.append(r)
        out[name] = group[:top]
        out[name + "_picks"] = picks
    for r in scored:
        if r["side"] == 0:
            r["rule"] = "flat or choppy"
    return out
