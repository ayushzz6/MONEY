"""Market-neutral sector-pair divergence (intraday, NIFTY 200, MIS).

Pairs: two stocks of the same sector whose daily returns were the most correlated over the TRAIN period
only (to 2024-12-31), at most 3 pairs a sector, 40 pairs. Each session, the spread of their returns since
the open is compared with how wide that spread usually is at that time of day (standard deviation over the
previous 40 sessions). When it is z >= `z_in` (10:00-14:00) buy the laggard and short the leader in equal
rupee amounts; out when the spread closes (z back to `z_out`), at z >= `z_stop`, or at 15:15. Each leg
Rs 1.25 lakh (two pairs fill the Rs 5 lakh MIS allows on Rs 1 lakh), 3 bps slippage a side, NSE intraday charges.
"""
import itertools
from collections import defaultdict

import numpy as np

from portfolio.pnl import order_charges
from research import data
from research.sim import FLAT_SLOT, slot_of_close

TRAIN_END = np.datetime64("2024-12-31")
LEG = 125000.0
SLIP = 3 / 10000.0


def pick_pairs(mats, sectors, n_pairs=40, per_sector=3):
    by_sec = defaultdict(list)
    for s, m in mats.items():
        if sectors.get(s):
            by_sec[sectors[s]].append(s)
    cands = []
    for sec, syms in by_sec.items():
        rets = {}
        for s in syms:
            m = mats[s]
            c = m.c[:, -1]
            d = m.dates
            keep = d <= TRAIN_END
            rets[s] = dict(zip(d[keep][1:], (c[keep][1:] / c[keep][:-1] - 1)))
        sec_c = []
        for a, b in itertools.combinations(syms, 2):
            common = sorted(set(rets[a]) & set(rets[b]))
            if len(common) < 300:
                continue
            x = np.array([rets[a][d] for d in common])
            y = np.array([rets[b][d] for d in common])
            sec_c.append((np.corrcoef(x, y)[0, 1], a, b, sec))
        sec_c.sort(reverse=True)
        cands += sec_c[:per_sector]
    cands.sort(reverse=True)
    return [(a, b, sec, round(c, 2)) for c, a, b, sec in cands[:n_pairs] if c >= 0.5]


def pair_trades(ma, mb, z_in, z_out, z_stop, costs):
    common = sorted(set(ma.dates) & set(mb.dates))
    ia = {d: i for i, d in enumerate(ma.dates)}
    ib = {d: i for i, d in enumerate(mb.dates)}
    ra = np.array([ma.c[ia[d]] / ma.o[ia[d], 0] - 1 for d in common])
    rb = np.array([mb.c[ib[d]] / mb.o[ib[d], 0] - 1 for d in common])
    spread = ra - rb
    trades = []
    s0, s1 = slot_of_close("10:00"), slot_of_close("14:00")
    for k in range(40, len(common)):
        sd = spread[k - 40:k].std(0)
        if not np.all(np.isfinite(sd[s0:FLAT_SLOT + 1])) or (sd[s0:s1 + 1] <= 0).any():
            continue
        z = spread[k] / np.where(sd > 0, sd, np.nan)
        j = next((t for t in range(s0, s1 + 1) if abs(z[t]) >= z_in), None)
        if j is None:
            continue
        sign = -1 if z[j] > 0 else 1          # z > 0: A ran ahead of B -> short A, buy B
        x = FLAT_SLOT
        for t in range(j + 1, FLAT_SLOT + 1):
            if (z[j] > 0 and z[t] <= z_out) or (z[j] < 0 and z[t] >= -z_out) or abs(z[t]) >= z_stop:
                x = t
                break
        d = common[k]
        pa_in, pa_out = ma.c[ia[d], j], ma.c[ia[d], x]
        pb_in, pb_out = mb.c[ib[d], j], mb.c[ib[d], x]
        qa, qb = int(LEG // pa_in), int(LEG // pb_in)
        if qa < 1 or qb < 1:
            continue
        # leg A: side `sign`, leg B: the opposite
        fa_in, fa_out = pa_in * (1 + SLIP * sign), pa_out * (1 - SLIP * sign)
        fb_in, fb_out = pb_in * (1 - SLIP * sign), pb_out * (1 + SLIP * sign)
        gross = sign * (fa_out - fa_in) * qa - sign * (fb_out - fb_in) * qb
        ch = (order_charges(fa_in * qa, "buy" if sign > 0 else "sell", costs) + order_charges(fa_out * qa, "sell" if sign > 0 else "buy", costs)
              + order_charges(fb_in * qb, "sell" if sign > 0 else "buy", costs) + order_charges(fb_out * qb, "buy" if sign > 0 else "sell", costs))
        trades.append({"day": d, "slot": j, "exit_slot": x, "net": gross - ch, "gross": gross, "charges": ch})
    return trades


def run_pairs(mats, sectors, costs, log=print):
    pairs = pick_pairs(mats, sectors)
    log(f"  pairs: {len(pairs)} picked on 2021-24 correlation (top: {', '.join(f'{a}/{b}' for a, b, _, _ in pairs[:6])})")
    rows, daily_by = [], {}
    dates = sorted({d for m in mats.values() for d in m.dates})
    pos = {d: i for i, d in enumerate(dates)}
    month = np.array([str(d)[:7] for d in dates])
    train = np.array(dates) <= TRAIN_END
    for z_in, z_out, z_stop in itertools.product((2.0, 2.5, 3.0), (0.0, 0.5), (4.0, 5.0)):
        trades = []
        for a, b, sec, c in pairs:
            trades += pair_trades(mats[a], mats[b], z_in, z_out, z_stop, costs)
        # at most 2 pairs open at once: take each day's earliest two signals
        by_day = defaultdict(list)
        for t in trades:
            by_day[t["day"]].append(t)
        daily = np.zeros(len(dates))
        n = 0
        for d, ts in by_day.items():
            for t in sorted(ts, key=lambda t: t["slot"])[:2]:
                daily[pos[d]] += t["net"]
                n += 1
        r = {"family": "pairs", "setup": f"Sector-pair divergence z {z_in:g} in / {z_out:g} out / {z_stop:g} stop", "trades": n}
        for nm, m in (("train", train), ("test", ~train)):
            q = daily[m]
            eq = np.cumsum(q)
            r[f"{nm}_per_month"] = round(float(q.sum()) / len(set(month[m])), 0)
            r[f"{nm}_maxdd"] = round(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()), 0)
        rows.append(r)
        daily_by[r["setup"]] = daily
    return rows, daily_by, np.array(dates), pairs
