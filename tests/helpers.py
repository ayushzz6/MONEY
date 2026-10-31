"""Synthetic NSE sessions for tests."""
import datetime as dt
import json
from pathlib import Path

from market.calendar import IST
from market.candles.bars import Bar


def session_start(date: str) -> int:
    d = dt.date.fromisoformat(date)
    return int(dt.datetime.combine(d, dt.time(9, 15), IST).timestamp())


def make_day(date: str, rows, volume: float = 1000.0):
    """rows: [(o, h, l, c) or (o, h, l, c, v)] one per 5-minute bar from 09:15."""
    t0 = session_start(date)
    out = []
    for i, r in enumerate(rows):
        o, h, l, c = r[:4]
        v = r[4] if len(r) > 4 else volume
        out.append(Bar(t0 + 300 * i, o, h, l, c, v))
    return out


def flat_day(date: str, price: float = 100.0, n: int = 75, wiggle: float = 0.2):
    return make_day(date, [(price, price + wiggle, price - wiggle, price)] * n)


def ist(date: str, hhmm: str) -> dt.datetime:
    h, m = map(int, hhmm.split(":"))
    return dt.datetime.combine(dt.date.fromisoformat(date), dt.time(h, m), IST)


class FakeSettings:
    """Settings stand-in: the real config file, with overrides."""

    def __init__(self, live=False, **over):
        from core.config.settings import CONFIG_PATH
        self._raw = json.loads(Path(CONFIG_PATH).read_text())
        self._raw.update(over)
        self._raw["mode"] = "LIVE" if live else "PAPER"

    @property
    def mode(self):
        return self._raw["mode"]

    @property
    def live(self):
        return self.mode == "LIVE"

    def get(self, dotted, default=None):
        node = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def strategy(self, name):
        return dict(self.get(f"strategies.{name}", {}) or {})

    @staticmethod
    def env(name, default=None):
        return default
