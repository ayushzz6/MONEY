"""Loads configs/trading.json and the environment into one Settings object.

The JSON file holds the trading rules (risk limits, strategy parameters).
The environment (.env) holds secrets and the TRADING_MODE override.
"""
import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

logger = logging.getLogger("Settings")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "trading.json"
DATA_DIR = ROOT / "data"
STATE_DIR = DATA_DIR / "state"
CACHE_DIR = DATA_DIR / "cache"

VALID_MODES = ("PAPER", "LIVE")


class Settings:
    """Thread-safe view of the trading config. Values are read with get('a.b.c')."""

    def __init__(self, path: Path = CONFIG_PATH):
        load_dotenv(ROOT / ".env")
        self.path = path
        self._lock = threading.Lock()
        self._raw: Dict[str, Any] = {}
        self.reload()

    def reload(self):
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        env_mode = os.getenv("TRADING_MODE")
        if env_mode:
            raw["mode"] = env_mode.strip().strip('"').upper()
        mode = str(raw.get("mode", "PAPER")).upper()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        raw["mode"] = mode
        env_provider = os.getenv("MONEYV2_DATA_PROVIDER")      # auto | angel | yahoo, overrides data.provider
        if env_provider:
            raw.setdefault("data", {})["provider"] = env_provider.strip().lower()
        with self._lock:
            self._raw = raw
        for d in (DATA_DIR, STATE_DIR, CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def mode(self) -> str:
        return self._raw["mode"]

    @property
    def live(self) -> bool:
        return self.mode == "LIVE"

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def strategy(self, name: str) -> Dict[str, Any]:
        return dict(self.get(f"strategies.{name}", {}) or {})

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._raw)

    @staticmethod
    def env(name: str, default: str = None) -> str:
        val = os.getenv(name, default)
        return val.strip().strip('"') if isinstance(val, str) else val

    def angel_credentials(self):
        keys = ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET")
        vals = [self.env(k) for k in keys]
        return vals if all(vals) else None
