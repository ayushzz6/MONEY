"""Kill switch (plan.md section 26): global, per strategy and per symbol. Persisted, so a
restart does not silently re-arm trading. A tripped global switch blocks every new entry;
exits are never blocked."""
import json
import threading
import time
from typing import Optional

from core.config.settings import STATE_DIR

PATH = STATE_DIR / "kill_switch.json"


class KillSwitch:
    def __init__(self, path=PATH):
        self.path = path
        self._lock = threading.Lock()
        self.state = {"global": None, "strategies": {}, "symbols": {}}
        if self.path.exists():
            try:
                self.state.update(json.loads(self.path.read_text()))
            except ValueError:
                pass

    def _save(self):
        self.path.write_text(json.dumps(self.state, indent=2))

    def trip(self, reason: str, by: str = "system"):
        with self._lock:
            self.state["global"] = {"reason": reason, "by": by, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._save()

    def reset(self):
        with self._lock:
            self.state["global"] = None
            self._save()

    def disable_strategy(self, name: str, reason: str):
        with self._lock:
            self.state["strategies"][name] = reason
            self._save()

    def enable_strategy(self, name: str):
        with self._lock:
            self.state["strategies"].pop(name, None)
            self._save()

    def block_symbol(self, symbol: str, reason: str):
        with self._lock:
            self.state["symbols"][symbol] = reason
            self._save()

    def unblock_symbol(self, symbol: str):
        with self._lock:
            self.state["symbols"].pop(symbol, None)
            self._save()

    @property
    def active(self) -> bool:
        return self.state.get("global") is not None

    def blocked(self, strategy: Optional[str] = None, symbol: Optional[str] = None) -> Optional[str]:
        g = self.state.get("global")
        if g:
            return f"kill switch: {g['reason']}"
        if strategy and strategy in self.state["strategies"]:
            return f"strategy {strategy} disabled: {self.state['strategies'][strategy]}"
        if symbol and symbol in self.state["symbols"]:
            return f"{symbol} blocked: {self.state['symbols'][symbol]}"
        return None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.state))
