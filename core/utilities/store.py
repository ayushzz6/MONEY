"""SQLite journal: every signal, trade and engine event, so a restart can pick the day up and the
dashboard can show history. One file: data/moneyv2.db."""
import json
import sqlite3
import threading
import time
from typing import List, Optional

from core.config.settings import DATA_DIR

DB_PATH = DATA_DIR / "moneyv2.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, date TEXT, symbol TEXT, strategy TEXT, mode TEXT,
    status TEXT, net_pnl REAL, updated REAL, body TEXT);
CREATE INDEX IF NOT EXISTS trades_date ON trades(date);
CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, date TEXT, symbol TEXT,
    strategy TEXT, side TEXT, decision TEXT, reason TEXT, body TEXT);
CREATE INDEX IF NOT EXISTS signals_date ON signals(date);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, date TEXT, kind TEXT,
    symbol TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS events_date ON events(date);
"""


class Journal:
    def __init__(self, path=DB_PATH):
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def save_trade(self, t: dict):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?)",
                            (t["id"], t["date"], t["symbol"], t["strategy"], t["mode"], t["status"],
                             t.get("net_pnl"), time.time(), json.dumps(t, default=str)))
            self.db.commit()

    def trades(self, date: Optional[str] = None, since: Optional[str] = None, mode: Optional[str] = None) -> List[dict]:
        q, args = "SELECT body FROM trades WHERE 1=1", []
        if date:
            q += " AND date = ?"
            args.append(date)
        if since:
            q += " AND date >= ?"
            args.append(since)
        if mode:
            q += " AND mode = ?"
            args.append(mode)
        with self._lock:
            rows = self.db.execute(q + " ORDER BY date, updated", args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_signal(self, date: str, row: dict):
        with self._lock:
            self.db.execute("INSERT INTO signals (ts, date, symbol, strategy, side, decision, reason, body) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (time.time(), date, row.get("symbol"), row.get("strategy"), row.get("side"),
                             row.get("decision"), row.get("reason"), json.dumps(row, default=str)))
            self.db.commit()

    def signals(self, date: str, limit: int = 300) -> List[dict]:
        with self._lock:
            rows = self.db.execute("SELECT body FROM signals WHERE date = ? ORDER BY id DESC LIMIT ?",
                                   (date, limit)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def event(self, date: str, kind: str, text: str, symbol: Optional[str] = None):
        with self._lock:
            self.db.execute("INSERT INTO events (ts, date, kind, symbol, text) VALUES (?,?,?,?,?)",
                            (time.time(), date, kind, symbol, text))
            self.db.commit()

    def events(self, date: str, limit: int = 200) -> List[dict]:
        with self._lock:
            rows = self.db.execute("SELECT ts, kind, symbol, text FROM events WHERE date = ? ORDER BY id DESC LIMIT ?",
                                   (date, limit)).fetchall()
        return [{"ts": r[0], "kind": r[1], "symbol": r[2], "text": r[3]} for r in rows]
