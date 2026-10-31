"""Angel One feed: real connection state, reconnects, and the market-hours watchdog."""
import json

import pytest

from helpers import ist


class FakeApp:
    """Stands in for websocket.WebSocketApp: keeps the callbacks SmartApi hands it."""
    made = []

    def __init__(self, url, header=None, **callbacks):
        self.cb = callbacks
        self.sent = []
        FakeApp.made.append(self)

    def run_forever(self, **kw):
        pass

    def close(self):
        pass

    def send(self, data):
        self.sent.append(json.loads(data))


@pytest.fixture
def feed(monkeypatch, tmp_path):
    import SmartApi.smartWebSocketV2 as sdk
    from brokers.angelone.market_data import AngelOneMarketData
    monkeypatch.chdir(tmp_path)                  # SmartApi writes logs/<date>/app.log in the cwd
    monkeypatch.setattr(sdk.websocket, "WebSocketApp", FakeApp)
    FakeApp.made = []
    md = AngelOneMarketData("jwt", "key", "client", "feed")
    quotes = []
    md.set_callback(lambda *a: quotes.append(a))
    md.subscribe({"RELIANCE": "2885"})
    return md, quotes


def _start(md):
    md.connect()
    md._thread.join(2)
    return FakeApp.made[-1]


def _tokens(app):
    return [t for m in app.sent for tl in m["params"]["tokenList"] for t in tl["tokens"]]


def test_close_from_websocket_client_marks_the_feed_down(feed):
    md, _ = feed
    app = _start(md)
    app.cb["on_open"](app)
    assert md.connected and _tokens(app) == ["2885"]
    # websocket-client 1.8 calls on_close(wsapp, status_code, reason); SmartApi's own
    # handler takes only wsapp and used to raise, leaving the feed "connected"
    app.cb["on_close"](app, 1000, "bye")
    assert not md.connected


def test_sdk_giving_up_is_noticed(feed):
    md, _ = feed
    app = _start(md)
    app.cb["on_open"](app)
    md.sws.on_error("Max retry attempt reached", "Connection closed")
    assert md.gave_up and not md.connected


def test_reconnect_opens_a_new_socket_and_ignores_the_old_one(feed):
    md, quotes = feed
    old = _start(md)
    old.cb["on_open"](old)
    old_sws = md.sws
    md.subscribe({"TCS": "11536"})
    md.reconnect("jwt2", "feed2")
    md._thread.join(2)
    new = FakeApp.made[-1]
    assert md.sws is not old_sws and new is not old and md.reconnects == 1
    assert not md.connected
    old.cb["on_open"](old)                       # a late callback from the replaced socket
    assert not md.connected
    new.cb["on_open"](new)
    assert md.connected and sorted(_tokens(new)) == ["11536", "2885"]
    tick = {"token": "2885", "last_traded_price": 250000, "exchange_timestamp": 1_790_000_000_000,
            "volume_trade_for_the_day": 10, "closed_price": 249000}
    old_sws.on_data(old, tick)                   # ignored
    md.sws.on_data(new, tick)
    assert quotes == [("RELIANCE", 2500.0, 1_790_000_000.0, 10.0, 2490.0)]


class StubFeed:
    def __init__(self, connected=True, silent=5.0, tokens=True, gave_up=False):
        self.connected, self._silent, self.gave_up = connected, silent, gave_up
        self.token_to_symbol = {"2885": "RELIANCE"} if tokens else {}
        self.opened_at = None
        self.reconnects = 0

    def silent_for(self, now=None):
        return self._silent


class StubAdapter:
    def __init__(self, md):
        self.market_data = md
        self.calls = 0

    def reconnect_feed(self):
        self.calls += 1


def _manager(md):
    from apps.market_data_service.main import FeedManager
    from helpers import FakeSettings
    fm = FeedManager(FakeSettings(), None, None)
    fm.mode, fm.adapter = "angel", StubAdapter(md)
    return fm


@pytest.mark.parametrize("when, md, expect", [
    ("2026-10-26 10:00", dict(connected=False, silent=400), 1),     # down in market hours
    ("2026-10-26 09:05", dict(connected=False, silent=400), 1),     # down before the bell
    ("2026-10-26 10:00", dict(connected=False, silent=10), 0),      # still opening
    ("2026-10-26 10:00", dict(connected=False, silent=10, gave_up=True), 1),
    ("2026-10-26 10:00", dict(silent=200), 1),                      # connected but silent
    ("2026-10-26 10:00", dict(silent=200, tokens=False), 0),        # nothing subscribed yet
    ("2026-10-26 10:00", dict(silent=5), 0),                        # healthy
    ("2026-10-26 09:15", dict(silent=600), 0),                      # pre-open lull, not dead
    ("2026-10-26 08:30", dict(connected=False, silent=9999), 0),    # too early
    ("2026-10-26 16:00", dict(connected=False, silent=9999), 0),    # after the close
    ("2026-10-25 10:00", dict(connected=False, silent=9999), 0),    # Sunday
])
def test_watchdog_reconnects_only_when_it_should(when, md, expect):
    fm = _manager(StubFeed(**md))
    fm._watch_angel(ist(*when.split()))
    assert fm.adapter.calls == expect


def test_watchdog_backs_off_between_attempts():
    fm = _manager(StubFeed(connected=False, silent=400))
    now = ist("2026-10-26", "10:00")
    fm._watch_angel(now)
    fm._watch_angel(now)
    assert fm.adapter.calls == 1                 # no second login straight away
    fm._last_reconnect -= 61                     # 61 s later: the 2nd wait is 60 s
    fm._watch_angel(now)
    assert fm.adapter.calls == 2
    fm._last_reconnect -= 61                     # the 3rd wait is 120 s
    fm._watch_angel(now)
    assert fm.adapter.calls == 2


def test_smartapi_failed_request_logs_are_redacted(tmp_path, monkeypatch):
    import logging
    import logzero
    import brokers.angelone  # noqa: F401  (installs the filter)
    from logzero import logger as sdk_logger     # the logger SmartApi writes to
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "app.log"
    logzero.logfile(str(log), loglevel=logging.INFO)
    try:
        sdk_logger.error("Error occurred while making a POST request to https://x. Headers: "
                         "{'X-PrivateKey': 'APIKEY123', 'Authorization': 'Bearer eyJabc.def'}, "
                         "Request: {\"clientcode\": \"A1\", \"password\": \"1234\", \"totp\": \"987654\"}")
    finally:
        logzero.logfile(None)
    text = log.read_text()
    assert "Error occurred" in text and "clientcode" in text
    for secret in ("APIKEY123", "eyJabc", "1234", "987654"):
        assert secret not in text
