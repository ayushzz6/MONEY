import logging

from brokers.angelone.auth import AngelOneAuth
from brokers.angelone.market_data import AngelOneMarketData

logger = logging.getLogger(__name__)


class AngelOneAdapter:
    """Login once; hands out the REST client (orders) and the WebSocket feed (quotes)."""

    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: str):
        self.auth = AngelOneAuth(api_key, client_id, password, totp_secret)
        self.market_data = None
        self.session = None

    @property
    def smart_api(self):
        return self.auth.smart_api

    def start(self, connect_feed: bool = True):
        try:
            self.session = self.auth.login()
            if connect_feed:
                self.market_data = AngelOneMarketData(
                    auth_token=self.session['jwtToken'],
                    api_key=self.auth.api_key,
                    client_code=self.auth.client_id,
                    feed_token=self.session['feedToken'],
                )
                self.market_data.connect()
            return self
        except Exception as e:
            logger.error(f"Failed to start Angel One Adapter: {e}")
            raise

    def reconnect_feed(self):
        """Log in again (fresh tokens for the socket and the REST client) and open a new socket."""
        self.session = self.auth.login()
        self.market_data.reconnect(self.session['jwtToken'], self.session['feedToken'])
