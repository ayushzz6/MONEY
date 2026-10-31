import logging

import pyotp
from SmartApi import SmartConnect

logger = logging.getLogger(__name__)


class AngelOneAuth:
    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: str):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret
        self.smart_api = SmartConnect(api_key=self.api_key)
        self.session = None

    def login(self):
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self.smart_api.generateSession(self.client_id, self.password, totp)
            if data and data.get('status'):
                self.session = data['data']
                self.session['feedToken'] = self.session.get('feedToken') or self.smart_api.getfeedToken()
                logger.info("Successfully logged into Angel One.")
                return self.session
            logger.error(f"Login failed: {(data or {}).get('message')}")
            raise Exception(f"Angel One login failed: {(data or {}).get('message')}")
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            raise
