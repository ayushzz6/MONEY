"""Notifications: macOS desktop banners and, optionally, Telegram.

Telegram needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env and notify.telegram true.
Sending never blocks the engine: it runs in a short-lived thread and failures are logged.
"""
import json
import logging
import platform
import subprocess
import threading
import urllib.parse
import urllib.request

logger = logging.getLogger("notify")


class Notifier:
    def __init__(self, settings):
        self.settings = settings

    def send(self, title: str, text: str, important: bool = False):
        threading.Thread(target=self._send, args=(title, text, important), daemon=True).start()

    def _send(self, title, text, important):
        cfg = self.settings.get("notify", {}) or {}
        if cfg.get("desktop", True) and platform.system() == "Darwin":
            try:
                script = f"display notification {json.dumps(text[:240])} with title {json.dumps('MONEYV2 ' + title)}"
                if important:
                    script += ' sound name "Glass"'
                subprocess.run(["osascript", "-e", script], timeout=5, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.debug("desktop notification failed: %s", e)
        if cfg.get("telegram"):
            token, chat = self.settings.env("TELEGRAM_BOT_TOKEN"), self.settings.env("TELEGRAM_CHAT_ID")
            if token and chat:
                try:
                    body = urllib.parse.urlencode({"chat_id": chat, "text": f"{title}\n{text}"}).encode()
                    urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", body, timeout=10).read()
                except Exception as e:
                    logger.warning("telegram failed: %s", e)
