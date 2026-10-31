"""Angel One SmartAPI integration.

SmartApi logs every failed REST call with its headers and body (API key, session token,
and on a failed login the PIN and one-time code) to logzero's logger, which writes to
logs/<date>/app.log and stderr. The filter below masks those values before any handler
sees them. It is installed on import, so every user of this package gets it.
"""
import logging
import re

_SECRET_KEYS = ("X-PrivateKey", "Authorization", "x-api-key", "x-feed-token", "password", "totp",
                "jwtToken", "refreshToken", "feedToken", "X-ClientPublicIP", "X-ClientLocalIP", "X-MACAddress")
_PATTERN = re.compile(r"""(['"](?:%s)['"]\s*:\s*['"])[^'"]*""" % "|".join(map(re.escape, _SECRET_KEYS)),
                      re.IGNORECASE)


def redact(text: str) -> str:
    return _PATTERN.sub(r"\1***", text)


class RedactSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, ()
        return True


def _install():
    names = ["logzero_default", "SmartApi", "SmartApi.smartConnect", "SmartApi.smartWebSocketV2"]
    try:
        import logzero
        names.append(logzero.logger.name)
    except Exception:
        pass
    for name in dict.fromkeys(names):
        lg = logging.getLogger(name)
        if not any(isinstance(f, RedactSecrets) for f in lg.filters):
            lg.addFilter(RedactSecrets())


_install()
