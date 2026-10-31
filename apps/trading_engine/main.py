"""Start the trading engine and its dashboard: ./run.sh, then open http://127.0.0.1:8050"""
import argparse
import logging
import os
import sys

import uvicorn

from apps.api_service.main import create_app
from apps.trading_engine.engine import Engine
from core.config.settings import DATA_DIR, Settings


def main(argv=None):
    ap = argparse.ArgumentParser(description="MONEYV2 NSE trading engine + dashboard")
    ap.add_argument("--port", type=int, default=int(os.getenv("MONEYV2_PORT", "8050")))
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: the API can approve and exit trades")
    a = ap.parse_args(argv)

    logs = DATA_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs / "engine.log")],
    )
    for noisy in ("urllib3", "uvicorn.access", "SmartApi"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings()
    log = logging.getLogger("TradingEngine")
    log.info("Starting MONEYV2 in %s mode", settings.mode)
    if settings.live:
        log.warning("LIVE MODE: real orders through Angel One. Every entry waits for your Approve.")
    engine = Engine(settings)
    engine.start()
    log.info("Dashboard: http://%s:%d", a.host, a.port)
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
