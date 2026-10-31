#!/bin/bash
# Start the MONEYV2 trading engine and its dashboard (http://127.0.0.1:8050)
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
exec venv/bin/python -m apps.trading_engine.main "$@"
