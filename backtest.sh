#!/bin/bash
# Backtest the trading pipeline on recent NSE data (see backtesting/run.py --help)
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
exec venv/bin/python -m backtesting.run "$@"
