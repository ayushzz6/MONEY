#!/bin/bash
# 15-strategy, 5-year NSE research backtest (see research/run.py --help)
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
exec venv/bin/python -m research.run "$@"
