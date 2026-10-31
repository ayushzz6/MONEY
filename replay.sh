#!/bin/bash
# Replay a past NSE session through the live engine components (see backtesting/replay.py --help)
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
exec venv/bin/python -m backtesting.replay "$@"
