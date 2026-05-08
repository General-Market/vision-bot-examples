#!/usr/bin/env bash
# Side-by-side live-trading race between two strategies.
#
# Usage:
#     ./race.sh <strategy_a> <strategy_b> [deposit]
#
# Example:
#     ./race.sh ensemble all_yes 0.1
#
# Requires two funded wallets: BOT_PRIVATE_KEY_A and BOT_PRIVATE_KEY_B
# in .env. Generate + print addresses with:
#     python main.py gen-keys
# Then fund both addresses with L3 USDC before running the race.

set -euo pipefail
cd "$(dirname "$0")"

A="${1:-ensemble}"
B="${2:-all_yes}"
DEP="${3:-0.1}"

source .env 2>/dev/null || true

if [ -z "${BOT_PRIVATE_KEY_A:-}" ] || [ -z "${BOT_PRIVATE_KEY_B:-}" ]; then
    echo "Need BOT_PRIVATE_KEY_A and BOT_PRIVATE_KEY_B in .env."
    echo "Run: python main.py gen-keys"
    exit 2
fi

mkdir -p logs

echo "[race] wallet A: $A  deposit=$DEP"
BOT_PRIVATE_KEY="$BOT_PRIVATE_KEY_A" \
    .venv/bin/python live_trader.py \
        --strategy "$A" --deposit "$DEP" --db "pnl-$A.db" \
        > "logs/trader-$A.log" 2>&1 &
PID_A=$!

echo "[race] wallet B: $B  deposit=$DEP"
BOT_PRIVATE_KEY="$BOT_PRIVATE_KEY_B" \
    .venv/bin/python live_trader.py \
        --strategy "$B" --deposit "$DEP" --db "pnl-$B.db" \
        > "logs/trader-$B.log" 2>&1 &
PID_B=$!

echo "[race] running — pids: $PID_A (A) $PID_B (B)"
echo "[race] tail logs:  tail -f logs/trader-$A.log  logs/trader-$B.log"
echo "[race] report pnl: python race_report.py pnl-$A.db pnl-$B.db"
echo "[race] stop:       kill $PID_A $PID_B"

trap "echo; echo '[race] stopping'; kill $PID_A $PID_B 2>/dev/null; wait; exit 0" INT TERM
wait
