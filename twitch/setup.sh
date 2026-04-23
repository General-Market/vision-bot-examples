#!/usr/bin/env bash
# One-shot bootstrap. Expected time: ≈ 2 minutes on a fresh machine.
#
# What it does:
#   1. Creates a Python 3.11+ venv and installs requirements.
#   2. Generates BOT_PRIVATE_KEY into .env if one doesn't exist.
#   3. Optionally funds the wallet from the L3 testnet faucet (--auto-fund).
#   4. Prints the bot address and next steps.
#
# Flags:
#   --auto-fund    After key gen, seed the wallet with 0.01 ETH native +
#                  1 USDC via the public L3 testnet faucet. Lets a fresh
#                  agent go clone → trading-ready without manual funding.
#
# Model training is NOT part of setup — it is an extra ~3 min step for
# users who want the `xgb` / `ensemble` strategies:
#
#     .venv/bin/python main.py train-xgb --hours 72 --max-assets 500 --out models/xgb.pkl
#
# Out-of-the-box strategies (momentum / rolling / contrarian / all_yes)
# run without a trained model.
#
# Idempotent: safe to re-run. Existing keys are preserved.

AUTO_FUND=0
INSTALL_ML=0
for arg in "$@"; do
    case "$arg" in
        --auto-fund) AUTO_FUND=1 ;;
        --with-ml)   INSTALL_ML=1 ;;
        -h|--help)
            sed -n '/^#!/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

set -euo pipefail
cd "$(dirname "$0")"

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GRN}[setup]${NC} $*"; }
warn()  { echo -e "${YEL}[setup]${NC} $*"; }
die()   { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 not found"

# Detect a 3.11+ interpreter. macOS default 3.9 won't work (PEP-604 unions).
PYBIN=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])' 2>/dev/null)
        if [ "$ver" -ge 311 ]; then PYBIN="$candidate"; break; fi
    fi
done
[ -n "$PYBIN" ] || die "need Python 3.11+ (Homebrew: brew install python@3.14)"
info "using $PYBIN ($($PYBIN --version))"

if [ ! -d .venv ]; then
    info "creating venv"
    "$PYBIN" -m venv .venv
fi

.venv/bin/pip install -q --upgrade pip
info "installing core requirements (~30 s)"
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import web3, eth_account, requests, dotenv" \
    || die "core deps failed to install"

if [ "$INSTALL_ML" = "1" ]; then
    info "installing ML stack (~90 s) — rolling / xgb / ensemble / claude"
    .venv/bin/pip install -q -r requirements-ml.txt
    .venv/bin/python -c "import pandas, numpy, xgboost, sklearn, anthropic" \
        || die "ML stack failed — retry: pip install -r requirements-ml.txt"
else
    info "skipping ML stack — momentum / all_yes / all_no / contrarian run on core alone"
    info "                    (later: pip install -r requirements-ml.txt for xgb/ensemble)"
fi

# macOS libomp reminder (xgboost)
if [ "$(uname)" = "Darwin" ] && ! ls /opt/homebrew/Cellar/libomp 2>/dev/null | head -1 >/dev/null 2>&1; then
    warn "macOS: xgboost needs libomp → run: brew install libomp"
fi

if [ ! -f .env ]; then
    info "creating .env from .env.example"
    cp .env.example .env
fi

# Generate BOT_PRIVATE_KEY if missing.
if ! grep -q '^BOT_PRIVATE_KEY=0x[0-9a-fA-F]\{64\}' .env; then
    info "generating bot private key"
    KEYLINE=$(.venv/bin/python -c 'from eth_account import Account; a=Account.create(); print(f"{a.key.hex()}\t{a.address}")')
    KEY=$(echo "$KEYLINE" | cut -f1)
    ADDR=$(echo "$KEYLINE" | cut -f2)
    # Prepend 0x if missing
    case "$KEY" in 0x*) ;; *) KEY="0x$KEY" ;; esac
    # Replace the line — portable sed for macOS and Linux.
    if grep -q '^BOT_PRIVATE_KEY=' .env; then
        python3 -c "
import re, sys
p='.env'
s=open(p).read()
s=re.sub(r'^BOT_PRIVATE_KEY=.*$', 'BOT_PRIVATE_KEY=$KEY', s, flags=re.M)
open(p,'w').write(s)
"
    else
        printf '\nBOT_PRIVATE_KEY=%s\n' "$KEY" >> .env
    fi
    info "bot address: $ADDR"
    info "→ fund with at least 0.5 L3 USDC before running live_trader.py"
fi

BOT_ADDR=$(.venv/bin/python -c "
import os
from dotenv import load_dotenv
load_dotenv('.env')
from eth_account import Account
key = os.getenv('BOT_PRIVATE_KEY')
if key: print(Account.from_key(key).address)
")

if [ "$AUTO_FUND" = "1" ] && [ -n "$BOT_ADDR" ]; then
    info "auto-funding ${BOT_ADDR} from the L3 testnet faucet"
    .venv/bin/python main.py faucet --to "$BOT_ADDR" --usdc 1.0 --eth 0.01 || \
        warn "faucet failed — fund manually before running live_trader.py"
fi

info ""
info "${GRN}setup complete${NC}"
info ""
echo -e "${GRN}"
cat <<EOF
╭──────────────────────────────────────────────────────────────────╮
│                        YOUR BOT WALLET                            │
│  ${BOT_ADDR}  │
EOF
if [ "$AUTO_FUND" = "1" ]; then
    echo -e "│              ${GRN}funded via the L3 testnet faucet${NC}                    │"
fi
cat <<EOF
╰──────────────────────────────────────────────────────────────────╯
${NC}
EOF
info "next steps:"
if [ "$AUTO_FUND" != "1" ]; then
    info "  fund:        .venv/bin/python main.py faucet --to ${BOT_ADDR} --usdc 1"
    info "  probe:       .venv/bin/python main.py probe"
    info "  trade:       .venv/bin/python main.py trade --strategy momentum --deposit 0.1"
else
    info "  probe:       .venv/bin/python main.py probe"
    info "  trade:       .venv/bin/python main.py trade --strategy momentum --deposit 0.1"
    info "  watch pnl:   .venv/bin/python -c 'from pnl_logger import report; report(\"pnl.db\")'"
fi
info "  ml add-on:   pip install -r requirements-ml.txt  (unlocks rolling / xgb / ensemble / claude)"
info "  train model: .venv/bin/python main.py train-xgb --hours 72 --max-assets 500 --out models/xgb.pkl  (needs ml add-on)"
