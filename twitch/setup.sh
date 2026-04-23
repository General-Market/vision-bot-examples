#!/usr/bin/env bash
# One-shot bootstrap. Expected time: 3–5 minutes on a fresh machine.
#
# What it does:
#   1. Creates a Python 3.11+ venv and installs requirements.
#   2. Generates a BOT_PRIVATE_KEY into .env if one doesn't exist.
#   3. Trains a baseline XGBoost model on ~200 assets × 24 h of history.
#   4. Prints the funded-bot address and next steps.
#
# Idempotent: safe to re-run. Existing keys and models are preserved
# unless explicitly removed.

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
info "installing requirements"
.venv/bin/pip install -q -r requirements.txt

# pip sometimes claims "already satisfied" on a fresh venv when a transitive
# requirement chain finishes before top-level lines are visited. Explicitly
# install the ML stack to make sure it actually lands.
info "verifying ML stack"
.venv/bin/python -c "import pandas, numpy, xgboost, sklearn, anthropic" 2>/dev/null || {
    warn "ML stack missing after requirements.txt — installing explicitly"
    .venv/bin/pip install -q --no-cache-dir \
        "pandas>=2.1.0" "numpy>=1.24.0" "xgboost>=2.0.0" \
        "scikit-learn>=1.3.0" "anthropic>=0.40.0" "scipy>=1.11.0"
    .venv/bin/python -c "import pandas, numpy, xgboost, sklearn, anthropic" \
        || die "ML stack still broken after explicit install — debug manually"
}

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

if [ ! -f models/xgb.pkl ]; then
    info "training baseline XGBoost model (≈ 3 min)…"
    mkdir -p models
    # 500 × 72 h produces ~560 k training rows, converges with real edge:
    # lift_over_naive ≈ 0.5 pp, flip-catch ≈ 20%. For stronger edge, users
    # can re-run `main.py train-xgb --hours 168 --max-assets 500`.
    .venv/bin/python main.py train-xgb --hours 72 --max-assets 500 --out models/xgb.pkl
    info "model saved to models/xgb.pkl"
else
    info "existing model found at models/xgb.pkl — skipping training"
fi

info ""
info "${GRN}setup complete${NC}"
BOT_ADDR=$(.venv/bin/python -c "
import os
from dotenv import load_dotenv
load_dotenv('.env')
from eth_account import Account
key = os.getenv('BOT_PRIVATE_KEY')
if key: print(Account.from_key(key).address)
")
info ""
info "bot address : ${BOT_ADDR}"
info "next steps  :"
info "  1. Fund this address with L3 testnet USDC (≥ 0.5 USDC recommended)"
info "  2. Run:   .venv/bin/python main.py probe     (sanity check)"
info "  3. Run:   .venv/bin/python live_trader.py --deposit 0.1 --strategy ensemble"
info "  4. Watch: .venv/bin/python -c 'from pnl_logger import report; report(\"pnl.db\")'"
