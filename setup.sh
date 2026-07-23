#!/usr/bin/env bash
# Repo-root bootstrap. Creates .venv, installs one bot's dependencies, and
# seeds its .env from .env.example. Idempotent — re-run it freely.
#
# When no source is specified, defaults to `twitch`.
#
#   ./setup.sh                     # = ./setup.sh --source twitch
#   ./setup.sh --source polymarket
#   SOURCE=polymarket ./setup.sh   # env-var form, same result
#
# Requires Python 3.11+. macOS system Python 3.9 will not work — install
# Homebrew's python@3.14 and re-run.

set -euo pipefail
cd "$(dirname "$0")"

SOURCE="${SOURCE:-twitch}"

while [ $# -gt 0 ]; do
    case "$1" in
        --source)
            SOURCE="$2"; shift 2 ;;
        --source=*)
            SOURCE="${1#--source=}"; shift ;;
        -h|--help)
            sed -n '/^#!/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "error: unknown argument '$1'" >&2
            exit 1 ;;
    esac
done

if [ ! -f "$SOURCE/requirements.txt" ]; then
    echo "error: no bot directory for source '$SOURCE'" >&2
    echo "Available sources:" >&2
    for d in */; do
        d="${d%/}"
        [ -f "$d/requirements.txt" ] && echo "  - $d" >&2
    done
    exit 1
fi

# ── Interpreter ────────────────────────────────────────────────────
PY=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "error: need Python 3.11+; none found on PATH" >&2
    echo "  macOS: brew install python@3.14" >&2
    exit 1
fi

echo "[bootstrap] source=$SOURCE  python=$("$PY" --version)"

# ── Virtualenv + dependencies ──────────────────────────────────────
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r "$SOURCE/requirements.txt"

# xgboost needs OpenMP on macOS. Warn rather than fail — the pure-feature
# paths still run without it.
if [ "$(uname -s)" = "Darwin" ] && ! .venv/bin/python -c 'import xgboost' >/dev/null 2>&1; then
    echo "[bootstrap] warning: xgboost failed to load — run 'brew install libomp'" >&2
fi

# ── Local env file ─────────────────────────────────────────────────
if [ -f "$SOURCE/.env" ]; then
    echo "[bootstrap] $SOURCE/.env already exists — left untouched"
else
    cp "$SOURCE/.env.example" "$SOURCE/.env"
    echo "[bootstrap] wrote $SOURCE/.env from .env.example — fill in your keys"
fi

cat <<EOF

[bootstrap] done.

  cd $SOURCE
  ../.venv/bin/python -m pytest tests/ -v      # smoke test, no creds needed
  ../.venv/bin/python main.py markets          # list live Vision markets
  ../.venv/bin/python main.py pipeline         # load -> features -> train -> predict

Leave BOT_PRIVATE_KEY blank in $SOURCE/.env unless you want the bot to sign
real testnet transactions.
EOF
