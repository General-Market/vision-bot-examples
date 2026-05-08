#!/usr/bin/env bash
# Pulls fresh vision-bot data from VPS 1 (where index_prices Postgres lives)
# and lands it in ./public/ for the Vite app to serve.
#
# Override defaults via env:
#   VISION_BOT_PLAYER=0x...   — bot wallet address
#   VISION_BOT_LIMIT=50       — max recent batches to consider
#   VISION_BOT_DAYS=7         — price-history window
#   VISION_BOT_MAX_ASSETS=120 — cap on per-asset history files
#   VISION_BOT_SSH=...        — ssh host alias (default: index-maker/prod/be)
set -euo pipefail

PLAYER="${VISION_BOT_PLAYER:-0x9d757d97d7a5fa8a20a70e1ac301887558f1ea3d}"
LIMIT="${VISION_BOT_LIMIT:-50}"
DAYS="${VISION_BOT_DAYS:-7}"
MAX_ASSETS="${VISION_BOT_MAX_ASSETS:-120}"
# rsync chokes on slash-bearing aliases like "index-maker/prod/be";
# use the slash-free alias by default.
SSH_HOST="${VISION_BOT_SSH:-vps1-new}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="$HERE/public"
REMOTE_OUT="/tmp/viz-$$"

cleanup() { ssh "$SSH_HOST" "rm -rf $REMOTE_OUT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

printf 'refresh: player=%s limit=%s days=%s host=%s\n' \
  "$PLAYER" "$LIMIT" "$DAYS" "$SSH_HOST"

ssh "$SSH_HOST" "cd /home/max/index/vision-bot/visualizer && \
  python3 build_from_db.py \
    --player '$PLAYER' \
    --limit  '$LIMIT' \
    --days   '$DAYS' \
    --max-assets '$MAX_ASSETS' \
    --out    '$REMOTE_OUT'"

mkdir -p "$PUBLIC"
rsync -az --delete "$SSH_HOST:$REMOTE_OUT/" "$PUBLIC/"

count=$(find "$PUBLIC/data" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
printf 'done: %s asset files in %s — click "refresh" in the UI\n' "$count" "$PUBLIC"
