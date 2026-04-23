#!/usr/bin/env bash
# Repo-root bootstrap. Forwards to the chosen source's own setup.
# When no source is specified, defaults to `twitch`.
#
#   ./setup.sh                   # = ./setup.sh --auto-fund --source twitch
#   ./setup.sh --auto-fund       # same, explicit
#   SOURCE=twitch ./setup.sh     # env-var form, same result
#
# Other sources ship in this repo later; once they do, pick one with
# --source <name> or SOURCE=<name>.

set -euo pipefail
cd "$(dirname "$0")"

SOURCE="${SOURCE:-twitch}"
ARGS=()

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
            ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "$SOURCE" ] || [ ! -x "$SOURCE/setup.sh" ]; then
    echo "error: no bot directory for source '$SOURCE'" >&2
    echo "Available sources:" >&2
    for d in */; do
        d="${d%/}"
        [ -x "$d/setup.sh" ] && echo "  - $d" >&2
    done
    exit 1
fi

echo "[bootstrap] source=$SOURCE  (override with --source <name> or SOURCE=<name>)"
exec "./$SOURCE/setup.sh" "${ARGS[@]}"
