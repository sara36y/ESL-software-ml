#!/usr/bin/env bash
# ESL desktop demo — full (threaded) or sprint (single-loop)
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
elif [ -d ".venv" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

exec python demo.py "$@"
