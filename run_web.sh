#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
elif [ -d ".venv" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

exec uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
