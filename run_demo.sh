#!/usr/bin/env bash
# ESL desktop demo — uses .venv if valid (Python 3.10–3.11), else first matching python3.11 / python3.10
set -euo pipefail
cd "$(dirname "$0")"

_py_ok() {
  "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" 2>/dev/null
}

resolve_python() {
  for vd in .venv venv; do
    if [[ -x "$vd/bin/python" ]] && _py_ok "$vd/bin/python"; then
      echo "$vd/bin/python"
      return 0
    fi
  done
  for cmd in python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1 && _py_ok "$(command -v "$cmd")"; then
      command -v "$cmd"
      return 0
    fi
  done
  echo "ERROR: Need Python 3.10 or 3.11 (TensorFlow). Install python3.11 then: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  return 1
}

PY="$(resolve_python)" || exit 1
exec "$PY" demo.py "$@"
