#!/usr/bin/env bash
# Create .venv with python3.11 or python3.10 and install requirements.
set -euo pipefail
cd "$(dirname "$0")"

pick_python() {
  for cmd in python3.11 python3.10; do
    if command -v "$cmd" >/dev/null 2>&1; then
      if "$cmd" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" 2>/dev/null; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  echo "Install Python 3.11 (e.g. apt install python3.11-venv) then re-run ./setup_venv.sh" >&2
  return 1
}

PY="$(pick_python)" || exit 1
echo "Using $PY"
"$PY" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Run: ./run_demo.sh  or  ./run_web.sh"
