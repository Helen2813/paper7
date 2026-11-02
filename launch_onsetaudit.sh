#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if ! command -v python3 >/dev/null 2>&1; then
  echo "[STOP] python3 not found. Install Python 3.10+ from python.org or Homebrew."
  exit 1
fi
python3 - <<'PY'
import sys
if not (3,10) <= sys.version_info[:2] < (3,14):
    raise SystemExit("Python 3.10–3.13 required for OnsetAudit. Current: "+sys.version)
PY
VENV="$ROOT/.venv-onsetaudit"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[SETUP] Creating isolated OnsetAudit environment"
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import pandas,numpy; assert pandas.__version__ == "2.3.3" and numpy.__version__ == "2.2.6"' >/dev/null 2>&1; then
  echo "[DEPENDENCIES] Installing numpy and pandas into isolated venv"
  "$VENV/bin/python" -m pip install --disable-pip-version-check -r onsetaudit/requirements.txt
fi
if [ "${1:-}" = "--test" ]; then
  exec "$VENV/bin/python" -m unittest discover -s onsetaudit/tests -v
fi
exec "$VENV/bin/python" onsetaudit/app.py
