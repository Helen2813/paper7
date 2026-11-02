#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

compatible() {
  command -v "$1" >/dev/null 2>&1 &&
  "$1" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' >/dev/null 2>&1
}

PY=""

for candidate in \
  "$ROOT/.venv-onsetaudit/bin/python" \
  python3.13 python3.12 python3.11 python3.10 python3 \
  /opt/homebrew/bin/python3.13 \
  /opt/homebrew/bin/python3.12 \
  /usr/local/bin/python3.13 \
  /usr/local/bin/python3.12 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  "$HOME/Desktop/paper8/endo-research/.venv-shih-h5/bin/python" \
  "$HOME/Desktop/paper8/.venv-shih-h5/bin/python"; do

  if compatible "$candidate"; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  BREW="$(command -v brew || true)"

  if [[ -z "$BREW" ]]; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [[ -x "$b" ]]; then
        BREW="$b"
        break
      fi
    done
  fi

  if [[ -z "$BREW" ]]; then
    echo "[STOP] Compatible Python not found."
    echo "[STOP] Homebrew is unavailable."
    exit 1
  fi

  echo "[SETUP] Installing Python 3.12"
  "$BREW" install python@3.12

  PY="$("$BREW" --prefix python@3.12)/bin/python3.12"

  if ! compatible "$PY"; then
    echo "[STOP] Python installation verification failed."
    exit 1
  fi
fi

echo "[PYTHON] $("$PY" -c 'import sys; print(sys.version.split()[0])')"
echo "[EXECUTABLE] $PY"

VENV="$ROOT/.venv-onsetaudit"

if [[ -x "$VENV/bin/python" ]] && ! compatible "$VENV/bin/python"; then
  BACKUP="${VENV}.old.$(date +%Y%m%dT%H%M%S)"
  mv "$VENV" "$BACKUP"
  echo "[BACKUP] $BACKUP"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[SETUP] Creating virtual environment"
  "$PY" -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import numpy,pandas; assert numpy.__version__=="2.2.6" and pandas.__version__=="2.3.3"' >/dev/null 2>&1; then
  echo "[DEPENDENCIES] Installing requirements"
  "$VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "$ROOT/onsetaudit/requirements.txt"
fi

echo "[START] OnsetAudit"
exec "$VENV/bin/python" "$ROOT/onsetaudit/app.py"
