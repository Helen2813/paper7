#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PAPER7_PYTHON:-$HOME/Desktop/paper7/.venv-onsetaudit/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3 || true)"; fi
[[ -n "$PY" ]] || { echo "[STOP] Python 3 unavailable"; exit 1; }
cd "$HERE"
"$PY" -c 'import numpy,pandas,matplotlib; print("[ENV] numpy",numpy.__version__,"pandas",pandas.__version__)'
"$PY" scripts/compute_gse246591_delta.py
"$PY" scripts/p7_selection_identifiability_audit.py
"$PY" scripts/p7_dual_component_audit.py
"$PY" scripts/plot_manuscript_figures.py
printf '\n[DONE] GEO-verified conditional diagnostics and manuscript figures regenerated\n'
