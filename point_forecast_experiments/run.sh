#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  PY="${PYTHON:-python}"
fi

cd "$HERE"
"$PY" -m src.run_experiments
"$PY" -m src.analyze_results
"$PY" -m src.make_figures
"$PY" -m src.verify_outputs
