#!/usr/bin/env bash
# Kangaroo setup for Linux/macOS: creates .venv, installs PyTorch + Kangaroo, installs the desktop app.
#   bash scripts/setup.sh [cu128|cu126|cu124|cpu]
set -euo pipefail
TORCH="${1:-cu128}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d .venv ] || python3 -m venv .venv
PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then
  "$PY" -m pip install torch            # macOS wheels (CPU/MPS)
else
  "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH}"
fi
# bitsandbytes (4-bit) is CUDA-only; skip it on macOS.
if [ "$(uname)" = "Darwin" ]; then "$PY" -m pip install -e ".[hub]"; else "$PY" -m pip install -e ".[quant,hub]"; fi

(cd app && npm install && { [ -f node_modules/electron/path.txt ] || node node_modules/electron/install.js; })
echo
echo "Done. Start the app with:  cd app && npm start"
