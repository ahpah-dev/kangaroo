#!/usr/bin/env bash
# End-to-end check of the real desktop app: launches Electron, clicks Train in the UI on a tiny
# offline model, then Test and Export, and saves a screenshot. No downloads, runs on CPU.
#   bash scripts/smoke-e2e.sh [python-executable]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-${KANGAROO_PYTHON:-python}}"
WORK="${KANGAROO_WORK:-$(mktemp -d)}"; mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
to_native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
W="$(to_native "$WORK")"

PATHS="$("$PY" "$ROOT/tests/fixtures.py" "$W" 2>/dev/null | tail -n 1)"
MODEL="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["model"].replace("\\","/"))' "$PATHS")"
DATA="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["dataset"].replace("\\","/"))' "$PATHS")"

export KANGAROO_PYTHON="$PY"
export KANGAROO_SMOKE_SHOT="${KANGAROO_SMOKE_SHOT:-$(to_native "$ROOT/app/smoke-e2e.png")}"
export KANGAROO_SMOKE_E2E="{\"train\":{\"model\":\"$MODEL\",\"dataset\":\"$DATA\",\"output_dir\":\"$W/out\",\"max_seq_len\":64,\"batch_tokens\":256,\"grad_accum\":2,\"epochs\":0,\"max_steps\":40,\"learning_rate\":0.003,\"lora_r\":8,\"lora_alpha\":16,\"precision\":\"fp32\",\"val_fraction\":0.1},\"test\":{\"prompt\":\"say red one\",\"temperature\":0,\"max_new_tokens\":6},\"export\":{\"output_dir\":\"$W/merged\",\"precision\":\"fp32\"}}"

cd "$ROOT/app"
node node_modules/electron/cli.js . --smoke 2>&1 | grep -E '^\{"smoke"' | tail -n 1
