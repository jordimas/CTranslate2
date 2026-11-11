#!/usr/bin/env bash
# Run release gates: baseline = latest PyPI, candidate = newest locally built wheel.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
wheel="$(find "$HERE/.." -name 'ctranslate2-*.whl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$wheel" ]] || { echo "no local ctranslate2 wheel found under $HERE/.." >&2; exit 2; }
echo "[gate] candidate wheel: $wheel"
exec python3 "$REPO/release-gates/run_gate.py" --candidate-wheel "$wheel" "$@"
