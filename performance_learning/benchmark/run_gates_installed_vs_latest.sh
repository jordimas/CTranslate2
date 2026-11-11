#!/usr/bin/env bash
# Run release gates: baseline = latest PyPI, candidate = newest locally built wheel.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="$HERE/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "run $HERE/provision_env.sh first" >&2; exit 2; }
tag="$("$PYTHON" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
wheel="$(find "$HERE/.." -name "ctranslate2-*-$tag-$tag-*.whl" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$wheel" ]] || { echo "no $tag ctranslate2 wheel found under $HERE/..; rebuild the candidate with $PYTHON" >&2; exit 2; }
echo "[gate] candidate wheel: $wheel"
exec "$PYTHON" "$REPO/release-gates/run_gate.py" --candidate-wheel "$wheel" --report "$HERE/gate-summary.md" "$@"
