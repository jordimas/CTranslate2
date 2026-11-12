#!/usr/bin/env bash
# Compare baseline and candidate on CUDA or HIP using the same gate tasks.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/run_gates_installed_vs_latest.sh" --device cuda --report "$HERE/gate-summary-gpu.md" "$@"
