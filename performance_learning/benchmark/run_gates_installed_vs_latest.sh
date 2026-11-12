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
device="cpu"
if [[ $# -gt 0 && "$1" != -* ]]; then
    device="$1"; shift
fi
echo "[gate] candidate wheel: $wheel"
echo "[gate] device: $device"

baseline=()
if [[ "$device" == "cuda" && "$*" != *--baseline-wheel* ]]; then
    d="$HERE/wheels/baseline-rocm"; mkdir -p "$d"
    bw="$(find "$d" -name "ctranslate2-*-$tag-$tag-*x86_64.whl" | head -1 || true)"
    if [[ -z "$bw" ]]; then
        echo "[gate] fetching ROCm baseline wheels v4.9.0"
        curl -fL -o "$d/w.zip" \
          "https://github.com/OpenNMT/CTranslate2/releases/download/v4.9.0/rocm-python-wheels-Linux.zip"
        (cd "$d" && unzip -qo w.zip && rm w.zip)
        bw="$(find "$d" -name "ctranslate2-*-$tag-$tag-*x86_64.whl" | head -1 || true)"
    fi
    [[ -n "$bw" ]] || { echo "no $tag ROCm baseline wheel found in $d" >&2; exit 2; }
    echo "[gate] baseline wheel: $bw"
    baseline=(--baseline-wheel "$bw")
fi
exec "$PYTHON" "$REPO/release-gates/run_gate.py" --candidate-wheel "$wheel" \
    --device "$device" --report "$HERE/gate-summary.md" "${baseline[@]}" "$@"
