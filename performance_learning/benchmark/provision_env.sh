#!/usr/bin/env bash
# Provision a local venv once with release-gate dependencies + latest PyPI ctranslate2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
REQ="$HERE/../../release-gates/requirements.txt"
# PyAV <13 (required by faster-whisper) has no Python 3.13+ wheels.
# The pinned transformers version requires Python >=3.11.
supported_python='import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))'
requested_python="${PYTHON:-}"
if [[ -n "${PYTHON:-}" ]]; then
    candidates=("$PYTHON")
else
    candidates=(python3.11 python3.12 python3)
fi
PYTHON=""
for candidate in "${candidates[@]}"; do
    if path="$(command -v "$candidate")" && "$path" -c "$supported_python" 2>/dev/null; then
        PYTHON="$(realpath "$path")"
        break
    fi
done
if [[ -z "$PYTHON" && -z "$requested_python" ]] && command -v uv >/dev/null 2>&1; then
    export UV_PYTHON_INSTALL_DIR="$HERE/.cache/python"
    export UV_CACHE_DIR="$HERE/.cache/uv"
    uv python install --no-bin 3.11
    PYTHON="$(uv python find --managed-python --no-project 3.11)"
fi
if [[ -z "$PYTHON" ]]; then
    echo "[provision] Python 3.11–3.12 is required; install one (or uv), or set PYTHON=/path/to/python." >&2
    exit 1
fi
version="$("$PYTHON" -c 'import sys; print(sys.version_info[:2])')"
if ! "$VENV/bin/python" -c "import sys; assert sys.version_info[:2] == $version" 2>/dev/null; then
    "$PYTHON" -m venv --clear "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade
fi
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install ctranslate2 -r "$REQ"
echo "[provision] ready: $VENV"
