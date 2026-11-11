#!/usr/bin/env bash
# Provision a local venv once with release-gate dependencies + latest PyPI ctranslate2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
REQ="$HERE/../../release-gates/requirements.txt"
# PyAV <13 (required by faster-whisper) has no Python 3.13+ wheels.
PYTHON="$(realpath "$(command -v python3.11)")"
if ! "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
    "$PYTHON" -m venv --clear "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q ctranslate2 -r "$REQ"
echo "[provision] ready: $VENV"
