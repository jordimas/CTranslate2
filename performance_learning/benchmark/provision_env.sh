#!/usr/bin/env bash
# Provision a local venv once with release-gate dependencies + latest PyPI ctranslate2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
REQ="$HERE/../../release-gates/requirements.txt"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q ctranslate2 -r "$REQ"
echo "[provision] ready: $VENV"
