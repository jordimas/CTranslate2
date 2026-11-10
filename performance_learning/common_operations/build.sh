#!/usr/bin/env bash
# Build locally on each OS/architecture. Never deletes existing directories.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/.build/profiling}"
JOBS="${JOBS:-4}"
WITH_CUDA="${WITH_CUDA:-OFF}"
WITH_MPS="${WITH_MPS:-OFF}"
mkdir -p "$BUILD_ROOT"
BUILD_ROOT="$(cd "$BUILD_ROOT" && pwd)"
case "$(uname -s)" in
  Linux) BACKEND=(-DWITH_ACCELERATE=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP) ;;
  Darwin) BACKEND=(-DWITH_ACCELERATE=ON -DWITH_OPENBLAS=OFF -DOPENMP_RUNTIME=NONE) ;;
  *) echo "Supported build hosts: Linux and macOS" >&2; exit 1 ;;
esac
"$PYTHON" -m venv "$BUILD_ROOT/venv"
PY="$BUILD_ROOT/venv/bin/python"
"$PY" -m pip install -r "$ROOT/python/install_requirements.txt" build
cmake -S "$ROOT" -B "$BUILD_ROOT/cmake" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/install" \
  -DBUILD_CLI=OFF -DBUILD_TESTS=ON -DENABLE_PROFILING=ON \
  -DWITH_MKL=OFF -DWITH_RUY=ON \
  -DWITH_CUDA="$WITH_CUDA" -DWITH_CUDNN="$WITH_CUDA" -DWITH_MPS="$WITH_MPS" \
  "${BACKEND[@]}" "$@"
cmake --build "$BUILD_ROOT/cmake" --target install ctranslate2_test --parallel "$JOBS"
"$BUILD_ROOT/cmake/tests/ctranslate2_test" "$ROOT/tests/data" --gtest_filter='ProfilerTest.*'
# Stage the Python sources so setuptools does not modify the source checkout.
"$PY" - "$ROOT/python" "$BUILD_ROOT/python" <<'PY'
import shutil, sys
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '*.so', '__pycache__'))
PY
CTRANSLATE2_ROOT="$BUILD_ROOT/install" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
  "$PY" -m build --wheel --no-isolation --outdir "$BUILD_ROOT/wheels" "$BUILD_ROOT/python"
"$PY" - "$BUILD_ROOT" "$ROOT" <<'PY'
import json, pathlib, platform, subprocess, sys
out, root = map(pathlib.Path, sys.argv[1:])
info = {'platform': platform.platform(), 'machine': platform.machine(),
        'python': sys.version,
        'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', '-C', str(root), 'status', '--short'], text=True),
        'profile_format': 1}
(out / 'build-metadata.json').write_text(json.dumps(info, indent=2) + '\n')
(out / 'requirements-build.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
PY
printf 'Wheel: %s/wheels\nKeep this local library install: %s/install\n' "$BUILD_ROOT" "$BUILD_ROOT"
