#!/usr/bin/env bash
# Build a profiling-enabled CTranslate2 wheel for CPU+MPS on Apple Silicon.
# Reuses the host venv-mps for Python (Python 3.13). Produces a wheel in
# python/dist/ and installs it into a dedicated profiling venv.
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LIBOMP_PREFIX=$(brew --prefix libomp)
export LDFLAGS="-L${LIBOMP_PREFIX}/lib"
export CPPFLAGS="-I${LIBOMP_PREFIX}/include"
export CMAKE_PREFIX_PATH="${LIBOMP_PREFIX}"
export OpenMP_C_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_C_LIB_NAMES="omp"
export OpenMP_CXX_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_CXX_LIB_NAMES="omp"
export OpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

BUILD_DIR="$ROOT/build-profiling"
INSTALL_DIR="$ROOT/install-profiling"
rm -rf "$BUILD_DIR" "$INSTALL_DIR"
mkdir -p "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CLI=OFF \
  -DENABLE_PROFILING=ON \
  -DCMAKE_CXX_FLAGS="-O3 -mcpu=apple-m4" \
  -DCMAKE_C_FLAGS="-O3 -mcpu=apple-m4" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE \
  -DWITH_RUY=ON -DWITH_MPS=ON \
  -DOpenMP_C_FLAGS="${OpenMP_C_FLAGS}" \
  -DOpenMP_CXX_FLAGS="${OpenMP_CXX_FLAGS}" \
  -DOpenMP_omp_LIBRARY="${OpenMP_omp_LIBRARY}"

cmake --build "$BUILD_DIR" --target install -- -j"$(sysctl -n hw.physicalcpu_max)"

# Prepare profiling venv using same Python as venv-mps.
PROF_VENV="$ROOT/performance_learning/common_operations/venv-prof"
PY="$ROOT/venv-mps/bin/python"
rm -rf "$PROF_VENV"
"$PY" -m venv "$PROF_VENV"
PROF_PY="$PROF_VENV/bin/python"
"$PROF_PY" -m pip install -q -U pip wheel build delocate

cd "$ROOT/python"
rm -rf dist build *.egg-info
"$PROF_PY" -m pip install -q -r install_requirements.txt
CTRANSLATE2_ROOT="$INSTALL_DIR" "$PROF_PY" -m build --wheel --outdir dist
DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:${DYLD_LIBRARY_PATH:-}" delocate-wheel -w dist dist/*.whl

"$PROF_PY" -m pip install --force-reinstall dist/*.whl
"$PROF_PY" -m pip install -q \
  faster-whisper transformers sentencepiece sacrebleu jiwer \
  soundfile datasets accelerate

"$PROF_PY" -c "import ctranslate2; ctranslate2._ext.init_profiling('cpu'); print('profiling OK:', ctranslate2._ext.dump_profiling())"
echo "wheel ready: $PROF_VENV"
