# Build profiling instrumentation on each machine

The source now includes profiling bindings, call counts, and shape annotations.
Build the same Git commit separately for each OS, architecture, Python ABI, and
CPU/GPU backend. The historical Apple wheel is not needed for a new build.

## Linux CPU (first validation target)

Install a C++17 compiler, CMake, Git, Python with venv/development headers,
OpenBLAS development libraries, and an OpenMP runtime. On Debian/Ubuntu these
are typically `build-essential cmake git python3-dev python3-venv libopenblas-dev`.
Initialize submodules from the repository root:

```sh
git submodule update --init --recursive
cd performance_learning/common_operations
PYTHON=python3 JOBS=4 bash build.sh
```

Linux defaults: OpenBLAS + Ruy, compiler OpenMP, CPU only. macOS defaults:
Accelerate + Ruy, no OpenMP, CPU only. `WITH_MPS=ON bash build.sh` enables MPS on
macOS. CUDA is opt-in with `WITH_CUDA=ON bash build.sh` and requires a compatible
CUDA toolkit and cuDNN; GPU execution must be validated on the target machine.
Additional CMake options can be passed as arguments to `build.sh`.

The script builds into `.build/profiling`, runs `ProfilerTest.*`, and produces a
wheel plus `build-metadata.json`, `requirements-build.txt`, and the CMake cache.
It does not delete build directories. Use a fresh `BUILD_ROOT` when changing
Python, architecture, or backend. Preserve these metadata files with results.

## Install and smoke-test

```sh
.build/profiling/venv/bin/python -m pip install .build/profiling/wheels/*.whl
.build/profiling/venv/bin/python -m pip install -r requirements.txt
.build/profiling/venv/bin/python run_profiling.py \
  --devices cpu --tasks opus --compute-types float32 --opus-limit 2 \
  --out-dir results-linux-smoke
.build/profiling/venv/bin/python summarize.py --results results-linux-smoke \
  --json-out summary-linux-smoke.json
```

The report is named `reports_<cpu_id>.html` using the CPU recorded in the run
metadata, even when it is generated on another machine.

Use the newly built environment, not the historical `venv-prof`. The smoke run
must produce positive `calls` and Gemm shapes in the profile and a populated
report. Model/dataset downloads require network access. Then run the desired
matrix with `--devices cpu` (or `cpu,cuda`) and another fresh output directory.

These are local wheels: they reference the library under the build root's
`install` directory. Keep that directory in place. Do not copy a wheel alone to
another host; rebuild there. Distribution as standalone wheels requires separate
platform-specific dependency bundling/repair and validation.

## Profile format and interpretation

Format 1 adds a comment header and `calls=N` to each existing self/inclusive
percentage and self-time row. Shapes are bracketed, for example
`Gemm[2x512,512x512]`. Scalar shapes are `scalar`; absent brackets mean no shape
annotation. Gemm, MatMul, SoftMax/LogSoftMax, LayerNorm, and RMSNorm entry points
are annotated; other operations and paths bypassing these entry points remain
unannotated. Shapes use host metadata only.

The global profiler must be initialized/reset/dumped only while inference is
idle. Worker-thread scope aggregation is protected by a mutex. Scope boundaries
synchronize the selected device; recorded timings are not unprofiled throughput.
Normal builds compile profiling scopes out; calling `init_profiling` on such a
build reports that profiling is disabled.

Methodology 2 excludes warm-up from both wall time and scope timing, consumes
Whisper's lazy warm-up generator, and honors sample-limit flags. Original saved
results predate these fixes. Reconstructed instrumentation is not claimed to be
binary-identical to the historical wheel. Quality evaluation and repeated
unprofiled benchmarks remain separate work.
