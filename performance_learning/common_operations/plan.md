# Common Operations Profiling Plan

## Goal

Understand which CTranslate2 kernels and ops dominate typical workloads: *which
operations, with which tensor sizes, at what frequency, and with what latency*.
Use the release-gate workloads as the fixed benchmark surface so results are
reproducible and directly comparable to what we already gate on in CI.

Output: a per-workload table of `(op_name, input_shapes, call_count, total_time,
mean_time)` plus a cross-workload summary of the heaviest ops.

## Current implementation and validation

The historical macOS report used a custom wheel whose source changes were not
present in the checkout. Profiling bindings, call counters, and shape annotations
have now been reconstructed in source, with a portable build recipe in `build.md`.
The source changes and build workflow were committed in `08883514`. Build the
same revision separately on each target; the preserved Apple wheel is not portable.

Implemented: source instrumentation, profiling tests, build workflow, warm-up
measurement fixes, sample-limit forwarding, and run metadata/resume checks.
Pending: target Linux/CUDA execution, quality validation, repeated unprofiled
benchmarks, and broader shape coverage. The reconstructed profiler is not claimed
to be binary-identical to the old wheel.

Validated on macOS CPU: a fresh C++ library and Python wheel, two enabled
profiler tests, one separately compiled disabled-profiler test, three harness
tests, and a two-sentence OPUS run through report generation. The validation
build used Accelerate with MPS and Ruy disabled; this does not establish that
the Linux OpenBLAS/Ruy configurations pass.

The full Apple M4 Pro CPU/MPS rerun has now completed: 40 successful
configurations using the fresh Accelerate + Ruy + MPS build from source revision
`1b9a9c75`. Results are in `results/apple_m4_pro_methodology2/`; the updated
`reports_apple_m4_pro.html` and `summary.json` are generated from that directory.
All sample counts, positive timings/call counts, Gemm shape annotations, and
methodology markers were checked. This validates execution and profile output,
not model quality or unprofiled throughput. T5Gemma's affected MPS variants still
produce 1280 tokens across 20 prompts, so the decoding-limit concern remains.

Current defaults are 100 OPUS sentences, 20 T5Gemma prompts, and 20 Whisper
samples. The release-gate-sized workload below remains a planned extension;
changing counts or methodology must be explicit in comparisons.

## Workload set

Reuse the release-gate task definitions in `release-gates/tasks.py`:

1. **whisper_wer** — replace the hard-coded `tiny` model with:
   - `openai/whisper-medium`
   - `openai/whisper-large-v3`
   (Keep LibriSpeech dummy clean, `beam_size=1`, `temperature=0.0`.)
2. **opus_bleu** — Helsinki-NLP/opus-mt-en-de, WMT14 en-de, `beam_size=4`,
   `max_batch_size=16`, `--bleu-limit 500` (release-gate default).
3. **t5gemma_generation** — jordimas/t5gemma-2-270m-270m, SQuAD prompts,
   `beam_size=1`, `sampling_topk=1`, `max_batch_size=8`, 50 prompts.

The intended release-gate comparison uses 500 OPUS sentences and 50 T5Gemma
prompts. Request these explicitly with `--opus-limit 500 --t5gemma-limit 50`;
the current defaults remain 100 and 20. The harness does not yet compute
WER/BLEU or establish full release-gate equivalence.

## Compute-type matrix

Enumerate every `ComputeType` from `include/ctranslate2/types.h`:
`float32`, `float16`, `bfloat16`, `int16`, `int8`, `int8_float32`,
`int8_float16`, `int8_bfloat16`.

For each `(workload, compute_type)` pair:
- Skip combinations the device/CPU reports unsupported
  (`mayiuse_float16`/`bfloat16`/`int16`/`int8`) — log skipped, don't fail.
- Convert each model once per `compute_type` (via
  `TransformersConverter.convert(..., quantization=<type>)`) and cache under
  `performance_learning/common_operations/models/<task>/<compute_type>/`.

Device: start on `cpu` (matches release-gate CI). Add `cuda` and `mps` as
follow-on runs — same harness, different device flag.

## Profiling mechanism

CTranslate2 has a built-in scope profiler (`include/ctranslate2/profiler.h`)
that is **compiled out unless `CT2_ENABLE_PROFILING` is defined**. It is exposed
through private Python profiling bindings.

Build and use:
1. Build a dedicated profiling wheel with `-DENABLE_PROFILING=ON` (document
   the exact CMake invocation in `build.md` alongside this plan).
2. Use the Python bindings `init_profiling(device, num_threads)` before the
   workload and `dump_profiling(path)` afterward.
3. Each workload wrapper writes
   `performance_learning/common_operations/results/<task>__<device>__<compute_type>.prof`.

The profiler aggregates self and inclusive time and call counts by scope name
across threads; its output is consumed by `summarize.py`.
The CMake option is `ENABLE_PROFILING`; `CT2_ENABLE_PROFILING` is the C++
preprocessor definition it enables.

Source instrumentation implemented in the main repository:
- `src/profiler.cc`: initialize and increment a per-scope call counter and append
  `calls=N` to each timing row. Preserve self/inclusive timing semantics and
  thread-safe aggregation; reset counters when profiling is initialized again.
- `python/cpp/module.cc`: expose `init_profiling(device, num_threads)` and
  `dump_profiling(path)` used by `workloads.py`, with clear file-write errors.
- `include/ctranslate2/profiler.h` and relevant `src/ops/*.cc`: add a reusable
  shape annotation mechanism, beginning with Gemm, MatMul, SoftMax/LogSoftMax,
  LayerNorm, and RMSNorm. Emit names compatible with the parser, for example
  `Gemm[1x1500x1280,5120x1280]`. Avoid tensor reads or device transfers for shapes.
  Compile instrumentation out when profiling is disabled.
- Document which operations have shape coverage and the profile format version.
  Missing annotations must remain distinguishable from measured scalar shapes.

## C++ build and Python wheel workflow

The build is now source-based. `build.sh` configures CMake, builds and installs
`libctranslate2`, builds the C++ test executable, runs `ProfilerTest.*`, and then
builds the Python wheel against that library using `CTRANSLATE2_ROOT`.
See [build.md](build.md) for prerequisites and installation commands.

- Enable instrumentation with `-DENABLE_PROFILING=ON`; CMake defines
  `CT2_ENABLE_PROFILING` for C++. A normal build compiles scope instrumentation
  out, and its Python `init_profiling` call reports that profiling is disabled.
- Linux defaults: OpenBLAS + Ruy, compiler OpenMP, CPU only, MPS disabled.
- macOS defaults: Accelerate + Ruy, no OpenMP, CPU only. MPS is opt-in.
- CUDA is opt-in with `WITH_CUDA=ON` and requires CUDA/cuDNN prerequisites.
- Outputs live under `.build/profiling`: `cmake/`, `install/`, `venv/`, staged
  Python sources, and `wheels/`. Use a fresh `BUILD_ROOT` when changing the
  architecture, Python ABI, or backend. Existing directories are not deleted.
- Keep `install/` in place: the wheel references this local native library.
  Rebuild on every target OS/architecture/Python ABI; these wheels are not
  standalone distributable artifacts.
- Preserve `build-metadata.json`, `requirements-build.txt`, and the CMake cache
  with each run. They record source state, platform/Python, build dependencies,
  compiler and backend configuration. Run metadata also records the extension
  checksum and installed package versions.

The committed `tests/profiler_test.cc` checks nested/repeated scopes, reset,
thread aggregation, shape formatting, and disabled macros. The build script
runs the enabled tests; the disabled test needs a separate non-profiling build.
`test_harness.py` checks warm-up ordering, sample-limit forwarding, and report
regeneration after relocation.

## Linux acceptance and remaining work

1. On a clean Linux checkout of the same revision, initialize submodules and
   install the prerequisites from `build.md`. Run `PYTHON=python3 bash build.sh`.
   Verify the C++ profiling tests pass and install the locally produced wheel
   and workload requirements into its new virtual environment.
2. Run the two-sentence CPU float32 OPUS smoke test from `build.md` into a fresh
   directory. Require positive call counts, annotated Gemm shapes, two measured
   sentences, methodology version 2, and a nonempty generated report. An empty
   init/dump smoke check is insufficient.
3. Run the supported CPU compute-type matrix into a new machine-specific results
   directory. Check logged unsupported types and failures. The runner rejects
   incompatible resume metadata; do not reuse the original Mac results directory.
4. Validate a non-profiling Linux build and its disabled instrumentation test.
   Run repeated unprofiled measurements separately from synchronized profiles.
5. Validate WER/BLEU and T5Gemma output/EOS behavior before treating precision
   variants as equivalent workloads. Expand shape coverage where useful.
6. Validate CUDA separately on a suitable GPU host. Do not infer CUDA readiness
   from successful CPU compilation or the macOS smoke run.

Linux acceptance: a clean checkout plus documented prerequisites must build the
instrumented native library and wheel, run the CPU workload, and generate the
report without any binaries from the original Mac.

Measurement fixes are already implemented as methodology 2: warm-up completes
before both measurements, Whisper's segment generator is consumed, and sample
limits reach the worker. Preserve the historical Mac baseline and its documented
report/raw-data mismatch; do not silently compare old and new timing methods.

## Harness

Script: `performance_learning/common_operations/run_profiling.py`.
- Mirrors `release-gates/run_gate.py`'s structure but single-venv (profiling
  build only, no PyPI baseline).
- CLI: `--tasks whisper-medium,whisper-large-v3,opus,t5gemma`
         `--compute-types float32,float16,int8,...`
         `--whisper-limit 20 --opus-limit 100 --t5gemma-limit 20`
         `--devices cpu`
         `--out-dir results-linux-cpu/`.
- For each `(task, compute_type)`:
  1. Convert model if cache miss.
  2. Set env: `OMP_NUM_THREADS=2` etc. (same as release-gate).
  3. Run workload; capture profiler dump and wall-clock timing.
  4. Record metadata: ctranslate2 git SHA, device, CPU/arch, compute_type,
     model, sample count.

## Analysis

Script: `performance_learning/common_operations/summarize.py`.
- Parse the `*.prof` dumps (format defined in `src/profiler.cc`
  `dump_profiling`).
- Primary output: a single static **`reports_<cpu_id>.html`** with sortable tables.
  - Per-run section for each `(task, compute_type)`:
    `op | shape | calls | total ms | mean ms | % of run`, sorted by total ms.
  - Cross-workload roll-up table: top-N ops by cumulative time across all
    `(workload, compute_type)` runs, grouped by op name and by `(op, shape)`.
  - Rank-shift table: ops whose rank changes significantly between compute
    types (e.g. gemm dominance under `int8` vs `float32`) — tuning targets.
  - Implementation: self-contained HTML — plain `<table>` markup plus a small
    inline JS click-to-sort snippet (no external CDN, no build step). Opens
    directly in a browser; easy to share as a single file.
- Derive `<cpu_id>` from benchmark run metadata, not the report-rendering host.
  Historical runs can supply `--cpu-id`; `--out` still overrides the filename.
- Also emit `summary.json` (same data, machine-readable) so other tools can
  consume it.

## Deliverables

Checked into `performance_learning/common_operations/`:
- `plan.md` (this file)
- `build.sh` and `build.md` — native library, tests, and per-platform wheel build
- `requirements.txt` — workload dependencies
- `run_profiling.py` — orchestrator
- `summarize.py` — analyzer
- `results/` — raw `*.prof` dumps (one per `(task, compute_type)`)
- `reports_<cpu_id>.html` and `summary.json` — report and machine-readable data
- Main-repository C++/Python binding changes — required source instrumentation
- `tests/profiler_test.cc` and `test_harness.py` — profiling and harness checks

Native libraries, wheels, models, and virtual environments remain local ignored
artifacts. Target Linux/CUDA validation results are still outstanding.

## Open questions

1. Shape annotations: how many ops currently emit shapes in their `PROFILE`
   scope name? Audit before trusting the output; extend if sparse.
2. Should we also capture per-decoding-step cost for autoregressive workloads
   (whisper, t5gemma) separately from encoder cost? Likely yes — split by a
   `encoder`/`decoder_step` wrapper scope.
3. Thread count: release-gate uses 2 threads. Also run at 1 and 8 to see which
   ops' share changes with concurrency.
