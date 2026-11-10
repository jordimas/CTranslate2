# Common Operations Profiling Plan

## Goal

Understand which CTranslate2 kernels and ops dominate typical workloads: *which
operations, with which tensor sizes, at what frequency, and with what latency*.
Use the release-gate workloads as the fixed benchmark surface so results are
reproducible and directly comparable to what we already gate on in CI.

Output: a per-workload table of `(op_name, input_shapes, call_count, total_time,
mean_time)` plus a cross-workload summary of the heaviest ops.

## Current status and Linux prerequisite

The historical macOS report used a custom wheel whose source changes were not
present in the checkout. Profiling bindings, call counters, and shape annotations
have now been reconstructed in source, with a portable build recipe in `build.md`.
Build separately on Linux; the preserved Apple wheel is not portable.

Implemented: source instrumentation, profiling tests, build workflow, warm-up
measurement fixes, sample-limit forwarding, and run metadata/resume checks.
Pending: target Linux/CUDA execution, quality validation, repeated unprofiled
benchmarks, and broader shape coverage. The reconstructed profiler is not claimed
to be binary-identical to the old wheel.

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

Same sample counts as `release-gate.yml` (`--limit 20 --bleu-limit 500`) so one
run reflects the same work CI already does.

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

## Linux port and acceptance criteria

1. Recover the original instrumentation if available; otherwise implement the
   source changes above. Test nested scopes, repeated calls, reset behavior,
   multithread aggregation, and a build with profiling disabled. Do not claim
   the reconstructed profiler is identical to the preserved binary without
   verification.
2. Add a Linux build recipe in `build.md` or a dedicated script. Use an isolated
   build directory and virtual environment, `-DENABLE_PROFILING=ON`, and
   `-DWITH_MPS=OFF`. Select and document a CPU math backend appropriate to the
   target architecture; add CUDA only when requested and available. Remove
   Homebrew, Accelerate, Apple M4 compiler flags, macOS wheel repair, and local
   venv assumptions from the Linux recipe. Record compiler, CMake options,
   Python/dependency versions, source revision/dirty state, and hardware.
3. Build and install the Linux wheel from that source. Confirm both Python
   profiling functions exist, then run a small CPU float32 OPUS workload. Require
   a nonempty profile with positive call counts and annotated Gemm shapes, and
   a nonempty generated report. An empty init/dump smoke check alone is not enough.
4. Fix measurement boundaries before cross-machine benchmarking: fully consume
   Whisper warm-up segments, finish warm-up before both timer and profiler reset,
   and forward sample-limit arguments to the worker. Record the methodology
   version; do not silently compare corrected timing to the original wall times.
5. Run repeated unprofiled measurements separately from synchronized operation
   profiles. Validate WER/BLEU and T5Gemma outputs/EOS behavior before treating
   precision variants as equivalent workloads.
6. Run the supported compute-type matrix into a fresh machine-specific results
   directory. Record unsupported combinations explicitly. Never resume from
   the old machine's results; the current runner skips successful cached runs.
7. Verify report regeneration from a relocated results directory. Preserve the
   original macOS report and raw data, documenting their known snapshot mismatch.

Linux acceptance: a clean checkout plus documented prerequisites must build the
instrumented wheel, run the CPU smoke workload, and generate the report without
any files from the original Mac. CUDA validation is a separate follow-on step.

## Harness

Script: `performance_learning/common_operations/run_profiling.py`.
- Mirrors `release-gates/run_gate.py`'s structure but single-venv (profiling
  build only, no PyPI baseline).
- CLI: `--tasks whisper-medium,whisper-large-v3,opus,t5gemma`
         `--compute-types float32,float16,int8,...`
         `--whisper-limit 20 --opus-limit 100 --t5gemma-limit 20`
         `--devices cpu`
         `--out-dir results/`.
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
- Primary output: a single static **`report.html`** with sortable tables.
  - Per-run section for each `(task, compute_type)`:
    `op | shape | calls | total ms | mean ms | % of run`, sorted by total ms.
  - Cross-workload roll-up table: top-N ops by cumulative time across all
    `(workload, compute_type)` runs, grouped by op name and by `(op, shape)`.
  - Rank-shift table: ops whose rank changes significantly between compute
    types (e.g. gemm dominance under `int8` vs `float32`) — tuning targets.
  - Implementation: self-contained HTML — plain `<table>` markup plus a small
    inline JS click-to-sort snippet (no external CDN, no build step). Opens
    directly in a browser; easy to share as a single file.
- Also emit `summary.json` (same data, machine-readable) so other tools can
  consume it.

## Deliverables

Checked into `performance_learning/common_operations/`:
- `plan.md` (this file)
- `build.md` — how to build the profiling-enabled wheel
- `run_profiling.py` — orchestrator
- `summarize.py` — analyzer
- `results/` — raw `*.prof` dumps (one per `(task, compute_type)`)
- `report.html` and `summary.json` — report and machine-readable data
- Main-repository C++/Python binding changes — required source instrumentation
- Linux build instructions and instrumentation tests — clean-checkout validation

## Open questions

1. Shape annotations: how many ops currently emit shapes in their `PROFILE`
   scope name? Audit before trusting the output; extend if sparse.
2. Should we also capture per-decoding-step cost for autoregressive workloads
   (whisper, t5gemma) separately from encoder cost? Likely yes — split by a
   `encoder`/`decoder_step` wrapper scope.
3. Thread count: release-gate uses 2 threads. Also run at 1 and 8 to see which
   ops' share changes with concurrency.
