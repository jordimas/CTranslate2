# Common Operations Profiling Plan

## Goal

Understand which CTranslate2 kernels and ops dominate typical workloads: *which
operations, with which tensor sizes, at what frequency, and with what latency*.
Use the release-gate workloads as the fixed benchmark surface so results are
reproducible and directly comparable to what we already gate on in CI.

Output: a per-workload table of `(op_name, input_shapes, call_count, total_time,
mean_time)` plus a cross-workload summary of the heaviest ops.

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
that is **compiled out unless `CT2_ENABLE_PROFILING` is defined**. It is not
exposed to Python.

Required steps:
1. Build a dedicated profiling wheel with `-DCT2_ENABLE_PROFILING=ON` (document
   the exact CMake invocation in `build.md` alongside this plan).
2. Add a thin Python binding (or a small C++ CLI shim) that calls
   `init_profiling(device, num_threads)` before the workload and
   `dump_profiling(ostream)` after, writing to a file.
3. Each workload wrapper writes
   `performance_learning/common_operations/results/<task>_<compute_type>.prof`.

The profiler already aggregates by scope name across threads and reports call
counts + total time. Op scopes are named via `PROFILE(op_name)` in
`src/ops/*.cc`; many ops include tensor shapes in the scope string — audit a
few (`gemm`, `layer_norm`, `softmax`, attention variants) and extend any
missing shape annotations so the output is useful.

## Harness

New script: `performance_learning/common_operations/run_profiling.py`.
- Mirrors `release-gates/run_gate.py`'s structure but single-venv (profiling
  build only, no PyPI baseline).
- CLI: `--tasks whisper_wer,opus_bleu,t5gemma_generation`
         `--compute-types float32,float16,int8,...`
         `--whisper-models medium,large-v3`
         `--device cpu`
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
- `report.md` — final tables and the shortlist of hot ops/shapes

## Open questions

1. Shape annotations: how many ops currently emit shapes in their `PROFILE`
   scope name? Audit before trusting the output; extend if sparse.
2. Should we also capture per-decoding-step cost for autoregressive workloads
   (whisper, t5gemma) separately from encoder cost? Likely yes — split by a
   `encoder`/`decoder_step` wrapper scope.
3. Thread count: release-gate uses 2 threads. Also run at 1 and 8 to see which
   ops' share changes with concurrency.
