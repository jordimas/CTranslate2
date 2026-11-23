# Experimental oneDNN full-shape plan with tiled execution

This is a diagnostic prototype, not a production CT2 backend. The normal build
does not compile these files. `build.py` creates a separate library snapshot,
replacing only the fused-attention object and adding `plan.cc`; it does not edit
the production kernel or overwrite the normal build.

## Numerical contract and scope

The target is compatibility with CT2's existing **oneDNN SGEMM** attention,
including its float bits and resulting quantized bytes/scales. This prototype
uses the original SGEMM microkernels, not a new BRGEMM implementation with a
potentially different reduction order. It tests the full-plan/tiled-execution
architecture without simultaneously changing the backend or arithmetic kernel.

Eligibility is deliberately narrow: oneDNN **3.1.1**, effective ISA **AVX2**,
unmasked FP32 attention, query/key lengths **1500**, query/key/value depth **64**,
scale **0.125**, and at least two batch/head entries. CT2's existing validation
checks tensor dtype/shape and active backend before calling the adapter. Other
shapes, masks, ISAs and versions retain the committed whole-head path. This is
not a validated production dispatch policy for all AVX2 CPUs or input values.

## What is planned

The immutable plan is constructed once per process, from the **full** shape:

1. QK uses oneDNN's copy-based SGEMM plan at `(m,n,k)=(1500,1500,64)` in
   column-major notation. The plan exposes its original copy functions and
   compute function pointers. The current kernel has `um=24`, `un=4`.
   Keys are packed/scaled once per head per invocation, preserving the reference
   GEMM's alpha placement. Queries are packed per outer tile. The original
   compute kernel executes each key microblock against that query tile, with
   the complete 64-element reduction.
2. PV fixes the original full-shape **no-copy** strategy at `(64,1500,1500)`.
   Its full-query schedule comes from the oneDNN 3.1.1 NN no-copy policy:
   thirty 48-query blocks followed by a 60-query tail. The adapter executes
   those preplanned ranges through the explicitly selected no-copy driver,
   which retains the reference K-block policy. It never submits smaller
   matrices to generic SGEMM's copy/no-copy heuristic.
3. Outer tiles group consecutive preplanned PV ranges within a workspace
   target. A target of 256 produces six 240-query tiles and a 60-query tail.
   The effective size therefore follows the reference execution boundaries;
   it is not an independently selected replacement for all oneDNN tiles.
4. CT2's regular per-row softmax runs between QK and PV for each outer tile.
   There is no online softmax, changed scale placement, or reduced precision.

For the default target, each worker has about **1.89 MB** of explicit buffers:
1.44 MB scores, 0.384 MB packed keys, 0.061 MB packed queries. This excludes
allocator bookkeeping and any internal no-copy driver scratch. The whole-head
score buffer alone is 9 MB per worker for this shape. Packed keys are local to
the current invocation; model unload does not leave a large thread-local cache.
These are buffer-size calculations, not measured process RSS. Plan/JIT setup is
amortized across layers and falls in the benchmark warm-up, outside timed runs.

The PV adapter still calls a private driver per preplanned block rather than
holding direct PV JIT function pointers. This is a conservative prototype with
remaining dispatch overhead. It is not OpenVINO's BRGEMM implementation and does
not establish what a newer public oneDNN microkernel API would do.

## Build and validate

Use the **matching oneDNN 3.1.1 source tree** with the existing configured Release
CT2 build. No oneDNN rebuild is needed: the current static archive contains the
private SGEMM symbols, but its installed SDK does not include their headers.

```bash
curl -L --fail https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.1.1.tar.gz -o /tmp/ct2-onednn-v3.1.1.tar.gz
tar -xzf /tmp/ct2-onednn-v3.1.1.tar.gz -C /tmp
python3 tools/onednn_attention_plan/build.py --onednn-source /tmp/oneDNN-3.1.1
```

The scripts expect the earlier attention probe and operation benchmark binaries
(`/tmp/ct2-attention-probe`, `/tmp/ct2-attention-bench`), built from
`benchmark-results/cpu-attention-epyc/investigation/probe.cc` and
`investigation/performance-exploration/bench.cc`, respectively. Link them against
the existing CT2 build with OpenMP and the repository include directory.

```bash
c++ -O3 -std=c++17 -fopenmp -Iinclude benchmark-results/cpu-attention-epyc/investigation/probe.cc -L/tmp/ct2-attn-variants/planned -lctranslate2 -Wl,-rpath,/tmp/ct2-attn-variants/planned -o /tmp/ct2-attention-probe
c++ -O3 -std=c++17 -fopenmp -Iinclude benchmark-results/cpu-attention-epyc/investigation/performance-exploration/bench.cc -L/tmp/ct2-attn-variants/planned -lctranslate2 -Wl,-rpath,/tmp/ct2-attn-variants/planned -o /tmp/ct2-attention-bench
python3 tools/onednn_attention_plan/validate.py --output benchmark-results/cpu-attention-epyc/investigation/planned-attention
.cache/cpu-plan/venv/bin/python tools/onednn_attention_plan/benchmark.py
.cache/cpu-plan/venv/bin/python tools/onednn_attention_plan/transcribe.py
```

Run these **sequentially**, with no other CPU benchmarks. `benchmark.py` also
expects the earlier `head` and `tile` library snapshots described in the
performance-exploration report. `transcribe.py` uses only the new planned
library, disables/enables fusion for its regular/planned controls, and retains
all warm-up and timed outputs. It uses temperature 0, beam 5 and English.

Diagnostic variables, recognized only in this experimental library:

- `CT2_DIAG_DNNL_PLAN=1`: enable the eligible adapter.
- `CT2_DIAG_PLAN_TARGET`: outer workspace target in query rows (default 256,
  clamped to 96–1500); the process caches the first selected plan.
- `CT2_DIAG_PLAN_VERIFY`: when present, recompute reference QK/final outputs
  per head, compare exact bits, print counts and abort on a mismatch. Do not
  set this for timings.

Private oneDNN structures/functions are not ABI-stable. The runtime version
check cannot make mismatched headers/libraries safe; the build must use matching
sources. A supported production solution needs an explicit dependency decision
and broader numerical/performance validation.

## Follow-up latency experiments

`tune.py` isolates PV/QK call grouping and sweeps workspace targets, writing
results under `investigation/latency-tuning/`. `CT2_DIAG_PLAN_PV_GROUP` and
`CT2_DIAG_PLAN_QK_GROUP` enable those changes when present; both are off by
default. Grouped PV execution is accepted only if its generated local block
schedule equals the full plan. These changes passed exact comparisons but did
not produce a meaningful encoder latency improvement in the first sweep.

`thread_screen.py` checks the encoder before timing reduced thread counts. Both
4 and 6 threads changed encoder float patterns versus the 8-thread control and
were rejected under the strict compatibility criterion.

The optional **INT8 weight-packing** adapter targets decoding rather than
attention. Build it separately:

```bash
python3 tools/onednn_attention_plan/build.py --onednn-source /tmp/oneDNN-3.1.1 --output /tmp/ct2-attn-variants/hybrid-pack --int8-pack
```

This changes generated experimental copies of `backend.cc`, `primitives.cc`
and `model.cc`; normal CT2 sources/builds remain unchanged. Set
`CT2_DIAG_INT8_PACK=1` to enable it. **Compact is the default packing mode.**
It selects decoder weights, excluding embeddings and cross-attention KV
projections, and stores only their packed representations. Larger prompt GEMMs
execute in blocks of at most **5 queries** against that same representation;
the complete K reduction and existing dequantization are retained. The encoder
and cross-attention KV weights remain unpacked. Compatible layout, alpha 1 and
unsigned-input compensation are required by the compact prototype.

The generated model loader snapshots names rather than shared pointers, so
original matrices can be released as each replacement finishes. The earlier
shared-pointer snapshot kept originals alive throughout packing and increased
construction peak memory. Aliases still retain storage when needed (for
example, the embedding table shared with output projection). There is no
global pointer cache or independent buffer lifetime.

`CT2_DIAG_INT8_PACK_MODE=selective` keeps the decoder-only selection but retains
raw fallback copies; `all` reproduces the earlier all-linear hybrid experiment.
Those modes exist for comparison. The earlier 6.17 GiB peak-RSS all-linear
variant is rejected by the user's memory constraint. Do not treat it as the
default optimization.

The query extent is part of the private packing plan: a plan created for one
query did not compute all five output rows, despite returning success. A
five-query plan is therefore **never** reused for a larger query count. Packing
for 1500 queries gave correct small-query results but regressed several small
GEMMs. A universal replacement for CT2's MKL packing policy is not established.

Retaining both representations in the historical hybrid modes increases weight
storage. The compact mode removes those raw fallback copies. The benchmark
records peak RSS plus RSS after model load, warm-up and timed runs; load and
packing are outside transcription timing.
This private-API prototype has only been validated on the documented CPU/model
configuration and is not included in production builds.

`CT2_DIAG_INT8_PACK_VERIFY`, when present **before model loading**, retains raw
oracle copies solely for verification, recomputes **every packed INT8 GEMM**
(including tiled prompt calls) through the reference and aborts on any differing
int32 output. That diagnostic allocation is not the measured compact mode. Use
`hybrid_validate.py` with the hybrid library, this flag, and the same environment
as the prior 8-thread tests. It checks the real encoder and full transcription.
`hybrid_benchmark.py` runs regular attention/raw weights versus planned
attention/historical hybrid weights with verification disabled. The new
`memory_benchmark.py` compares compact decoder-only packing against the regular
control, with oracle copies disabled. It retains every output and enforces a
20% experiment ceiling on both peak and post-inference RSS ratios, substantially
stricter than the user's prohibition on doubling memory. This ceiling is a
test gate, not a user-approved general memory budget. Run validation and timing
sequentially.

On the measured large-v3 INT8 clip, compact packing plus planned attention took
54.61 ± 0.31 s versus 57.87 ± 0.13 s for the regular control (5.6% less time).
Peak RSS was 3.44 versus 3.28 GiB (+4.9%); post-inference RSS was 3.14 versus
3.03 GiB (+3.5%). Exact encoder, per-packed-GEMM and transcription/timestamp
checks passed. See the [memory-constrained report](../../benchmark-results/cpu-attention-epyc/investigation/memory-bounded-packing/report.md)
for the single-CPU/clip scope, timing setup and memory definitions.

## Broader conformance and application checks

`packing_conformance.cc` compares the actual compact adapter with independent
public oneDNN GEMMs across real medium/large-v3 widths, odd SIMD tails, padded
input/output strides, query counts on both sides of five, extreme INT8 values,
and accumulation modes. Its supported-context matrix includes 6,348 exact
numerical comparisons and 2,116 expected unsupported-beta rejections, with
address/undefined-behavior sanitizers, canaries and a packed-buffer immutability
check. The prebuilt oneDNN archive/JIT kernels are not sanitizer-instrumented;
canaries and numerical comparisons supplement that limitation. LeakSanitizer
is disabled; a separate model unload/reload smoke test measures RSS.

For standalone checks, compile each of `packing_conformance.cc`,
`packing_guards.cc` and `packing_thread_probe.cc` together with `int8_pack.cc`,
the project/adapter/oneDNN include directories, the matching `libdnnl.a`,
`-fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -ldl -lpthread`.
Run with `CT2_DIAG_INT8_PACK_MODE=compact`,
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`. The conformance executable takes an optional
`cross-threads` argument for expected thread-context rejections. These tests
were run sequentially before the application matrix.

The broader tests found two limits that are now guarded:

- Fractional beta can differ between public and private packed oneDNN drivers.
  Compact mode accepts only the checked beta values 0, 1 and -1; other values
  fail explicitly. Historical raw-retaining modes use the raw fallback.
  Whisper's tested linear layers use beta 0.
- Private packing includes thread partitions. Using an eight-thread plan in a
  one-thread context can return incomplete output. Compact mode requires the
  original fixed OpenMP thread context, and dynamic teams are excluded from
  INT8 packing. The [replica pool](../../include/ctranslate2/replica_pool.h)
  already configures matching thread counts for model loading and execution.
  Cross-context misuse now fails immediately; raw-retaining historical modes
  can fall back. `packing_thread_probe.cc` and `packing_guards.cc` test this.

`whisper_matrix.py` compares regular and compact results on additional audio,
beams 1/5/8, silence/noise, 75-second windowing, a long prompt, native batch two,
medium INT8, large-v3 float32 fallback, concurrent workers and model reloads.
It compares encoder float patterns, tokens, text, timestamps, probabilities and
score bits. These are correctness/memory checks with single-pass elapsed times,
not repeated performance measurements. Encoder tensors remain in `/tmp`;
committed artifacts contain hashes and comparison results.

The lifecycle test preserves ordinary unload-RSS observations separately. Both
the regular control and compact variant retain substantial allocator RSS after
unloading; a fixed 64 MiB untrimmed-growth test failed and is not presented as a
pass. The ownership diagnostic then uses `mkl_free_buffers` and `malloc_trim`
after model destruction, between four reload cycles, to check that live model
storage is no longer retained. That explicit cleanup is test-only and does not
change normal inference or promise immediate OS memory return on unload. See
the [extended validation report](../../benchmark-results/cpu-attention-epyc/investigation/extended-validation/report.md).

## Upstream references

- [oneDNN 3.1.1 GEMM plan and copy/compute pointers](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/gemm_info.hpp)
- [oneDNN full-shape algorithm selection and copy-based execution](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/gemm_driver.cpp)
- [oneDNN AVX no-copy execution and blocking policy](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/f32/jit_avx_gemm_f32.cpp)
- [OpenVINO's full-shape planning / tiled attention precedent](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/scaled_attn.cpp)
