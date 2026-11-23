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

## Upstream references

- [oneDNN 3.1.1 GEMM plan and copy/compute pointers](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/gemm_info.hpp)
- [oneDNN full-shape algorithm selection and copy-based execution](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/gemm_driver.cpp)
- [oneDNN AVX no-copy execution and blocking policy](https://github.com/uxlfoundation/oneDNN/blob/v3.1.1/src/cpu/x64/gemm/f32/jit_avx_gemm_f32.cpp)
- [OpenVINO's full-shape planning / tiled attention precedent](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/scaled_attn.cpp)
