# Extended compact oneDNN packing validation

Date: 2026-11-23. Same AMD EPYC 7B13 / oneDNN 3.1.1 / effective AVX2 environment. These are tests of the opt-in experimental adapter, not production activation or a supported public oneDNN API. The user requested committing only after confirming the supported implementation works.

## Results

- Full CPU suite with the guarded experimental library enabled: **371 passed, 18 skipped**.
- Standalone adapter conformance: **6,348 exact numerical cases**, **2,116 expected unsupported-beta rejections**, zero failures.
- Mixed-thread-context misuse: **576 expected rejections**, zero failures.
- Additional rejection/raw-fallback checks: **18 passed**.
- AddressSanitizer/UndefinedBehaviorSanitizer, packed/output canaries, input preservation and packed-buffer immutability checks passed. LeakSanitizer was disabled; the prebuilt oneDNN archive and JIT kernels are not sanitizer-instrumented. Canaries and reference comparisons supplement that limitation.
- **27 application comparisons** across five groups match exactly, including encoder float patterns, tokens, text, timestamps, probability/score bits and native batched generation results.

| Application group | Compared cases | Exact | Peak RSS ratio, compact/control | Post-run RSS ratio |
|---|---|---|---|---|
| large_int8 | 11 | yes | 1.011 | 1.043 |
| medium_int8 | 3 | yes | 1.004 | 1.097 |
| large_float32 | 2 | yes | 1.000 | 1.034 |
| concurrent_int8 | 3 | yes | 1.014 | 0.834 |
| lifecycle_int8 | 8 | yes | 1.011 | diagnostic cleanup; see below |

The application matrix covers:

- large-v3 INT8: independent JFK speech, beams 1/5/8, silence, deterministic noise, a 75-second multi-window recording, a long initial prompt and native batch size two.
- medium INT8: JFK and the full local English clip.
- large-v3 float32: the INT8 packing path stays disabled, while the attention path is checked against its regular control.
- Concurrent large-v3 INT8: two workers with four threads each, using matching packing/execution contexts.
- Four medium-model destroy/reload cycles in each configuration; outputs and encoder hashes also remain identical between cycles.

The 27 comparison count includes encoder probes and the native-batch test; it is not 27 independent recordings. There are 16 high-level transcription comparisons, 10 standalone encoder comparisons and one native batch comparison with its own encoder check. New single-pass elapsed times are diagnostic only and are not new repeated speedup estimates. The earlier repeated 60-second performance/memory comparison remains in [the memory report](../memory-bounded-packing/report.md).

## Bugs/limits discovered and guarded

### Fractional beta

The initial supported-context sweep found 660 mismatching cases, all with beta=0.5. Other tested beta values (0, 1, -1) were exact. The private packed and public drivers can round fractional accumulation differently. Compact mode now accepts only those checked integer beta values and throws explicitly otherwise; historical raw-retaining modes use their raw reference fallback. Whisper's tested linear layers use beta=0, so the prior Whisper results are not invalidated. `conformance-before.log` preserves the negative evidence.

### Packing/execution thread context

The mixed-context sweep stalled and was interrupted. A narrowed independent comparison then reproduced 220 numerical failures in 960 total cases, involving eight-thread packing followed by one-thread compute; some output rows were not computed. A bare call could return success, so absence of an API error is insufficient. We do not claim a deterministic deadlock: the isolated numeric failure is the reproducible acceptance issue.

The packed header now records the original fixed OpenMP context. Compact mode rejects mismatched or dynamic contexts before dispatch; raw-retaining historical modes can fall back. The normal CT2 replica pool already sets the same number of threads for loading and execution (`include/ctranslate2/replica_pool.h`). Direct mixed-context reuse is intentionally unsupported, even where an individual private plan happens to tolerate it. `cross-narrow-before.log` and the before/after thread probes preserve this evidence.

## Unload/RSS investigation

The original fixed 64 MiB untrimmed-RSS growth gate failed after model destruction. It failed as a useful ownership metric for the regular baseline too: both configurations retain substantial allocator memory. Those failed observations are preserved in `lifecycle-untrimmed*.json/log`; they are not relabeled as passing tests.

After four no-cleanup cycles, ordinary unloaded RSS was about 1.75 GiB for regular versus 1.86 GiB for compact. Calling `mkl_free_buffers` followed by `malloc_trim(0)` after all models were destroyed reduced them to about 153 MiB and 162 MiB. That demonstrates releasable allocator retention rather than a live model-sized packed/raw copy in this diagnostic. It does not prove a universal leak-free bound.

The final ownership diagnostic repeats four cycles with that cleanup **between** cycles. Compact cleaned RSS grew only 9.3 MiB between first and last cycles and ended around 175 MiB. The test gate is <64 MiB growth after diagnostic cleanup plus release of at least 25% of the pre-cleanup retained RSS. This procedure is test-only: normal inference/unload was not changed to call heap trimming. Ordinary unload RSS does not immediately return to the cleaned floor, and no untrimmed plateau guarantee is claimed.

## Reproduction and scope

[Adapter instructions](../../../../tools/onednn_attention_plan/README.md) describe the isolated SDK build. `packing_conformance.cc`, `packing_guards.cc`, `packing_thread_probe.cc` and `whisper_matrix.py` generate the new checks. The final lifecycle check was rerun on its own after investigating allocator retention; the other four groups retained their already passing comparisons.

Audio source/hash and synthetic generation details are in `audio-fixtures.json`. The JFK sample is fetched from the faster-whisper repository; the other speech input is the existing local clip. Tensor comparison arrays stay under `/tmp/ct2-extended-validation`; only hashes/metadata/results are committed.

This validates the documented current CPU/library/model configurations. It does not establish corpus-level ASR accuracy, other ISAs/oneDNN versions, arbitrary GEMM accumulation modes, changed thread contexts, or unbounded model-lifetime behavior. Production source files/builds remain unchanged.
