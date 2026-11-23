# Further Whisper latency investigation

Baseline investigation/prototype commit: `e828c1c8` (pushed to `origin/perf/cpu-attention-transpose-softmax`). Date: 2026-11-23. Same AMD EPYC 7B13, oneDNN 3.1.1 / effective AVX2, eight physical cores pinned to 0–7. Production sources/build remain unchanged; all new kernels/adapters are in isolated experimental libraries.

## Attention overhead and workspace tuning: rejected

Two exact-preserving changes were isolated: let the original QK kernel execute its own M loop instead of calling it separately for each key microblock; group PV private-driver calls only when the resulting local boundaries exactly match the full-query plan. Both together passed 228 randomized exact QK/attention/quantization comparisons and the two existing fused-attention regressions. Neither produced a meaningful latency improvement.

Attention operation: 20 heads, lengths 1500/1500, depth 64, alternating regular/fused measurements, three processes per variant, 30 samples per mode per process. Table values are mean ± sample SD of the three **process means**, not of the 90 individual samples.

| Variant | Fused ms |
|---|---|
| original | 24.684 ± 0.217 |
| pv | 24.726 ± 0.131 |
| qk | 24.608 ± 0.088 |
| both | 24.680 ± 0.131 |

A one-process workspace sweep selected target 128 (effective outer tiles mostly 96) by a very small operation-time margin. That selection did not translate into a reliable encoder improvement. Both grouping flags remain off by default.

| Compute | Variant | Encoder mean ± sample SD, s | Changed float patterns |
|---|---|---|---|
| int8_float32 | regular | 2.915 ± 0.057 | 0 |
| int8_float32 | original | 2.565 ± 0.007 | 0 |
| int8_float32 | grouped256 | 2.571 ± 0.017 | 0 |
| int8_float32 | grouped_best | 2.554 ± 0.005 | 0 |
| float32 | regular | 4.203 ± 0.025 | 0 |
| float32 | original | 3.848 ± 0.036 | 0 |
| float32 | grouped256 | 3.853 ± 0.021 | 0 |
| float32 | grouped_best | 3.943 ± 0.115 | 0 |

## Reduced thread counts: rejected for strict compatibility

Before timing full transcription at 6/4 threads, the runner compared the real first-window encoder against the 8-thread reference. Six threads changed all 1,920,000 float patterns; four changed 1,919,998. Both candidates were rejected before full transcription. This is a compatibility result, not evidence that recognition accuracy would necessarily deteriorate. No latency result is claimed for them.

## Persistent INT8 weight packing

CT2 currently enables persistent GEMM weight packing for MKL but not oneDNN (`src/cpu/backend.cc`). The installed oneDNN 3.1.1 archive contains private INT8 pack/compute interfaces. This targets repeated linear GEMMs during decoding, extending beyond the earlier encoder-attention optimization.

### Important negative controls

- Pack for one query, then execute five: the private API returned success but left the fifth output row unwritten in the probe (1,280 changed int32 outputs). A direct translation of CT2's MKL pack-for-one policy is therefore rejected.
- Pack for five queries, then execute sixteen: rows 8–15 were unwritten in the probe. Packed shape metadata must not be reused indiscriminately.
- Pack for 1500 queries, then execute 1/5/16/1500: all 16 tested combinations (two weight ranges, two output widths, four query counts) match the reference, but several small-query GEMMs regress substantially. Correctness alone is insufficient.

Raw controls and timings are in `pack-probe*.log`. Timing samples are microbenchmarks, not end-to-end speedup estimates; timings associated with mismatching outputs are invalid as optimization evidence. The probe exits nonzero on the first mismatch.

### Hybrid implementation

An optional experimental build (`build.py --int8-pack`) generates modified copies of the backend/primitives objects. It keeps both a five-query packed representation and a raw matrix inside one model-owned allocation. Calls with at most five queries, alpha 1, supported layout and u8 compensation use the packed representation. Larger calls (including encoder/prefill) and unsupported cases use the original oneDNN GEMM on the retained raw matrix. Normal source files are not changed. No global pointer cache is introduced; both representations follow model lifetime.

This is INT8-only, private oneDNN 3.1.1 / AVX2 functionality. It is not a supported public packing API or a production backend change. Keeping raw and packed forms increases weight storage. Encoder weights currently receive a packed form even though their large-query calls use the raw fallback; restricting packing to layers that actually benefit is an untested memory/setup optimization.

### Real-model numerical validation

The large-v3 first-window encoder matches all 1,920,000 float patterns. A full 60-second transcription also matches the regular reference text and segment timestamps, with **every packed INT8 GEMM recomputed through the raw reference and compared exactly**. No int32 mismatch occurred. The diagnostic transcription took 67.74 s, including the duplicate computation; this is not a performance measurement. See `hybrid-validation.json/log`.

### Full-transcription timing

The same experimental library runs with regular attention/raw weights versus planned attention/hybrid weights, verification disabled. Both use temperature 0, beam 5, English, eight threads, one warm-up and three timed runs. Model load and audio decoding are outside the timer. The hybrid runs first, followed by the regular control; each configuration uses one process.

| Configuration | Transcription mean ± SD, s | Process peak RSS, GiB | Model load, s |
|---|---|---|---|
| regular | 58.18 ± 0.13 | 3.23 | 3.17 |
| hybrid | 53.77 ± 0.45 | 6.17 | 3.45 |

The hybrid reduces total time by **7.6% (1.082x)** versus the fresh regular control. All eight outputs (warm-up plus three timed runs per configuration) have identical text and segment timestamps. This gain is from the combined planned-attention/hybrid-weight configuration; the run does not include a separate attention-only control, so it does not isolate weight packing's contribution from attention fusion. The earlier attention-only full transcription was 56.87 s in a separate run.

Peak RSS increases by **2.93 GiB**, from 3.23 to 6.17 GiB. This is process peak memory, including construction and inference, not steady-state retained model memory. Model-load values are individual observations, not repeated latency statistics. Both raw and packed eligible weights remain model-owned; duplicated representations and packing temporaries matter. Encoder-only packing is currently unused on this workload and could be avoided in a future memory/setup optimization.

This is one CPU and one clip, whose reference output contains repeated ending text. The same ending is present in every candidate/control run, so the gain is not explained by a shorter generated transcript. No corpus-level recognition accuracy claim or broad hardware/shape performance claim is made. Float32 was not rerun: the new packing optimization applies only to INT8. The private API dependency and memory cost keep this an experimental result, not a production recommendation.

## Reproduction

New experiment code and commands are documented in [the prototype README](../../../../tools/onednn_attention_plan/README.md). `tune.py`, `pack_probe.cc`, `thread_screen.py`, `hybrid_validate.py` and `hybrid_benchmark.py` generate this directory's evidence. Run CPU measurements sequentially. The committed original prototype/results are separate from this follow-up experiment.
