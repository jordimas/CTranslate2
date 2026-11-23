# Recovering oneDNN attention performance without changing outputs

Date: 2026-11-23. Baseline code: `eb9f7fb0` (whole-head oneDNN compatibility fix). CPU: AMD EPYC 7B13, eight physical cores pinned to 0–7, eight threads, oneDNN 3.1.1, FP32 attention activations in both model compute types. The model is Whisper large-v3. These experiments do not change production source.

## Isolated encoder

First 30 seconds of `english_60s.mp3`, padded to 3000 feature frames / 1500 attention queries and keys, depth 64. One warm-up and three timed repetitions per process. All comparisons use the same features. The last encoder tensor from each variant is retained for comparison. Variants were measured sequentially, so small differences should not be overinterpreted.

| Compute | Variant | Mean ± sample SD (s) | Time reduction vs regular |
|---|---|---|---|
| int8_float32 | regular | 2.893 ± 0.050 | 0.0% |
| int8_float32 | head | 2.740 ± 0.004 | 5.3% |
| int8_float32 | workspace | 2.707 ± 0.008 | 6.4% |
| int8_float32 | tile192 | 2.871 ± 0.142 | 0.8% |
| int8_float32 | tile240 | 2.485 ± 0.014 | 14.1% |
| float32 | regular | 4.221 ± 0.062 | 0.0% |
| float32 | head | 4.039 ± 0.070 | 4.3% |
| float32 | workspace | 3.966 ± 0.038 | 6.0% |
| float32 | tile192 | 3.725 ± 0.034 | 11.7% |
| float32 | tile240 | 3.738 ± 0.022 | 11.4% |

`head` is the committed whole-head fix; `workspace` changes only scratch allocation; `tile192` and `tile240` use that scratch allocation plus the respective query-block size. Thus comparisons against `head` include both changes. Tile240 reduces encoder time by 9.3% versus `head` in int8 and 7.5% in float32.

## Attention operation and numerical checks

Alternating regular/fused operation timings, 20 measured samples per mode per process, three processes per variant (order reversed in the middle round), 20 heads, 1500 queries/keys, depth 64. Values below average the three process means. Each variant has its own paired regular baseline.

| Variant | Fused (ms) | Paired regular (ms) | Speedup |
|---|---|---|---|
| head | 29.72 | 34.96 | 1.18x |
| workspace | 28.65 | 35.66 | 1.24x |
| tile192 | 21.33 | 35.07 | 1.64x |
| tile240 | 22.19 | 36.27 | 1.63x |
| tile256 | 21.54 | 34.72 | 1.61x, numerically incompatible |

The randomized probe checked exact float bits for 20 seeds × three input magnitudes (0.01, 1, 10) × two thread counts (1, 8) × two tiles (192, 240): 240 comparisons, all matching. This covers the 1500 × 1500, depth-64 attention shape with two heads; it is not a proof for all values/shapes. The real encoder tests cover large-v3's 20 heads. Tile256 changes all 1,920,000 real int8 encoder output elements, with max difference 10.5271, reproducing the original issue. Earlier [shape sweeps](../block-shapes.log) contain counterexamples for 192/240 on other dimensions.

## Framework findings and next steps

See [framework approaches and hypotheses](framework-hypotheses.md) for primary sources and H1–H7. The most useful longer-term idea is OpenVINO-style full-shape planning followed by tiled execution, with explicit packing/reduction control. A supported microkernel API or backend change is needed; it does not automatically reproduce the existing SGEMM rounding.

A production tile optimization would need explicit shape, version, ISA and threading eligibility plus fallback and numerical regressions. We have not added such a policy on the basis of one CPU and one clip. Simply selecting tile240 for every oneDNN operation is contradicted by the shape sweeps.

## Reproduction artifacts

- `workspace.patch`: scratch-only change against the committed fix.
- `tile-experiment.patch`: scratch change plus diagnostic tile override. Apply only in an isolated experimental build.
- `bench.cc`, `tile-op-paired.log`, `workspace-paired.log`: operation measurements.
- `../probe.cc`, `tiles-random-bitwise.log`: randomized exact comparisons.
- `encoder_bench.py`, `run_encoder_bench.py`, `encoder-bench.json`: real encoder measurements.
- `profile_whisper.py`, `profile-whisper.json`: full-run encoder-time fraction.
- `whisper_e2e.py`, `e2e.json`: full transcription; the runner retains every warm-up and timed transcript and segment timestamp. Uses temperature 0, beam 5, English, one warm-up and three timed runs. Audio decoding and model load are outside timing.

The scripts use local `/tmp/ct2-attn-variants/{head,workspace,tile}` library snapshots through `LD_LIBRARY_PATH`, `CT2_USE_MKL=0` and `CT2_CPU_FUSED_ATTENTION=0/1`. Only the experimental tile library recognizes `CT2_DIAG_DNNL_BLOCK`. Rebuild each snapshot from the same configuration as `.cache/cpu-plan/branch-build`, applying the corresponding patch; the diagnostic override is not a supported runtime option.

## Full transcription results

60 seconds of English audio, temperature 0, beam 5, eight physical cores. One warm-up and three timed runs; same experimental library with fusion disabled/enabled, avoiding unrelated build differences. Order: regular then tile240 for int8; tile240 then regular for float32. Means ± sample SD.

| Compute | Regular (s) | Tile240 (s) | Time reduction | Speedup |
|---|---|---|---|---|
| int8_float32 | 57.69 ± 0.13 | 56.43 ± 0.31 | 2.2% | 1.022x |
| float32 | 49.48 ± 1.21 | 47.69 ± 1.23 | 3.6% | 1.038x |

All eight outputs per compute type (four per variant including warm-up) have identical text and segment start/end times. The final first-window encoder tensors also match bit-for-bit for every tested variant in both compute types; see `encoder-bitwise.json`. The workspace-only variant passes both focused fused-attention regression tests (`workspace-tests.log`).

The float32 timing spread is material relative to the measured gain, and each variant was run in one process. This is encouraging local evidence, not a statistically established cross-workload speedup. INT8 encoding occupied 15.95% of a separate profiled full run (9.35 of 58.61 s across three encoder calls), explaining why a 14.1% isolated encoder reduction produces only about a 2.2% total reduction. Work outside encoding includes decoding and other transcription processing.

No corpus-level accuracy claim is made: the local clip includes repeated hallucinated ending text, and compatibility here means preserving the regular implementation output, including that ending. No production dispatch change was made.
