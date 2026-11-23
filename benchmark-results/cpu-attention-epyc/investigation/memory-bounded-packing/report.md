# Reducing oneDNN packing memory for Whisper

Date: 2026-11-23. AMD EPYC 7B13, eight physical cores pinned to 0–7, oneDNN 3.1.1 / effective AVX2, Whisper large-v3 int8_float32. The user explicitly rejects doubling memory. The earlier all-linear hybrid's 6.17 GiB peak RSS is rejected on that basis, despite its measured latency gain.

## Changes

The new compact prototype addresses both retained storage and construction peak:

1. Select decoder weights only: exclude every encoder weight, embeddings used for lookup, and the 32 cross-attention KV projection matrices. The policy selects 192 stored decoder matrices plus the output projection alias. The shared embedding table still retains the storage required for lookup.
2. Store only the packed representation of selected weights. Larger prompt GEMMs execute in query blocks of at most five, using the same private packing plan and complete K reduction. This is INT8/int32 GEMM tiling, not a change to the FP32 attention GEMMs that caused the original discrepancy. No raw fallback matrix is retained in the measured compact configuration.
3. Snapshot variable names in the generated model-loading loop instead of copying the map of shared pointers. The original loop kept all source matrices alive until packing completed. The new loop releases each replaced original when its remaining aliases permit it.
4. Measure peak RSS and current RSS after model load, warm-up, and all timed runs. Model load/packing and audio decoding remain outside transcription timing.

The adapter, generated model loader and backend changes remain isolated under `/tmp/ct2-attn-variants/memory-pack`; normal production source files/builds are unchanged. Compact becomes the default experimental INT8 packing mode. Historical all-linear/raw-copy modes require explicit selection and are retained only to reproduce prior experiments.

## Numerical validation

The real first-window encoder matches all 1,920,000 float patterns. A full 60-second transcription also matches the original text and segment timestamps. During that diagnostic pass, every packed INT8 GEMM—including larger prompt operations executed as five-query chunks—is recomputed as one regular raw-weight reference operation and compared exactly. No int32 mismatch occurred.

**Verification-only oracle:** enabling `CT2_DIAG_INT8_PACK_VERIFY` before model loading deliberately retains raw copies for the numerical oracle. Those copies are absent in the timing/memory run, whose log explicitly reports `mode=compact raw_copy=0`. The 69.46-second diagnostic pass includes duplicate computation and is not a latency or compact-memory measurement.

## End-to-end timing and memory

Same 60-second English clip, temperature 0, beam 5, eight threads, one warm-up and three timed repetitions per configuration. Compact runs first, then the fresh regular-attention/raw-weight control using the same library. Model load/packing and audio decoding are outside the timer. All eight texts and segment timestamps match, including warm-ups.

| Configuration | Time mean ± SD, s | Peak RSS, GiB | Post-inference RSS, GiB |
|---|---|---|---|
| regular | 57.87 ± 0.13 | 3.28 | 3.03 |
| compact | 54.61 ± 0.31 | 3.44 | 3.14 |

Compact packing plus planned attention reduces total time by **5.6% (1.060x)** versus the fresh regular control. Peak RSS increases **4.9%, or 0.16 GiB**, and RSS after inference increases **3.5%, or 0.11 GiB**. Both pass the experiment's stricter 20% rejection gate; memory is nowhere near doubled. These are measurements from two processes on one clip, not a universal bound on other workloads/models.

The earlier all-linear hybrid had a 6.17 GiB peak and is rejected. Compact avoids its raw fallback copies and unused packed encoder/KV weights. The loading-loop lifetime fix also removes the shared-pointer snapshot that retained old matrices during replacement. The new selection includes 193 decoder weights; output projection remains tied to the embedding table, whose original lookup storage is still necessary.

Additional current-RSS observations (GiB):

| Configuration | After model load | After warm-up | After timed runs | Model load, s |
|---|---|---|---|---|
| regular | 1.69 | 2.97 | 3.03 | 2.99 |
| compact | 1.82 | 3.14 | 3.14 | 3.05 |

Peak RSS is process high-water memory, including construction and inference. Post-inference RSS is a current process snapshot after three completed transcriptions, not a proof of a universal steady-state maximum. Model-load timings are single observations. The gain is for the combined attention-plan/compact-packing configuration; this run does not include an attention-only control.

The measured library used **explicit compact mode**. After timing, the experimental default was changed from historical all-linear mode to compact and unknown mode names were made errors. The measured binary is preserved separately; a rebuilt-default smoke check verifies `mode=compact raw_copy=0` and exact encoder output. These configuration-default changes do not change the explicitly selected numerical path used in the full benchmark.

## Scope and limitations

This still uses private oneDNN 3.1.1 interfaces and is limited to the validated CPU/layout/compute configuration. Compact selected-weight GEMMs require the supported layout, alpha 1 and u8-input compensation; unsupported selected-weight calls fail explicitly rather than consume an invalid private plan. Larger query counts are processed through bounded five-query calls, never passed wholesale to the undersized packing metadata.

The unit of validation is exact reference compatibility on these inputs. It does not establish corpus-level recognition accuracy or compatibility across oneDNN releases, ISAs, thread counts or all inputs. The local reference clip has repeated ending text; the same ending must appear in both modes to avoid measuring a different decode workload. Float32 packing is not part of this experiment.

## Reproduction and artifacts

[Prototype instructions](../../../../tools/onednn_attention_plan/README.md) describe the isolated build. Build with `--int8-pack`, choose `CT2_DIAG_INT8_PACK_MODE=compact`, run `hybrid_validate.py --output benchmark-results/cpu-attention-epyc/investigation/memory-bounded-packing` with the verification flag enabled, then run `memory_benchmark.py` with it disabled. Run CPU measurements sequentially.

Files: `decoder-weight-inventory.json`, `selection.json`, `validation.log`, `hybrid-validation.json`, and `e2e*.json/log`. `selection.json` describes the stored weights and alias selection; the C++ predicate plus model inventory determine the selected set. Normal and measured-library hashes and final summaries are recorded after completion.
