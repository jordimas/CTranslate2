# Why fused attention changes int8 outputs on EPYC

Follow-up: a oneDNN-specific whole-head fused path has since been implemented; see [the implementation and validation results](head-fusion/report.md). The diagnosis below describes the original query-blocked kernel.

Investigation on 2026-11-23, existing branch CPU library (`877d7efe`), oneDNN 3.1.1 / MKL 2025.3, EPYC 7B13, AVX2, 8 physical cores. Inputs: `english_60s.mp3`, Whisper large-v3. Encoder comparisons use the first 30 seconds, padded/truncated to 3000 feature frames. This is a numerical diagnosis, not a performance benchmark or an accuracy evaluation.

## Finding

The remaining difference is reproduced by **shape-dependent rounding in oneDNN float32 GEMM**, amplified by activation quantization. Attention itself uses float32 in both compute types. Blocking changes the GEMM row count from 1500 to at most 256. The same softmax implementation does not guarantee bit-identical results when its input GEMMs differ.

The libraries were built with both MKL and oneDNN, but AMD defaults to oneDNN: `mayiuse_mkl_init` in `src/cpu/backend.cc` selects MKL by default only on genuine Intel CPUs. Verbose logs confirm DNNL for both SGEMM and int8 GEMM in this reproduction. `CT2_USE_MKL=1` explicitly selects MKL.

## Evidence

1. `probe.cc` compares regular and blocked attention on identical seeded Gaussian Q/K/V tensors (1 batch, 2 heads, 1500 queries/keys, depth 64). With oneDNN, the attention output differs in 2810/192000 values, maximum 1.04308e-7. The discrepancy also occurs with one thread. Separately changing only the matrix blocking reproduces differences in QK and in PV using the same probabilities. With MKL forced, all three comparisons are bit-identical. This reproduces the initiating numerical difference without int8 weights, model loading, decoding, or sampling.

2. Temporary instrumentation compares fused attention with the regular MatMul/SoftMax/MatMul sequence on the **same real layer inputs**, then quantizes the merged-head outputs using the same rounding and unsigned-shift settings as this model. On the fused int8 trajectory it records a quantized-byte difference at attention call 6 (the seventh encoder layer), despite a maximum attention-output difference of only 2.16067e-7 at that call.

3. A reference-output control computes both paths, logs their difference, then feeds the regular result into the next layer. At attention call 13, one value illustrates the threshold crossing:

   | | fused | regular |
   |---|---:|---:|
   | attention value | -0.0192635097 | -0.0192635134 |
   | per-row scale | 337.425690 | 337.425720 |
   | scaled value, including +128 unsigned shift | 121.500000 | 121.499992 |
   | quantized byte | 122 | 121 |

   Both the attention value and its row's quantization scale can change. Rows with equal quantized bytes can also have different scales, affecting subsequent dequantization.

4. Final encoder outputs for the same first-30-second input:

   | backend / compute | different elements | max absolute difference |
   |---|---:|---:|
   | oneDNN / float32, fused vs regular | 1,890,839 / 1,920,000 | 0.00124416 |
   | oneDNN / int8_float32, fused vs regular | 1,920,000 / 1,920,000 | 10.52707 |
   | MKL / int8_float32, fused vs regular | 0 / 1,920,000 | 0 |
   | oneDNN / int8_float32, replace fused results with regular results | 0 / 1,920,000 | 0 |

   The installed, uninstrumented fused library produces exactly the same encoder output as the instrumented fused run. The instrumentation did not introduce the reproduced discrepancy. Replacing only attention results with regular results restores the regular encoder exactly, locating the source in the attention-path numerical differences.

5. Full 60-second transcription with `CT2_USE_MKL=1`, int8_float32, temperature 0, beam 5, 8 threads, one separate process per switch setting: **identical transcripts**, 247 words each. See `mkl-transcripts.json` and `mkl-transcripts.log`. This checks one transcription per mode, not repeated-process determinism or performance.

## Implications

- The new block-wise softmax removed the original online-softmax difference, but universal bit identity also requires GEMM results invariant to blocking. That condition fails for oneDNN on this tested CPU/build.
- A race is not needed to explain this reproduction: a single-thread fixed-input probe already differs. This does not constitute a general race audit.
- Large pairwise transcript disagreement is not a measurement of absolute accuracy. The previous temperature-zero transcript difference is concentrated in an added repetitive ending, which can magnify a small upstream perturbation.
- Forcing MKL removes the tested encoder discrepancy, but changes other GEMM paths too. It is a diagnostic control, not a validated performance recommendation.
- A fix needs either a GEMM strategy that preserves the regular path's results for the relevant shapes, or a backend/compute-type fallback when strict output compatibility is required. Increasing softmax precision alone does not address GEMM differences. Neither fix is implemented here.

## Artifacts and reproduction

`probe.cc`, `probe-dnnl-1.log`, `probe-dnnl-8.log`, and `probe-mkl-8.log` contain the isolated reproduction. Build the probe with:

```
c++ -O2 -std=c++17 -fopenmp -Iinclude \
  benchmark-results/cpu-attention-epyc/investigation/probe.cc \
  -L.cache/cpu-plan/branch-install/lib \
  -Wl,-rpath,/home/jupyter/CTranslate2/.cache/cpu-plan/branch-install/lib \
  -lctranslate2 -o /tmp/ct2-attention-probe
CT2_USE_MKL=0 taskset -c 0-7 /tmp/ct2-attention-probe 1
CT2_USE_MKL=0 taskset -c 0-7 /tmp/ct2-attention-probe 8
CT2_USE_MKL=1 taskset -c 0-7 /tmp/ct2-attention-probe 8
```

`diagnostic-instrumentation.patch` records the temporary source instrumentation. It was built only into `.cache/cpu-plan/branch-build/libctranslate2.so`; source changes were then restored. Installed benchmark libraries remain unchanged. `run_encoders.py`, `encode.py`, `encoders.log`, per-mode logs, `replaced.log`, and `uninstrumented.log` preserve the model-level controls. Encoder arrays are in `/tmp/ct2-encoder-*.npy` and are not committed. `transcribe_control.py` checks the full 60-second temperature-zero transcript with MKL in separate processes.
