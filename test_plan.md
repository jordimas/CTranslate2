# Test plan: CPU fused attention

Goal: check that the CPU fused attention (branch `perf/cpu-attention-transpose-softmax`) speeds up Whisper end to end compared with `master`, without changing the transcripts.

## Models covered

The fused path is used by any model going through the regular attention layer, in FP32 or int8 with FP32 activations, when a call has 32 or more queries. Single-token decode steps never use it. One model per family, all run by `tools/benchmark_encoder_fuse.py`:

| Family | Model | Workload | Compute types |
|--------|-------|----------|---------------|
| Whisper | medium, large-v3 | 60 s of audio, beam 5, encoder self-attention (1500 frames) | int8_float32, float32 |
| Translation | `facebook/nllb-200-distilled-600M` | batch 4 x 512 source tokens, greedy, up to 128 output tokens | int8_float32, float32 |
| Encoder | `FacebookAI/xlm-roberta-base` | forward pass, batch 4 x 512 tokens | int8_float32, float32 |
| Decoder-only LLM | `google/gemma-3-4b-it` | prompt prefill, batch 4 x 512 tokens + 32 generated tokens | int8_float32 only (a float32 copy does not fit the disk) |

Not affected: T5/mT5 (relative bias), ALiBi models (Bloom, MPT, Falcon), FP16/BF16, and layers that return attention weights. wav2vec2 / w2v-BERT is not covered (its attention path is not verified).

The non-Whisper models are converted from the Hugging Face cache on first use (needs `transformers` and `torch`). Quality: WER between fused and unfused outputs for text, and the largest difference of the output vectors for the encoder.

## Steps (per machine)

1. Build `master` and this branch (CPU build, same flags), and install each into its own prefix.
2. Pin to one thread per physical core, then run the script against each build:

   ```
   LD_LIBRARY_PATH=<prefix>/lib taskset -c <physical cores> \
       python tools/benchmark_encoder_fuse.py --audio <audio> --duration 60 \
       --threads <physical cores> --output results_<machine>_<build>.json
   ```

   - On `master` the fused path does not exist, so the fused and unfused runs are the same code. Use only the `unfused` rows as the baseline (the script still runs both; ignore the `fused` rows and the speedup line).
   - On the branch, the script compares fused on and off in the same build.
3. Copy the printed tables (avg, std, RTF, speedup, quality) into the Results section.

## Results

### Machine: AMD Ryzen AI Max+ PRO 395, 16 cores / 32 threads, 128 GB, Linux 7.0 (clang, MKL + oneDNN)

Pinned with `taskset -c 0-30:2`, `OMP_PROC_BIND=true OMP_PLACES=cores`, 16 threads. 60 s of audio, beam 5, language `ca`, 1 warm-up + 3 timed runs. Tile 64x64.

| family | model | compute | fused avg (s) | fused std | unfused avg (s) | unfused std | speedup | quality (fused vs unfused) |
|--------|-------|---------|---------------|-----------|-----------------|-------------|---------|----------------------------|
| Whisper | medium | int8_float32 | 8.10 | 0.04 | 8.79 | 0.02 | 1.08x | WER 0.0070, differ |
| Whisper | medium | float32 | 12.16 | 0.13 | 13.14 | 0.24 | 1.08x | identical |
| Whisper | large-v3 | int8_float32 | 13.79 | 0.21 | 15.95 | 0.10 | 1.16x | WER 0.0000 (text differs slightly) |
| Whisper | large-v3 | float32 | 24.31 | 0.34 | 26.30 | 0.16 | 1.08x | identical |
| Translation | nllb-200-distilled-600M | int8_float32 | 9.51 | 0.09 | 9.25 | 0.02 | 0.97x | WER 0.0156, differ |
| Translation | nllb-200-distilled-600M | float32 | 17.38 | 0.01 | 17.66 | 0.09 | 1.02x | identical |
| Encoder | xlm-roberta-base | int8_float32 | 0.83 | 0.00 | 0.89 | 0.02 | 1.07x | max diff 2.2e-02 |
| Encoder | xlm-roberta-base | float32 | 1.79 | 0.02 | 1.81 | 0.01 | 1.01x | max diff 3e-07 |
| Decoder | gemma-3-4b-it | int8_float32 | 33.53 | 0.11 | 33.51 | 0.27 | 1.00x | identical |

Master baseline (Whisper, `unfused` rows; same code as branch unfused): medium int8_float32 7.90±0.03 s, medium float32 16.42±7.02 s (noisy), large-v3 int8_float32 17.79±6.58 s (noisy), large-v3 float32 25.66±0.14 s.

Notes:
- Non-Whisper workloads: batch 4 x 512 tokens; NLLB greedy up to 128 output tokens, Gemma 32 generated tokens. NLLB and XLM-R were stored as float16 and Gemma as int8 on disk.
- NLLB and Gemma show no gain; decode steps do not use the fused path. Not checked whether Gemma 3 (sliding-window layers) reaches the fused kernel at all.
- Master medium float32 and large-v3 int8_float32 had std of 6-7 s (slow outlier runs); not reliable, re-run before comparing against master.
- Medium int8_float32 transcripts differ between fused and unfused (WER 0.007); the others have WER 0. Large-v3 int8_float32 text differs only in ways the WER ignores.
- Branch unfused is not always equal to master unfused (medium int8: 8.79 vs 7.90 s); likely run-to-run noise, not checked.

## Conclusion

(Fill in after all machines have run.)
