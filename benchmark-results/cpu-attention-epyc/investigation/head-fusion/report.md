#### oneDNN conditional retaining fused attention (2026-11-23)

Implemented a backend-specific path: oneDNN fuses the attention stages per whole head, keeping the full query and key dimensions of the regular GEMMs, including masked keys. Other backends keep the existing query-blocked kernel. This preserves fusion without switching GEMM providers. oneDNN score scratch is capped at 64 MiB per worker; shapes exceeding that cap fall back to regular attention. The existing limit of 8192 keys remains.

For large-v3 (20 heads, 1500 queries/keys, 8 workers), score workspace is about 72 MB instead of the regular 180 MB tensor. This uses more scratch than query blocking (about 1.5 MB per worker for this shape), trading some memory savings for numerical compatibility.

Validation on this EPYC / oneDNN 3.1.1 build: 30 fixed-input shapes match regular attention bit-for-bit; masked exact-comparison regression cases pass; real large-v3 encoder outputs match bit-for-bit in both int8_float32 and float32. The new regression fails against the original installed library and passes against the patched library. Full CPU suite: 371 passed, 18 skipped. Other hardware/backends were not rerun.

Same benchmark setup as the controlled EPYC rerun: 8 physical cores, beam 5, 60 s of English audio, temperature 0, one warm-up and three timed runs. Non-Whisper workloads are unchanged. Medium was not rerun. Means ± sample standard deviations:

| model | compute | fused (s) | unfused (s) | speedup | output agreement |
|---|---|---|---|---|---|
| large-v3 | int8_float32 | 57.73 ± 0.05 | 57.65 ± 0.20 | 1.00x | identical |
| large-v3 | float32 | 49.12 ± 0.35 | 47.77 ± 1.93 | 0.97x | identical |
| nllb | int8_float32 | 5.78 ± 0.30 | 5.80 ± 0.03 | 1.00x | identical |
| nllb | float32 | 11.62 ± 0.62 | 11.90 ± 1.38 | 1.02x | identical |
| xlm-roberta | int8_float32 | 0.43 ± 0.00 | 0.48 ± 0.02 | 1.10x | identical |
| xlm-roberta | float32 | 0.66 ± 0.01 | 0.66 ± 0.00 | 1.00x | identical |
| gemma-3-4b | int8_float32 | 20.08 ± 0.69 | 20.01 ± 0.66 | 1.00x | identical |

Raw data, logs, shape probes, and validation runner: `benchmark-results/cpu-attention-epyc/investigation/head-fusion/`. As before, the benchmark retains only the last timed output in each mode; these comparisons do not establish corpus-level accuracy or bit identity on untested oneDNN versions and CPUs.
