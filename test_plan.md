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

   Whisper must pass `temperature=0.0` explicitly to `model.transcribe` (as `tools/benchmark_encoder_fuse.py` does). Do not use faster-whisper's default temperature sequence: it can retry with nonzero-temperature sampling, changing transcripts and decode times. Use the same fixed temperature for fused, unfused, and master runs, and record it with the results. Earlier results that did not fix the temperature must be identified as fallback-enabled and cannot establish deterministic output agreement.

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

#### Rerun with the final block-wise kernel (2026-11-22, branch `d4403ff8d`, same source as `bc96a06c7`)

Same machine and setup as above (16 threads, `taskset -c 0-30:2`, `OMP_PROC_BIND=true OMP_PLACES=cores`, 60 s of audio), branch and master built and run one after the other. Master (`build/plan_master`) has no fused path, so its unfused row is the baseline. Mean +- std in seconds, 3 timed runs unless stated. The change is the time of the fused branch (negative = faster). NLLB, XLM-R and Gemma ran with transformers 4.57.6 because `tools/benchmark_encoder_fuse.py` fails on the transformers 5 `NllbTokenizer` (no `build_inputs_with_special_tokens`). The page `perf.html` has the same table.

| family | model | compute | branch unfused | branch fused | fused vs unfused | master unfused | fused vs master | quality (fused vs unfused) |
|--------|-------|---------|----------------|--------------|------------------|----------------|-----------------|----------------------------|
| Whisper | medium | int8_float32 | 9.39 +- 0.05 | 8.31 +- 0.03 | -11.5% | 10.11 +- 0.21 (5 runs) | -17.8% | identical |
| Whisper | medium | float32 | 13.19 +- 0.06 | 12.52 +- 0.01 | -5.1% | 12.61 +- 0.05 | -0.7% | identical |
| Whisper | large-v3 | int8_float32 | 14.45 +- 0.05 | 13.11 +- 0.05 | -9.3% | 15.46 +- 0.03 | -15.2% | identical |
| Whisper | large-v3 | float32 | 25.90 +- 0.17 | 23.47 +- 0.08 | -9.4% | 23.22 +- 0.08 | +1.1% | identical |
| Translation | nllb-200-distilled-600M | int8_float32 | 9.41 +- 0.02 | 9.26 +- 0.01 | -1.6% | 9.33 +- 0.01 | -0.8% | identical |
| Translation | nllb-200-distilled-600M | float32 | 17.68 +- 0.01 | 17.55 +- 0.05 | -0.7% | 17.94 +- 0.07 | -2.2% | identical |
| Encoder | xlm-roberta-base | int8_float32 | 0.92 +- 0.03 | 0.77 +- 0.00 | -16.3% | 0.93 +- 0.04 | -17.2% | max diff 0 |
| Encoder | xlm-roberta-base | float32 | 1.85 +- 0.01 | 1.73 +- 0.01 | -6.5% | 1.87 +- 0.01 | -7.5% | max diff 0 |
| Decoder | gemma-3-4b-it | int8_float32 | 33.70 +- 0.05 | 33.57 +- 0.10 | -0.4% | 34.04 +- 0.14 | -1.4% | identical |

- Every fused vs unfused comparison gives identical outputs (WER 0.0000, or max difference 0 for the encoder). The first-kernel table above had WER differences on Whisper medium int8 (0.0070) and NLLB int8 (0.0156); they are gone, as the block-wise softmax is meant to give.
- Gains against master are real for the Whisper int8 and XLM-R int8 rows (15-18%); float32, NLLB and Gemma are within 0-2% of master, which is inside the noise. The earlier 1.29x for large-v3 float32 came from a noisy unfused row.
- The large-v3 float32 row is not settled. A first 3-run pass gave the numbers above; a 5-run repeat gave 30.73 +- 5.62 (unfused) and 28.44 +- 7.42 (fused), 22.5% slower than master, with a load average of about 16 from other jobs on the machine. Both builds slowed together, which points to interference, but this was not proven. It needs a rerun on an idle machine, alternating the builds.
- Master medium int8_float32 had std 7.11 s in the first pass (one slow run); the 5-run repeat gave 10.11 +- 0.21 and replaces it.
- Still one machine, one thread count (16) and one workload per family. Not checked: other thread counts, small batches, Intel, FP16/BF16, T5/ALiBi.

### Machine: GitHub Actions `ubuntu-22.04-arm` (aarch64, OpenBLAS), release gate

The CI release gate (`tools/release-gates/run_gate.py`, `--limit 20`) compares the released `ctranslate2==4.9.0` from PyPI (baseline) with the wheel built from the code under test (candidate), on Whisper WER, Opus BLEU and T5Gemma tokens/s. Each measurement is repeated 3 times. Only the ARM job fails; the x86_64, macOS arm64 and Windows gates pass.

`whisper_wer`, WER baseline -> candidate (median of 3; the 3 runs of each side are identical, so the difference is deterministic, not noise):

| candidate code | compute `default` (float32) | compute `auto` (int8_float32) | gate |
|----------------|-----------------------------|-------------------------------|------|
| master + CI-only changes (upstream PR 2170, run 44931865129) | 0.1942 -> 0.1942 | 0.1853 -> 0.1853 | ok |
| this branch, fused attention on (fork run 44942740078, `ef20cc8d8`) | 0.1942 -> 0.1942 | 0.1853 -> **0.2009** (+0.0156) | **FAIL** |
| this branch, `CT2_CPU_FUSED_ATTENTION=0` (fork run 44981387047, branch `exp/arm-fused-attention-off`) | 0.1942 -> 0.1942 | 0.1853 -> 0.1853 | ok |

Inverse real-time factor, `auto`: 14.88 -> 15.16 with the fused path on (+1.9%), 14.01 -> 14.05 with it off. Opus BLEU and T5Gemma tokens/s pass in every run.

Finding:
- The fused attention is what changes the ARM WER: the same code with the switch off gives the baseline WER exactly, and master gives the baseline WER exactly.
- It only shows with int8 weights (`auto` resolves to int8_float32 on ARM). In float32 the transcripts are identical, as on the AMD machine.
- This is the ARM counterpart of the AMD results above, where int8_float32 transcripts also differ between fused and unfused (Whisper medium WER 0.0070, NLLB WER 0.0156) and float32 is identical. On ARM the difference lands on the gate's WER tolerance.
- The cost/benefit on ARM is about +1.9% speed for +0.0156 WER on this sample, which the gate rejects.
- Not found yet: why the int8 path is affected and float32 is not. Candidates are rounding in the fused path's `gemm` with `beta = 1` accumulation over key chunks, `cpu_primitives::exp` on ARM, and how the quantized activations feed the Q/K/V products. This is not verified.
- Not checked: whether the changed ARM transcripts are better or worse in absolute terms (the gate flags any WER increase beyond its tolerance).

What llama.cpp does on ARM (read from the ggml copy vendored in `whisper.cpp/ggml/src/ggml-cpu`, commit `bd9e1a1`; not tested, upstream not consulted):
- Same algorithm as the fused path: 64x64 query/key tiles (`GGML_FA_TILE_Q/KV`), online softmax, F32 accumulators for the output. The branch already borrows the 64x64 tile size.
- The tiled path needs F32 Q, F32/F16 K and V of the same type, and at least 64 queries; otherwise it uses a per-row vec-dot path. On non-x86 SIMD builds it also needs `DV % f32_epr == 0` (`svcntw()` on SVE, NEON width otherwise); the fused path has no such check. Whisper's head dim is 64, so it should not matter here.
- Q·Kᵀ uses its own fixed-order NEON microkernel (`simd-gemm.h`, 4x4 vectors) on K repacked as F32. The scale is a separate pass (`ggml_vec_scale_f32`). The fused path calls the platform `sgemm` (OpenBLAS in the ARM wheel) with the scale folded in, and accumulates with `beta = 1` across key chunks.
- Softmax uses `ggml_v_expf` (NEON, from Arm's optimized routines, about 1.5 ulp; flushes to inf above 88.38 and to 0 below -103.97), four lanes at a time, and sums the row in double precision (`ggml_float`). The fused path uses `cpu_primitives::exp`; how its sum accumulates on ARM was not checked.
- The ARM-optimized kernels (KleidiAI, `repack.cpp`, `arch/arm`) are for quantized weight matmul; no flash-attention code was found in `kleidiai.cpp` or `repack.cpp`. Attention does not depend on the weight quantization, so llama.cpp has no counterpart of the int8-only difference.
- Conclusion: no ARM-specific attention trick to copy. The remaining differences are the `beta = 1` accumulation over 512-key chunks through OpenBLAS and the precision of the exp and the softmax sum. Untested experiment: 64-key chunk plus a double-precision sum on ARM (one CI run on the experiment branch).

Options to resolve (none done yet): do not use the fused path for int8_float32 on ARM; find and fix the numeric difference in the fused kernel; relax the gate's WER tolerance.

### Machine: Apple M4 Pro (arm64, 8 performance + 4 efficiency cores, 24 GB, macOS 15)

Branch = working tree of this branch (4.9.0 base plus the fused attention); master = `88e735ac` (4.9.1). Both built Release with `OPENMP_RUNTIME=COMP` (Homebrew libomp), `WITH_ACCELERATE=ON`, `WITH_RUY=ON`, in `~/dev/ct2-bench` (out of tree), and loaded through Python 3.13 extensions built against each install. The only C++ source difference between the two is `src/ops/fused_attention.cc`. This is not the CI configuration: float32 goes through Accelerate, not OpenBLAS; int8 goes through ruy, as on the CI ARM runner.

Pinned by thread count only (no `taskset` on macOS): 4 threads, `audio/15GdH1.mp3` from 1:00, 60 s, beam 5, language `ca`, 1 warm-up + 3 timed runs, no reference transcript (WER is fused vs unfused only).

| family | model | compute | branch fused (s) | branch unfused (s) | master (s) | fused vs branch unfused | fused vs master | quality (fused vs unfused) |
|--------|-------|---------|------------------|--------------------|------------|-------------------------|-----------------|----------------------------|
| Whisper | medium | int8_float32 | 20.61 ± 0.09 | 28.54 ± 0.58 | 32.36 ± 0.18 | 1.38x | 1.57x | WER 0.0207, differ |
| Whisper | medium | float32 | 28.97 ± 1.36 | 39.25 ± 2.28 | 38.37 ± 2.06 | 1.35x | 1.32x | identical |
| Whisper | large-v3 | int8_float32 | 85.24 ± 6.81 | 51.99 ± 0.30 | 47.09 ± 0.48 | 0.61x | 0.55x | WER 0.0654, differ |
| Whisper | large-v3 | float32 | 44.27 ± 0.71 | 59.77 ± 1.02 | 60.98 ± 0.83 | 1.35x | 1.38x | identical |
| Translation | nllb-200-distilled-600M | int8_float32 | 3.89 ± 0.00 | 4.53 ± 0.01 | 4.58 ± 0.09 | 1.16x | 1.18x | WER 0.0391, differ |
| Translation | nllb-200-distilled-600M | float32 | 7.44 ± 0.13 | 8.42 ± 0.13 | 8.31 ± 0.09 | 1.13x | 1.12x | identical |

The master column uses the `unfused` rows (the fused switch does nothing on master; the `fused` rows were within noise: 32.50, 40.09, 48.03, 60.61, 4.64, 8.39 s). Not run: xlm-roberta-base and gemma-3-4b-it (not in the local HF cache, not downloaded).

Findings:
- The ARM WER problem reproduces here. With int8_float32 the fused transcripts differ from the unfused ones on Whisper medium (WER 0.0207), Whisper large-v3 (0.0654) and NLLB (0.0391); with float32 they are identical in all three. Same pattern as the CI ARM runner and the AMD machine, and ruy is the int8 GEMM in all of them. Master fused/unfused are identical in every row, as expected.
- Whisper medium int8 is deterministic: 3 separate fused processes give the same transcript, 2 unfused give the same transcript, and the two differ (145 words each, 60 s). At 20 s all of fused/unfused/1 thread/4 threads were identical, so the difference needs longer audio (later windows condition on earlier text).
- Whisper large-v3 int8 fused is not deterministic: 3 separate processes (same command, 4 threads, 60 s) gave 175, 158 and 156 words and 45 s, 75 s and 88 s, against 153 words and a steady 52 s for unfused. The first matrix run (91.1, 86.8, 77.8 s) and a rerun (51.4, 83.4, 70.1 s) were erratic as well. The slow runs look like longer decodes, not a slower attention. A different transcript on every run points to something other than rounding (a race, or uninitialised or aliased scratch memory in the fused path) and is worth checking before the rounding candidates listed under the ARM runner. Not checked: whether it depends on the thread count (only 4 threads tried on large-v3), and why medium does not show it.
- Speed with fused on is 1.35-1.38x for Whisper float32 and medium int8, and 1.12-1.18x for NLLB, except large-v3 int8, where it is slower because of the point above.
- Thread count matters a lot on this machine. At 8 and 12 threads master slows down (medium int8, 10 s of audio: 5.4 s at 4 threads, about 15-16 s at 8 and 12 threads), so the baseline gets worse as threads increase (probably OpenMP threads contending with Accelerate; not verified). At 8 threads the fused path avoids it (medium int8, 20 s of audio: fused 7.99 s, unfused 30.02 s on the branch, 30.44 s on master, so 3.8x), which is mostly a baseline artifact. 4 threads gives the best baseline for master and is used for the table.
- Branch unfused is not always equal to master unfused (medium int8 28.54 vs 32.36 s, large-v3 int8 51.99 vs 47.09 s). The branch is on the 4.9.0 base and master is 4.9.1, and the only source difference is the fused attention file, so the cause was not isolated.
- Setup notes: NLLB needed `transformers<5` (5.24 `NllbTokenizer` has no `build_inputs_with_special_tokens`, which the script calls). `faster-whisper` 1.2.1 needs `av<17` to decode mp3; the audio was extracted with `ffmpeg` to a `.f32` file instead.

Release-gate `whisper_wer` on the M4 Pro (not reproduced): the gate's own worker (`tools/release-gates/tasks.py run whisper_wer`, Whisper tiny, 2 threads, `faster-whisper==1.0.2`, same LibriSpeech dummy clean set) run against the branch with the fused path on and off and against master, WER against the references:

| samples | compute | master | branch, fused off | branch, fused on |
|---------|---------|--------|-------------------|------------------|
| 20 (gate setting) | `default` (float32) | 0.1942 | 0.1942 | 0.1942 |
| 20 (gate setting) | `auto` (int8_float32) | 0.1987 | 0.1987 | **0.1897** |
| all | `default` (float32) | 0.2330 | - | 0.2330 |
| all | `auto` (int8_float32) | 0.2383 | 0.2383 | **0.2313** |

- The float32 WER (0.1942) equals the CI baseline on the ARM runner exactly, so the setup follows the gate. The int8 baseline differs from CI (0.1987 vs 0.1853), as expected with another platform and build (Accelerate/ruy vs OpenBLAS/ruy).
- Here the fused path changes the int8 transcripts but lowers the WER (-0.0090 on 20 samples, -0.0070 on all), while on the CI ARM runner it raised it by +0.0156. With 20 samples (about 1,100 words) a 0.0156 change is about 17 words, so the sign of the change on this sample is not a measure of quality in either direction. This suggests the CI failure may be a different transcript that happens to land above the 0.01 tolerance on that sample, not a systematic degradation, but this is not shown: the CI runner was not reproduced, and the M4 Pro uses a different float32 GEMM and a different build.
- This does not explain the Whisper large-v3 int8 non-determinism above, which is a separate problem.

### Local reproduction of the CI ARM configuration (Linux aarch64 container on the M4 Pro)

Built the same way as the CI aarch64 wheel, in `quay.io/pypa/manylinux_2_28_aarch64` (native arm64 via Docker Desktop, 12 CPUs): OpenBLAS 0.3.26 (`TARGET=ARMV8 USE_OPENMP=1`), CMake 3.24 with `-DOPENMP_RUNTIME=COMP -DWITH_OPENBLAS=ON -DWITH_RUY=ON`, CPython 3.12 wheel repaired with `auditwheel` (the steps of `python/tools/prepare_build_environment_linux.sh`), from the working tree of this branch. Then, in `python:3.12-slim`, the gate's own worker (`tools/release-gates/tasks.py run whisper_wer`, 2 threads, `faster-whisper==1.0.2`) with the released `ctranslate2==4.9.0` from PyPI as baseline (as the gate does) and this wheel as candidate:

| samples | compute | baseline PyPI 4.9.0 | candidate, fused off | candidate, fused on |
|---------|---------|---------------------|----------------------|---------------------|
| 20 (gate setting) | `default` (float32) | 0.1942 | 0.1942 | 0.1942 |
| 20 (gate setting) | `auto` (int8_float32) | 0.1897 | 0.1897 | **0.1830** |
| all | `default` (float32) | 0.2330 | 0.2330 | 0.2330 |
| all | `auto` (int8_float32) | 0.2330 | 0.2330 | 0.2330 |

- The CI regression is not reproduced with the CI build configuration either: with the fused path on, the int8 WER goes down (-0.0067 on 20 samples) or stays equal (all samples), not up by +0.0156. Fused off gives the baseline exactly, as in CI.
- The float32 baseline (0.1942) equals the CI value; the int8 baseline (0.1897) does not (CI: 0.1853), so the int8 numerics depend on something that differs between this container and the CI runner.
- Remaining differences from the CI runner: CPU (the container exposes `asimddp`, `bf16`, no `i8mm`, no SVE; the runner's CPU features were not checked), the exact candidate revision (CI ran `ef20cc8d8`; this is the working tree), and dependency versions (installed unpinned except `faster-whisper`). Which of these matters was not isolated.
- To reproduce on the real runner, run the same worker there with the fused path on and off and diff the transcripts sample by sample; the sign of a 20-sample WER change is not informative by itself.

### ARM runner, per-sample diagnosis (GitHub Actions `ubuntu-22.04-arm`, run 45008753816)

Throwaway workflow on the fork (branch `exp/arm-wer-diagnosis`, not meant to be merged) that runs the gate's own Whisper task on the real runner and dumps every transcript. Candidate = the wheel of the failing run 44942740078 (`ef20cc8d8`), baseline = `ctranslate2==4.9.0` from PyPI, `faster-whisper==1.0.2`, 2 threads, same pins as the gate. The runner CPU reports `asimddp`, `bf16`, `i8mm`, `sve`, `sve2`, `svei8mm` (full list in the run's `cpu.txt`); the local container had `asimddp` and `bf16` only.

| samples | compute | baseline 4.9.0 | candidate, fused off | candidate, fused on |
|---------|---------|----------------|----------------------|---------------------|
| 20 | `default` (float32) | 0.1942 (87/448) | 0.1942 | 0.1942 |
| 20 | `auto` (int8_float32) | 0.1853 (83/448) | 0.1853 | **0.2009** (90/448) |
| all (73) | `auto` (int8_float32) | 0.2296 (264/1150) | 0.2296 | **0.2374** (273/1150) |

- **The CI failure reproduces on the runner**, with the same numbers as the gate (baseline 0.1853, fused on 0.2009, +0.0156). Fused off gives the baseline exactly, and float32 is identical. It is deterministic: 4 separate processes and 3 passes in one process all gave the same transcripts.
- It did not reproduce on the M4 Pro, native or in a container with the CI build recipe, where the int8 baseline was different (0.1897 in the container vs 0.1853 here). int8 results therefore depend on the CPU, and the missing `i8mm`/SVE in the container is the most visible difference. Which code path depends on them was not identified.
- 6 of the first 20 clips change with the fused path: 3 get worse (clip 2: 4 to 6 errors, clip 4: 17 to 21, clip 7: 1 to 4), 2 get better (clips 13 and 17: 4 to 3), 1 changes without changing the error count. Net +7 errors in 448 words.
- Over all 73 clips, 20 change: 8 get worse (+17 errors), 5 get better (-8), 7 are unchanged in error count. Net +9 errors in 1150 words (+0.0078 WER). More clips get worse than better, but the sample is small, so this is a mild, not clearly systematic, degradation.
- The gate tolerance is 0.01 WER, and 20 clips are 448 words (one word is 0.0022), so a net change of 7 words is enough to fail it.
- Not checked: the cause of the int8 difference on this CPU, and whether the Whisper large-v3 non-determinism seen on the M4 Pro (OpenMP, Accelerate) also happens on this runner; the tiny model gave identical output across processes here.

### Fix: block-wise softmax in the fused attention

The online softmax of the first kernel rounds differently from the regular path. With int8 weights the attention output is quantized to int8 before the next GEMM, which turns those rounding differences into different quantized activations and different transcripts. The kernel now computes, for a block of queries, the scores against all the keys, normalizes them with the same `cpu::softmax` kernel as the regular path and multiplies by the values (one GEMM each). Every block repacks the keys and values in its two GEMMs, so large blocks are faster: up to 256 queries, fewer for many keys (2 MB of scores); with more than 8192 keys (block under 64 queries) the fused path is not used and the regular path runs, as before the fused attention existed.

Check on the CI recipe (manylinux_2_28 aarch64, OpenBLAS 0.3.26 ARMV8, ruy, OpenMP; container on the M4 Pro): Whisper tiny encoder output of 4 samples, fused vs unfused, and the gate's `whisper_wer` task (released 4.9.0 as baseline):

| | float32 | int8_float32 |
|---|---|---|
| old kernel (online softmax), encoder output vs unfused | differs in 98% of values, max 1.0e-3 | differs in 100% of values, max 2.87 |
| new kernel, encoder output vs unfused | bit-identical | bit-identical |
| new kernel, gate WER (20 samples / all), fused on vs baseline | 0.1942 / 0.2330, equal | 0.1897 / 0.2330, equal |

- On the M4 Pro (Accelerate, 4 threads), Whisper large-v3 int8_float32, 60 s of audio: 3 separate fused processes give the same transcript as unfused (153 words, identical); the old kernel gave a different transcript each time. Fused 38.7 s vs unfused 47.6 s. Why the old kernel was not deterministic was not identified; it does not happen with the new one.
- The int8 WER change is of the size of rounding noise. On unmodified master (M4 Pro, int8_float32, all 73 samples), adding 1e-6 or 1e-5 amplitude noise to the audio changes 13-24 transcripts and moves the WER of the gate's first 20 samples with a standard deviation of 0.006-0.008 (range -0.018 to +0.009 around the unperturbed 0.1987; 12 runs per amplitude). The CI change of +0.0156 is about 2 standard deviations. The 0.01 tolerance on 20 samples is close to that noise.
- Speed, final kernel vs old kernel vs unfused (4 threads). Whisper medium, 60 s, M4 Pro: int8_float32 21.76 +- 0.35 s (unfused 29.11 s, 1.34x; old kernel 20.61 s; master 32.36 s), float32 30.34 +- 1.96 s (unfused 38.17 s, 1.26x; old kernel 28.97 s); fused and unfused outputs identical. Whisper small encoder only, CI recipe (OpenBLAS aarch64 container): float32 1012-1031 ms unfused, 1055-1060 old kernel, 991-1012 final; int8_float32 556-561 ms, 576 old, 511-516 final. Whisper small encoder only, M4 Pro, 64-query blocks: float32 384 ms unfused, 343 old, 342 new; int8_float32 727, 666, 648 ms.
- Op level (one attention op, 4 threads, ratio of regular to fused time, so above 1 is faster). The regular path allocates its [batch, heads, queries, keys] score tensor in every layer and the CPU allocator does not cache (plain `posix_memalign`), so on macOS each layer pays page faults on a block of 100+ MB (16 heads, 1500 keys: 18.7 ms with the tensors reused between runs, 31.3 ms with fresh ones, as in a layer). "reuse" below keeps the tensors allocated between runs (an artificial best case for the regular path), "fresh" allocates them in every run. The first kernel used blocks of 64 queries (256 KB of scores) and was slower than the regular path in reuse mode (0.7-0.87x for Whisper and NLLB shapes on the M4, 0.88-0.93x on OpenBLAS aarch64); the packing overhead of the many small blocks was the cause, and 256-query blocks remove it. Final kernel, M4 Pro / OpenBLAS aarch64 container: Whisper encoder 16 heads 0.99-1.00x reuse, 1.6-2.2x fresh / 1.06x, 1.14x; 20 heads 0.97-0.99x, 1.9x / 1.08x, 1.09x; NLLB encoder 0.97-1.03x, 1.3-1.7x / 1.00x, 1.10x; cross-attention with 448 queries 0.99-1.03x, 1.2x / 1.03x, 1.08x; causal prefill 512x512, head dim 128: 1.2-1.3x, 1.3-1.4x / 1.24x, 1.36x; causal 4096x4096: 1.3x, 1.7-1.8x / 1.9x; 16384 keys: regular path. At 12 threads the worst case was 0.98x (M4) and 1.03x (container). The 3% spread between repeated runs is noise. So most of the end-to-end gain on the M4 comes from not materializing the score tensor, not from faster attention arithmetic; on the CI recipe the allocation costs little (+0.7 ms for the same case) and the gain is small. Not measured: x86 (MKL, oneDNN) timings, and machines with many threads, where fewer, larger blocks could balance worse (72 tasks for Whisper small on 32 threads).
- Limits: the bit-identical result needs a GEMM whose per-element result does not depend on the number of rows, which holds for OpenBLAS (checked above). Accelerate gives identical transcripts end to end (Whisper medium, large-v3, NLLB, WER 0.0000, fused vs unfused); MKL and oneDNN were not checked, and neither were x86 timings. The AMD (first table), M4 Pro and first ARM CI tables above are from the first kernel unless stated. x86 timings with the new kernel are now in the AMD "Rerun with the final block-wise kernel" table; the fused and unfused outputs are identical there (MKL and oneDNN), apart from the large-v3 float32 timing noted in that section.
- ARM release gate on the CI runner with this kernel: passes, see the next section.

### Release gate with the block-wise kernel (fork run 45018654845, commit `bc96a06c7`)

All four gates (x86_64 Linux, Windows, Linux ARM, macOS arm64) end in success. Baseline = `ctranslate2==4.9.0` from PyPI, candidate = the wheel of this branch; median of 3, `default` = float32, `auto` = int8_float32. Speed is `rtf_inverse` (Whisper) and tokens/s (Opus, T5Gemma), change of the candidate against the baseline.

| gate | Whisper WER default | Whisper WER auto | Whisper speed default / auto | Opus BLEU float32 / int8 | T5Gemma speed float32 / int8 |
|------|---------------------|------------------|------------------------------|--------------------------|------------------------------|
| Linux x86_64 | 0.1942 -> 0.1942 | 0.1897 -> 0.1897 | +10.8% / +15.3% | identical | +7.3% / +21.9% |
| Windows | 0.1942 -> 0.1942 | 0.1987 -> 0.1987 | +8.3% / +13.4% | identical | +0.6% / +0.6% |
| **Linux ARM** | 0.1942 -> 0.1942 | **0.1853 -> 0.1853** | +4.4% / +7.2% | identical | -0.6% / -2.4% |
| macOS arm64 | 0.1942 -> 0.1942 | 0.1920 -> 0.1942 (+0.0022) | -2.3% / +3.2% | 29.2039 -> 29.2039 / 29.2644 -> 29.2841 | +0.6% / +1.4% (see below) |

- The ARM failure is gone: `auto` WER is 0.1853 -> 0.1853, where the old kernel gave 0.1853 -> 0.2009 (run 44942740078) and failed the gate. This matches the diagnosis above: the old online softmax rounded differently, and with int8 weights that changed the quantized activations. The new kernel was bit-identical to the unfused path in the container check; the runner result confirms it on the real CPU (`i8mm`, SVE).
- macOS: the first T5Gemma float32 pass was flagged (17.51 -> 15.10 tokens/s, -13.8%, 3 samples ranging 13.9-21.3). The gate's confirmation step repeated it with 5 samples and found no regression (16.20 -> 16.29, +0.6%). The job passed. The small `auto` WER change on macOS (+0.0022, one word) is inside the gate tolerance.
- The speed changes are against PyPI 4.9.0, so they include everything merged since 4.9.0, not only the fused attention. Each metric is 3 runs on shared runners; the T5Gemma x86 gain (+21.9%) is larger than I would attribute to this change.
- Not checked: which of the commits between `ef20cc8d8` and `bc96a06c7` matters on ARM (the plan above attributes it to the block-wise softmax), and a gate run with `CT2_CPU_FUSED_ATTENTION=0` on this commit to confirm the fused path runs on ARM for Whisper.
- The llama.cpp comparison above is superseded for the ARM WER question: the cause was the online softmax of the old kernel, not the `beta = 1` accumulation or the exp precision, so the untested 64-key-chunk experiment is not needed for it.

### Machine: AMD EPYC 7B13 (8 physical cores / 16 threads, 62 GiB RAM, Linux VM)

**Temperature caveat for the original EPYC matrix:** these Whisper measurements used faster-whisper’s default temperature fallback sequence, not fixed temperature zero. The large-v3 rows are superseded by the controlled rerun below; the other Whisper rows have not been rerun.

Run on 2026-11-22. Branch `877d7efe`; master baseline `origin/master` at `88e735ac` (4.9.1). The local `master` ref was older, so the baseline uses the locally available `origin/master` ref; no fetch was performed. Both built CPU-only, Release, GCC 10.2.1, `-msse4.1`, MKL 2025.3, oneDNN 3.1.1, `OPENMP_RUNTIME=COMP`, with separate install prefixes and Python extensions. Pinned to CPUs 0–7 (one logical CPU per physical core), 8 threads, `OMP_PROC_BIND=true OMP_PLACES=cores`. Linux 5.10.0-46-cloud-amd64, Python 3.11, Transformers 4.57.6.

Whisper: local `english_60s.mp3`, exactly 60 s, language `en`, beam 5; this is different audio from the Catalan benchmarks above. One warm-up and 3 timed runs per mode, each mode in a separate process. Branch then master for each model, sequentially. NLLB and XLM-R: batch 4 × 512 tokens; NLLB greedy up to 128 output tokens; Gemma: batch 4 × 512 prompt tokens plus 32 generated tokens, int8 only. Means ± sample standard deviations in seconds. Master uses only the `unfused` rows.

| model | compute | branch fused (s) | branch unfused (s) | master unfused (s) | speedup vs branch unfused | speedup vs master | quality (fused vs unfused) |
|---|---|---|---|---|---|---|---|
| medium | int8_float32 | 16.60 ± 0.68 | 16.78 ± 0.28 | 16.96 ± 0.34 | 1.01x | 1.02x | WER(fused vs unfused) = 0.0077 |
| medium | float32 | 18.82 ± 0.38 | 20.56 ± 0.26 | 19.84 ± 0.40 | 1.09x | 1.05x | identical |
| large-v3 | int8_float32 | 155.15 ± 5.65 | 152.48 ± 10.78 | 145.17 ± 16.71 | 0.98x | 0.94x | WER(fused vs unfused) = 0.4521 |
| large-v3 | float32 | 141.33 ± 21.66 | 166.86 ± 5.85 | 162.30 ± 18.49 | 1.18x | 1.15x | WER(fused vs unfused) = 0.3946 |
| nllb | int8_float32 | 5.52 ± 0.01 | 5.58 ± 0.03 | 6.05 ± 0.54 | 1.01x | 1.09x | WER(fused vs unfused) = 0.0625 |
| nllb | float32 | 11.73 ± 0.86 | 11.97 ± 0.75 | 12.42 ± 0.17 | 1.02x | 1.06x | identical |
| xlm-roberta | int8_float32 | 0.41 ± 0.00 | 0.46 ± 0.02 | 0.46 ± 0.02 | 1.12x | 1.12x | max \|fused - unfused\| = 2.73e-02 |
| xlm-roberta | float32 | 0.66 ± 0.01 | 0.65 ± 0.00 | 0.65 ± 0.00 | 0.98x | 0.99x | max \|fused - unfused\| = 1.19e-07 |
| gemma-3-4b | int8_float32 | 19.57 ± 0.82 | 19.82 ± 0.69 | 19.80 ± 0.59 | 1.01x | 1.01x | WER(fused vs unfused) = 0.0645 |

Whisper real-time factor (elapsed seconds / audio seconds; lower is faster):

| model | compute | branch fused RTF | branch unfused RTF | master RTF |
|---|---|---|---|---|
| medium | int8_float32 | 0.277 | 0.280 | 0.283 |
| medium | float32 | 0.314 | 0.343 | 0.331 |
| large-v3 | int8_float32 | 2.586 | 2.541 | 2.419 |
| large-v3 | float32 | 2.355 | 2.781 | 2.705 |

- CPU correctness checks: 370 passed, 18 skipped; the dedicated `OpTest.FusedAttentionCPUMatchesMaterializedPath` test passed separately.
- The unchanged-output goal is not met on this machine. Medium int8 differs only by one comma (the script reports WER 0.0077 because it retains punctuation). NLLB int8 and Gemma int8 have token differences; their reported WER is over tokenizer pieces, not decoded natural-language words. XLM-R int8 has a maximum difference of 0.0273 in the pooled vectors. Medium and NLLB float32 outputs are identical; XLM-R float32 differs by 1.19e-7.
- Large-v3 is unusually slow (141–167 s for 60 s of audio) and its retained transcripts have long, differing continuations after the passage transcribed by medium. Both float32 and int8 differ between fused and unfused. Timings are variable and include these different decode workloads, so they do not establish an attention speedup or regression. This run does not isolate the cause.
- Crucially, master large-v3 also gives different transcripts between its two switch settings (which execute the same code): WER 0.3067 int8 and 0.2857 float32. All other master rows are identical between settings. Large-v3 therefore has a reproducibility problem beyond the branch fused path in this setup; its branch differences cannot be attributed solely to fusion.
- Quality compares fused and unfused outputs, without a reference transcript; it does not measure absolute transcription accuracy. For XLM-R the benchmark compares mean-pooled hidden-state vectors, not every hidden-state element. For translation and generation it compares the first sequence of the repeated batch, and for all models it retains the last timed output.
- Artifacts: `benchmark-results/cpu-attention-epyc/` contains raw JSON, logs, build/linkage details, environment metadata, and the runner scripts. Models and conversions are cached in `/tmp/ct2-cpu-plan-models` and `/tmp/ct2-cpu-plan-hf`.

#### Large-v3 rerun with temperature fixed at zero (2026-11-23)

Same CPU builds, affinity, threads, audio, language, beam size, one warm-up and three timed runs as the EPYC matrix above. The benchmark now explicitly passes `temperature=0.0` to Whisper, disabling nonzero-temperature sampling fallback. Only large-v3 was rerun. Original results are preserved in the preceding table and the original JSON files. The requirement is committed as `dcc918b0`; the libraries remain built from branch `877d7efe` and master `88e735ac`.

| compute | branch fused (s) | branch unfused (s) | master unfused (s) | speedup vs branch unfused | speedup vs master | quality (branch fused vs unfused) |
|---|---|---|---|---|---|---|
| int8_float32 | 56.31 ± 0.25 | 58.70 ± 0.14 | 58.57 ± 0.26 | 1.04x | 1.04x | WER(fused vs unfused) = 0.5380 |
| float32 | 46.85 ± 0.86 | 49.84 ± 1.23 | 50.03 ± 1.09 | 1.06x | 1.07x | identical |

Real-time factors:

| compute | branch fused RTF | branch unfused RTF | master RTF |
|---|---|---|---|
| int8_float32 | 0.938 | 0.978 | 0.976 |
| float32 | 0.781 | 0.831 | 0.834 |

- Fixing temperature substantially reduces runtime and timing variability relative to the original fallback-enabled runs. Float32 branch outputs are now identical. Int8 branch outputs still differ (WER 0.5380): they match through the passage transcribed by medium, then diverge in an added ending, with unfused repeating “I saw the stars” many times. Different decode workloads limit the interpretation of the int8 speedup.
- This script retains only the last timed transcript per mode; it does not prove that every warm-up and timed repetition produces the same output.
- Branch unfused and master unfused retained transcripts match exactly in both compute types. The remaining int8 output difference is between the branch fused path and these matching baselines.
- Raw results, logs, and metadata: `benchmark-results/cpu-attention-epyc/temperature-zero/`.
- Master `int8_float32` switch comparison (same code in both settings): WER(fused vs unfused) = 0.0000; outputs identical.
- Master `float32` switch comparison (same code in both settings): WER(fused vs unfused) = 0.0000; outputs identical.

#### EPYC int8 numerical diagnosis (2026-11-23)

The remaining final-kernel difference was reproduced without decoding or sampling. This AMD build defaults to **oneDNN for both float32 and int8 GEMM**, despite being compiled with MKL too; verbose backend logs confirm it. A fixed-input attention probe shows that oneDNN changes float32 GEMM rounding when the query matrix is blocked (1500 rows versus blocks of at most 256). Attention outputs differ by about 1e-7 even with one thread. With `CT2_USE_MKL=1`, the same probe is bit-identical.

On real large-v3 encoder inputs (first 30 s of the local audio), a same-input layer comparison records an int8 activation-byte change despite a maximum attention-output difference of only 2.16e-7 at that layer. Another captured example scales almost equal attention values to 121.500000 and 121.499992, which round to different unsigned bytes (122 and 121). Per-row quantization scales can also differ. These discrete changes propagate through later layers: final encoder max difference is 10.5271 for int8_float32 versus 0.00124416 for float32 on this input.

Two controls localize the problem: replacing each fused attention result with its regular-path result restores the unfused int8 encoder bit-for-bit; forcing MKL also gives bit-identical fused/unfused int8 encoder outputs (all 1,920,000 elements). The uninstrumented installed library reproduces the instrumented fused encoder output exactly. The evidence supports shape-dependent float32 GEMM rounding amplified by activation quantization; a race is not needed to explain this reproduction. The block-wise softmax fix alone cannot guarantee equivalence across GEMM backends.

Full 60-second large-v3 transcription with MKL forced, int8_float32, temperature 0 and beam 5 also gives identical fused/unfused transcripts (247 words; one separate process per setting). This verifies the workaround on this input, not its performance or generality.

Detailed evidence, the quantization-boundary values, probe source, logs, and the temporary diagnostic patch are in [the investigation report](benchmark-results/cpu-attention-epyc/investigation/report.md). Diagnostic source changes were restored before implementing the follow-up conditional described below. The existing approximate attention unit test can pass while these small float differences cross int8 thresholds.

#### Remedies considered for the oneDNN/int8 output difference

- **Immediate compatibility workaround:** set `CT2_CPU_FUSED_ATTENTION=0` before starting the process. This retains the existing oneDNN backend and runs regular attention, giving the matching branch-unfused/master baseline seen above. It gives up the CPU fused-attention optimization for the process.
- **Tested alternative backend:** set `CT2_USE_MKL=1` before starting the process, when the library was built with MKL. The EPYC fixed-input probe, int8 encoder, and full 60-second int8 transcript all match between fused and unfused with this setting. This changes both float32 and int8 GEMM backends, including other layers; its performance and compatibility on other inputs are not established. Do not select it automatically as a presumed speed improvement.
- **Conservative source fallback considered:** bypass CPU fused attention for `int8_float32` models when the active float32 GEMM backend is oneDNN. Check the runtime-selected backend, not merely whether oneDNN was compiled in. The attention tensors themselves are float32 in both float32 and int8_float32 models, so the model's effective compute type would need to reach the dispatch decision; inspecting Q/K/V dtype alone cannot make this distinction. This would give up fusion for the affected int8 case. It was not implemented: the follow-up uses whole-head fusion on oneDNN instead, covering both compute types without needing a model-type flag.
- **Longer-term optimization:** find a query-blocked GEMM strategy whose per-element results match regular attention across the relevant shapes and threading modes, then test both attention output and the resulting quantized bytes/scales. Merely adjusting the softmax precision or tolerance does not fix a difference already introduced in GEMM. Sweeps found block sizes that fix some shapes but fail others; no generally compatible query-blocked strategy has been validated here. The following conditional preserves the full GEMM dimensions instead.

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

#### Follow-up: performance exploration and framework-derived hypotheses (2026-11-23)

The committed oneDNN whole-head compatibility fix remains unchanged during these experiments. Experimental libraries live under `/tmp/ct2-attn-variants/`. The detailed [framework comparison and hypothesis log](benchmark-results/cpu-attention-epyc/investigation/performance-exploration/framework-hypotheses.md) distinguishes source evidence, tested ideas and untested proposals; it is not a benchmark of those frameworks.

Sources inspected: [PyTorch CPU FlashAttention](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L1106) uses workload-dependent tiles and capability checks; [OpenVINO oneDNN attention](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/scaled_attn.cpp#L299) caches explicit BRGEMM plans, manages scratch and packs keys per KV head; [oneDNN Graph](https://github.com/uxlfoundation/oneDNN/blob/main/src/graph/backend/dnnl/kernels/sdp_decomp_config.cpp#L90) considers head/batch parallelism; [ONNX Runtime CPU attention](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h#L274) exposes batch/row work to MLAS and combines mask/bias with softmax; [ggml CPU attention](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp#L8512) owns its tiled SIMD matmul and scratch layout. None establishes bit identity with CT2's existing oneDNN GEMMs.

Hypotheses under investigation:

- **H1, aligned uninitialized scratch:** avoid zero-initializing scores that GEMM overwrites. Initial paired op measurements show only a small (about 2–4%) gain over whole-head fusion. Allocation policy and alignment change together; no end-to-end benefit is assumed.
- **H2, shape-specific query tiling:** 192- and 240-query tiles match the tested large-v3 encoder in both compute types, unlike 256-query tiles. For the 1500-query/key, depth-64 shape, both candidates passed a total of 240 randomized bitwise comparisons (120 each), using two thread counts and three input magnitudes. Other shapes have counterexamples, so these constants are not a general replacement. Full transcription checks completed: all eight outputs per compute type (warm-up plus three timed runs in each mode) have identical text and segment timestamps. Final encoder tensors also match bit-for-bit.
- **H3, explicit BRGEMM and packing:** cache kernel plans and pack keys once across query blocks. This requires an appropriate API and numerical validation; our oneDNN 3.1.1 install lacks the newer public ukernel interface used in PyTorch's wrapper. A follow-up below implements the planning/packing architecture with the existing private SGEMM kernels; a new BRGEMM backend remains unimplemented.
- **H4/H5, workspace lifetime and scheduling:** test bounded workspace reuse and adapt outer-head versus inner-GEMM parallelism. Persistent worker buffers need explicit release semantics, and changing threading can change rounding. Not implemented.
- **H6, scale placement:** investigate GEMM alpha versus explicit post-GEMM scaling, which upstream attention kernels use. Changing this only in the fused path can introduce further reference differences; it is a hypothesis, not a validated fix.
- **H7, end-to-end ceiling:** a one-pass int8 profile took 58.61 s, of which 9.35 s was encoding across three calls (15.95%). A 14.1% reduction in first-window encoder time would therefore suggest only about a 2% total-time reduction if representative. This is a projection; full-transcription timing is measured separately.


Measured large-v3 follow-up results, same EPYC, oneDNN 3.1.1, eight cores, temperature 0, beam 5, one warm-up and three timed repetitions. Tile240 includes the aligned scratch change. Encoder timing is the first 30-second window; full transcription uses the entire 60-second clip.

| compute | encoder regular → tile240 (s) | encoder time reduction | full regular → tile240 (s, mean ± sample SD) | full time reduction |
|---|---|---|---|---|
| int8_float32 | 2.893 → 2.485 | 14.1% | 57.69 ± 0.13 → 56.43 ± 0.31 | 2.2% |
| float32 | 4.221 → 3.738 | 11.4% | 49.48 ± 1.21 → 47.69 ± 1.23 | 3.6% |

These are local results on one CPU and clip; float32 variability is material relative to the gain. Relative to the committed whole-head fix, isolated encoder time falls 9.3% (int8) / 7.5% (float32). No production tile whitelist was introduced: other shapes have numerical counterexamples and other backend versions/ISAs remain unvalidated. The strongest longer-term hypothesis is **full-shape GEMM planning with explicit tiled execution and key packing**, following OpenVINO's architecture; it requires API work and its own parity tests. See [measurements, limitations and reproduction patches](benchmark-results/cpu-attention-epyc/investigation/performance-exploration/report.md).


#### Prototype: full-shape planning with tiled execution (2026-11-23)

Implemented an isolated [experimental adapter and runners](tools/onednn_attention_plan/README.md), leaving the production kernel unchanged. It links against the existing oneDNN 3.1.1 archive using matching private headers. QK creates a full-shape SGEMM plan, reuses its copy/compute kernels and packs keys once per head. PV explicitly preserves the reference no-copy strategy and its full-query block/tail schedule. It uses the original SGEMM arithmetic, not a new BRGEMM kernel. A 256-query workspace target groups the preplanned ranges into 240-query outer tiles plus a 60-query tail.

Scope: unmasked 1500-query/key, depth-64 attention, scale 0.125, effective AVX2, oneDNN 3.1.1 and at least two batch/head entries. Other cases retain whole-head fusion. Matching private headers/archive are required; this is not a production dependency or dispatch policy.

Validation: 228 randomized cases across seven outer targets, two thread counts and three input magnitudes match QK scores, attention, quantized bytes and scales exactly. Both fused-attention regressions pass. Real large-v3 final first-window encoder tensors match bit-for-bit in both compute types. The single-thread PV control still differs under generic 256-query blocking, demonstrating why internal block/tail boundaries belong in the plan.

Fresh comparisons, eight physical cores, one encoder warm-up and three timed runs: planned operator 24.67 ms versus whole-head 29.02 ms (15.0% less elapsed time); encoder regular → planned is 2.884 → 2.569 s for int8_float32 (10.9% less) and 4.144 → 3.850 s for float32 (7.1% less). The earlier tile240 experiment remains faster than this conservative private-driver prototype. Full-transcription results (temperature 0, beam 5, one warm-up plus three timed runs, mean ± sample SD):

| Compute | Regular mean ± SD, s | Planned mean ± SD, s | Time reduction | Speedup |
|---|---|---|---|---|
| int8_float32 | 58.44 ± 0.19 | 56.87 ± 0.17 | 2.7% | 1.028x |
| float32 | 49.04 ± 1.16 | 48.80 ± 1.43 | 0.5% | 1.005x |

All eight outputs per compute type have identical text and segment timestamps. The 0.5% float32 difference is within timing variability and is treated as flat. This is one CPU and one clip, not corpus-level accuracy or a general speedup. Detailed evidence, source/API limitations and reproduction commands are in [the planned execution report](benchmark-results/cpu-attention-epyc/investigation/planned-attention/report.md).


#### Further latency investigation: hybrid oneDNN INT8 weight packing (2026-11-23)

The original prototype/results were committed and pushed as `e828c1c8`. Follow-up experiments rejected PV/QK call grouping and workspace tuning as meaningful encoder latency improvements. Grouped execution passed another 228 exact numerical cases and both regressions, but operator means stayed around 24.6–24.7 ms. Reducing threads from 8 to 6/4 changed the real encoder output and was rejected under the strict compatibility criterion before transcription timing.

A larger candidate targets decoding: persistent INT8 weight packing using oneDNN 3.1.1's private API. **The packing plan depends on query extent:** packing for one query and computing five returned success but left the fifth row unwritten; packing for five and computing sixteen left later rows unwritten. Packing for 1500 preserved the tested outputs but regressed several small-query GEMMs. Blindly adopting MKL's pack-for-one policy is therefore rejected.

Implemented an optional isolated hybrid adapter: keep a five-query packed representation and a raw copy in one model-owned allocation; use packed compute only for compatible calls with at most five queries, and the raw reference for larger/unsupported calls. The production kernel/backend sources remain unchanged. The real first-window encoder matches bit-for-bit. A full-transcription diagnostic run recomputed **every packed INT8 GEMM** through the raw reference and found no int32 differences; final text/timestamps also match.

Full large-v3 **int8_float32**, same 60-second English clip, eight physical cores, temperature 0, beam 5, one warm-up and three timed runs, verification disabled. The same library provides both controls. Mean ± sample SD; load/audio decode excluded:

| Configuration | Transcription mean ± SD, s | Process peak RSS, GiB | Model load, s |
|---|---|---|---|
| regular | 58.18 ± 0.13 | 3.23 | 3.17 |
| hybrid | 53.77 ± 0.45 | 6.17 | 3.45 |

Combined planned attention + hybrid packing reduces transcription time by **7.6% (1.082x)** versus the fresh regular control. All eight warm-up/timed outputs have identical text and segment timestamps. This comparison does not isolate packing from attention fusion. Peak RSS rises by **2.93 GiB**; the memory and private-API costs remain material. One CPU and clip do not establish corpus-level accuracy or a general speedup. Float32 was not rerun for this INT8-only change.

Detailed results, rejected alternatives, query-extent controls, memory definitions, and reproduction commands: [further latency investigation](benchmark-results/cpu-attention-epyc/investigation/latency-tuning/report.md).


#### Memory constraint: compact decoder packing (2026-11-23)

**The user rejects doubling memory.** The previous all-linear hybrid's 6.17 GiB peak is rejected as the default experimental path. Implemented compact packing: select only useful decoder weights (193 matrices/aliases), exclude encoder and cross-attention KV weights, store no raw fallback copies, and execute larger prompt INT8 GEMMs in validated five-query blocks. The generated loading loop snapshots names instead of shared pointers, releasing replaced originals when aliases permit.

Numerical validation: the first-window encoder matches all 1,920,000 float patterns; every packed INT8 GEMM in a full transcription, including tiled prompt calls, matches a regular raw-weight int32 oracle. The oracle copies are present only in diagnostic verification, not in measured compact mode (`raw_copy=0`). All eight benchmark warm-up/timed transcripts and segment timestamps also match.

Fresh large-v3 int8_float32 comparison, same CPU/clip/temperature 0/beam 5/eight cores, one warm-up and three timed runs per configuration:

| Configuration | Time mean ± SD, s | Peak RSS, GiB | Post-inference RSS, GiB |
|---|---|---|---|
| regular | 57.87 ± 0.13 | 3.28 | 3.03 |
| compact | 54.61 ± 0.31 | 3.44 | 3.14 |

Total time falls **5.6% (1.060x)**. Peak RSS rises **4.9% (+0.16 GiB)** and RSS after inference **3.5% (+0.11 GiB)**, replacing the near-doubling with a small measured increase. Both pass the experiment's 20% rejection gate; that gate is not a user-approved general memory allowance. Load/packing/audio decoding are excluded from transcription time. Process peak and post-inference RSS are measured separately; neither is a universal memory bound.

Compact becomes the default only in the opt-in experimental packing adapter; production sources/builds remain unchanged. The earlier hybrid mode remains explicitly selectable solely for reproduction. One CPU and clip do not establish corpus-level accuracy, general latency improvements or compatibility with other oneDNN versions/ISAs. Details, configuration-default checks, source/API limitations and evidence: [compact packing report](benchmark-results/cpu-attention-epyc/investigation/memory-bounded-packing/report.md).


#### Extended validation before committing compact packing (2026-11-23)

Full CPU suite with the guarded experimental library enabled: **371 passed, 18 skipped**. Standalone ASan/UBSan adapter checks pass **6,348 exact numerical cases**, **2,116 expected unsupported-beta rejections**, **576 mixed-thread-context rejections**, and **18 additional guard/raw-fallback checks**. Canaries, padded output strides, input preservation and packed-buffer immutability also pass. The prebuilt oneDNN archive/JIT are not sanitizer-instrumented; LeakSanitizer is disabled.

The broader tests discovered two private-API limitations that are now guarded. Fractional beta caused 660 mismatches in the initial sweep; compact mode now accepts checked beta values 0, 1 and -1 only, with explicit rejection otherwise. A narrowed mixed-thread comparison reproduced incomplete outputs when an eight-thread packing plan ran in a one-thread context. Compact mode now records/enforces the original fixed OpenMP context and excludes dynamic teams. Historical raw-retaining modes use their raw fallback. Whisper's tested linears use beta 0, and CT2's normal replica pool already configures matching load/run thread counts.

**27 application comparisons** match encoder float patterns, tokens/text/timestamps and score/probability bits: JFK speech, beams 1/5/8, silence/noise, 75-second windowing, long-prompt prefill, native batch two, medium INT8, large-v3 float32 without INT8 packing, two concurrent four-thread workers, and four medium-model reload cycles. This includes encoder probes; it is not 27 independent recordings. All active-memory comparisons pass the experiment gate; new single-pass elapsed times are not performance claims.

Unload testing initially failed a 64 MiB untrimmed-RSS growth gate for both ordinary and compact allocators. Those failed observations are preserved. After model destruction, explicit diagnostic `mkl_free_buffers` plus `malloc_trim` releases the retained memory; cleaned compact RSS grows about 9.3 MiB across four reload cycles and ends around 175 MiB. That demonstrates releasable allocator retention in this test, not a universal leak-free bound or an ordinary-unload RSS plateau. Cleanup is test-only and was not inserted into normal inference/unload.

Detailed results, negative controls, limitations, source/library hashes and reproduction: [extended validation report](benchmark-results/cpu-attention-epyc/investigation/extended-validation/report.md). These results confirm the documented current CPU/library/model configurations, not arbitrary oneDNN versions, ISAs, GEMM modes or thread-context changes. Production sources/builds remain unchanged.

## Conclusion

(Fill in after all machines have run.)
