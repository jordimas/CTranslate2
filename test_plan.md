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

The online softmax of the first kernel rounds differently from the regular path. With int8 weights the attention output is quantized to int8 before the next GEMM, which turns those rounding differences into different quantized activations and different transcripts. The kernel now computes, for a block of queries, the scores against all the keys, normalizes them with the same `cpu::softmax` kernel as the regular path and multiplies by the values (one GEMM each). The block has up to 64 queries and fewer for many keys (256 KB of scores); with more than 2048 keys (block under 32 queries) the fused path is not used and the regular path runs, as before the fused attention existed.

Check on the CI recipe (manylinux_2_28 aarch64, OpenBLAS 0.3.26 ARMV8, ruy, OpenMP; container on the M4 Pro): Whisper tiny encoder output of 4 samples, fused vs unfused, and the gate's `whisper_wer` task (released 4.9.0 as baseline):

| | float32 | int8_float32 |
|---|---|---|
| old kernel (online softmax), encoder output vs unfused | differs in 98% of values, max 1.0e-3 | differs in 100% of values, max 2.87 |
| new kernel, encoder output vs unfused | bit-identical | bit-identical |
| new kernel, gate WER (20 samples / all), fused on vs baseline | 0.1942 / 0.2330, equal | 0.1897 / 0.2330, equal |

- On the M4 Pro (Accelerate, 4 threads), Whisper large-v3 int8_float32, 60 s of audio: 3 separate fused processes give the same transcript as unfused (153 words, identical); the old kernel gave a different transcript each time. Fused 38.7 s vs unfused 47.6 s. Why the old kernel was not deterministic was not identified; it does not happen with the new one.
- The int8 WER change is of the size of rounding noise. On unmodified master (M4 Pro, int8_float32, all 73 samples), adding 1e-6 or 1e-5 amplitude noise to the audio changes 13-24 transcripts and moves the WER of the gate's first 20 samples with a standard deviation of 0.006-0.008 (range -0.018 to +0.009 around the unperturbed 0.1987; 12 runs per amplitude). The CI change of +0.0156 is about 2 standard deviations. The 0.01 tolerance on 20 samples is close to that noise.
- Speed, new vs old kernel vs unfused (4 threads): Whisper medium, 60 s, M4 Pro, new kernel: int8_float32 22.39 +- 0.29 s (unfused 29.36, 1.31x; old kernel 20.61 s), float32 31.29 +- 3.32 s (unfused 39.42, 1.26x; old kernel 28.97 s). Whisper small encoder only, M4 Pro: float32 384 ms unfused, 343 ms old, 342 ms new; int8_float32 727, 666, 648 ms. Same encoder on the CI recipe (OpenBLAS aarch64): float32 1017 ms unfused, 1042-1064 old, 1000-1003 new; int8_float32 567, 578, 537 ms.
- Op level (one attention op, 4 threads). The regular path allocates its [batch, heads, queries, keys] score tensor in every layer and the CPU allocator does not cache (plain `posix_memalign`), so on macOS each layer pays page faults on a block of 100+ MB: 16 heads, 1500 keys take 18.2 ms with the tensors reused between runs and 28.7 ms with fresh ones, as in a layer. The fused path never allocates it. With fresh tensors fused/regular is 1.24-1.44x for Whisper encoder shapes and 1.16-1.26x for NLLB (0.89-0.92x for cross-attention with 448 queries, about even for causal prefill); with reused tensors, a benchmark artifact, it was 0.7-0.87x. So most of the end-to-end gain on the M4 comes from not materializing the score tensor, not from faster attention arithmetic. On the CI recipe (OpenBLAS aarch64 container) the allocation costs little (+0.7 ms for the same case) and fused is only slightly faster: new kernel 1.04-1.08x for Whisper encoder shapes, 1.03-1.04x for NLLB, 0.96-1.02x for cross-attention, 1.7x for causal prefill (512x512, head dim 128); the old kernel was 0.88-0.93x for the first three. The first versions of the new kernel were 0.55-0.20x for 4096-16384 keys (blocks of 8-16 queries), which is why those lengths now use the regular path.
- Limits: the bit-identical result needs a GEMM whose per-element result does not depend on the number of rows, which holds for OpenBLAS (checked above). Accelerate gives identical transcripts end to end (Whisper medium, large-v3, NLLB, WER 0.0000, fused vs unfused); MKL and oneDNN were not checked, and neither were x86 timings. The AMD, M4 Pro and CI tables above, except where stated, are from the first kernel and have not been repeated. x86 timings are the open item: the gain there came from the old kernel and was not measured with the new one.
- Not done yet: the ARM release gate on the CI runner with this kernel (pushed, result pending).

## Conclusion

(Fill in after all machines have run.)
