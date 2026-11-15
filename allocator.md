# CPU caching allocator

Branch `macos-cpu-caching-allocator`. Only code change: `src/cpu/allocator.cc`. macOS only: the cache is compiled
under `#ifdef __APPLE__`, other platforms use the base allocator directly (see the Linux results below).

## Problem

CTranslate2 allocates a fresh `StorageView` for every op output, and the CPU allocator was a bare
`mkl_malloc`/`mkl_free` (or `posix_memalign`/`free` without MKL). Decoding one token allocates and
frees many temporaries, some large (for Whisper, the `[heads, 1500, 1500]` fp32 attention scores and the
fp32 cross-attention tensors). On macOS, `free` of a large block goes to libmalloc `free_medium` ->
`madvise`, which returns the pages to the OS. The next allocation page-faults them back in. Profiling
(macOS `sample`) showed this as a large share of decode and encode time.

## Change

`CachingAllocator` wraps the existing allocator (`MklAllocator` or `AlignedAllocator`):

- Blocks >= 16 KB are kept on a free list per size when freed, and reused by the next allocation of the same size.
  Smaller blocks pass straight through to the base allocator.
- Sizes are rounded up to 4 steps per power of two (worst-case waste 25%, mean about 9%), so tensors that
  grow a little each step (KV cache, growing sequences) still hit the same bucket.
- The block size is stored in a 64-byte header in front of the user pointer (keeps 64-byte alignment), so
  `free` needs no lookup table.
- The cache is bounded: `CT2_CPU_ALLOCATOR_CACHE_MB` (default 1024). Blocks freed beyond the cap go to the
  base allocator. `0` disables caching (every block is passed through).
- `clear_cache()` releases all cached blocks, then calls the base `clear_cache()`.
- One mutex protects the free lists. It is only taken for blocks >= 16 KB.
- Only built on macOS. On Linux and Windows `get_allocator<Device::CPU>()` returns the base allocator, and
  `CT2_CPU_ALLOCATOR_CACHE_MB` is ignored.

Not covered: no splitting or coalescing (unlike ONNX Runtime's BFC arena), no per-thread caches.

## Results (this machine only)

Intel i9-9880H (AVX2, no AVX512/VNNI), macOS (Darwin 25.5), 8 threads, Whisper int8, greedy, English, no timestamps,
`jfk.wav` (11 s, 25 tokens), medians of 5 interleaved rounds, with a second unmodified build as noise control.

| Model | Total time change |
|---|---|
| base | -21% (496 ms -> 392 ms) |
| small | -16 to -23% (noisier) |
| medium | -12% |

Base-model peak RSS: 981 MB -> 856 MB (lower, because blocks are reused instead of fragmenting).
Noise control differed from the baseline by +0.9%. Transcripts are identical. 367 unit tests pass.

Grain-size and reciprocal-scale changes in `kernels.cc` were also tried and gave no gain; they were dropped.

## Results: Apple M4 Pro (macOS, arm64)

Darwin 24.6.0, 12 cores, Accelerate + ruy, no MKL (`OPENMP_RUNTIME=NONE`, `-mcpu=apple-m4`), base allocator
`AlignedAllocator`. Whisper int8, greedy, English, no timestamps, first 30 s of a podcast clip (about 100 tokens).
A fresh process per run (1 warmup + 3 timed passes), cache off (`CT2_CPU_ALLOCATOR_CACHE_MB=0`), on (1024) and a
second off run as noise control, interleaved over 5 rounds. Medians:

| Model | cache off | cache on | time change | noise control | peak RSS off -> on |
|---|---|---|---|---|---|
| tiny | 0.269 s | 0.244 s | -9.3% | 0.0% | 427 -> 365 MB |
| small | 0.917 s | 0.832 s | -9.2% | +0.6% | 1125 -> 1153 MB |
| medium | 3.147 s | 3.031 s | -3.7% | -0.2% | 3006 -> 3106 MB |

Generated tokens are identical. The gain is smaller than on the Intel machine and shrinks with model size,
probably because larger models are dominated by GEMM and weight reads. large-v3 was not measured.

## Results: Linux CPU (AMD Ryzen AI Max+ 395)

These Linux measurements were made with the cache compiled in on Linux, before it was restricted to macOS.

Build from commit 50097ab57: CPU only, MKL + oneDNN, clang, `OPENMP_RUNTIME=COMP` (base allocator is
`MklAllocator`). Whisper large-v3, first 60 s of `podcast120` (Catalan), greedy, no timestamps,
`cpu_threads=16`. A fresh process per run (1 warmup + 2 timed), cache off (`CT2_CPU_ALLOCATOR_CACHE_MB=0`)
and on (1024) alternated over 3 rounds, so each median covers 6 samples. No noise-control pair.

| compute_type | cache off | cache on | time change | peak RSS off -> on |
|---|---|---|---|---|
| float32 | 15.19 s | 14.91 s | -1.8% | 6.8 GB -> 7.7 GB |
| int8 | 8.10 s | 8.12 s | +0.2% | 3.1 GB -> 3.7 GB |
| int8_float32 | 8.17 s | 8.10 s | -0.9% | 3.1 GB -> 3.7 GB |

Transcripts are identical in all runs. Only float32, int8 and int8_float32 are supported on this CPU.

Conclusion: on Linux the cache gives no measurable speedup (all differences within ~2%, within noise) and
costs 0.6-0.9 GB of extra peak RSS. Probable reasons (not profiled): decode time is dominated by weight
reads and GEMM, and glibc already reuses large freed blocks, unlike macOS libmalloc which `madvise`s them back.
For this reason the cache is compiled only on macOS.

## Results: Linux ROCm GPU (AMD Ryzen AI Max+ 395, Radeon 8060S)

Same machine and workload as above (cache compiled in, as noted there), but a ROCm build (`build_amd.sh`, MKL + oneDNN) with `device="cuda"`. Only
the CPU allocator (host-side tensors) is affected by the cache; GPU memory uses a different allocator.
Cache off/on alternated over 3 rounds, a fresh process per run, 3 timed passes after 1 warmup, so each median
covers 9 samples. No noise-control pair.

| compute_type | cache off | cache on | time change | peak RSS off -> on |
|---|---|---|---|---|
| float32 | 16.69 s | 16.69 s | 0.0% | 6.2 GB -> 7.1 GB |
| float16 | 7.96 s | 7.96 s | 0.0% | 3.2 GB -> 3.2 GB |
| bfloat16 | 8.15 s | 8.11 s | -0.6% | 3.6 GB -> 3.4 GB |
| int8 | 7.78 s | 7.83 s | +0.6% | 3.2 GB -> 3.8 GB |
| int8_float32 | 9.53 s | 9.54 s | +0.1% | 3.2 GB -> 3.8 GB |
| int8_float16 | 7.78 s | 7.74 s | -0.5% | 3.2 GB -> 3.8 GB |
| int8_bfloat16 | 7.92 s | 7.95 s | +0.4% | 3.2 GB -> 3.8 GB |

Transcripts are identical in all runs. All differences are within +-0.6% (noise; the sign flips between
compute types). Peak RSS rises by 0.6-0.9 GB for float32 and the int8 types, and is unchanged or slightly
lower for float16/bfloat16 (not checked against noise). Conclusion: no speed benefit on ROCm.

## Not measured

- Windows.
- drama3 (the benchmark script is CUDA/ROCm only; the model is 46 GB).
- Per-token decode time, `MALLOC_MMAP_THRESHOLD_` and jemalloc comparisons on Linux.
- AVX512/VNNI/AMX machines.
- Multi-threaded servers with several models or many concurrent translators (cache is process-wide).
- Peak memory under variable-length batches.

## How to benchmark on another machine

The cache is only built on macOS. To measure another platform, remove the `#ifdef __APPLE__` guards in
`src/cpu/allocator.cc`. Run the same workload twice, changing only the environment variable:

```
CT2_CPU_ALLOCATOR_CACHE_MB=0    python bench.py   # baseline (passthrough, same code path)
CT2_CPU_ALLOCATOR_CACHE_MB=1024 python bench.py   # caching
```

Alternate the two runs several times, take medians, and run the baseline twice to measure noise.
Record total time, per-token decode time and peak RSS (`resource.getrusage(...).ru_maxrss`; bytes on
macOS, KB on Linux). Check that the transcripts match. Also try `MALLOC_MMAP_THRESHOLD_=33554432` (Linux,
without the patch's cache) and `LD_PRELOAD=libjemalloc.so`, to see how much of the gain a system
allocator already gives.

## Expectation for Linux and Windows

Not compiled there. For reference, the earlier reasoning:

- Linux/glibc: moderate at best. Blocks under 32 MB are probably already reused by glibc. The large
  attention score tensors can exceed it, and glibc trims the heap on free, so some gain is plausible.
  This needs measuring.
- Windows: unknown. The Windows heap and `_aligned_malloc` differ from both. I would expect some gain,
  but I have no basis for a number.
- Wherever the system allocator already caches, the cost of the patch is just the mutex and the header.

## Risks

- Memory held in the cache is not returned to the OS until the cap is hit or `clear_cache()` is called.
  Lower `CT2_CPU_ALLOCATOR_CACHE_MB` for memory-constrained deployments.
- Process-wide cache: with many models loaded, the cap is shared, not per model.
- Rounding wastes up to 25% of each cached block.
