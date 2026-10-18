# Environment variables

Some environment variables can be configured to customize the execution. When using Python, these variables should be set before importing the `ctranslate2` module, e.g.:

```python
import os
os.environ["CT2_VERBOSE"] = "1"

import ctranslate2
```

```{note}
Boolean environment variables can be enabled with `"1"` or `"true"`.
```

## `CT2_CUDA_ALLOCATOR`

Allocating memory on the GPU with `cudaMalloc` is costly and is best avoided in high-performance code. For this reason CTranslate2 integrates caching allocators which enable a fast reuse of previously allocated buffers. The following allocators are integrated:

* `cuda_malloc_async` (default for CUDA >= 11.2)<br/>Uses the [asynchronous allocator with memory pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html) introduced in CUDA 11.2.
* `cub_caching` (default for CUDA < 11.2)<br/>Uses the caching allocator from the [CUB project](https://github.com/NVIDIA/cub).

## `CT2_CUDA_ALLOW_BF16`

Allow using BF16 computation on GPU even if the device does not have efficient BF16 support.

## `CT2_CUDA_ALLOW_FP16`

Allow using FP16 computation on GPU even if the device does not have efficient FP16 support.

## `CT2_CUDA_TRUE_FP16_GEMM`

Allow using true FP16 computation in GEMM operations. When disabled, the computation or accumulation may use FP32 instead.

This flag is enabled by default, but some models may automatically disable it when they are known to work better with the increased precision.

## `CT2_CUDA_CACHING_ALLOCATOR_CONFIG`

The `cub_caching` allocator can be configured to tradeoff memory usage and speed. By default, CTranslate2 uses the following values which have been selected experimentally:

* `bin_growth = 4`
* `min_bin = 3`
* `max_bin = 12`
* `max_cached_bytes = 209715200` (200MB)

You can override these parameters with comma-separated values in the same order as the list above:

```bash
export CT2_CUDA_CACHING_ALLOCATOR_CONFIG=8,3,7,6291455
```

See the description of each parameter in the [allocator implementation](https://github.com/NVIDIA/cub/blob/main/cub/util_allocator.cuh).

## `CT2_FORCE_CPU_ISA`

Force CTranslate2 to select a specific instruction set architecture (ISA). Possible values are:

* `GENERIC`
* `AVX`
* `AVX2`
* `AVX512`

```{attention}
This does not impact backend libraries (such as Intel MKL) which usually have their own environment variables to configure ISA dispatching.
```

## `CT2_DRAMA3_MAX_MEMORY_GB`

Set an absolute CPU memory budget for Drama3 in **decimal GB** (1 GB =
1,000,000,000 bytes), for example `CT2_DRAMA3_MAX_MEMORY_GB=20`.

Unset, empty, or `0` disables this option. Positive values, including fractions,
automatically enable mapped INT8 expert weights. Invalid, negative, nonfinite,
or out-of-range values raise an error. Other model types ignore this variable.

**This limit applies only to CPU system RAM. It does not limit GPU VRAM.**
The option currently requires Linux cgroup v2, a local standard INT8 Drama3
checkpoint, and CPU `int8` or `int8_float32` compute. GPU requests fail explicitly.

The cap covers the **entire calling process**, its charged file cache, kernel
memory, and subsequent children. Swap is disabled. The limit persists until
process exit, including after model unloading; use a dedicated inference worker
when embedding CT2 in a larger application. It does not reserve RAM or change
quantization. A budget below the non-reclaimable working set can cause an OOM kill.

CT2 reuses an equal or tighter effective cgroup memory limit if swap is already
disabled. Otherwise it uses `libsystemd.so.0` and the systemd user manager to create a
temporary scope containing this process only, preserving any tighter existing
memory limit. This does not modify the parent scope. Missing systemd libraries/user manager or
denied scope creation produces an error before loading weights. Containers
without systemd can instead configure a sufficient cgroup cap and disable swap
externally. Configure limits for all workers together if the container runs
multiple inference processes.

Set the variable before loading the model. Unsetting it later does not remove
an already-applied process limit. The existing `CT2_DRAMA_CPU_MMAP=1` flag still
enables mapping alone, without establishing a cap.

## `CT2_PACKED_GEMM`

Enable or disable the packed GEMM API for Intel MKL. Packed GEMM is enabled by default when using the MKL backend and improves decoding performance by pre-packing weight matrices at model load time. Set to `0` to disable it. See [Intel's article](https://software.intel.com/content/www/us/en/develop/articles/introducing-the-new-packed-apis-for-gemm.html) to learn more about packed GEMM.

## `CT2_USE_MKL`

Force CTranslate2 to use (or not) Intel MKL. By default, the runtime automatically decides whether to use Intel MKL or not based on the CPU vendor.

## `CT2_VERBOSE`

Configure the default logs verbosity:

* -3 = off
* -2 = critical
* -1 = error
* 0 = warning (default)
* 1 = info
* 2 = debug
* 3 = trace

```{tip}
The log level can also be controlled by API. See for example the Python function [`ctranslate2.set_log_level`](python/ctranslate2.set_log_level.rst).
```
