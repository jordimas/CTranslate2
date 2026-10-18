# Drama3 GPU inference

Drama3 INT8 GPU inference uses resident backbone and tower execution by default.
Routing, grouped expert projections and recurrent/attention state stay on the
GPU. Waveform decoding also runs on the GPU, using float32 intermediates and
bounded convolution workspace. CPU inference remains available. CUDA and ROCm
builds both expose the device name `cuda`.

The resident backbone currently supports `int8_float32`; other GPU compute types
retain projection-based execution. Waveform GPU execution supports float32
independently of projection compute type.

For ROCm, the measured configuration uses the caching allocator:

```bash
CT2_CUDA_ALLOCATOR=cub_caching \
CT2_CUDA_CACHING_ALLOCATOR_CONFIG=2,6,28,536870912 \
uv run benchmark.py --device cuda --compute-type int8 --threads 8
```

Run this command from the faster_tts benchmark directory with a matching native
library installed. The allocator setting is explicit; this change does not
change the process-wide allocator default. The 512 MiB cap limits free cached
buffers, not live weights or active workspace.

On Radeon 8060S, the development candidate achieved 2.20 times CPU throughput
on one completed prompt, measured by real-time factor. Broader speech validation
is in progress. A Spanish two-sentence request reached the frame limit during
that validation; baseline attribution is pending. These results do not establish
general speech-quality acceptance or NVIDIA hardware validation.

All routed expert weights are cached on the GPU; approximately 30 GiB of INT8
expert payload is required for the tested model, in addition to scales, other
weights, state and workspace. CPU source weights remain retained. Process RSS
alone does not capture total GPU or unified-memory use.

For diagnostics, `CT2_DRAMA_GPU_RESIDENT_BACKBONE=0` restores projection-based
backbone/tower execution and `CT2_DRAMA_GPU_WAVEFORM=0` restores CPU waveform
decoding. Both settings must be present before model construction. Intermediate
nonfinite-input and transactional cache-error parity with CPU remains incomplete;
the implementation is experimental.
