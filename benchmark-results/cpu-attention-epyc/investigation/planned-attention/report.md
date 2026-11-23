# Full-shape oneDNN planning with tiled execution

Date: 2026-11-23. Prototype built against the unchanged `eb9f7fb0` CT2 compatibility implementation and its existing oneDNN 3.1.1 static archive. CPU: AMD EPYC 7B13, effective ISA AVX2, eight physical cores pinned to 0–7. The normal CT2 source/build is unchanged.

## Implementation

The [isolated adapter](../../../../tools/onednn_attention_plan/README.md) creates an immutable plan for the full Whisper encoder shape, then executes query tiles using the original SGEMM kernels. This is an architectural prototype of full-shape planning/tiled execution, **not a BRGEMM backend or a oneDNN upgrade**. Keeping the original kernels isolates planning and scheduling from changes in arithmetic implementation.

- **QK:** construct the internal `gemm_info_t<float,float,float>` for the full 1500×1500×64 GEMM. Reuse its copy and compute pointers (`um=24`, `un=4`); pack/scale keys once per head; pack queries per outer tile. Preserve the complete depth-64 reduction and alpha placement.
- **PV:** explicitly select the reference no-copy strategy, preserve its full-query block schedule (30 blocks of 48, final block of 60), and execute those ranges through the private no-copy driver with the full 1500-key reduction and its original K-block policy. It still has per-block private-driver overhead.
- **Workspace:** group preplanned ranges within an outer target of 256 queries, yielding six 240-query tiles and a 60-query tail. Explicit worker buffers total about 1.89 MB (scores + packed keys + packed queries), versus a 9 MB whole-head score buffer; internal driver scratch is excluded. Softmax remains CT2's normal per-row kernel.

Eligibility is restricted to oneDNN 3.1.1 / effective AVX2, unmasked FP32 attention with query/key lengths 1500, all head depths 64, scale 0.125 and at least two batch/head entries. Other cases use the committed compatibility path. The private headers must match the linked archive; the runtime version check does not create ABI stability. No production dispatch rule has been added.

## Exact numerical checks

228 randomized cases passed exact QK, final attention, quantized-byte and quantization-scale checks. The default target covered 20 seeds × three magnitudes (0.01, 1, 10) × two thread counts (1, 8), or 120 cases. Six additional outer targets (96, 128, 192, 320, 512, 1500) covered three seeds × three magnitudes × two thread counts, adding 108 cases. Targets group full-plan blocks rather than splitting them arbitrarily.

Each case computes a reference full QK and final attention inside the same head-parallel context and compares bits. The two existing fused-attention regressions also pass, exercising the adapter plus unchanged fallback cases. Final first-window encoder tensors match bit-for-bit for every measured variant in both compute types (1,920,000 floats each). These are tested inputs, not a proof for all values, AVX2 machines, or full models/corpora.

A useful control from the single-thread, seed-0, magnitude-1 case: ordinary 256-query blocking changes 50,332/4,500,000 QK values and 459/192,000 PV values with identical probability inputs. The planned path changes zero QK/final values. Using the single-thread control avoids conflating the block-size effect with different internal GEMM thread counts. The PV result shows why preserving the selected algorithm alone is insufficient: internal block/tail boundaries matter too. See `randomized.log`.

## Attention operation

Twenty heads, 1500 queries/keys, depth 64. Alternating regular/fused measurements, 30 measured samples per mode per process, three processes per variant. Variant order was reversed in the middle round. Values average the three process means; each variant has its own paired regular control. No other benchmark ran concurrently.

| Variant | Fused ms | Paired regular ms | Speedup |
|---|---|---|---|
| committed whole-head | 29.02 | 34.36 | 1.184x |
| earlier tile240 | 22.13 | 33.95 | 1.534x |
| explicit SGEMM plan | 24.67 | 33.18 | 1.345x |

The planned operator takes 15.0% less time than the committed whole-head operator. The earlier unconstrained tile240 experiment is faster; explicit planning has not surpassed it in this prototype.

## Real large-v3 encoder

First 30 seconds of the local audio, same features padded to 3000 frames. One warm-up, three timed repetitions per worker process; mean ± sample standard deviation. Final tensors are compared by their 32-bit patterns, including signed zeros. Variants run sequentially; small differences and timing noise should not be overinterpreted.

| Compute | Variant | Mean ± SD, s | Time reduction vs regular | Changed float bits |
|---|---|---|---|---|
| int8_float32 | regular | 2.884 ± 0.073 | 0.0% | 0 |
| int8_float32 | head | 2.822 ± 0.159 | 2.1% | 0 |
| int8_float32 | tile240 | 2.445 ± 0.020 | 15.2% | 0 |
| int8_float32 | planned | 2.569 ± 0.022 | 10.9% | 0 |
| float32 | regular | 4.144 ± 0.040 | 0.0% | 0 |
| float32 | head | 4.239 ± 0.136 | -2.3% | 0 |
| float32 | tile240 | 3.744 ± 0.023 | 9.7% | 0 |
| float32 | planned | 3.850 ± 0.017 | 7.1% | 0 |

The planned encoder reduces time by 10.9% (int8) / 7.1% (float32) versus regular attention, or 9.0% / 9.2% versus the whole-head means from this run. The whole-head samples vary more, especially float32.

## Full transcription

Same 60-second English clip, beam 5, temperature 0, one warm-up and three timed runs per mode. Regular and planned use the **same prototype library** with fusion disabled/enabled, respectively. All warm-up and measured texts and segment start/end times are retained. The order is regular then planned for int8; planned then regular for float32. Values are mean ± sample standard deviation.

| Compute | Regular mean ± SD, s | Planned mean ± SD, s | Time reduction | Speedup |
|---|---|---|---|---|
| int8_float32 | 58.44 ± 0.19 | 56.87 ± 0.17 | 2.7% | 1.028x |
| float32 | 49.04 ± 1.16 | 48.80 ± 1.43 | 0.5% | 1.005x |

All eight outputs per compute type (16 total) have identical text and segment timestamps. The INT8 reduction is 2.7% in this run; the 0.5% float32 difference is much smaller than the timing spread and is treated as effectively flat. These measurements come from one process per variant, one CPU and one clip; they do not establish a general speedup or a statistically significant float32 end-to-end improvement. The regular clip output includes repeated ending text; matching it establishes compatibility here, not corpus-level recognition accuracy.

The original separate int8 profile placed encoding at about 16% of total time. That limits the end-to-end effect even when isolated encoder/operator gains are measurable. No claim is made that the measured total reduction is entirely explained by that earlier profile.

## Conclusions and limits

The prototype shows that preserving the original SGEMM algorithm, packing/scale handling and internal boundaries can recover part of the tiled-attention gain while retaining exact results on these tests. It does not show that arbitrary BRGEMM kernels will match SGEMM or that full-shape planning by itself guarantees numerical compatibility.

The current private-driver PV execution leaves optimization opportunities: group driver calls only when their generated block schedule is identical to the full plan, or hold direct compatible PV kernel pointers. Each change needs the same stage/quantization/encoder checks. A supported newer microkernel API is a separate dependency experiment and may change arithmetic. No production API/dependency change is justified solely by this single-CPU/single-clip result.

Artifacts: `stage-probe.log`, `randomized.json/log`, `ops-tests.log`, `op.json`, `op-paired.log`, `encoder.json`, encoder logs, and the full `e2e*.json/log` files. [Build, validation and benchmark tools](../../../../tools/onednn_attention_plan/README.md) include source/API limitations and reproduction commands.
