# Experimental Drama 3 CPU inference

This package requires the matching experimental CTranslate2 wheel exposing
`Drama3` and a converted checkpoint. Ordinary upstream CT2 wheels do not guarantee
that binding. This is a local functional prototype; held-out generation reliability
and portable installation remain open.

```python
from faster_tts import Model

model = Model("/path/to/converted-drama3", compute_type="int8_float32")
result = model.synthesize("Hello, how are you?")
print(result.sample_rate, result.finish_reason)
```

A bounded retry is optional and disabled by default:

```python
result = model.synthesize("Hello, how are you?", seed=0, max_retries=1)
print(result.attempts)  # Includes capped attempts and their cost.
```

Only sampled generation ending at the frame limit retries once, on the same
loaded model with fresh native state and the next seed. EOS, greedy decoding,
and errors never retry. This can double computation; a second capped attempt
returns its partial audio. Attempt metadata is retained, discarded audio/codes
are not, and `result.timings` sums native phase timings across all attempts.
Retrying does not detect incomplete speech that ends with EOS or establish
speech quality.

## Compute type and memory

Use `compute_type="int8_float32"` (the default) with a checkpoint converted using
the intended quantization policy. With `experts_int8_sensitive_float32`, routed
experts remain INT8 while the other 519 matrix weights remain FLOAT32, including
embeddings, routers, shared experts and output heads. The independently audited
mixed checkpoint has 47,825,164,164 bytes of tensor payload (44.54 GiB); runtime
memory additionally includes caches, scratch buffers and process overhead. This
payload size is not a measured inference peak or a quality acceptance result.

`compute_type="float32"` dequantizes the routed expert bank during native loading;
it does not merely select FLOAT32 activations. Those 30,720 expert matrices alone
expand from 30 GiB of INT8 values to **120 GiB** of FLOAT32 values. The complete
FLOAT32 tensor payload is approximately **134.42 GiB** (144,336,099,204 bytes),
before runtime allocations or conversion temporaries. It exceeds both this
machine's approximately 122 GiB RAM and the intended 60 GiB memory budget. This
estimate follows the audited tensor inventory and loader conversion behavior;
no full FLOAT32 load was attempted. FLOAT32 remains available for machines with
sufficient memory.

The mixed checkpoint's on-disk audit covers all 519 FLOAT32 matrices. Native tiny
fixtures verify representative weight preservation and FLOAT32 Dense execution;
they do not enumerate all 519 variables in a freshly loaded full model.
