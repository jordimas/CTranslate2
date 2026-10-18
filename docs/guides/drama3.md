# Drama 3

Drama 3 supports experimental offline text-to-speech synthesis on CPU, with
optional CUDA or ROCm acceleration for backbone and tower projections.
The native runtime generates audio codes and decodes them to a
mono waveform.

The examples below use conversion and inference APIs included in the
`ctranslate2` package. Prompt preparation and WAV writing are shown in full;
no companion TTS package, sample scripts, or separate source checkout is required.

## Installation

Use a CTranslate2 build with the `Drama3` extension. Build the
native library and Python bindings as described in the
[installation guide](../installation.md). The Drama3 bindings require
pybind11 >=2.12 for NumPy 2 compatibility, or >=3.1 with CPython 3.14.
Portable release wheels for this feature are not yet provided.

```bash
python -c 'from ctranslate2._ext import Drama3'
```

## Model conversion

Use the complete official Drama 3 checkpoint, including indexed safetensors
shards, `codec.pth`, configuration, tokenizer, template, and licensing files.
Install the `ctranslate2` Python package, NumPy, and PyTorch for conversion.
PyTorch reads the codec checkpoint; it is not needed for synthesis.

Use the packaged CTranslate2 converter with a local checkpoint and a new output
directory:

```bash
ct2-transformers-converter \
  --model /path/to/drama-3-official \
  --output_dir /path/to/drama-3-ct2 \
  --quantization float32 --dry-run
```

Remove `--dry-run` to convert. The dry run audits mappings and shapes without
loading weight payloads or creating the model directory. `--report plan.json`
saves the audit. The converter performs no downloads. Use `--force` explicitly
to replace an existing output directory.

The Python API uses the standard CTranslate2 `Converter` interface:

```python
from ctranslate2.converters import TransformersConverter

converter = TransformersConverter("/path/to/drama-3-official")
converter.convert("/path/to/drama-3-ct2", quantization="float32")
```

The converter detects `model_type: fish_omni` in the local checkpoint's
configuration and dispatches to the Drama 3 loader. No separate command or
Fish Speech installation is needed. It uses the standard CTranslate2 model
specification, validation, optimization, and serialization. Routed experts retain
original E4M3FN FP8 bytes and 128-by-128 block scales. Sensitive matrices and the
codec remain FLOAT32 with default or `float32` conversion. Other standard conversion
types (`float16`, `bfloat16`, `int8`, `int8_float32`, `int8_float16`,
`int8_bfloat16`, and `int16`) decode the FP8 weights before applying the standard
CTranslate2 quantizer to non-codec matrices. Normalization, recurrent parameters,
and codec weights remain FLOAT32. For example, use `--quantization int8_float32`
for INT8 storage with FLOAT32 activations. Vocabulary mapping is unsupported. The codec
tensor contract is included in the installed Python package; conversion does
not require development scripts or a Fish Speech checkout.

The converter holds the model and codec weights in memory before serialization.
A dry run checks mappings and shapes, but does not validate the converted
waveform or speech quality.

## Text-to-speech

For speaker tags, bracketed performance directions, paired delivery tags, and
phoneme markers, see the [official Drama 3 prompting guide](https://docs.fish.audio/features/text-to-speech/drama-3).
Bracketed directions are ordinary text rather than a fixed token vocabulary.
That reference describes the upstream model; CTranslate2's supported scope is
listed below.

The native Python binding accepts prepared prompt rows. The following example
builds a single-speaker, text-only prompt using the converted model's local
tokenizer and saves the result as a mono PCM16 WAV. Install `tokenizers` for
prompt preparation.

```python
from pathlib import Path
import wave

import numpy as np
from tokenizers import Tokenizer
from ctranslate2._ext import Drama3

model_path = Path("/path/to/drama-3-ct2")
tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
tokenizer.no_padding()
tokenizer.no_truncation()
text = "Hello. This is a speech synthesis test."
segments = (
    "<|im_start|>system\n", "convert the provided text to speech",
    "<|im_end|>\n", "<|im_start|>user\n", "<|speaker:0|> " + text,
    "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
)
ids = [token for segment in segments for token in tokenizer.encode(segment).ids]
prompt_rows = np.zeros((len(ids), 11), dtype=np.int32)
prompt_rows[:, 0] = ids

model = Drama3(str(model_path), intra_threads=8, compute_type="default")
result = model.generate(prompt_rows, max_frames=256, seed=0)
print(result["sample_rate"], result["audio"].shape, result["finish_reason"])
pcm = np.rint(np.clip(result["audio"], -1, 1) * 32767).astype("<i2")
with wave.open("hello.wav", "wb") as wav:
    wav.setparams((1, 2, result["sample_rate"], 0, "NONE", "not compressed"))
    wav.writeframes(pcm.tobytes())
```

Encode each segment separately to preserve the TTS prompt's token boundaries.
The generic chat template is not a replacement for this prompt. This example
expects ordinary text without reserved prompt tokens or speaker tags.
`prompt_rows` is a contiguous INT32 array with shape `[rows, 11]`. Its first
column contains tokenizer IDs; the ten codec columns are zero for text rows.
The native API does not accept a plain text string or encode reference audio.

Local model paths do not trigger downloads. Synthesis requires neither PyTorch
nor Transformers, and does not depend on audio.cpp or GGML.

### Generation controls and results

* `max_frames` bounds the generated audio frames (default 512). Each frame
  represents 2,048 samples at 44,100 Hz, approximately 46 milliseconds.
* `temperature`, `top_k`, and `top_p` control sampling, with defaults 0.8, 30,
  and 0.8 respectively. `seed` defaults to 0. Identical seeds do not promise
  identical random trajectories across runtimes or compute types.
* `greedy=True` enables greedy decoding for diagnosis.
* `prefill_chunk_size`, supplied to `generate`, controls prompt processing
  chunk size (default 16); it does not enable streaming audio.
* `result["finish_reason"]` is `"eos"` when the model generates its end token,
  or `"length"` when it reaches the frame budget and may cut off the utterance.
* `result["audio"]` is mono FLOAT32 audio and `result["codes"]` is an INT32
  array with shape `[frames, 10]`.
* `result["timings"]` reports native prefill, backbone, tower, codec, and total
  execution durations, excluding model loading and Python-side processing.

The native constructor accepts `model_path`, `intra_threads=8`,
`compute_type="default"`, `device="cpu"`, and `device_index=0`.
It accepts the standard CTranslate2 compute types,
including `auto`, `float32`, `int8`, and `int8_float32`. Availability follows the
CPU backend when `device="cpu"`: explicit unsupported types raise the standard error, while `default`
and `auto` use standard fallback rules. FLOAT16 and BFLOAT16 storage can therefore
be converted for CPU execution at load time. The synthesis graph and codec use
FLOAT32 activations.

With default FP8 storage and FLOAT32 execution, selected expert weights are
decoded on demand. Requesting INT8 or INT16 execution decodes and quantizes the
matrices during loading, which requires additional temporary memory. By default,
all expert weights remain resident; sparse selection reduces computation, not
weight storage. The optional CPU memory budget below enables mapped experts.
Requests on one model instance execute sequentially with independent per-call
state.

### CUDA and ROCm

Build the native library and Python extension with `WITH_CUDA=ON` for NVIDIA,
or `WITH_HIP=ON` for AMD, as described in the
[installation guide](../installation.md). Both builds use `device="cuda"`:

```python
model = Drama3(
    str(model_path), device="cuda", device_index=0, compute_type="float16"
)
```

`device="auto"` selects a GPU when one is available. Projection precision follows
the selected GPU backend's standard compute-type rules, including FLOAT32,
FLOAT16, BFLOAT16, and supported INT8 variants. Unsupported explicit types raise
an error; `default` and `auto` use standard fallback rules.

This is hybrid execution: backbone and tower dense projections, including routed
experts, execute on the GPU. Attention, recurrent state updates, routing,
sampling, embeddings, and the entire waveform codec execute on CPU. Activations
are transferred for each projection; this is not an entirely GPU-resident graph.
There is no tensor parallelism.

The host checkpoint remains resident, with non-FP8 parameters loaded as FLOAT32
even when GPU projections use lower precision. Projection weights are uploaded on first
use and cached on the selected GPU, so GPU memory usage grows as new experts
are selected. FP8 expert bytes remain FP8 in that cache for floating-point
execution and are decoded on the GPU for each projection. INT8 execution instead
quantizes each projection once when it is first used. Converted checkpoints do
not need to change, and output audio and codes remain CPU arrays.

### Memory settings

For local standard INT8 checkpoints, two optional environment variables control memory use:

- [`CT2_DRAMA3_MAX_MEMORY_GB`](../environment_variables.md#ct2_drama3_max_memory_gb)
  limits the whole process's CPU memory in decimal GB; CPU execution only.
- [`CT2_DRAMA3_GPU_EXPERT_CACHE_GB`](../environment_variables.md#ct2_drama3_gpu_expert_cache_gb)
  limits cached GPU expert weights in decimal GB on CUDA or ROCm, not total VRAM.

Both are disabled when unset or `0`. See the linked environment variable documentation
for supported compute types, requirements, and limitations.

## Supported scope

The implementation provides offline synthesis and waveform
decoding. It does not provide reference-audio voice cloning, audio encoding,
voice design, a dedicated multi-speaker API, speech editing or history,
automatic long-text splitting, public streaming, or batching.
Transcription is a separate speech-recognition task.

Voice cloning would require a reference-audio encoder and reference-conditioned
prompt preparation. The existing waveform decoder alone cannot turn a reference
recording into the codes needed to condition generation.

