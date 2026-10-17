# Drama 3

Drama 3 is supported experimentally for offline text-to-speech synthesis on CPU
in this branch. The native runtime generates audio codes and decodes them to a
mono waveform.

## Installation

Use a CTranslate2 build from this branch with the `Drama3` extension. Build the
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
Install the Python package from this branch, NumPy, and PyTorch for conversion.
PyTorch reads the codec checkpoint; it is not needed for synthesis.

Use the packaged CTranslate2 converter with a local checkpoint and a new output
directory:

```bash
ct2-transformers-converter \
  --model /path/to/drama-3-official \
  --output_dir /path/to/drama-3-ct2 \
  --quantization int8_float32 --dry-run
```

Remove `--dry-run` to convert. The dry run audits mappings and shapes without
loading weight payloads or creating the model directory. `--report plan.json`
saves the audit. The converter performs no downloads. Use `--force` explicitly
to replace an existing output directory.

The Python API uses the standard CTranslate2 `Converter` interface:

```python
from ctranslate2.converters import TransformersConverter

converter = TransformersConverter("/path/to/drama-3-official")
converter.convert("/path/to/drama-3-ct2", quantization="int8_float32")
```

The converter detects `model_type: fish_omni` in the local checkpoint's
configuration and dispatches to the Drama 3 loader. No separate command or
Fish Speech installation is needed. It uses the standard CTranslate2 model
specification, validation, optimization, and serialization. By default, routed
expert matrices use INT8 while sensitive matrices and the codec remain FLOAT32.
`--quantization float32` disables INT8 conversion. Other quantization types and
vocabulary mapping are not supported. Source FP8 block scales are applied in
FLOAT32, then routed expert weights are stored temporarily in BF16 before INT8
quantization. This rounding can change the final quantized weights. Sensitive
tensors remain FLOAT32. The codec tensor contract is included in the installed Python package;
conversion does not require the development scripts or a Fish Speech checkout.

```{warning}
The converter holds the model and codec weights in memory before optimization.
BF16 expert staging approximately halves expert storage; temporary FP32 buffers
are still needed during decoding and quantization. Full FLOAT32 conversion needs
approximately 134.42 GiB for weights alone, plus conversion overhead. A successful dry run
checks mappings and shapes, but does not validate the converted waveform or
speech quality.
```

## Text-to-speech

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

model = Drama3(str(model_path), intra_threads=8, compute_type="int8_float32")
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

The native constructor accepts `model_path`, `intra_threads=8`, and
`compute_type="int8_float32"`. Both `int8_float32` and `float32` are supported.
FLOAT32 requires substantially more model memory. All expert weights remain
resident; sparse expert selection reduces computation, not weight storage.
Requests on one model instance execute sequentially with independent per-call
state.

## Supported scope

The implementation currently provides offline CPU synthesis and waveform
decoding. It does not provide reference-audio voice cloning, audio encoding,
voice design, a dedicated multi-speaker API, speech editing or history,
automatic long-text splitting, public streaming, batching, or GPU execution.
Transcription is a separate speech-recognition task.

Voice cloning would require a reference-audio encoder and reference-conditioned
prompt preparation. The existing waveform decoder alone cannot turn a reference
recording into the codes needed to condition generation.

English and Spanish smoke tests have produced intelligible speech, but broader
evaluation has also found repeated codes, prolonged silence, and frame-limit
failures. Quality, speaker similarity, and production reliability have not been
established. Component numerical parity and successful conversion do not by
themselves establish speech quality.
