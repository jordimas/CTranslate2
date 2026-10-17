The standalone `faster_tts` Python library is developed in
[jordimas/faster_tts](https://github.com/jordimas/faster_tts), on the `dev` branch.
Clone it into `tts/faster_tts` for local development.

Native runtime, bindings, and converters remain in CTranslate2.

Conversion uses the standard CTranslate2 `ModelSpec.validate()`, `optimize()`
and `save()` methods. Install the Python package from this branch and PyTorch
(needed only to load the codec checkpoint), then run:

```sh
python tts/dev/convert_drama3_model.py --source /path/to/official-checkpoint \
  --output /path/to/converted-model --dry-run
```

Remove `--dry-run` to convert. The default preserves sensitive matrices and the
codec in F32 and quantizes routed experts with standard INT8 quantization,
matching the current mixed checkpoint policy. Source FP8 block scales are
applied in F32; there is no intermediate BF16 cast. `--quantization float32`
disables INT8 conversion entirely.

The official checkpoint requires **134.42 GiB for F32 weights alone**, plus
optimization and serialization overhead. Full conversion has not completed on
this 122.8 GiB machine; the RAM preflight rejects it before loading weights.
There is no disk-backed staging or alternate streaming converter. A full
conversion and weight-parity check on a machine with sufficient RAM remain
required before replacing the existing checkpoint.
