"""ASR WER task.

Model: openai/whisper-tiny (converted on the fly).
Data:  LibriSpeech test-clean subset, streamed from HF datasets.
Metric: WER (jiwer) + tokens_per_sec.
"""

import os
import tempfile
import time

MODEL_NAME = "openai/whisper-tiny"
DEFAULT_LIMIT = 300


def _convert_model(output_dir):
    from ctranslate2.converters.transformers import TransformersConverter
    TransformersConverter(MODEL_NAME).convert(output_dir, force=True)


def _load_samples(limit):
    from datasets import load_dataset
    ds = load_dataset(
        "librispeech_asr", "clean", split="test", streaming=True,
        trust_remote_code=True,
    )
    samples = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        samples.append((ex["audio"]["array"], ex["audio"]["sampling_rate"], ex["text"]))
    return samples


def run(limit=None, device="cpu"):
    import ctranslate2
    import jiwer
    import transformers

    limit = limit or DEFAULT_LIMIT

    processor = transformers.WhisperProcessor.from_pretrained(MODEL_NAME)
    samples = _load_samples(limit)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "ct2")
        _convert_model(model_dir)
        model = ctranslate2.models.Whisper(model_dir, device=device)

        hypotheses = []
        references = []
        total_tokens = 0

        prompt = processor.tokenizer.convert_tokens_to_ids(
            [
                "<|startoftranscript|>",
                "<|en|>",
                "<|transcribe|>",
                "<|notimestamps|>",
            ]
        )

        start = time.perf_counter()
        for audio, sr, text in samples:
            inputs = processor(audio, sampling_rate=sr, return_tensors="np")
            features = ctranslate2.StorageView.from_array(inputs.input_features)
            results = model.generate(features, [prompt], beam_size=1)
            tokens = results[0].sequences_ids[0]
            total_tokens += len(tokens)
            hyp = processor.decode(tokens, skip_special_tokens=True)
            hypotheses.append(hyp.strip().lower())
            references.append(text.strip().lower())
        elapsed = time.perf_counter() - start

    wer = jiwer.wer(references, hypotheses)

    return {
        "wer": wer,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "num_samples": len(samples),
        "elapsed_sec": elapsed,
    }
