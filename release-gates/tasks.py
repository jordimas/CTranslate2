"""Release-gate tasks. Each `run_*` returns a flat dict of metrics."""

import tempfile
import time


def _load_librispeech(limit):
    import io
    import soundfile
    from datasets import Audio, load_dataset
    # Parquet-backed mirror maintained by HF for CI; the canonical
    # `librispeech_asr` loading script was removed from the Hub.
    # decode=False avoids pulling torchcodec/ffmpeg system libs; we decode
    # the raw FLAC bytes with soundfile instead.
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for x in list(ds.take(limit)):
        audio, sr = soundfile.read(io.BytesIO(x["audio"]["bytes"]))
        out.append((audio, sr, x["text"]))
    return out


def run_whisper_wer(limit=300, device="cpu"):
    import jiwer
    from faster_whisper import WhisperModel

    samples = _load_librispeech(limit)
    model = WhisperModel("tiny", device=device, compute_type="int8")

    refs, hyps, audio_sec = [], [], 0.0
    start = time.perf_counter()
    for audio, sr, text in samples:
        audio_sec += len(audio) / sr
        segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                       vad_filter=False)
        hyps.append(" ".join(s.text for s in segments).strip().lower())
        refs.append(text.strip().lower())
    elapsed = time.perf_counter() - start

    return {"wer": jiwer.wer(refs, hyps),
            "rtf_inverse": audio_sec / elapsed,
            "num_samples": len(samples)}


def run_opus_bleu(limit=500, device="cpu"):
    import ctranslate2
    import sacrebleu
    import transformers
    from ctranslate2.converters.transformers import TransformersConverter

    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    src_path = sacrebleu.utils.get_source_file("wmt14", "en-de")
    ref_paths = sacrebleu.utils.get_reference_files("wmt14", "en-de")
    sources = open(src_path, encoding="utf-8").read().splitlines()[:limit]
    refs = [open(p, encoding="utf-8").read().splitlines()[:limit]
            for p in ref_paths]

    with tempfile.TemporaryDirectory() as tmp:
        TransformersConverter(model_name).convert(tmp, force=True)
        translator = ctranslate2.Translator(tmp, device=device)
        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
                     for s in sources]
        total_tokens = sum(len(t) for t in tokenized)
        start = time.perf_counter()
        results = translator.translate_batch(tokenized, beam_size=4,
                                             max_batch_size=16)
        elapsed = time.perf_counter() - start

    hyps = [tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                             skip_special_tokens=True)
            for r in results]
    return {"bleu": sacrebleu.corpus_bleu(hyps, refs).score,
            "tokens_per_sec": total_tokens / elapsed,
            "num_sentences": len(sources)}


TASKS = {"whisper_wer": run_whisper_wer, "opus_bleu": run_opus_bleu}


if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    kw = {"device": a.device}
    if a.limit is not None:
        kw["limit"] = a.limit
    json.dump(TASKS[a.task](**kw), sys.stdout)
    sys.stdout.write("\n")
