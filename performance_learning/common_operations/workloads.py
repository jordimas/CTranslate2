"""Release-gate-shaped workloads with profiler bracketing.

Each run_* function wraps the actual inference between
ctranslate2._ext.init_profiling / dump_profiling and writes the raw dump
to the path provided. Metrics are also returned for sanity checks.
"""

import contextlib
import os
import tempfile
import time

import ctranslate2

CPU_THREADS = 2
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
OPUS_MODEL = "Helsinki-NLP/opus-mt-en-de"
T5GEMMA_MODEL = "jordimas/t5gemma-2-270m-270m"
T5GEMMA_REVISION = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"


def _profile(device, num_threads, prof_path, fn):
    ctranslate2._ext.init_profiling(device, num_threads)
    try:
        start = time.perf_counter()
        result = fn()
        result["wall_sec"] = time.perf_counter() - start
    finally:
        ctranslate2._ext.dump_profiling(prof_path)
    return result


def _load_librispeech(limit):
    import io
    import soundfile
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for x in list(ds.take(limit)):
        audio, sr = soundfile.read(io.BytesIO(x["audio"]["bytes"]))
        out.append((audio, sr, x["text"]))
    return out


def run_whisper(whisper_model, compute_type, device, prof_path,
                limit=20, model_dir=None):
    """whisper_model: HF id like 'openai/whisper-medium' OR a converted dir."""
    from faster_whisper import WhisperModel

    samples = _load_librispeech(limit)
    model_arg = model_dir if model_dir else whisper_model
    # faster-whisper accepts a size string, an HF id, or a local CT2 dir.
    model = WhisperModel(model_arg, device=device, compute_type=compute_type,
                         cpu_threads=CPU_THREADS, num_workers=1)

    def do():
        # Warm-up on first sample (not profiled with final dump).
        model.transcribe(samples[0][0], language="en", beam_size=1,
                         temperature=0.0, vad_filter=False)
        # Reset profiler to drop warmup.
        ctranslate2._ext.init_profiling(device, CPU_THREADS)
        hyps = []
        audio_sec = 0.0
        for audio, sr, _ in samples:
            segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                            temperature=0.0, vad_filter=False)
            hyps.append(" ".join(s.text for s in segments).strip())
            audio_sec += len(audio) / sr
        return {"num_samples": len(samples), "audio_sec": audio_sec,
                "hyps_sample": hyps[:3]}

    return _profile(device, CPU_THREADS, prof_path, do)


def run_opus(compute_type, device, prof_path, limit=100, model_dir=None):
    import sacrebleu
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(OPUS_MODEL)
    src_path = sacrebleu.utils.get_source_file("wmt14", "en-de")
    sources = open(src_path, encoding="utf-8").read().splitlines()[:limit]

    translator = ctranslate2.Translator(model_dir, device=device,
                                         compute_type=compute_type,
                                         intra_threads=CPU_THREADS, inter_threads=1)
    tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
                 for s in sources]

    def do():
        translator.translate_batch(tokenized[:4], beam_size=4, max_batch_size=16)
        ctranslate2._ext.init_profiling(device, CPU_THREADS)
        results = translator.translate_batch(tokenized, beam_size=4,
                                              max_batch_size=16)
        total_tokens = sum(len(r.hypotheses[0]) for r in results)
        return {"num_sentences": len(sources), "total_tokens": total_tokens}

    return _profile(device, CPU_THREADS, prof_path, do)


def run_t5gemma(compute_type, device, prof_path, limit=20, model_dir=None):
    import transformers
    from datasets import load_dataset

    samples = load_dataset("rajpurkar/squad", split="validation",
                           revision=SQUAD_REVISION).shuffle(seed=42)[:limit]
    prompts = [f"Context: {c} Question: {q} Answer:"
               for c, q in zip(samples["context"], samples["question"])]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        T5GEMMA_MODEL, revision=T5GEMMA_REVISION)
    inputs = [tokenizer.convert_ids_to_tokens(
                tokenizer.encode(p, truncation=True, max_length=512))
              for p in prompts]
    translator = ctranslate2.Translator(model_dir, device=device,
                                         compute_type=compute_type,
                                         intra_threads=CPU_THREADS, inter_threads=1)

    def do():
        translator.translate_batch(inputs[:2], beam_size=1, sampling_topk=1,
                                    max_decoding_length=64, max_batch_size=8)
        ctranslate2._ext.init_profiling(device, CPU_THREADS)
        results = translator.translate_batch(inputs, beam_size=1,
                                              sampling_topk=1,
                                              max_decoding_length=64,
                                              max_batch_size=8)
        total_tokens = sum(len(r.hypotheses[0]) for r in results)
        return {"num_prompts": len(prompts), "total_tokens": total_tokens}

    return _profile(device, CPU_THREADS, prof_path, do)
