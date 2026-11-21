#!/usr/bin/env python3
"""End-to-end benchmark for the CPU fused attention (CT2_CPU_FUSED_ATTENTION).

One model per family that uses the fused attention:
  medium, large-v3   Whisper transcription (encoder self-attention, 1500 frames)
  nllb               NLLB-200 distilled 600M translation (encoder self-attention)
  xlm-roberta        XLM-RoBERTa base encoder forward pass
  gemma-3-4b         Gemma 3 4B IT prompt prefill plus a short generation (int8 only, disk)

For each model and compute type, the workload runs with the fused attention on and off,
each in its own process (the switch is read once per process). Every process loads the
model, does one untimed warm-up, then times --runs repetitions. Reported: average latency,
standard deviation, real-time factor (Whisper only) and quality metrics. The quality
metrics are the word error rate (WER) between the outputs of the two modes and whether
they match exactly (for the encoder: the largest difference of the output vectors), plus
the WER of each mode against --reference when given (Whisper only).

Models other than Whisper are converted from the Hugging Face cache on first use, which
needs `transformers` and `torch` in the Python environment.

Example, pinned to one thread per physical core, using the library found on the path:

    LD_LIBRARY_PATH=build/install/lib taskset -c 0-30:2 \\
        python tools/benchmark_encoder_fuse.py --audio speech.wav --duration 60
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000


def load_audio(path, seconds):
    import numpy as np

    path = Path(path)
    if path.suffix == ".f32":  # Raw 16 kHz float32 samples.
        audio = np.fromfile(path, dtype=np.float32)
    else:
        from faster_whisper import decode_audio

        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    return audio[: int(seconds * SAMPLE_RATE)] if seconds > 0 else audio


FAMILIES = {
    "nllb": dict(kind="translation", hf="facebook/nllb-200-distilled-600M", quantization="float16",
                 compute_types=["int8_float32", "float32"]),
    "xlm-roberta": dict(kind="encoder", hf="FacebookAI/xlm-roberta-base", quantization="float16",
                        compute_types=["int8_float32", "float32"]),
    # Stored as int8 (a float32 copy of 4B parameters does not fit the disk), so int8 only.
    "gemma-3-4b": dict(kind="decoder", hf="google/gemma-3-4b-it", quantization="int8",
                       compute_types=["int8_float32"]),
}

PARAGRAPH = (
    "The city council met on Tuesday evening to discuss the new transport plan, which would "
    "add three bus lines and extend the cycle network to the northern districts. Several "
    "residents spoke in favour, arguing that the current service is too slow and too "
    "infrequent, while shop owners worried about the loss of parking spaces near the market. "
    "After two hours of debate, the council agreed to run a one year pilot and to review the "
    "results before deciding on the permanent changes. "
)


def source_tokens(tokenizer, count):
    """Token ids of an English text of exactly `count` tokens (special tokens included)."""
    ids = tokenizer(PARAGRAPH * (count // 40 + 2), add_special_tokens=False)["input_ids"]
    body = ids[: count - len(tokenizer.build_inputs_with_special_tokens([]))]
    return tokenizer.build_inputs_with_special_tokens(body)


def timed(fn, runs):
    fn()  # Warm-up.
    times, result = [], None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return times, result


def worker(args):
    import numpy as np

    family = FAMILIES.get(args.model)
    if family is None:  # Whisper.
        from faster_whisper import WhisperModel

        audio = load_audio(args.audio, args.duration)
        model = WhisperModel(args.model_path, device="cpu", compute_type=args.compute_type,
                             cpu_threads=args.threads)

        def transcribe():
            segments, _ = model.transcribe(audio, language=args.language, beam_size=args.beam_size)
            return " ".join(segment.text.strip() for segment in segments)

        times, text = timed(transcribe, args.runs)
        out = {"text": text, "audio_seconds": len(audio) / SAMPLE_RATE}
    else:
        import ctranslate2
        from transformers import AutoTokenizer

        kind = family["kind"]
        options = dict(device="cpu", compute_type=args.compute_type, intra_threads=args.threads)
        if kind == "translation":
            tokenizer = AutoTokenizer.from_pretrained(family["hf"], src_lang="eng_Latn")
            translator = ctranslate2.Translator(args.model_path, **options)
            ids = source_tokens(tokenizer, args.tokens)
            batch = [tokenizer.convert_ids_to_tokens(ids)] * args.batch
            results_ = lambda: translator.translate_batch(
                batch, target_prefix=[["cat_Latn"]] * args.batch, beam_size=1,
                max_decoding_length=128)
            times, results = timed(results_, args.runs)
            out = {"text": " ".join(results[0].hypotheses[0])}
        elif kind == "encoder":
            tokenizer = AutoTokenizer.from_pretrained(family["hf"])
            encoder = ctranslate2.Encoder(args.model_path, **options)
            ids = [source_tokens(tokenizer, args.tokens)] * args.batch
            ids = [tokenizer.convert_ids_to_tokens(i) for i in ids]
            times, output = timed(lambda: encoder.forward_batch(ids), args.runs)
            hidden = np.array(output.last_hidden_state)
            out = {"vector": hidden.mean(axis=1).ravel().tolist()}
        else:
            tokenizer = AutoTokenizer.from_pretrained(family["hf"])
            generator = ctranslate2.Generator(args.model_path, **options)
            ids = source_tokens(tokenizer, args.tokens)
            prompt = tokenizer.convert_ids_to_tokens(ids)
            times, results = timed(lambda: generator.generate_batch(
                [prompt] * args.batch, max_length=args.new_tokens, beam_size=1,
                include_prompt_in_result=False), args.runs)
            out = {"text": " ".join(results[0].sequences[0])}
    print(json.dumps(dict(out, times=times)))


def convert(name, family, path):
    if path.exists():
        return
    import ctranslate2

    print(f"Converting {family['hf']} to {path} ...", file=sys.stderr, flush=True)
    ctranslate2.converters.TransformersConverter(family["hf"]).convert(
        str(path), quantization=family["quantization"])


def word_error_rate(reference, hypothesis):
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    if not ref:
        return float(len(hyp) > 0)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def run_worker(args, model_path, compute_type, fused):
    command = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--model-path", str(model_path), "--compute-type", compute_type,
        "--audio", str(args.audio), "--duration", str(args.duration),
        "--runs", str(args.runs), "--threads", str(args.threads),
        "--language", args.language, "--beam-size", str(args.beam_size),
        "--model", args.model_name, "--tokens", str(args.tokens), "--batch", str(args.batch),
        "--new-tokens", str(args.new_tokens),
    ]
    env = dict(os.environ, CT2_CPU_FUSED_ATTENTION="1" if fused else "0")
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def summarize(result):
    times = result["times"]
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    rtf = mean / result["audio_seconds"] if "audio_seconds" in result else None
    return mean, std, rtf


def compare(on, off, reference):
    """Returns (quality text, whether the outputs are identical)."""
    if "vector" in on:
        diff = max(abs(a - b) for a, b in zip(on["vector"], off["vector"]))
        return f"max |fused - unfused| = {diff:.2e}", on["vector"] == off["vector"]
    text = f"WER(fused vs unfused) = {word_error_rate(off['text'], on['text']):.4f}"
    if reference:
        text += (f"; WER vs reference: fused {word_error_rate(reference, on['text']):.4f}, "
                 f"unfused {word_error_rate(reference, off['text']):.4f}")
    return text, on["text"] == off["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["medium", "large-v3", *FAMILIES],
                        help="Whisper model directories or names under --models-dir, or one of: "
                             + ", ".join(FAMILIES) + ".")
    parser.add_argument("--models-dir", default=str(ROOT / "bench_run" / "models"),
                        help="Where the Whisper models are, and where the other models get converted.")
    parser.add_argument("--compute-types", nargs="+", help="Default: those of each model.")
    parser.add_argument("--audio", default=str(ROOT / "performance" / "whisper_compare" / "audio" / "podcast120.f32"))
    parser.add_argument("--duration", type=float, default=60, help="Seconds of audio to use (0 = all).")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per configuration.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--language", default="ca")
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size.")
    parser.add_argument("--tokens", type=int, default=512, help="Input tokens (non-Whisper models).")
    parser.add_argument("--batch", type=int, default=4, help="Sequences per batch (non-Whisper models).")
    parser.add_argument("--new-tokens", type=int, default=32, help="Generated tokens (decoder model).")
    parser.add_argument("--reference", help="Reference transcript text file, for the WER of each mode.")
    parser.add_argument("--output", help="Write the raw results to this JSON file.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help=argparse.SUPPRESS)
    parser.add_argument("--compute-type", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args)

    reference = Path(args.reference).read_text() if args.reference else None
    results = []
    print(f"{'model':<12} {'compute':<13} {'mode':<8} {'avg (s)':>8} {'std (s)':>8} {'RTF':>6}")
    for name in args.models:
        family = FAMILIES.get(name)
        if family:
            model_path = Path(args.models_dir) / (family["hf"].split("/")[1] + "-ct2")
            convert(name, family, model_path)
        else:
            model_path = Path(name) if Path(name).is_dir() else Path(args.models_dir) / name
        args.model_name = name if family else ""
        for compute_type in args.compute_types or (family or {}).get("compute_types", ["int8_float32", "float32"]):
            runs = {}
            for fused in (True, False):
                runs[fused] = run_worker(args, model_path, compute_type, fused)
                mean, std, rtf = summarize(runs[fused])
                print(f"{name if family else model_path.name:<12} {compute_type:<13} "
                      f"{'fused' if fused else 'unfused':<8} {mean:8.2f} {std:8.2f} "
                      f"{'-' if rtf is None else format(rtf, '6.3f'):>6}", flush=True)

            on, off = runs[True], runs[False]
            speedup = statistics.mean(off["times"]) / statistics.mean(on["times"])
            quality, identical = compare(on, off, reference)
            print(f"  -> fused is {speedup:.2f}x vs unfused; {quality}; "
                  f"outputs {'identical' if identical else 'differ'}", flush=True)
            results.append({"model": name if family else model_path.name, "compute_type": compute_type,
                            "fused": on, "unfused": off, "speedup": speedup, "quality": quality,
                            "identical": identical})

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
