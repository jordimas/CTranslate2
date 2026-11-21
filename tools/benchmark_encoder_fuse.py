#!/usr/bin/env python3
"""End-to-end Whisper benchmark for the CPU fused attention (CT2_CPU_FUSED_ATTENTION).

For each model and compute type, the transcription runs with the fused attention on and
off, each in its own process (the switch is read once per process). Every process loads
the model, does one untimed warm-up, then times --runs transcriptions. Reported: average
latency, standard deviation, real-time factor, and quality metrics. The quality metrics are
the word error rate (WER) between the transcripts of the two modes and whether they match
exactly, plus the WER of each mode against --reference when given.

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


def worker(args):
    from faster_whisper import WhisperModel

    audio = load_audio(args.audio, args.duration)
    model = WhisperModel(
        args.model_path,
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )

    def transcribe():
        start = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language=args.language, beam_size=args.beam_size
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return time.perf_counter() - start, text

    transcribe()  # Warm-up.
    times, text = [], ""
    for _ in range(args.runs):
        elapsed, text = transcribe()
        times.append(elapsed)
    print(json.dumps({"times": times, "text": text, "audio_seconds": len(audio) / SAMPLE_RATE}))


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
    ]
    env = dict(os.environ, CT2_CPU_FUSED_ATTENTION="1" if fused else "0")
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def summarize(result):
    times = result["times"]
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, mean / result["audio_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["medium", "large-v3"],
                        help="Model directories, or names under --models-dir.")
    parser.add_argument("--models-dir", default=str(ROOT / "bench_run" / "models"))
    parser.add_argument("--compute-types", nargs="+", default=["int8_float32", "float32"])
    parser.add_argument("--audio", default=str(ROOT / "performance" / "whisper_compare" / "audio" / "podcast120.f32"))
    parser.add_argument("--duration", type=float, default=60, help="Seconds of audio to use (0 = all).")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per configuration.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--language", default="ca")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--reference", help="Reference transcript text file, for the WER of each mode.")
    parser.add_argument("--output", help="Write the raw results to this JSON file.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help=argparse.SUPPRESS)
    parser.add_argument("--compute-type", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args)

    reference = Path(args.reference).read_text() if args.reference else None
    results = []
    print(f"{'model':<10} {'compute':<13} {'mode':<10} {'avg (s)':>8} {'std (s)':>8} {'RTF':>6}")
    for name in args.models:
        model_path = Path(name) if Path(name).is_dir() else Path(args.models_dir) / name
        for compute_type in args.compute_types:
            runs = {}
            for fused in (True, False):
                runs[fused] = run_worker(args, model_path, compute_type, fused)
                mean, std, rtf = summarize(runs[fused])
                print(f"{model_path.name:<10} {compute_type:<13} {'fused' if fused else 'unfused':<10} "
                      f"{mean:8.2f} {std:8.2f} {rtf:6.3f}", flush=True)

            on, off = runs[True], runs[False]
            speedup = statistics.mean(off["times"]) / statistics.mean(on["times"])
            quality = word_error_rate(off["text"], on["text"])
            line = (f"  -> fused is {speedup:.2f}x vs unfused; WER(fused vs unfused) = {quality:.4f}, "
                    f"transcripts {'identical' if on['text'] == off['text'] else 'differ'}")
            if reference:
                line += (f"; WER vs reference: fused {word_error_rate(reference, on['text']):.4f}, "
                         f"unfused {word_error_rate(reference, off['text']):.4f}")
            print(line, flush=True)
            results.append({"model": model_path.name, "compute_type": compute_type,
                            "fused": on, "unfused": off, "speedup": speedup, "wer_fused_vs_unfused": quality})

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
