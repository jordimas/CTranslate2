"""CPU synthesis: python -m faster_tts MODEL --text TEXT --output out.wav.

MODEL is a local converted directory or explicit hf://organization/repository.
Hub example (replace placeholders; requires faster-tts-ct2[hub]):
python -m faster_tts hf://YOUR_ORG/YOUR_CONVERTED_REPO --revision PINNED_COMMIT
    --text "Hello" --output out.wav
"""

import argparse
from pathlib import Path
import time
import wave

import numpy as np

from .model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="local converted directory or explicit hf://organization/repository")
    parser.add_argument("--revision", help="Hub revision; prefer a pinned commit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face snapshot cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="resolve Hub assets only from the local cache")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compute-type", default="int8_float32")
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true", help="diagnostic greedy decoding instead of sampling")
    args = parser.parse_args()
    model = Model(args.model, intra_threads=args.threads, compute_type=args.compute_type,
                   revision=args.revision, cache_dir=args.cache_dir,
                   local_files_only=args.local_files_only)
    started = time.perf_counter()
    result = model.synthesize(
        args.text, max_frames=args.max_frames, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed, greedy=args.greedy)
    inference_seconds = time.perf_counter() - started
    pcm = np.rint(np.clip(result.audio, -1, 1) * 32767).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(result.sample_rate)
        output.writeframes(pcm.tobytes())
    print(f"Saved {result.audio.size / result.sample_rate:.3f}s to {args.output}; "
          f"finish_reason={result.finish_reason}; decoding={'greedy' if args.greedy else 'sampled'}")
    print(f"Inference time (excluding model loading): {inference_seconds:.3f}s")


if __name__ == "__main__":
    main()
