"""Dump per-sample transcripts of the release-gate Whisper task (diagnostic, not part of the gate).

Uses the gate's own sample loader and model setup, so the transcripts are the ones the gate scores.
With --repeat N the whole sample set is transcribed N times in the same process, to detect
non-determinism inside a process (separate processes are compared by running the script again).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jiwer

import tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-type", choices=("default", "auto"), default="auto")
    parser.add_argument("--limit", type=int, default=20, help="0 = full set")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import ctranslate2

    samples = tasks._load_librispeech(args.limit or None)
    model = tasks.load_task_model("whisper_wer", "cpu", args.compute_type)
    refs = [text.strip().lower() for _, _, text in samples]

    runs = []
    for _ in range(args.repeat):
        hyps = []
        for audio, _, _ in samples:
            segments, _ = model.transcribe(audio, language="en", temperature=0.0, vad_filter=False)
            hyps.append(" ".join(s.text for s in segments).strip().lower())
        runs.append(hyps)

    per_sample = []
    for ref, hyp in zip(refs, runs[0]):
        out = jiwer.process_words(ref, hyp)
        per_sample.append({
            "ref": ref, "hyp": hyp, "words": len(ref.split()),
            "errors": out.substitutions + out.deletions + out.insertions,
        })

    result = {
        "ctranslate2": ctranslate2.__version__,
        "fused_env": os.environ.get("CT2_CPU_FUSED_ATTENTION"),
        "compute_type": tasks.effective_compute_type("whisper_wer", model),
        "wer": [jiwer.wer(refs, hyps) for hyps in runs],
        "per_sample": per_sample,
        "repeat_hyps": runs,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    print(args.output, result["compute_type"], result["wer"])


if __name__ == "__main__":
    main()
