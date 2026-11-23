"""Full transcription comparison; every warm-up and timed output is retained."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
out = root / "benchmark-results/cpu-attention-epyc/investigation/planned-attention"

if len(sys.argv) > 1 and sys.argv[1] == "worker":
    from faster_whisper import WhisperModel, decode_audio
    model = WhisperModel("/tmp/ct2-cpu-plan-models/large-v3", device="cpu",
                         compute_type=sys.argv[2], cpu_threads=8)
    audio = decode_audio(str(root / "english_60s.mp3"))[:60 * 16000]
    def run():
        begin = time.perf_counter()
        segments, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0)
        segments = [dict(start=s.start, end=s.end, text=s.text) for s in segments]
        return dict(seconds=time.perf_counter() - begin,
                    text=" ".join(s["text"].strip() for s in segments), segments=segments)
    warmup = run()
    print("warm-up", warmup["seconds"], file=sys.stderr, flush=True)
    runs = []
    for i in range(3):
        runs.append(run())
        print("timed", i, runs[-1]["seconds"], file=sys.stderr, flush=True)
    print(json.dumps(dict(warmup=warmup, runs=runs)))
else:
    env = dict(os.environ, PYTHONPATH=str(root / ".cache/cpu-plan/branch-python"),
               LD_LIBRARY_PATH="/tmp/ct2-attn-variants/planned", CT2_USE_MKL="0",
               CT2_DIAG_DNNL_PLAN="1", CT2_DIAG_PLAN_TARGET="256",
               OMP_NUM_THREADS="8", OMP_PROC_BIND="true", OMP_PLACES="cores")
    env.pop("CT2_DIAG_PLAN_VERIFY", None)
    results = {}
    for compute in ("int8_float32", "float32"):
        results[compute] = {}
        order = ("regular", "planned") if compute == "int8_float32" else ("planned", "regular")
        for variant in order:
            print("Starting", compute, variant, flush=True)
            with (out / f"e2e-{compute}-{variant}.log").open("w") as log:
                result = subprocess.run(
                    ["taskset", "-c", "0-7", sys.executable, str(__file__), "worker", compute],
                    env=dict(env, CT2_CPU_FUSED_ATTENTION="0" if variant == "regular" else "1"),
                    stdout=subprocess.PIPE, stderr=log, text=True, check=True)
            if variant == "planned":
                assert "CT2 planned SGEMM prototype" in (out / f"e2e-{compute}-{variant}.log").read_text(), "Planned path did not run"
            row = json.loads(result.stdout.strip().splitlines()[-1])
            results[compute][variant] = row
            (out / f"e2e-{compute}-{variant}.json").write_text(json.dumps(row, indent=2) + "\n")
            times = [r["seconds"] for r in row["runs"]]
            print("Finished", compute, variant, statistics.mean(times), statistics.stdev(times), flush=True)
        reference = results[compute]["regular"]["runs"][0]
        rows = [r for v in results[compute].values() for r in [v["warmup"], *v["runs"]]]
        results[compute]["all_texts_identical"] = all(r["text"] == reference["text"] for r in rows)
        results[compute]["all_segments_identical"] = all(r["segments"] == reference["segments"] for r in rows)
        (out / "e2e.json").write_text(json.dumps(results, indent=2) + "\n")
        print(compute, "exact text:", results[compute]["all_texts_identical"],
              "exact segments:", results[compute]["all_segments_identical"], flush=True)
        assert results[compute]["all_texts_identical"] and results[compute]["all_segments_identical"]
