"""Release gate: compare candidate ctranslate2 against latest PyPI release.

Creates two venvs, runs the same tasks in each, exits non-zero on hard
regressions.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
import venv

ROOT = pathlib.Path(__file__).resolve().parent
DEPS = ["faster-whisper==1.0.2", "jiwer", "datasets", "soundfile", "librosa",
        "transformers", "torch", "sacrebleu", "sentencepiece", "sacremoses"]

# (threshold, higher_is_better, severity). Severity: "hard" exits non-zero.
RULES = {
    "whisper_wer": {
        "wer":         (0.003, False, "hard"),   # +0.3 WER points
        "rtf_inverse": (0.10,  True,  "soft"),   # 10% slower
    },
    "opus_bleu": {
        "bleu":           (0.3,  True, "hard"),  # -0.3 BLEU points
        "tokens_per_sec": (0.10, True, "soft"),
    },
}


def latest_version(pkg="ctranslate2"):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
        return json.load(r)["info"]["version"]


def make_venv(path, ct2_spec):
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    py = os.path.join(path, "bin", "python")
    subprocess.check_call([py, "-m", "pip", "install", "-q", "-U", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-q", ct2_spec, *DEPS])
    return py


def run_task(py, task, limit, device):
    cmd = [py, str(ROOT / "tasks.py"), task, "--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    out = subprocess.check_output(cmd).decode()
    return json.loads([l for l in out.splitlines() if l.strip()][-1])


def judge(task, base, cand):
    rows, failed = [], False
    for metric, (threshold, higher_is_better, severity) in RULES[task].items():
        if metric not in base or metric not in cand:
            continue
        b, c = base[metric], cand[metric]
        regression = (b - c) if higher_is_better else (c - b)
        # For soft rules the threshold is a relative fraction.
        limit = threshold * abs(b) if severity == "soft" else threshold
        bad = regression > limit
        if bad and severity == "hard":
            failed = True
        rows.append((metric, b, c, c - b, severity, "FAIL" if bad else "ok"))
    return rows, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-wheel", default="ctranslate2",
                   help="pip spec or wheel path (defaults to PyPI = self-test)")
    p.add_argument("--limit", type=int, default=50, help="0 = full task set")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    limit = None if args.limit == 0 else args.limit

    baseline_spec = f"ctranslate2=={latest_version()}"
    print(f"[gate] baseline={baseline_spec} candidate={args.candidate_wheel}")

    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        print("[gate] creating baseline venv")
        base_py = make_venv(os.path.join(tmp, "baseline"), baseline_spec)
        print("[gate] creating candidate venv")
        cand_py = make_venv(os.path.join(tmp, "candidate"), args.candidate_wheel)

        hard_fail = False
        for task in RULES:
            print(f"[gate] {task}: baseline")
            b = run_task(base_py, task, limit, args.device)
            print(f"       {b}")
            print(f"[gate] {task}: candidate")
            c = run_task(cand_py, task, limit, args.device)
            print(f"       {c}")
            rows, failed = judge(task, b, c)
            for m, bv, cv, d, sev, st in rows:
                print(f"  {task}.{m}: {bv:.4f} -> {cv:.4f} ({d:+.4f}) [{sev}] {st}")
            hard_fail |= failed

    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
