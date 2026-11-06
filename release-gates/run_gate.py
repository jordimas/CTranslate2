"""Release gate: compare candidate ctranslate2 against latest PyPI release.

Creates two venvs, runs the same tasks in each, exits non-zero on hard
regressions.
"""

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
import venv

ROOT = pathlib.Path(__file__).resolve().parent
DEPS = ["faster-whisper==1.0.2", "requests", "jiwer", "datasets", "soundfile",
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
    "t5gemma_generation": {
        "tokens_per_sec": (0.10, True, "soft"),
    },
}


def latest_version(pkg="ctranslate2"):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
        return json.load(r)["info"]["version"]


def make_venv(path, ct2_spec):
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    if sys.platform == "win32":
        py = os.path.join(path, "Scripts", "python.exe")
    else:
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


def validate_metrics(task, result, label):
    for metric in RULES[task]:
        value = result.get(metric)
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(
                f"{task}: {label} metric {metric!r} "
                f"is missing or invalid: {value!r}")


def judge(task, base, cand):
    validate_metrics(task, base, "baseline")
    validate_metrics(task, cand, "candidate")
    rows, failed = [], False
    for metric, (threshold, higher_is_better, severity) in RULES[task].items():
        b, c = base[metric], cand[metric]
        regression = (b - c) if higher_is_better else (c - b)
        # For soft rules the threshold is a relative fraction.
        limit = threshold * abs(b) if severity == "soft" else threshold
        bad = regression > limit
        if bad and severity == "hard":
            failed = True
        rows.append((metric, b, c, c - b, severity, "FAIL" if bad else "ok"))
    return rows, failed


def compare_completions(base, cand):
    baseline, candidate = base["outputs"], cand["outputs"]
    if (not baseline or len(baseline) != len(candidate)
            or any(not isinstance(s, str) or not s.strip() for s in baseline + candidate)):
        raise ValueError("Expected equal, nonempty sets of nonempty completions")
    changed = [(i, b, c) for i, (b, c) in enumerate(zip(baseline, candidate), 1) if b != c]
    print(f"[gate] changed completions: {len(changed)}/{len(baseline)} [informational]")
    for i, b, c in changed:
        print(f"  prompt {i}: baseline={b!r} candidate={c!r}")
    return len(changed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-wheel", default="ctranslate2",
                   help="pip spec or wheel path (defaults to PyPI = self-test)")
    p.add_argument("--limit", type=int, default=50, help="0 = full task set")
    p.add_argument("--bleu-limit", type=int, default=500,
                   help="number of translation sentences (default: 500; overrides --limit)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.bleu_limit < 1:
        p.error("--bleu-limit must be positive")
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
            task_limit = args.bleu_limit if task == "opus_bleu" else limit
            if task == "t5gemma_generation":
                task_limit = 50
            print(f"[gate] {task}: baseline")
            b = run_task(base_py, task, task_limit, args.device)
            print(f"       {b}")
            print(f"[gate] {task}: candidate")
            c = run_task(cand_py, task, task_limit, args.device)
            print(f"       {c}")
            if task == "t5gemma_generation":
                compare_completions(b, c)
            rows, failed = judge(task, b, c)
            for m, bv, cv, d, sev, st in rows:
                print(f"  {task}.{m}: {bv:.4f} -> {cv:.4f} ({d:+.4f}) [{sev}] {st}")
            hard_fail |= failed

    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
