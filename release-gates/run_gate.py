"""Release gate orchestrator.

Creates two venvs (baseline = latest PyPI release of ctranslate2,
candidate = either an installed wheel or the already-installed version),
runs each task in both, diffs the resulting metrics against
`thresholds.yaml`, and exits non-zero on hard-severity regressions.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv

import yaml

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ["whisper_wer", "opus_bleu"]
TASK_DEPS = {
    "whisper_wer": [
        "transformers", "torch", "jiwer", "datasets", "soundfile", "librosa",
    ],
    "opus_bleu": [
        "transformers", "torch", "sacrebleu", "sentencepiece", "sacremoses",
    ],
}
COMMON_DEPS = ["pyyaml"]


def latest_pypi_version(pkg="ctranslate2"):
    url = f"https://pypi.org/pypi/{pkg}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return data["info"]["version"]


def make_venv(path, ctranslate2_spec, extras):
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(path)
    py = os.path.join(path, "bin", "python")
    pip = [py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"]
    subprocess.check_call(pip)
    install = [py, "-m", "pip", "install", "--quiet", ctranslate2_spec] + extras
    subprocess.check_call(install)
    return py


def run_task(py, task, limit, device):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [py, str(ROOT / "harness.py"), task, "--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    out = subprocess.check_output(cmd, env=env)
    # harness prints a single JSON line; take the last non-empty line to
    # tolerate any logging noise from models.
    last = [l for l in out.decode().splitlines() if l.strip()][-1]
    return json.loads(last)


def judge(task, baseline, candidate, thresholds):
    results = []
    for metric, rule in thresholds["metrics"].items():
        if metric not in baseline or metric not in candidate:
            continue
        b, c = baseline[metric], candidate[metric]
        direction = rule["direction"]
        regressed = False
        delta = c - b
        if direction == "lower_is_better":
            abs_delta = max(0.0, delta)
            rel_delta = (delta / b) if b else 0.0
        else:
            abs_delta = max(0.0, -delta)
            rel_delta = (-delta / b) if b else 0.0
        if "max_abs_delta" in rule and abs_delta > rule["max_abs_delta"]:
            regressed = True
        if "max_rel_delta" in rule and rel_delta > rule["max_rel_delta"]:
            regressed = True
        results.append({
            "metric": metric,
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "severity": rule["severity"],
            "regressed": regressed,
        })
    return results


def format_table(all_results):
    lines = ["| task | metric | baseline | candidate | delta | severity | status |",
             "|---|---|---|---|---|---|---|"]
    for task, rows in all_results.items():
        for r in rows:
            status = "FAIL" if r["regressed"] else "ok"
            lines.append(
                f"| {task} | {r['metric']} | {r['baseline']:.4f} | "
                f"{r['candidate']:.4f} | {r['delta']:+.4f} | "
                f"{r['severity']} | {status} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-wheel", default=None,
                        help="path or pip spec for the candidate build; "
                             "defaults to 'ctranslate2' (installed from PyPI, same as baseline — "
                             "useful only for smoke-testing the gate itself)")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--limit", type=int, default=50,
                        help="cap samples per task (default 50 for CI speed; "
                             "set higher or 0 for a full run)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    baseline_version = latest_pypi_version()
    baseline_spec = f"ctranslate2=={baseline_version}"
    candidate_spec = args.candidate_wheel or "ctranslate2"

    tmp = args.work_dir or tempfile.mkdtemp(prefix="release-gate-")
    print(f"[gate] work dir: {tmp}", flush=True)
    print(f"[gate] baseline: {baseline_spec}", flush=True)
    print(f"[gate] candidate: {candidate_spec}", flush=True)

    all_deps = sorted({d for t in args.tasks for d in TASK_DEPS[t]} | set(COMMON_DEPS))

    print("[gate] creating baseline venv...", flush=True)
    baseline_py = make_venv(os.path.join(tmp, "baseline"), baseline_spec, all_deps)
    print("[gate] creating candidate venv...", flush=True)
    candidate_py = make_venv(os.path.join(tmp, "candidate"), candidate_spec, all_deps)

    with open(ROOT / "thresholds.yaml") as f:
        thresholds = yaml.safe_load(f)["tasks"]

    all_results = {}
    hard_fail = False
    for task in args.tasks:
        print(f"[gate] running task={task} on baseline...", flush=True)
        baseline_metrics = run_task(baseline_py, task, limit, args.device)
        print(f"[gate]   baseline metrics: {baseline_metrics}", flush=True)
        print(f"[gate] running task={task} on candidate...", flush=True)
        candidate_metrics = run_task(candidate_py, task, limit, args.device)
        print(f"[gate]   candidate metrics: {candidate_metrics}", flush=True)

        rows = judge(task, baseline_metrics, candidate_metrics, thresholds[task])
        all_results[task] = rows
        for r in rows:
            if r["regressed"] and r["severity"] == "hard":
                hard_fail = True

    print()
    print(format_table(all_results))
    print()

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Release gate results\n\n")
            f.write(f"- baseline: `{baseline_spec}`\n")
            f.write(f"- candidate: `{candidate_spec}`\n\n")
            f.write(format_table(all_results))
            f.write("\n")

    if not args.work_dir:
        shutil.rmtree(tmp, ignore_errors=True)

    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
