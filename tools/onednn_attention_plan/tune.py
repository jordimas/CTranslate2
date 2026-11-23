"""Compare plan execution overhead and workspace targets on this machine."""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
out = root / "benchmark-results/cpu-attention-epyc/investigation/latency-tuning"
env = dict(os.environ, LD_LIBRARY_PATH="/tmp/ct2-attn-variants/planned-v2",
           CT2_USE_MKL="0", CT2_DIAG_DNNL_PLAN="1", OMP_NUM_THREADS="8",
           OMP_PROC_BIND="true", OMP_PLACES="cores",
           PYTHONPATH=str(root / ".cache/cpu-plan/branch-python"))
for key in ("CT2_DIAG_PLAN_VERIFY", "CT2_DIAG_PLAN_PV_GROUP", "CT2_DIAG_PLAN_QK_GROUP"):
    env.pop(key, None)

def config(qk=False, pv=False, target=256, fused=1):
    result = dict(env, CT2_DIAG_PLAN_TARGET=str(target), CT2_CPU_FUSED_ATTENTION=str(fused))
    if qk: result["CT2_DIAG_PLAN_QK_GROUP"] = "1"
    if pv: result["CT2_DIAG_PLAN_PV_GROUP"] = "1"
    return result

def run(command, child, logfile):
    result = subprocess.run(["taskset", "-c", "0-7", *command], env=child,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / logfile).write_text(result.stdout)
    result.check_returncode()
    return result.stdout

variants = [("original", config()), ("pv", config(pv=True)),
            ("qk", config(qk=True)), ("both", config(qk=True, pv=True))]
# First isolate the two call-grouping changes at the same workspace target.
rows = []
for repeat in range(3):
    order = variants if repeat != 1 else variants[::-1]
    for name, child in order:
        text = run(["/tmp/ct2-attention-bench", "20", "1500", "64", "30"], child,
                   f"op-{name}-{repeat}.log")
        row = dict(variant=name, repeat=repeat, **{
            k: float(re.search(k + r" mean_ms=([\d.]+)", text).group(1))
            for k in ("regular", "fused")})
        rows.append(row)
        (out / "op.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(row, flush=True)

# Sweep outer workspace sizes with both grouping changes enabled.
sweep = []
for target in (96, 128, 192, 256, 320, 384, 512, 768, 1500):
    text = run(["/tmp/ct2-attention-bench", "20", "1500", "64", "30"],
               config(qk=True, pv=True, target=target), f"target-{target}.log")
    row = dict(target=target, **{k:float(re.search(k + r" mean_ms=([\d.]+)",text).group(1))
                               for k in ("regular", "fused")})
    sweep.append(row)
    print(row, flush=True)
(out / "targets.json").write_text(json.dumps(sweep, indent=2) + "\n")

best = min(sweep, key=lambda x:x["fused"])["target"]
(out / "selected-target.json").write_text(json.dumps(dict(target=best)) + "\n")
print("Best operation target", best, flush=True)

import numpy as np
encoder = []
for compute in ("int8_float32", "float32"):
    reference = None
    cases = [("regular", config(fused=0)), ("original", config()),
             ("grouped256", config(qk=True,pv=True)),
             ("grouped_best", config(qk=True,pv=True,target=best))]
    for name, child in cases:
        target = f"/tmp/ct2-tuned-encoder-{compute}-{name}.npy"
        text = run([sys.executable, str(out.parent / "performance-exploration/encoder_bench.py"),
                    compute, target], child, f"encoder-{compute}-{name}.log")
        actual = np.load(target)
        if reference is None: reference = actual
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        changed = int(np.count_nonzero(actual.view(np.uint32) != reference.view(np.uint32)))
        row = dict(json.loads(text.strip().splitlines()[-1]), compute=compute, variant=name,
                   changed_float_bits=changed)
        row["mean"] = statistics.mean(row["times"])
        row["stdev"] = statistics.stdev(row["times"])
        encoder.append(row)
        (out / "encoder.json").write_text(json.dumps(encoder, indent=2) + "\n")
        print(row, flush=True)
        assert changed == 0
