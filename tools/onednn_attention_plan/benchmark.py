"""Run operation and real-encoder comparisons sequentially, avoiding CPU overlap."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
out = root / "benchmark-results/cpu-attention-epyc/investigation/planned-attention"
previous = out.parent / "performance-exploration"
env = dict(os.environ, PYTHONPATH=str(root / ".cache/cpu-plan/branch-python"),
           CT2_USE_MKL="0", CT2_DIAG_DNNL_PLAN="1", CT2_DIAG_DNNL_BLOCK="240",
           OMP_NUM_THREADS="8", OMP_PROC_BIND="true", OMP_PLACES="cores")
env.pop("CT2_DIAG_PLAN_VERIFY", None)

def run(command, library, fused, log):
    child = dict(env, LD_LIBRARY_PATH="/tmp/ct2-attn-variants/" + library,
                 CT2_CPU_FUSED_ATTENTION=str(fused))
    result = subprocess.run(["taskset", "-c", "0-7", *command], env=child,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    (out / log).write_text(result.stdout)
    if library == "planned" and fused:
        assert "CT2 planned SGEMM prototype" in result.stdout, "Planned path did not run"
    return result.stdout

with (out / "op-paired.log").open("w") as log:
    for repeat in range(3):
        order = ("head", "tile", "planned") if repeat != 1 else ("planned", "tile", "head")
        for variant in order:
            print("Operation", variant, repeat, flush=True)
            result = run(["/tmp/ct2-attention-bench", "20", "1500", "64", "30"],
                         variant, 1, f"op-{variant}-{repeat}.log")
            log.write(f"variant={variant} repeat={repeat}\n{result}\n")
            log.flush()

import numpy as np
rows = []
for compute in ("int8_float32", "float32"):
    ref = None
    for variant, library, fused in (("regular", "planned", 0), ("head", "head", 1),
                                     ("tile240", "tile", 1), ("planned", "planned", 1)):
        print("Encoder", compute, variant, flush=True)
        tensor = f"/tmp/ct2-planned-encoder-{compute}-{variant}.npy"
        result = run([sys.executable, str(previous / "encoder_bench.py"), compute, tensor],
                     library, fused, f"encoder-{compute}-{variant}.log")
        row = dict(json.loads(result.strip().splitlines()[-1]), compute=compute, variant=variant)
        actual = np.load(tensor)
        if ref is None:
            ref = actual
        assert actual.dtype == ref.dtype and actual.shape == ref.shape
        row["changed_float_bits"] = int(np.count_nonzero(actual.view(np.uint32) != ref.view(np.uint32)))
        row["mean"] = statistics.mean(row["times"])
        row["stdev"] = statistics.stdev(row["times"])
        rows.append(row)
        (out / "encoder.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(row, flush=True)
        assert row["changed_float_bits"] == 0
