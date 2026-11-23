"""Exercise the isolated plan with varying inputs and outer workspace targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--probe", default="/tmp/ct2-attention-probe")
p.add_argument("--library", default="/tmp/ct2-attn-variants/planned")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, LD_LIBRARY_PATH=args.library, CT2_USE_MKL="0",
           CT2_DIAG_DNNL_PLAN="1", CT2_DIAG_PLAN_VERIFY="1",
           OMP_PROC_BIND="true", OMP_PLACES="cores")
rows = []
with (args.output / "randomized.log").open("w") as log:
    # 20 seeds * 3 magnitudes * 2 thread counts at the default target;
    # plus a sweep of independently grouped full-shape blocks.
    cases = [(256, threads, seed, amplitude)
             for threads in (1, 8) for seed in range(20)
             for amplitude in (.01, 1, 10)]
    cases += [(target, threads, seed, amplitude)
              for target in (96, 128, 192, 320, 512, 1500)
              for threads in (1, 8) for seed in range(3)
              for amplitude in (.01, 1, 10)]
    for target, threads, seed, amplitude in cases:
        command = ["taskset", "-c", "0-7", args.probe, str(threads), "1500", "64",
                   str(seed), str(amplitude)]
        result = subprocess.run(command, env=dict(env, CT2_DIAG_PLAN_TARGET=str(target),
                                                  OMP_NUM_THREADS=str(threads)),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        label = dict(target=target, threads=threads, seed=seed, amplitude=amplitude)
        log.write(json.dumps(label) + "\n" + result.stdout + "\n")
        log.flush()
        passed = (result.returncode == 0
                  and "planned stage verification: QK changed=0; final changed=0" in result.stdout
                  and "attention changed=0/" in result.stdout
                  and "quantized changed=0/" in result.stdout
                  and "quantization-scales changed=0/" in result.stdout)
        rows.append(dict(**label, passed=passed))
        if not passed:
            raise RuntimeError(f"Plan comparison failed: {label}: {result.stdout}")
(args.output / "randomized.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"Passed {len(rows)} exact attention, QK, quantized-byte and scale comparisons")
