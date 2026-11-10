"""Executes exactly one (task, compute_type, device) and prints JSON metrics.
Argv: task compute_type device model_dir prof_path
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import workloads

task, compute_type, device, model_dir, prof_path = sys.argv[1:6]

if task == "opus":
    m = workloads.run_opus(compute_type, device, prof_path, model_dir=model_dir)
elif task == "t5gemma":
    m = workloads.run_t5gemma(compute_type, device, prof_path, model_dir=model_dir)
elif task in ("whisper-medium", "whisper-large-v3"):
    m = workloads.run_whisper(workloads.OPUS_MODEL,  # unused if model_dir set
                                compute_type, device, prof_path,
                                model_dir=model_dir)
else:
    raise SystemExit(f"unknown task {task}")

print(json.dumps(m))
