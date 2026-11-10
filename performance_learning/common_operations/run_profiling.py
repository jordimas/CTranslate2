"""Run (task x compute_type x device) matrix, write .prof + .json per run.

Usage:
  venv-prof/bin/python run_profiling.py \
      --tasks opus,whisper-medium,whisper-large-v3,t5gemma \
      --compute-types float32,float16,bfloat16,int8,int8_float32,int8_float16,int8_bfloat16,int16 \
      --devices cpu,mps \
      --out-dir results
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import traceback

import ctranslate2

ROOT = pathlib.Path(__file__).resolve().parent

WHISPER_MODELS = {
    "whisper-medium":   "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
}

OPUS_MODEL = "Helsinki-NLP/opus-mt-en-de"
T5GEMMA_MODEL = "jordimas/t5gemma-2-270m-270m"
T5GEMMA_REVISION = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"


def convert(hf_id, out_dir, quantization, revision=None, copy_files=()):
    """Convert an HF model to CT2 at a given quantization. Idempotent."""
    done = os.path.join(out_dir, "model.bin")
    if not os.path.exists(done):
        os.makedirs(out_dir, exist_ok=True)
        print(f"[conv] {hf_id} -> {out_dir} ({quantization})", flush=True)
        from ctranslate2.converters.transformers import TransformersConverter
        kw = {"revision": revision} if revision else {}
        TransformersConverter(hf_id, **kw).convert(out_dir, force=True,
                                                    quantization=quantization)
    # Copy supporting files (e.g. preprocessor_config.json for Whisper).
    import huggingface_hub
    for fname in copy_files:
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            continue
        try:
            src = huggingface_hub.hf_hub_download(
                hf_id, fname, revision=revision)
            import shutil
            shutil.copyfile(src, dest)
            print(f"[conv]   + {fname}", flush=True)
        except Exception as exc:
            print(f"[conv]   {fname} unavailable: {exc}", flush=True)


def ensure_model(task, compute_type, cache_root):
    """Return a converted model dir for (task, compute_type). None if N/A."""
    if task == "opus":
        d = os.path.join(cache_root, "opus", compute_type)
        convert(OPUS_MODEL, d, compute_type)
        return d
    if task == "t5gemma":
        d = os.path.join(cache_root, "t5gemma", compute_type)
        convert(T5GEMMA_MODEL, d, compute_type, revision=T5GEMMA_REVISION)
        return d
    if task in WHISPER_MODELS:
        d = os.path.join(cache_root, task, compute_type)
        convert(WHISPER_MODELS[task], d, compute_type,
                copy_files=("preprocessor_config.json", "tokenizer.json"))
        return d
    raise ValueError(task)


def filter_compute_types(requested, device):
    supported = ctranslate2.get_supported_compute_types(device)
    return [c for c in requested if c in supported]


def run_one(task, compute_type, device, model_dir, prof_path):
    """Shell out so a failure in one combo doesn't poison later runs."""
    cmd = [sys.executable, str(ROOT / "_run_one.py"),
           task, compute_type, device, model_dir, prof_path]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr, file=sys.stderr)
        raise RuntimeError(f"{task}/{compute_type}/{device} failed")
    return json.loads([ln for ln in out.stdout.splitlines() if ln.strip()][-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="opus,whisper-medium,whisper-large-v3,t5gemma")
    p.add_argument("--compute-types",
                   default="float32,float16,bfloat16,int8,int8_float32,int8_float16,int8_bfloat16,int16")
    p.add_argument("--devices", default="cpu,mps")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--cache-dir", default=str(ROOT / "models"))
    p.add_argument("--whisper-limit", type=int, default=20)
    p.add_argument("--opus-limit", type=int, default=100)
    p.add_argument("--t5gemma-limit", type=int, default=20)
    args = p.parse_args()

    tasks = args.tasks.split(",")
    requested_types = args.compute_types.split(",")
    devices = args.devices.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    manifest = []
    for device in devices:
        compute_types = filter_compute_types(requested_types, device)
        print(f"[matrix] device={device} compute_types={compute_types}", flush=True)
        for task in tasks:
            for ct in compute_types:
                tag = f"{task}__{device}__{ct}"
                prof_path = os.path.join(args.out_dir, tag + ".prof")
                meta_path = os.path.join(args.out_dir, tag + ".json")
                if os.path.exists(meta_path) and os.path.exists(prof_path):
                    with open(meta_path) as f:
                        entry = json.load(f)
                    if entry.get("status") == "ok":
                        print(f"[skip] {tag}", flush=True)
                        manifest.append(entry)
                        continue
                print(f"[run] {tag}", flush=True)
                try:
                    model_dir = ensure_model(task, ct, args.cache_dir)
                    metrics = run_one(task, ct, device, model_dir, prof_path)
                    entry = {"task": task, "device": device,
                             "compute_type": ct, "prof": prof_path,
                             "metrics": metrics, "status": "ok"}
                except Exception as exc:
                    entry = {"task": task, "device": device,
                             "compute_type": ct, "prof": prof_path,
                             "status": "fail", "error": repr(exc),
                             "trace": traceback.format_exc()}
                    print(f"[run] FAIL: {entry['error']}", flush=True)
                with open(meta_path, "w") as f:
                    json.dump(entry, f, indent=2)
                manifest.append(entry)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[matrix] done", flush=True)


if __name__ == "__main__":
    main()
