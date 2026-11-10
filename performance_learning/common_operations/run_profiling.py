"""Run (task x compute_type x device) matrix, write .prof + .json per run.

Usage:
  venv-prof/bin/python run_profiling.py \
      --tasks opus,whisper-medium,whisper-large-v3,t5gemma \
      --compute-types float32,float16,bfloat16,int8,int8_float32,int8_float16,int8_bfloat16,int16 \
      --devices cpu,mps \
      --out-dir results
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import socket
import subprocess
import sys
import traceback

import ctranslate2

from machine_info import detect_cpu_id

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


def run_one(task, compute_type, device, model_dir, prof_path, limit):
    """Shell out so a failure in one combo doesn't poison later runs."""
    cmd = [sys.executable, str(ROOT / "_run_one.py"),
           task, compute_type, device, model_dir, prof_path, str(limit)]
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
    p.add_argument("--devices", default="cpu")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--cache-dir", default=str(ROOT / "models"))
    p.add_argument("--whisper-limit", type=int, default=20)
    p.add_argument("--opus-limit", type=int, default=100)
    p.add_argument("--t5gemma-limit", type=int, default=20)
    args = p.parse_args()

    limits = {"opus": args.opus_limit, "t5gemma": args.t5gemma_limit,
              "whisper-medium": args.whisper_limit, "whisper-large-v3": args.whisper_limit}
    if any(n <= 0 for n in limits.values()):
        p.error("Sample limits must be positive")
    metadata = {"cpu_id": detect_cpu_id(), "hostname": socket.gethostname(), "platform": platform.platform(),
                "machine": platform.machine(), "python": sys.version,
                "ctranslate2": ctranslate2.__version__, "methodology_version": 2,
                "limits": limits, "cpu_threads": 2,
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    build_meta = pathlib.Path(sys.prefix).parent / "build-metadata.json"
    metadata["build"] = json.loads(build_meta.read_text()) if build_meta.is_file() else None
    metadata["extension_sha256"] = hashlib.sha256(pathlib.Path(ctranslate2._ext.__file__).read_bytes()).hexdigest()
    metadata["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    tasks = args.tasks.split(",")
    if any(task not in limits for task in tasks):
        p.error("Unknown task; choose opus,t5gemma,whisper-medium,whisper-large-v3")
    requested_types = args.compute_types.split(",")
    devices = args.devices.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Reject resume across machines/methodologies/settings rather than mixing results.
    meta_path = pathlib.Path(args.out_dir) / "run-metadata.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            p.error("Run metadata differs; use a fresh --out-dir")
    elif list(pathlib.Path(args.out_dir).glob("*.json")):
        p.error("Existing results lack run metadata; use a fresh --out-dir")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = []
    for device in devices:
        compute_types = filter_compute_types(requested_types, device)
        print(f"[matrix] device={device} compute_types={compute_types}", flush=True)
        for ct in requested_types:
            if ct not in compute_types:
                print(f"[unsupported] {device}/{ct}", flush=True)
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
                    metrics = run_one(task, ct, device, model_dir, prof_path, limits[task])
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
    if not manifest or any(e.get("status") != "ok" for e in manifest):
        raise SystemExit("No successful matrix, or one or more runs failed; inspect manifest.json")


if __name__ == "__main__":
    main()
