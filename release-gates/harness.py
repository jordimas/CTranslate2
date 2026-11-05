"""Worker entry point. Runs a single task inside the current venv and
writes a JSON metrics blob to stdout.

Invoked by `run_gate.py` as a subprocess so the baseline and candidate
builds of ctranslate2 can live in separate venvs without clashing.
"""

import argparse
import importlib
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task module name under release_gates.tasks")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of samples (for smoke runs)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    module = importlib.import_module(f"tasks.{args.task}")
    metrics = module.run(limit=args.limit, device=args.device)

    import ctranslate2
    metrics["_ctranslate2_version"] = ctranslate2.__version__
    json.dump(metrics, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
