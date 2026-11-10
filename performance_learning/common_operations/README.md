# Common operations profiling handoff

## Before shutting down

Commit this directory's scripts, documentation, raw `results/*.prof` and
`results/*.json`, report, and handoff metadata. Models and virtual environments
are ignored. Copy `handoff/ctranslate2-4.9.0-cp312-cp312-macosx_15_0_arm64.whl`
to the destination separately: it is deliberately ignored by Git.
The wheel SHA-256 is recorded in `handoff/machine.json`.

IMPORTANT: the installed profiling wheel contains custom Python bindings,
call counts, and tensor-shape instrumentation absent from the current source
checkout. The source HEAD in machine.json is the current checkout, NOT a proven
build revision. Preserve this wheel: a fresh build from HEAD cannot reproduce
this harness. Recover/reimplement the instrumentation before building for Linux,
CUDA, x86, or another Python ABI. `build.sh` is a historical Apple M4-specific
recipe, depends on a local venv, and deletes build directories; do not use it as
a portable setup script.

## View or regenerate the existing report (any machine)

Open `report.html` directly, or from this directory run:

```sh
python3 summarize.py --results results --out report.html --json-out summary.json
```

Only Python's standard library is required. Profile files are resolved relative
to the results directory, regardless of original absolute paths in metadata.
Missing/unsupported profile files now cause an error rather than empty tables.

## Run new profiles on a compatible Apple Silicon Mac

The preserved wheel requires CPython 3.12 and macOS 15 or newer. From this
directory, after separately copying the wheel:

```sh
python3.12 -m venv venv-prof
venv-prof/bin/python -m pip install -r handoff/requirements-reference.txt
venv-prof/bin/python -m pip install --force-reinstall --no-deps handoff/ctranslate2-4.9.0-cp312-cp312-macosx_15_0_arm64.whl
venv-prof/bin/python -c "import ctranslate2; ctranslate2._ext.init_profiling('cpu', 2); ctranslate2._ext.dump_profiling('/tmp/ct2-profile-smoke.prof')"
venv-prof/bin/python run_profiling.py --devices cpu,mps --out-dir results-new-machine
venv-prof/bin/python summarize.py --results results-new-machine --out report-new-machine.html --json-out summary-new-machine.json
```

Use a NEW results directory: the runner skips existing successful runs and would
otherwise reuse this machine's results. Downloads require network access and
substantial storage for converted models. The reference dependency snapshot is
from this machine, not a validated cross-platform lockfile. Start with
`--tasks opus --devices cpu --compute-types float32` for a small smoke run.

## Interpretation and known limitations

- This is synchronized operation profiling, not an unprofiled throughput benchmark.
- Wall timing includes warm-up; profiler tables exclude it. Whisper warm-up does
  not consume the lazy segment generator. Fix these before precise comparisons.
- The CLI sample-limit flags currently do not reach the worker. Effective defaults
  are 100 OPUS sentences, 20 T5Gemma prompts, and 20 Whisper samples.
- No WER/BLEU validation or repeated-run statistics are recorded.
- T5Gemma MPS float16/int8/int8_float16 produced 1280 tokens, equal to 20 times
  the 64-token decoding limit; inspect outputs/EOS behavior before speed claims.
- Preserve the original results as a baseline; runs with corrected timing should
  be marked as a different methodology.

The saved summary/report snapshot differs from current raw data for at least
`whisper-large-v3__mps__int8_bfloat16`. Regenerating may change reported numbers;
the original snapshot has deliberately been preserved.
