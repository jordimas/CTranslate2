# Release gate

Compares the candidate ctranslate2 build against the latest version published
on PyPI. Fails on hard-severity regressions in WER (whisper-tiny on
the LibriSpeech dummy clean validation set) or BLEU (opus-mt-en-de on
sacrebleu wmt14).

## CI, manual, and scheduled runs

The CI workflow builds wheels, then calls the reusable `release-gate.yml`
workflow. The gate downloads artifacts from that same run and requires exactly
one matching CPython 3.14 wheel per platform (Linux x86-64 and ARM64, macOS
ARM64, and Windows x86-64). It compares that wheel against the latest PyPI
release using 20 examples per task. PyPI publication depends on the gate passing.

To run manually, select **CI → Run workflow** in GitHub Actions and choose the
branch. CI also runs every Monday at 06:00 UTC on the default branch once these
workflow changes are merged there. Both triggers run CI, including fresh wheel
builds and the release gate; they do not publish packages.

Files:
- `run_gate.py` — orchestrator (venvs, judging, CLI).
- `tasks.py` — the two tasks; also runnable directly for local checks.

Local run:
```
python release-gates/run_gate.py --candidate-wheel dist/ctranslate2-*.whl
```

Without `--candidate-wheel`, both venvs install the latest PyPI release — a
smoke test of the gate itself.

Thresholds and metrics are inlined in `run_gate.py` (`RULES`).
