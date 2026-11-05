# Release gate

Pass/fail checkpoint that blocks a release if the candidate build regresses
in quality (WER, BLEU) vs. the latest `ctranslate2` published on PyPI.

Baselines are computed at run time by installing the latest published
version in a separate venv and running the identical task against the same
model files and inputs — no stored baseline JSON, no drift.

## Layout

- `run_gate.py` — orchestrator. Creates two venvs (baseline + candidate),
  runs each task in both, diffs the resulting metrics against
  `thresholds.yaml`, exits non-zero on regression.
- `harness.py` — worker invoked inside a venv. Loads the model, runs one
  task, writes a single JSON blob to stdout.
- `tasks/whisper_wer.py` — Whisper tiny on a LibriSpeech test-clean subset,
  WER via `jiwer`.
- `tasks/opus_bleu.py` — `opus-mt-en-de` on sacrebleu `wmt14` en-de, BLEU
  via `sacrebleu`.
- `thresholds.yaml` — per-metric regression thresholds and hard/soft flag.

## Running locally

```
python release-gates/run_gate.py --candidate-wheel dist/ctranslate2-*.whl
```

Omit `--candidate-wheel` to use the already-installed `ctranslate2` as the
candidate (useful for a quick in-tree check against the latest PyPI
release).

## Thresholds

See `thresholds.yaml`. Quality deltas are **hard** (non-zero exit);
performance deltas start as **soft** (warn only) until noise is
characterized.
