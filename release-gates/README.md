# Release gate

Compares the candidate ctranslate2 build against the latest version published
on PyPI. Fails on hard-severity regressions in WER (whisper-tiny on
LibriSpeech test-clean) or BLEU (opus-mt-en-de on sacrebleu wmt14).

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
