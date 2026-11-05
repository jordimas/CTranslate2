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
release using 500 translation sentences for BLEU and 20 audio samples for WER.
It also converts and runs `jordimas/t5gemma-2-270m-270m` at a pinned revision
on 50 fixed text prompts, using FP32 greedy decoding with at most 64 output
tokens. Missing or empty completions and runtime failures fail the gate;
changed completions are printed for review and throughput regressions are soft.
This is encoder-decoder generation through `Translator`, not decoder-only
`Generator` coverage or an answer-quality benchmark.
PyPI publication depends on the gate passing.

To run manually, select **CI → Run workflow** in GitHub Actions and choose the
branch. CI also runs every Monday at 06:00 UTC on the default branch once these
workflow changes are merged there. Both triggers run CI, including fresh wheel
builds and the release gate; they do not publish packages.

Files:
- `run_gate.py` — orchestrator (venvs, judging, CLI).
- `tasks.py` — the three tasks; also runnable directly for local checks.
- `generation_prompts.json` — the fixed text-generation inputs.

Local run:
```
python release-gates/run_gate.py --candidate-wheel dist/ctranslate2-*.whl
```

Without `--candidate-wheel`, both venvs install the latest PyPI release — a
smoke test of the gate itself.

Thresholds and metrics are inlined in `run_gate.py` (`RULES`).
Translation coverage defaults to 500 sentences and can be set independently
with `--bleu-limit` (a positive integer), overriding `--limit` for BLEU.
