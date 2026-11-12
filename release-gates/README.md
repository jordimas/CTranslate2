# Release gate

Compares candidate wheels against the latest PyPI release; publication requires
passing the gate. Thresholds are in `run_gate.py` (`RULES`).

| Task | CI workload | Failure threshold |
| --- | --- | --- |
| Whisper tiny | 20 LibriSpeech dummy clean samples | WER increase >0.01 (1 percentage point) |
| opus-mt-en-de | 500 WMT14 sentences | BLEU decrease >1 point |
| T5Gemma 2 270M | 50 SQuAD validation prompts, INT8/FP32 greedy, up to 64 output tokens | Missing or empty completions |
| All tasks | Median throughput | Confirmed drop >10% |

T5Gemma uses `jordimas/t5gemma-2-270m-270m` at a pinned revision through
`Translator`. Changed text is informational, not an answer-quality check.
Prompts use `rajpurkar/squad` at revision `7b6d24c440a36b6815f21b70d25016731768db1f`,
shuffled with seed 42 and capped at 512 input tokens.
T5Gemma catches encoding, inference, and decoding errors per prompt, including
empty completions, and continues through all prompts. Failed outputs retain their
position for comparison and fail the quality gate. The summary shows the lowest
successful completion count across the three measurements.
Errors outside per-prompt generation still abort the run.

## Performance

Each task runs three sequential baseline/candidate pairs in alternating order.
Each fresh process warms up on one batch (one sample for Whisper), then times
inference only. CPU thread counts are fixed at two, with one CTranslate2 worker.
T5Gemma runs one prompt per call to isolate failures. Its throughput counts tokens
from successful prompts over total attempt time, including failed attempts,
prompt tokenization, and decoding.
These timings are not directly comparable to earlier batches of eight prompts.

A median throughput drop >10% triggers three more pairs, starting with the
candidate. The same metric must regress in both rounds to fail; quality failures
from any measurement remain blocking. Individual measurements and medians are
logged. This reduces runner noise at the cost of 3–6× benchmark work plus warm-up.

## Running

CI tests one CPython 3.12 wheel per platform: Linux x86-64/ARM64, macOS ARM64,
and Windows x86-64. The gate runs automatically on pull requests and on tag
pushes, reusing the wheels built earlier in the CI pipeline.

```sh
python release-gates/run_gate.py --candidate-wheel dist/ctranslate2-*.whl
```

Use `--candidate-wheel-pattern "dist/*.whl"` to require exactly one matching wheel.
Omit `--candidate-wheel` to compare PyPI against itself. `--bleu-limit` sets
translation coverage (default 500), independently of `--limit`.
Task implementations are in `tasks.py`.

The summary prints separate quality and performance tables, including the actual
device and compute type on each side. Use `--report summary.md` to save Markdown.
The local `performance_learning/benchmark/run_gates_installed_vs_latest.sh` wrapper
saves `gate-summary.md`; `run_gates_gpu.sh` selects `--device cuda` and saves
`gate-summary-gpu.md`. The `cuda` device name also applies to the HIP backend.
Both wheels must support the host GPU. On AMD, provide a ROCm baseline wheel
instead of the default PyPI CUDA wheel:

```sh
bash performance_learning/benchmark/run_gates_gpu.sh --baseline-wheel /path/to/baseline-rocm.whl
```

All three tasks explicitly request `int8_float32` on CPU and GPU: INT8 weights
with FP32 nonquantized computation. Reported types reflect the backend's resolved
type. Comparing different releases/builds does not isolate an int8 kernel speedup.
