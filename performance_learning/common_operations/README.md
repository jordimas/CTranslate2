# Common operations profiling

Build the instrumented CTranslate2 source separately on each target system using
[build.md](build.md). Linux CPU, optional CUDA, and macOS CPU/MPS build options
are provided. The source includes Python profiling bindings, call counts, and
shape annotations; no historical Mac binaries are required for new builds.

Start with the two-sentence CPU OPUS smoke run in the build instructions. Use a
fresh machine-specific results directory. Runs record host, source revision,
package/Python versions, workload limits, and methodology version; incompatible
resume metadata is rejected. Preserve build metadata and the CMake cache too.

To regenerate the latest report with only Python's standard library:

```sh
python3 summarize.py --results results/apple_m4_pro_methodology2 --json-out summary.json
```

New reports default to `reports_<cpu_id>.html`, using the CPU identifier saved
on the benchmark machine. Use `--cpu-id` for historical results without that
metadata, or `--out` for an explicit filename.

Profiles are resolved in the supplied results directory, regardless of original
absolute paths. Missing or unsupported profiles fail explicitly.

## Historical baseline

`reports_apple_m4_pro.html` and `summary.json` are generated from the new
`results/apple_m4_pro_methodology2/` run. That directory also contains run
metadata, the CMake cache, and a native-library checksum in `build-provenance.txt`.

The files directly under `results/` preserve the original Mac measurements.
Their original HTML/summary snapshot is available in Git history at `1b9a9c75`.
That old snapshot differs from its raw data for at least
`whisper-large-v3__mps__int8_bfloat16`; regeneration can change those numbers.
The old wall timer included warm-up, and Whisper warm-up was not fully consumed.
New runs use methodology 2; do not silently compare their timings to that baseline.
T5Gemma MPS float16/int8/int8_float16 reached the aggregate 64-token-per-prompt
budget; inspect output quality and EOS behavior before drawing speed conclusions.

`handoff/` records the historical environment. Its separately preserved, ignored
wheel is specific to CPython 3.12/macOS 15+/arm64 and cannot run on Linux. Its
exact source revision is unknown; the new instrumentation is reconstructed.

Commit source, docs, raw baseline profiles, and metadata. Models, virtualenvs,
build outputs, new run outputs, and wheel binaries are ignored. WER/BLEU checks,
repeated unprofiled benchmarks, and target Linux/CUDA execution are still needed
before production performance claims. See [plan.md](plan.md) for remaining work.
