# Test plan: CPU fused attention

Goal: check that the CPU fused attention (branch `perf/cpu-attention-transpose-softmax`) speeds up Whisper end to end compared with `master`, without changing the transcripts.

## Steps (per machine)

1. Build `master` and this branch (CPU build, same flags), and install each into its own prefix.
2. Pin to one thread per physical core, then run the script against each build:

   ```
   LD_LIBRARY_PATH=<prefix>/lib taskset -c <physical cores> \
       python tools/benchmark_encoder_fuse.py --audio <audio> --duration 60 \
       --threads <physical cores> --output results_<machine>_<build>.json
   ```

   - On `master` the fused path does not exist, so the on and off runs are both the baseline.
   - On the branch, the script compares fused on and off in the same build.
3. Copy the printed tables (avg, std, RTF, speedup, WER) into the Results section.

## Results

### Machine: <CPU model, cores/threads, RAM, OS>

| build | model | compute | mode | avg (s) | std (s) | RTF | speedup | WER fused vs unfused |
|-------|-------|---------|------|---------|---------|-----|---------|----------------------|
|       |       |         |      |         |         |     |         |                      |

Notes:

## Conclusion

(Fill in after all machines have run.)
