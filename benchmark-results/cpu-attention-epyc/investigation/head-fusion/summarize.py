import json
from pathlib import Path
import statistics as stats

out = Path(__file__).resolve().parent
root = out.parents[3]

def mean(result):
    return stats.mean(result['times'])

def timing(result):
    return f"{mean(result):.2f} ± {stats.stdev(result['times']):.2f}"

lines = [
    '#### oneDNN conditional retaining fused attention (2026-11-23)', '',
    'Implemented a backend-specific path: oneDNN fuses the attention stages per whole head, '
    'keeping the full query and key dimensions of the regular GEMMs, including masked keys. '
    'Other backends keep the existing query-blocked kernel. This preserves fusion without '
    'switching GEMM providers. oneDNN score scratch is capped at 64 MiB per worker; shapes '
    'exceeding that cap fall back to regular attention. The existing limit of 8192 keys remains.', '',
    'For large-v3 (20 heads, 1500 queries/keys, 8 workers), score workspace is about 72 MB '
    'instead of the regular 180 MB tensor. This uses more scratch than query blocking '
    '(about 1.5 MB per worker for this shape), trading some memory savings for numerical compatibility.', '',
    'Validation on this EPYC / oneDNN 3.1.1 build: 30 fixed-input shapes match regular attention '
    'bit-for-bit; masked exact-comparison regression cases pass; real large-v3 encoder outputs '
    'match bit-for-bit in both int8_float32 and float32. The new regression fails against the '
    'original installed library and passes against the patched library. Full CPU suite: '
    '371 passed, 18 skipped. Other hardware/backends were not rerun.', '',
    'Same benchmark setup as the controlled EPYC rerun: 8 physical cores, beam 5, '
    '60 s of English audio, temperature 0, one warm-up and three timed runs. '
    'Non-Whisper workloads are unchanged. Medium was not rerun. Means ± sample standard deviations:', '',
    '| model | compute | fused (s) | unfused (s) | speedup | output agreement |',
    '|---|---|---|---|---|---|',
]
all_rows = []
for model in ('large-v3','nllb','xlm-roberta','gemma-3-4b'):
    rows = json.loads((out/(model+'.json')).read_text())
    all_rows.extend(rows)
    for row in rows:
        on, off = row['fused'], row['unfused']
        quality = 'identical' if row['identical'] else row['quality'].replace('|','\\|')
        lines.append(f"| {model} | {row['compute_type']} | {timing(on)} | {timing(off)} | {mean(off)/mean(on):.2f}x | {quality} |")
lines += ['', 'Raw data, logs, shape probes, and validation runner: '
          '`benchmark-results/cpu-attention-epyc/investigation/head-fusion/`. '
          'As before, the benchmark retains only the last timed output in each mode; '
          'these comparisons do not establish corpus-level accuracy or bit identity on untested '
          'oneDNN versions and CPUs.', '']
(out/'report.md').write_text('\n'.join(lines))
(out/'results.json').write_text(json.dumps(all_rows,indent=2)+'\n')
plan = root/'test_plan.md'
text = plan.read_text()
heading = '#### oneDNN conditional retaining fused attention'
if heading in text:
    start = text.index(heading)
    end = text.index('## Conclusion', start)
    text = text[:start] + '\n'.join(lines) + '\n' + text[end:]
else:
    text = text.replace('## Conclusion','\n'.join(lines)+'\n## Conclusion')
plan.write_text(text)
print('\n'.join(lines))
