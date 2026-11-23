"""Compare compact decoder-only packing against a fresh raw-weight control."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
out=root/'benchmark-results/cpu-attention-epyc/investigation/memory-bounded-packing'
env=dict(os.environ,PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
         LD_LIBRARY_PATH='/tmp/ct2-attn-variants/memory-pack',CT2_USE_MKL='0',
         CT2_DIAG_DNNL_PLAN='1',CT2_DIAG_INT8_PACK_MODE='compact',
         OMP_NUM_THREADS='8',OMP_PROC_BIND='true',OMP_PLACES='cores')
for key in ('CT2_DIAG_PLAN_PV_GROUP','CT2_DIAG_PLAN_QK_GROUP','CT2_DIAG_PLAN_VERIFY',
            'CT2_DIAG_INT8_PACK_VERIFY','CT2_DIAG_PLAN_TARGET'):
    env.pop(key,None)
results={}
for variant in ('compact','regular'):
    print('Starting',variant,flush=True)
    with (out/f'e2e-{variant}.log').open('w') as log:
        on=variant=='compact'
        child=dict(env,CT2_CPU_FUSED_ATTENTION='1' if on else '0',CT2_DIAG_INT8_PACK='1' if on else '0')
        result=subprocess.run(['taskset','-c','0-7',sys.executable,
                               str(root/'tools/onednn_attention_plan/transcribe.py'),
                               'worker','int8_float32'],env=child,stdout=subprocess.PIPE,
                              stderr=log,text=True,check=True)
    if on:
        text=(out/f'e2e-{variant}.log').read_text()
        assert 'mode=compact raw_copy=0' in text and 'CT2 hybrid INT8 packed compute active' in text
    row=json.loads(result.stdout.strip().splitlines()[-1])
    results[variant]=row
    (out/f'e2e-{variant}.json').write_text(json.dumps(row,indent=2)+'\n')
    times=[r['seconds'] for r in row['runs']]
    print('Finished',variant,statistics.mean(times),statistics.stdev(times),
          'peak GiB',row['peak_rss_kib']/1024**2,'final RSS GiB',row['rss_after_runs_kib']/1024**2,flush=True)
ref=results['regular']['runs'][0]
rows=[r for v in results.values() for r in [v['warmup'],*v['runs']]]
results['all_texts_identical']=all(r['text']==ref['text'] for r in rows)
results['all_segments_identical']=all(r['segments']==ref['segments'] for r in rows)
results['peak_rss_ratio']=results['compact']['peak_rss_kib']/results['regular']['peak_rss_kib']
results['post_run_rss_ratio']=results['compact']['rss_after_runs_kib']/results['regular']['rss_after_runs_kib']
(out/'e2e.json').write_text(json.dumps(results,indent=2)+'\n')
assert results['all_texts_identical'] and results['all_segments_identical']
# A near-doubling is explicitly unacceptable. Use a stricter 20% experiment gate.
assert results['peak_rss_ratio']<1.2 and results['post_run_rss_ratio']<1.2, 'Memory budget failed'
print('Exact outputs and memory gate passed',flush=True)
