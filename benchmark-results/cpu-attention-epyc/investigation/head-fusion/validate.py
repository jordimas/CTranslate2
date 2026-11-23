import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path('/home/jupyter/CTranslate2')
out = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
           LD_LIBRARY_PATH=str(root/'.cache/cpu-plan/branch-build'),
           CT2_USE_MKL='0', OMP_NUM_THREADS='8', OMP_PROC_BIND='true', OMP_PLACES='cores',
           HF_HOME='/tmp/ct2-cpu-plan-hf', HF_HUB_CACHE='/tmp/ct2-cpu-plan-hf',
           HUGGINGFACE_HUB_CACHE='/tmp/ct2-cpu-plan-hf', TRANSFORMERS_CACHE='/tmp/ct2-cpu-plan-hf',
           HF_HUB_OFFLINE='1')
for compute in ('int8_float32','float32'):
    arrays = []
    for fused in (0,1):
        name = f'{compute}-{fused}'
        print('Encoder', name, flush=True)
        target = '/tmp/ct2-head-fusion-'+name+'.npy'
        with (out/(name+'.log')).open('w') as stream:
            subprocess.run(['taskset','-c','0-7',sys.executable,str(out.parent/'encode.py'),compute,target],
                           env=dict(env,CT2_CPU_FUSED_ATTENTION=str(fused)),
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        arrays.append(np.load(target))
    print(compute, 'encoder changed', np.count_nonzero(arrays[0]!=arrays[1]),
          'max',float(np.max(np.abs(arrays[0]-arrays[1]))),flush=True)
    assert np.array_equal(*arrays)

for model in ('large-v3','nllb','xlm-roberta','gemma-3-4b'):
    print('Benchmark', model, flush=True)
    with (out/(model+'.log')).open('w') as stream:
        subprocess.run(['taskset','-c','0-7',sys.executable,
                        str(root/'tools/benchmark_encoder_fuse.py'),'--models',model,
                        '--models-dir','/tmp/ct2-cpu-plan-models','--audio',str(root/'english_60s.mp3'),
                        '--duration','60','--threads','8','--runs','3','--language','en',
                        '--output',str(out/(model+'.json'))],env=env,stdout=stream,
                       stderr=subprocess.STDOUT,check=True)
    rows = json.loads((out/(model+'.json')).read_text())
    for row in rows:
        print(model, row['compute_type'], row['quality'], 'identical', row['identical'],flush=True)
        assert row['identical']
print('VALIDATION COMPLETE',flush=True)
