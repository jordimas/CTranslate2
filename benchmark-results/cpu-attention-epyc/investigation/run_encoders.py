import os
from pathlib import Path
import subprocess
import numpy as np

root = Path('/home/jupyter/CTranslate2')
out = root / 'benchmark-results/cpu-attention-epyc/investigation'
env = dict(os.environ, PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
           LD_LIBRARY_PATH=str(root/'.cache/cpu-plan/branch-build'),
           OMP_PROC_BIND='true', OMP_PLACES='cores', OMP_NUM_THREADS='8',
           CT2_VERBOSE='1', CT2_DIAG_ATTENTION='1')
for backend, compute, fused in [('dnnl','int8_float32',0),('dnnl','int8_float32',1),
                              ('dnnl','float32',0),('dnnl','float32',1),
                              ('mkl','int8_float32',0),('mkl','int8_float32',1)]:
    name = f'{backend}-{compute}-{fused}'
    print('Starting', name, flush=True)
    with (out / (name+'.log')).open('w') as stream:
        subprocess.run(['taskset','-c','0-7',str(root/'.cache/cpu-plan/venv/bin/python'),
                        str(out/'encode.py'),compute,'/tmp/ct2-encoder-'+name+'.npy'],
                       env=dict(env, CT2_USE_MKL='1' if backend=='mkl' else '0',
                                CT2_CPU_FUSED_ATTENTION=str(fused)),
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    if fused:
        a = np.load('/tmp/ct2-encoder-'+name[:-1]+'0.npy')
        b = np.load('/tmp/ct2-encoder-'+name+'.npy')
        print(name, 'changed', np.count_nonzero(a!=b), '/', a.size,
              'max', float(np.max(np.abs(a-b))), flush=True)
