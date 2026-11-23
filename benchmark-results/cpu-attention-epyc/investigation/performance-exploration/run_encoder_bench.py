import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

root=Path('/home/jupyter/CTranslate2')
out=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
         CT2_USE_MKL='0',OMP_PROC_BIND='true',OMP_PLACES='cores',OMP_NUM_THREADS='8')
results=[]
for compute in ('int8_float32','float32'):
    reference=None
    for name,library,fused,block in [('regular','head',0,None),('head','head',1,None),
                                    ('workspace','workspace',1,None),
                                    ('tile192','tile',1,192),('tile240','tile',1,240)]:
        print('Encoder',compute,name,flush=True)
        child=dict(env,LD_LIBRARY_PATH='/tmp/ct2-attn-variants/'+library,
                   CT2_CPU_FUSED_ATTENTION=str(fused))
        if block:child['CT2_DIAG_DNNL_BLOCK']=str(block)
        target=f'/tmp/ct2-encoder-bench-{compute}-{name}.npy'
        result=subprocess.run(['taskset','-c','0-7',sys.executable,str(out/'encoder_bench.py'),compute,target],
                              env=child,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
        (out/f'encoder-bench-{compute}-{name}.log').write_text(result.stderr+result.stdout)
        row=dict(json.loads(result.stdout.strip().splitlines()[-1]),compute=compute,variant=name)
        actual=np.load(target)
        if reference is None:reference=actual
        row.update(changed=int(np.count_nonzero(reference!=actual)),max_diff=float(np.max(np.abs(reference-actual))))
        results.append(row)
        print(row,flush=True)
        (out/'encoder-bench.json').write_text(json.dumps(results,indent=2))
