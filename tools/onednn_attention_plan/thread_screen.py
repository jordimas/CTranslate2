"""Screen CPU thread counts using exact encoder and transcript controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
out = root / "benchmark-results/cpu-attention-epyc/investigation/latency-tuning"
if len(sys.argv)>1 and sys.argv[1]=='worker':
    import numpy as np
    from faster_whisper import WhisperModel,decode_audio
    from faster_whisper.audio import pad_or_trim
    threads=int(sys.argv[2])
    model=WhisperModel('/tmp/ct2-cpu-plan-models/large-v3',device='cpu',
                       compute_type='int8_float32',cpu_threads=threads)
    audio=decode_audio(str(root/'english_60s.mp3'))[:60*16000]
    features=pad_or_trim(model.feature_extractor(audio[:30*16000]),3000)
    encoded=np.array(model.encode(features))
    reference=np.load('/tmp/ct2-planned-encoder-int8_float32-regular.npy')
    assert encoded.dtype==reference.dtype and encoded.shape==reference.shape
    changed=int(np.count_nonzero(encoded.view(np.uint32)!=reference.view(np.uint32)))
    if changed:
        print(json.dumps(dict(threads=threads,encoder_changed_bits=changed,
                              rejected='encoder differs from 8-thread baseline')))
        sys.exit(0)
    def run():
        begin=time.perf_counter()
        segments,_=model.transcribe(audio,language='en',beam_size=5,temperature=0.0)
        segments=[dict(start=s.start,end=s.end,text=s.text) for s in segments]
        return dict(seconds=time.perf_counter()-begin,segments=segments,
                    text=' '.join(s['text'].strip() for s in segments))
    warmup=run();print('warm-up',threads,warmup['seconds'],file=sys.stderr,flush=True)
    timed=run();print('timed',threads,timed['seconds'],file=sys.stderr,flush=True)
    print(json.dumps(dict(threads=threads,encoder_changed_bits=changed,warmup=warmup,timed=timed)))
else:
    env=dict(os.environ,PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
             LD_LIBRARY_PATH='/tmp/ct2-attn-variants/planned-v2', CT2_USE_MKL='0',
             CT2_CPU_FUSED_ATTENTION='1',CT2_DIAG_DNNL_PLAN='1',
             OMP_PROC_BIND='true',OMP_PLACES='cores')
    for key in ('CT2_DIAG_PLAN_PV_GROUP','CT2_DIAG_PLAN_QK_GROUP','CT2_DIAG_PLAN_VERIFY',
                'CT2_DIAG_PLAN_TARGET'):
        env.pop(key,None)
    expected=json.loads((out.parent/'planned-attention/e2e-int8_float32-regular.json').read_text())['runs'][0]
    rows=[]
    for threads in (6,4):
        print('Starting threads',threads,flush=True)
        with (out/f'threads-{threads}.log').open('w') as log:
            result=subprocess.run(['taskset','-c','0-7',sys.executable,str(__file__),'worker',str(threads)],
                                  env=dict(env,OMP_NUM_THREADS=str(threads)),stdout=subprocess.PIPE,
                                  stderr=log,text=True,check=True)
        row=json.loads(result.stdout.strip().splitlines()[-1])
        if 'rejected' in row:
            rows.append(row)
            (out/'thread-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('Rejected',row,flush=True)
            continue
        row['all_texts_identical']=all(r['text']==expected['text'] for r in (row['warmup'],row['timed']))
        row['all_segments_identical']=all(r['segments']==expected['segments'] for r in (row['warmup'],row['timed']))
        rows.append(row)
        (out/'thread-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('Finished',threads,row['timed']['seconds'],row['all_texts_identical'],row['all_segments_identical'],flush=True)
        assert row['all_texts_identical'] and row['all_segments_identical']
