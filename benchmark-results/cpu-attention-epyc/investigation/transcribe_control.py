import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
root = out.parents[2]
if len(sys.argv) > 1:
    from faster_whisper import WhisperModel, decode_audio
    model = WhisperModel('/tmp/ct2-cpu-plan-models/large-v3', device='cpu',
                         compute_type='int8_float32', cpu_threads=8)
    segments, _ = model.transcribe(decode_audio(str(root/'english_60s.mp3'))[:60*16000],
                                   language='en', beam_size=5, temperature=0.0)
    print(json.dumps({'text': ' '.join(s.text.strip() for s in segments)}))
else:
    results = {}
    for fused in (0, 1):
        print('MKL transcription fused=', fused, flush=True)
        env = dict(os.environ, CT2_USE_MKL='1', CT2_CPU_FUSED_ATTENTION=str(fused),
                   PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
                   LD_LIBRARY_PATH=str(root/'.cache/cpu-plan/branch-install/lib'),
                   OMP_NUM_THREADS='8', OMP_PROC_BIND='true', OMP_PLACES='cores')
        result = subprocess.run(['taskset','-c','0-7',sys.executable,str(__file__),'worker'],
                                env=env,stdout=subprocess.PIPE,text=True,check=True)
        results[str(fused)] = json.loads(result.stdout.strip().splitlines()[-1])
        (out/'mkl-transcripts.json').write_text(json.dumps(results,indent=2))
    print('MKL transcripts identical:',results['0']==results['1'],flush=True)
