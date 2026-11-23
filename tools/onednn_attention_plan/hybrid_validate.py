"""Validate hybrid packing on the complete real encoder and transcription."""
import json
import os
import argparse
from pathlib import Path
import time
import numpy as np
from faster_whisper import WhisperModel,decode_audio
from faster_whisper.audio import pad_or_trim

root=Path(__file__).resolve().parents[2]
out=root/'benchmark-results/cpu-attention-epyc/investigation/latency-tuning'
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=out)
args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert os.environ.get('CT2_DIAG_INT8_PACK')=='1'
assert 'CT2_DIAG_INT8_PACK_VERIFY' in os.environ, 'Per-GEMM verification must be enabled'
model=WhisperModel('/tmp/ct2-cpu-plan-models/large-v3',device='cpu',
                   compute_type='int8_float32',cpu_threads=8)
audio=decode_audio(str(root/'english_60s.mp3'))[:60*16000]
features=pad_or_trim(model.feature_extractor(audio[:30*16000]),3000)
actual=np.array(model.encode(features))
expected=np.load('/tmp/ct2-planned-encoder-int8_float32-regular.npy')
assert actual.shape==expected.shape and actual.dtype==expected.dtype
changed=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
print('encoder changed float bits',changed,flush=True)
assert changed==0
begin=time.perf_counter()
segments,_=model.transcribe(audio,language='en',beam_size=5,temperature=0.0)
segments=[dict(start=s.start,end=s.end,text=s.text) for s in segments]
seconds=time.perf_counter()-begin
expected=json.loads((root/'benchmark-results/cpu-attention-epyc/investigation/planned-attention/e2e-int8_float32-regular.json').read_text())['runs'][0]
text=' '.join(s['text'].strip() for s in segments)
row=dict(seconds=seconds,encoder_changed_bits=changed,text=text,segments=segments,
         text_identical=text==expected['text'],segments_identical=segments==expected['segments'],
         per_gemm_verification=True)
(out/'hybrid-validation.json').write_text(json.dumps(row,indent=2)+'\n')
print('Full transcription',seconds,'text exact',row['text_identical'],'segments exact',row['segments_identical'],flush=True)
assert row['text_identical'] and row['segments_identical']
