"""Check that the experimental default selects compact packing without raw copies."""
import json
import os
from pathlib import Path
import resource
import numpy as np
from faster_whisper import WhisperModel,decode_audio
from faster_whisper.audio import pad_or_trim

assert 'CT2_DIAG_INT8_PACK_MODE' not in os.environ
assert 'CT2_DIAG_INT8_PACK_VERIFY' not in os.environ
assert os.environ.get('CT2_DIAG_INT8_PACK')=='1'
root=Path(__file__).resolve().parents[2]
out=root/'benchmark-results/cpu-attention-epyc/investigation/memory-bounded-packing'
model=WhisperModel('/tmp/ct2-cpu-plan-models/large-v3',device='cpu',
                   compute_type='int8_float32',cpu_threads=8)
audio=decode_audio(str(root/'english_60s.mp3'))[:30*16000]
actual=np.array(model.encode(pad_or_trim(model.feature_extractor(audio),3000)))
reference=np.load('/tmp/ct2-planned-encoder-int8_float32-regular.npy')
assert actual.shape==reference.shape and actual.dtype==reference.dtype
changed=int(np.count_nonzero(actual.view(np.uint32)!=reference.view(np.uint32)))
row=dict(encoder_changed_float_patterns=changed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(out/'default-smoke.json').write_text(json.dumps(row,indent=2)+'\n')
print(row)
assert changed==0
