import json
from pathlib import Path
import sys
import time
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim

model=WhisperModel('/tmp/ct2-cpu-plan-models/large-v3',device='cpu',
                   compute_type=sys.argv[1],cpu_threads=8)
audio=decode_audio('english_60s.mp3')[:30*16000]
features=pad_or_trim(model.feature_extractor(audio),3000)
model.encode(features)
times=[]
for _ in range(3):
    begin=time.perf_counter()
    encoded=model.encode(features)
    times.append(time.perf_counter()-begin)
np.save(sys.argv[2], np.array(encoded))
print(json.dumps({'times':times}))
