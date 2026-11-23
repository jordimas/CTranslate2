import os
import sys
import time
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim

start = time.perf_counter()
model = WhisperModel('/tmp/ct2-cpu-plan-models/large-v3', device='cpu',
                     compute_type=sys.argv[1], cpu_threads=8)
audio = decode_audio('english_60s.mp3')[:30*16000]
features = pad_or_trim(model.feature_extractor(audio), 3000)
encoded = np.array(model.encode(features))
np.save(sys.argv[2], encoded)
print('encoded', encoded.shape, 'seconds', time.perf_counter()-start, flush=True)
