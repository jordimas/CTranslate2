import json
from pathlib import Path
import time
from faster_whisper import WhisperModel,decode_audio

model=WhisperModel('/tmp/ct2-cpu-plan-models/large-v3',device='cpu',
                   compute_type='int8_float32',cpu_threads=8)
audio=decode_audio('english_60s.mp3')[:60*16000]
original=model.encode
encode_times=[]
def encode(features):
    begin=time.perf_counter()
    result=original(features)
    encode_times.append(time.perf_counter()-begin)
    return result
model.encode=encode
begin=time.perf_counter()
segments,_=model.transcribe(audio,language='en',beam_size=5,temperature=0.0)
text=' '.join(s.text.strip() for s in segments)
total=time.perf_counter()-begin
print(json.dumps({'total_seconds':total,'encode_seconds':sum(encode_times),
                  'encode_calls':len(encode_times),'encode_times':encode_times,
                  'encode_fraction':sum(encode_times)/total,
                  'other_seconds':total-sum(encode_times),'text':text},indent=2))
