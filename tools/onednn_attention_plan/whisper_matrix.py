"""Broader correctness/memory matrix. Single-pass times are not speedup claims."""
import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2]
out=root/'benchmark-results/cpu-attention-epyc/investigation/extended-validation'
tensor_dir=Path('/tmp/ct2-extended-validation');tensor_dir.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
p.add_argument('--group');p.add_argument('--variant');p.add_argument('--only-group');args=p.parse_args()
groups={
 'large_int8':dict(model='large-v3',compute='int8_float32',threads=8,workers=1),
 'medium_int8':dict(model='medium',compute='int8_float32',threads=8,workers=1),
 'large_float32':dict(model='large-v3',compute='float32',threads=8,workers=1),
 'concurrent_int8':dict(model='large-v3',compute='int8_float32',threads=4,workers=2),
 'lifecycle_int8':dict(model='medium',compute='int8_float32',threads=8,workers=1),
}
def bits(x):return struct.pack('!d',float(x)).hex()
def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])
    raise RuntimeError('No current RSS')

if args.worker:
    import numpy as np
    import ctranslate2
    from faster_whisper import WhisperModel,decode_audio
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    cfg=groups[args.group]
    model=WhisperModel('/tmp/ct2-cpu-plan-models/'+cfg['model'],device='cpu',
                       compute_type=cfg['compute'],cpu_threads=cfg['threads'],num_workers=cfg['workers'])
    english=decode_audio(str(root/'english_60s.mp3'))[:60*16000]
    jfk=decode_audio('/tmp/ct2-whisper-jfk.flac')
    silent=np.zeros(3*16000,dtype=np.float32)
    noise=np.random.default_rng(349).normal(0,.01,2*16000).astype(np.float32)
    long_audio=np.concatenate([english,english[:15*16000]])
    rows=[];lifecycle=[];cleanup={}
    def encoder_check(label,audio):
        features=pad_or_trim(model.feature_extractor(audio[:30*16000]),3000)
        actual=np.array(model.encode(features))
        path=tensor_dir/f'{args.group}-{label}-encoder.npy'
        if args.variant=='regular':
            np.save(path,actual);changed=0
        else:
            expected=np.load(path)
            assert actual.dtype==expected.dtype and actual.shape==expected.shape
            changed=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
        assert changed==0,f'{args.group}/{label}: encoder changed {changed}'
        return dict(shape=list(actual.shape),changed_float_patterns=changed,
                    sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    def transcribe(label,audio,beam=5,prompt=None):
        begin=time.perf_counter()
        segments,info=model.transcribe(audio,language='en',beam_size=beam,temperature=0.0,
                                       max_new_tokens=128,initial_prompt=prompt)
        segments=[dict(start=s.start,end=s.end,text=s.text,tokens=s.tokens,
                       avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob,
                       compression_ratio=s.compression_ratio,
                       score_bits=[bits(s.avg_logprob),bits(s.no_speech_prob),bits(s.compression_ratio)])
                  for s in segments]
        return dict(case=label,beam=beam,seconds=time.perf_counter()-begin,
                    audio_seconds=len(audio)/16000,segments=segments,
                    text=' '.join(s['text'].strip() for s in segments))
    if args.group=='large_int8':
        for label,audio in [('jfk',jfk),('silence',silent),('noise',noise)]:
            rows.append(dict(case=label+'_encoder',encoder=encoder_check(label,audio)))
        for beam in (1,5,8):
            rows.append(transcribe(f'jfk_beam{beam}',jfk,beam))
            print('Completed',args.group,rows[-1]['case'],file=sys.stderr,flush=True)
        rows.append(transcribe('silence',silent));rows.append(transcribe('noise',noise))
        rows.append(transcribe('multi_window_75s',long_audio))
        rows.append(transcribe('long_initial_prompt',jfk,prompt='The city, river, bakery and notebook. '*12))
        # Native batch two exercises joint prefill/beam rows beyond the pack extent.
        features=np.stack([pad_or_trim(model.feature_extractor(a),3000) for a in (jfk,english[:6*16000])])
        encoded=model.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(features)))
        actual=np.array(encoded);path=tensor_dir/'native-batch2-encoder.npy'
        if args.variant=='regular':np.save(path,actual);changed=0
        else:
            expected=np.load(path);assert actual.shape==expected.shape and actual.dtype==expected.dtype
            changed=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
        assert changed==0,'native batch-two encoder differs'
        tokenizer=Tokenizer(model.hf_tokenizer,model.model.is_multilingual,task='transcribe',language='en')
        prompt=tokenizer.sot_sequence+[tokenizer.no_timestamps]
        generated=model.model.generate(encoded,[prompt,prompt],beam_size=5,max_length=64,
                                       sampling_temperature=0,return_scores=True,return_no_speech_prob=True)
        rows.append(dict(case='native_batch2',encoder_changed_float_patterns=changed,
                         results=[dict(tokens=r.sequences_ids,scores=r.scores,
                                       score_bits=[bits(v) for v in r.scores],
                                       no_speech_prob=r.no_speech_prob,
                                       no_speech_bits=bits(r.no_speech_prob)) for r in generated]))
    elif args.group=='medium_int8':
        rows.append(dict(case='jfk_encoder',encoder=encoder_check('jfk',jfk)))
        rows.append(transcribe('jfk',jfk));rows.append(transcribe('english_60s',english))
    elif args.group=='large_float32':
        rows.append(dict(case='jfk_encoder',encoder=encoder_check('jfk',jfk)))
        rows.append(transcribe('jfk',jfk))
    elif args.group=='concurrent_int8':
        rows.append(dict(case='jfk_encoder',encoder=encoder_check('jfk',jfk)))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(transcribe,'concurrent_jfk',jfk),
                     pool.submit(transcribe,'concurrent_english',english[:6*16000])]
            rows.extend(f.result() for f in futures)
    else:
        # This is an ownership diagnostic, separate from ordinary inference RSS.
        # Keep the no-cleanup observations in the report; clearing free caches
        # between cycles tests whether live model storage remains referenced.
        import ctypes
        library=ctypes.CDLL('/tmp/ct2-attn-variants/extended-pack/libctranslate2.so.4.9.0')
        library.mkl_free_buffers.argtypes=[];library.mkl_free_buffers.restype=None
        libc=ctypes.CDLL(None);libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
        for cycle in range(4):
            if cycle:
                model=WhisperModel('/tmp/ct2-cpu-plan-models/medium',device='cpu',
                                   compute_type='int8_float32',cpu_threads=8)
            rows.append(dict(case=f'reload{cycle}_encoder',encoder=encoder_check(f'reload{cycle}',jfk)))
            rows.append(transcribe(f'reload{cycle}_jfk',jfk))
            alive=rss()
            model.model.unload_model();del model;gc.collect()
            unloaded=rss()
            library.mkl_free_buffers();libc.malloc_trim(0)
            lifecycle.append(dict(cycle=cycle,alive_rss_kib=alive,unloaded_rss_kib=unloaded,
                                  after_cleanup_rss_kib=rss()))
        cleanup=dict(before_kib=lifecycle[-1]['unloaded_rss_kib'],after_kib=rss(),diagnostic_only=True,
                     between_every_cycle=True)
    print(json.dumps(dict(group=args.group,variant=args.variant,config=cfg,rows=rows,
                          lifecycle=lifecycle,
                          cleanup=cleanup,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          post_run_rss_kib=rss())))
else:
    env=dict(os.environ,PYTHONPATH=str(root/'.cache/cpu-plan/branch-python'),
             LD_LIBRARY_PATH='/tmp/ct2-attn-variants/extended-pack',CT2_USE_MKL='0',
             CT2_DIAG_DNNL_PLAN='1',CT2_DIAG_INT8_PACK_MODE='compact',
             OMP_PROC_BIND='true',OMP_PLACES='cores')
    for k in ('CT2_DIAG_INT8_PACK_VERIFY','CT2_DIAG_PLAN_VERIFY',
              'CT2_DIAG_PLAN_QK_GROUP','CT2_DIAG_PLAN_PV_GROUP','CT2_DIAG_PLAN_TARGET'):
        env.pop(k,None)
    summary=[]
    if args.only_group:
        summary=[r for r in json.loads((out/'matrix-summary.json').read_text()) if r['group']!=args.only_group]
    for group,cfg in groups.items():
        if args.only_group and group!=args.only_group:continue
        variants={}
        for variant in ('regular','compact'):
            print('Starting',group,variant,flush=True)
            with (out/f'matrix-{group}-{variant}.log').open('w') as log:
                child=dict(env,OMP_NUM_THREADS=str(cfg['threads']),
                           CT2_CPU_FUSED_ATTENTION='0' if variant=='regular' else '1',
                           CT2_DIAG_INT8_PACK='0' if variant=='regular' else '1')
                result=subprocess.run(['taskset','-c','0-7',sys.executable,str(__file__),
                                       '--worker','--group',group,'--variant',variant],
                                      env=child,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
            if variant=='compact' and cfg['compute']=='int8_float32':
                text=(out/f'matrix-{group}-{variant}.log').read_text()
                assert 'mode=compact raw_copy=0' in text and 'CT2 hybrid INT8 packed compute active' in text
            row=json.loads(result.stdout.strip().splitlines()[-1]);variants[variant]=row
            (out/f'matrix-{group}-{variant}.json').write_text(json.dumps(row,indent=2)+'\n')
        a,b=variants['regular'],variants['compact']
        def strip_timing(rows):return [{k:v for k,v in r.items() if k!='seconds'} for r in rows]
        exact=strip_timing(a['rows'])==strip_timing(b['rows'])
        peak=b['peak_rss_kib']/a['peak_rss_kib'];post=b['post_run_rss_kib']/a['post_run_rss_kib']
        record=dict(group=group,config=cfg,cases=len(a['rows']),exact=exact,
                    peak_rss_ratio=peak,post_run_rss_ratio=post)
        if group=='lifecycle_int8':
            memory=b['lifecycle']
            record['unloaded_rss_growth_kib']=memory[1]['unloaded_rss_kib']-memory[0]['unloaded_rss_kib']
            record['released_kib']=[r['alive_rss_kib']-r['unloaded_rss_kib'] for r in memory]
            record['last_cycle_unloaded_growth_kib']=memory[-1]['unloaded_rss_kib']-memory[-2]['unloaded_rss_kib']
            record['cleaned_rss_growth_kib']=memory[-1]['after_cleanup_rss_kib']-memory[0]['after_cleanup_rss_kib']
            record['cleanup']=b['cleanup']
        summary.append(record)
        (out/'matrix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('Finished',summary[-1],flush=True)
        assert exact,f'{group}: outputs differ'
        assert peak<1.2,f'{group}: peak memory gate failed'
        if group=='lifecycle_int8':
            assert record['cleaned_rss_growth_kib']<64*1024,'Cleaned reload RSS does not plateau'
            assert b['cleanup']['after_kib']<b['cleanup']['before_kib']*.75,'Cleanup does not release retained RSS'
        else:assert post<1.2,f'{group}: post-run memory gate failed'
