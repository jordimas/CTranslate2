"""Standard-library tests; no model downloads or installed CTranslate2 required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent


def load(name, fake):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'ctranslate2': fake}):
        spec.loader.exec_module(mod)
    return mod


class HarnessTest(unittest.TestCase):
    def test_limit_reaches_worker(self):
        runner = load('run_profiling', types.SimpleNamespace())
        with patch.object(runner.subprocess, 'run') as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout='{"num_sentences": 2}', stderr='')
            self.assertEqual(runner.run_one('opus', 'float32', 'cpu', 'model', 'prof', 2), {'num_sentences': 2})
            self.assertEqual(run.call_args.args[0][-1], '2')

    def test_whisper_warmup_consumed_before_timer_and_profiler(self):
        events = []
        ext = types.SimpleNamespace(init_profiling=lambda *a: events.append('init'),
                                    dump_profiling=lambda *a: events.append('dump'))
        workloads = load('workloads', types.SimpleNamespace(_ext=ext))
        class Model:
            def __init__(self, *a, **kw): pass
            def transcribe(self, *a, **kw):
                def segments():
                    events.append('consume')
                    yield types.SimpleNamespace(text='hello')
                return segments(), None
        def clock():
            events.append('clock')
            return 1.0
        with patch.dict(sys.modules, {'faster_whisper': types.SimpleNamespace(WhisperModel=Model)}), \
             patch.object(workloads, '_load_librispeech', return_value=[([0.] * 16, 16, 'hello')]), \
             patch.object(workloads.time, 'perf_counter', side_effect=clock):
            result = workloads.run_whisper('model', 'float32', 'cpu', 'profile')
        self.assertEqual(events, ['consume', 'init', 'clock', 'consume', 'clock', 'dump'])
        self.assertEqual(result['methodology_version'], 2)

    def test_relocated_report_and_missing_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tag = 'opus__cpu__float32'
            (path / (tag + '.json')).write_text(json.dumps({
                'task': 'opus', 'device': 'cpu', 'compute_type': 'float32',
                'status': 'ok', 'metrics': {}, 'prof': '/old/machine/run.prof'}))
            (path / 'run-metadata.json').write_text('{}')
            profile = path / (tag + '.prof')
            profile.write_text('# ct2_profile_format=1\n 10.00% 10.00% 10.00% Gemm[2x3,3x4] 1.00ms calls=2\n')
            out = path / 'output'
            out.mkdir()
            cmd = [sys.executable, str(ROOT / 'summarize.py'), '--results', str(path),
                   '--out', str(out / 'report.html'), '--json-out', str(out / 'summary.json')]
            subprocess.run(cmd, check=True, capture_output=True)
            rows = json.loads((out / 'summary.json').read_text())['rows_by_tag'][tag]
            self.assertEqual(rows[0]['calls'], 2)
            self.assertEqual(rows[0]['shape'], '2x3,3x4')
            profile.unlink()
            failed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Missing profile', failed.stderr)


if __name__ == '__main__':
    unittest.main()
