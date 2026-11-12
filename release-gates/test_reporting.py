import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import run_gate as gate

class SummaryTest(unittest.TestCase):
    def test_models_are_removed_between_tasks_and_work_dir_is_used(self):
        converted = []
        evaluated = []
        def convert(py, root, tasks):
            self.assertTrue(all(not path.exists() for path in converted
                                if path.name != tasks[0]))
            path = pathlib.Path(root) / tasks[0]
            path.mkdir(parents=True)
            (path / 'model.bin').write_bytes(b'model')
            converted.append(path)
            return {tasks[0]: str(path)}

        def evaluate(base, cand, task, limit, device, model_dirs, summary):
            evaluated.append(task)
            for dirs in model_dirs.values():
                if task in gate.CONVERTIBLE_TASKS:
                    self.assertTrue((pathlib.Path(dirs[task]) / 'model.bin').exists())
                else:
                    self.assertEqual(dirs, {})
            return False

        with tempfile.TemporaryDirectory() as work_dir, \
             patch.object(gate, 'make_venv', side_effect=['b', 'c']), \
             patch.object(gate, 'convert_models', side_effect=convert), \
             patch.object(gate, 'evaluate_task', side_effect=evaluate), \
             patch('sys.argv', ['run_gate.py', '--baseline-wheel', 'baseline.whl',
                                '--work-dir', work_dir]), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
            gate.main()
        self.assertEqual(exited.exception.code, 0)
        self.assertEqual(evaluated, list(gate.RULES))
        self.assertEqual(len(converted), 2 * len(gate.CONVERTIBLE_TASKS))
        self.assertTrue(all(path.is_relative_to(work_dir) for path in converted))
        self.assertTrue(all(not path.exists() for path in converted))

    def test_conversion_failure_cleans_partial_model_and_reports_error(self):
        partial = []
        def convert(py, root, tasks):
            path = pathlib.Path(root) / tasks[0]
            path.mkdir(parents=True)
            (path / 'model.bin').write_bytes(b'partial')
            partial.append(path)
            raise OSError('Disk quota exceeded')

        with patch.object(gate, 'make_venv', side_effect=['b', 'c']), \
             patch.object(gate, 'convert_models', side_effect=convert), \
             patch.object(gate, 'evaluate_task', return_value=False), \
             patch.object(gate, 'publish_summary') as publish, \
             patch('sys.argv', ['run_gate.py', '--baseline-wheel', 'baseline.whl']), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(OSError, 'Disk quota exceeded'):
            gate.main()
        self.assertEqual(publish.call_args.args[3], 'ERROR (incomplete run)')
        self.assertTrue(partial)
        self.assertTrue(all(not path.exists() for path in partial))

    def test_gpu_baseline_override_and_preflight(self):
        with patch.object(gate, 'latest_version') as latest, \
             patch.object(gate, 'make_venv', side_effect=['b', 'c']) as make, \
             patch.object(gate, 'convert_models', return_value={}), \
             patch.object(gate, 'evaluate_task', return_value=False), \
             patch.object(gate.subprocess, 'check_call') as check, \
             patch('sys.argv', ['run_gate.py', '--device', 'cuda',
                                '--baseline-wheel', '/tmp/baseline-rocm.whl']), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
            gate.main()
        self.assertEqual(exited.exception.code, 0)
        latest.assert_not_called()
        self.assertEqual(make.call_args_list[0].args[1], '/tmp/baseline-rocm.whl')
        self.assertEqual([call.args[0][0] for call in check.call_args_list], ['b', 'c'])

    def test_metadata_medians_and_confirmation(self):
        rows = []
        def result(py, *args, **kwargs):
            return {'bleu': 30, 'tokens_per_sec': 100 if py == 'b' else 80,
                    'device': 'cpu', 'compute_type': 'int8_float32'}
        with patch.object(gate, 'run_task', side_effect=result), contextlib.redirect_stdout(io.StringIO()):
            failed = gate.evaluate_task('b', 'c', 'opus_bleu', 50, 'cpu',
                                       {'baseline': {}, 'candidate': {}}, summary=rows)
        self.assertTrue(failed)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1][2:4], ['cpu / int8_float32'] * 2)
        self.assertEqual(rows[1][5:9], ['100.0000', '80.0000', '-20.00%', 'threshold exceeded'])
        self.assertEqual(rows[3][1], 'confirmation')

    def test_generation_quality_and_report_file(self):
        rows = []
        def result(py, *args, **kwargs):
            return {'tokens_per_sec': 100, 'outputs': ['hello' if py == 'b' else 'world'],
                    'device': 'cpu', 'compute_type': 'float32'}
        with patch.object(gate, 'run_task', side_effect=result), contextlib.redirect_stdout(io.StringIO()):
            gate.evaluate_task('b', 'c', 't5gemma_generation', 50, 'cpu',
                               {'baseline': {}, 'candidate': {}}, summary=rows)
        self.assertEqual(rows[-1][4:8], ['nonempty completions', '1/1', '1/1', '1 changed (first pair)'])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = pathlib.Path(tmp) / 'summary.md'
            gate.publish_summary(rows, 'ctranslate2==4.9.0', 'new.whl', 'PASS', report)
            text = report.read_text()
        for expected in ('**PASS**', 'new.whl', 'cpu / float32', 'nonempty completions'):
            self.assertIn(expected, text)

    def test_runtime_failure_publishes_incomplete_report(self):
        with patch.object(gate, 'latest_version', return_value='4.9.0'), \
             patch.object(gate, 'make_venv', side_effect=['b', 'c']), \
             patch.object(gate, 'convert_models', return_value={}), \
             patch.object(gate, 'evaluate_task', side_effect=ValueError('runtime failure')), \
             patch.object(gate, 'publish_summary') as publish, \
             patch('sys.argv', ['run_gate.py']), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, 'runtime failure'):
            gate.main()
        self.assertEqual(publish.call_args.args[3], 'ERROR (incomplete run)')

    def test_prompt_failures_preserve_measurements_and_fail_quality(self):
        rows = []
        def result(py, *args, **kwargs):
            return {'tokens_per_sec': 100, 'outputs': ['', 'hello'] if py == 'b' else ['ok', 'hello'],
                    'prompt_errors': [{'prompt': 1, 'error': 'empty completion'}] if py == 'b' else []}
        with patch.object(gate, 'run_task', side_effect=result) as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            failed = gate.evaluate_task('b', 'c', 't5gemma_generation', 50, 'cpu',
                                       {'baseline': {}, 'candidate': {}}, summary=rows)
        self.assertTrue(failed)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(rows[-1][5:7], ['1/2', '2/2'])
        self.assertEqual(rows[-1][-1], 'FAIL; prompt errors')
        self.assertIn('baseline prompt 1: ERROR: empty completion', output.getvalue())

if __name__ == '__main__':
    unittest.main()
