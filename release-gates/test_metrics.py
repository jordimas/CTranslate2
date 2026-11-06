"""Validate measurements before applying release-gate thresholds."""

import contextlib
import io
import unittest
from unittest.mock import patch

import run_gate

from run_gate import judge


class MetricValidationTest(unittest.TestCase):
    def test_invalid_metrics_fail_on_either_side(self):
        valid = {"bleu": 30.0, "tokens_per_sec": 100.0}
        for label in ("baseline", "candidate"):
            for metric in valid:
                for value in (None, "30", True, False, float("nan"),
                              float("inf"), float("-inf")):
                    with self.subTest(label=label, metric=metric, value=value):
                        invalid = dict(valid, **{metric: value})
                        base, cand = (invalid, valid) if label == "baseline" else (valid, invalid)
                        with self.assertRaisesRegex(ValueError, f"opus_bleu: {label} metric '{metric}'"):
                            judge("opus_bleu", base, cand)

    def test_missing_metrics_fail_on_either_side(self):
        valid = {"bleu": 30.0, "tokens_per_sec": 100.0}
        for label in ("baseline", "candidate"):
            for metric in valid:
                with self.subTest(label=label, metric=metric):
                    invalid = {k: v for k, v in valid.items() if k != metric}
                    base, cand = (invalid, valid) if label == "baseline" else (valid, invalid)
                    with self.assertRaisesRegex(ValueError, f"opus_bleu: {label} metric '{metric}'"):
                        judge("opus_bleu", base, cand)

    def test_zero_values_are_valid(self):
        rows, failed = judge("whisper_wer", {"wer": 0, "rtf_inverse": 0.0},
                             {"wer": 0.0, "rtf_inverse": 0})
        self.assertFalse(failed)
        self.assertTrue(all(row[-1] == "ok" for row in rows))

    def test_quality_regression_fails(self):
        _, failed = judge("opus_bleu", {"bleu": 30, "tokens_per_sec": 100},
                          {"bleu": 29, "tokens_per_sec": 100})
        self.assertTrue(failed)

    def test_performance_thresholds_are_relative_and_fail_gate(self):
        for task, metric, quality in (
            ("whisper_wer", "rtf_inverse", {"wer": 0.1}),
            ("opus_bleu", "tokens_per_sec", {"bleu": 30}),
            ("t5gemma_generation", "tokens_per_sec", {}),
        ):
            for baseline in (1, 100, 1000):
                for ratio, expected_failure in ((0.8, True), (0.9, False),
                                                (0.95, False), (1.2, False)):
                    with self.subTest(task=task, baseline=baseline, ratio=ratio):
                        rows, failed = judge(
                            task, dict(quality, **{metric: baseline}),
                            dict(quality, **{metric: baseline * ratio}))
                        self.assertEqual(failed, expected_failure)
                        row = next(row for row in rows if row[0] == metric)
                        self.assertEqual(row[-1], "FAIL" if expected_failure else "ok")

    def test_main_exit_status_for_performance_regressions(self):
        baseline = {
            "whisper_wer": {"wer": 0.1, "rtf_inverse": 100},
            "opus_bleu": {"bleu": 30, "tokens_per_sec": 100},
            "t5gemma_generation": {"tokens_per_sec": 100, "outputs": ["hello"]},
        }
        for degraded_task in (None, *baseline):
            with self.subTest(degraded_task=degraded_task):
                def result(py, task, limit, device):
                    metrics = dict(baseline[task])
                    if py == "candidate" and task == degraded_task:
                        metric = "rtf_inverse" if task == "whisper_wer" else "tokens_per_sec"
                        metrics[metric] = 80
                    return metrics
                with patch.object(run_gate, "latest_version", return_value="4.8.5"), \
                     patch.object(run_gate, "make_venv", side_effect=["baseline", "candidate"]), \
                     patch.object(run_gate, "run_task", side_effect=result), \
                     patch("sys.argv", ["run_gate.py"]), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaises(SystemExit) as exited:
                    run_gate.main()
                self.assertEqual(exited.exception.code, 1 if degraded_task else 0)

    def test_confirmation_cannot_hide_quality_failures(self):
        for rounds, expected in (
            ([(False, {"tokens_per_sec"}), (False, set())], False),
            ([(False, {"tokens_per_sec"}), (False, {"tokens_per_sec"})], True),
            ([(True, {"tokens_per_sec"}), (False, set())], True),
            ([(False, {"tokens_per_sec"}), (True, set())], True),
            ([(True, set())], True),
            ([(False, set())], False),
        ):
            with self.subTest(rounds=rounds), \
                 patch.object(run_gate, "measure_round", side_effect=rounds) as measure, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_gate.evaluate_task("b", "c", "opus_bleu", 20, "cpu"),
                                 expected)
                self.assertEqual(measure.call_count, len(rounds))

    def test_median_ignores_outlier_and_alternates_order(self):
        values = iter((100, 100, 1))
        order = []

        def result(py, *args):
            order.append(py)
            return {"bleu": 30, "tokens_per_sec": next(values) if py == "c" else 100}

        with patch.object(run_gate, "run_task", side_effect=result), \
             contextlib.redirect_stdout(io.StringIO()):
            quality, performance = run_gate.measure_round("b", "c", "opus_bleu", 20, "cpu")
        self.assertFalse(quality)
        self.assertEqual(performance, set())
        self.assertEqual(order, ["b", "c", "c", "b", "b", "c"])

    def test_quality_failure_in_one_measurement_is_retained(self):
        values = iter((30, 29, 30))

        def result(py, *args):
            return {"bleu": next(values) if py == "c" else 30, "tokens_per_sec": 100}

        with patch.object(run_gate, "run_task", side_effect=result), \
             contextlib.redirect_stdout(io.StringIO()):
            quality, _ = run_gate.measure_round("b", "c", "opus_bleu", 20, "cpu")
        self.assertTrue(quality)

    def test_threshold_equality_is_not_a_regression(self):
        rows, failed = judge("whisper_wer", {"wer": 0, "rtf_inverse": 100},
                             {"wer": 0.003, "rtf_inverse": 90})
        self.assertFalse(failed)
        self.assertTrue(all(row[-1] == "ok" for row in rows))


if __name__ == "__main__":
    unittest.main()
