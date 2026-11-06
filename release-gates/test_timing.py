"""Check warm-up consumption and timing boundaries without model dependencies."""

import unittest
from unittest.mock import patch

from tasks import run_and_measure


class TimingTest(unittest.TestCase):
    def test_warmup_is_consumed_before_timing(self):
        events = []

        def infer(batch, scale):
            for value in batch:
                events.append(value)
                yield value * scale

        def clock():
            events.append("clock")
            return len(events)

        with patch("tasks.time.perf_counter", side_effect=clock):
            results, elapsed = run_and_measure(infer, [1, 2, 3], warmup_size=2, scale=10)
        self.assertEqual(events, [1, 2, "clock", 1, 2, 3, "clock"])
        self.assertEqual(results, [10, 20, 30])
        self.assertEqual(elapsed, 4)
