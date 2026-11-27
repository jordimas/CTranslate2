"""Resource and latency helpers used by the release-gate tasks."""

import unittest
from unittest.mock import patch

import tasks


class ResourceHelpersTest(unittest.TestCase):
    def test_percentile_is_bounded_and_ordered(self):
        values = [5, 1, 4, 2, 3]
        self.assertEqual(tasks.percentile(values, 0.0), 1)
        self.assertEqual(tasks.percentile(values, 0.95), 5)
        self.assertEqual(tasks.percentile([7], 0.95), 7)

    def test_rss_growth_handles_unsupported_platforms(self):
        with patch.object(tasks, "peak_rss_mb", return_value=None):
            self.assertIsNone(tasks.rss_growth_mb(100.0))
            self.assertIsNone(tasks.rss_growth_mb(None))

    def test_rss_growth_is_difference_of_peaks(self):
        with patch.object(tasks, "peak_rss_mb", return_value=150.0):
            self.assertEqual(tasks.rss_growth_mb(100.0), 50.0)


if __name__ == "__main__":
    unittest.main()
