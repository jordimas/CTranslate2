"""Regression checks for shared conversion output handling."""

import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from ctranslate2.converters.converter import Converter


class FixtureConverter(Converter):
    def __init__(self, atomic=False):
        self._atomic_conversion = atomic
        self.spec = Mock()
        self.spec.save.side_effect = lambda output: (
            Path(output) / "model.bin"
        ).write_bytes(b"model")

    def _load(self):
        return self.spec


class ConverterOutputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "nested" / "model"

    def test_standard_converter_preserves_conversion_arguments_and_force(self):
        converter = FixtureConverter()
        self.assertEqual(
            converter.convert(str(self.output), vmap="vmap", quantization="float32"),
            str(self.output),
        )
        converter.spec.register_vocabulary_mapping.assert_called_once_with("vmap")
        converter.spec.validate.assert_called_once_with()
        converter.spec.optimize.assert_called_once_with(quantization="float32")
        self.assertEqual((self.output / "model.bin").read_bytes(), b"model")
        converter.spec.reset_mock()
        with self.assertRaises(RuntimeError):
            converter.convert(str(self.output))
        converter.spec.save.assert_not_called()
        (self.output / "old").touch()
        converter.convert(str(self.output), force=True)
        self.assertFalse((self.output / "old").exists())

    def test_atomic_report_failure_preserves_previous_output_and_cleans_staging(self):
        converter = FixtureConverter(atomic=True)
        converter.convert(str(self.output))
        converter.conversion_report = {"dry_run": False}
        with patch.object(
            converter, "write_report", side_effect=OSError("report failed")
        ):
            with self.assertRaisesRegex(OSError, "report failed"):
                converter.convert(str(self.output), force=True)
        self.assertEqual((self.output / "model.bin").read_bytes(), b"model")
        self.assertEqual(list(self.output.parent.glob(".model.*")), [])

    def test_atomic_publication_does_not_replace_concurrent_output(self):
        converter = FixtureConverter(atomic=True)

        def save(output):
            (Path(output) / "model.bin").write_bytes(b"new")
            self.output.mkdir()
            (self.output / "existing").write_bytes(b"keep")

        converter.spec.save.side_effect = save
        with self.assertRaises(OSError):
            converter.convert(str(self.output))
        self.assertEqual((self.output / "existing").read_bytes(), b"keep")
        self.assertFalse((self.output / "model.bin").exists())
        self.assertEqual(list(self.output.parent.glob(".model.*")), [])

    @unittest.skipIf(os.name == "nt", "Creating symlinks requires Windows privileges")
    def test_atomic_force_rejects_symlink_without_changing_target(self):
        target = Path(self.temp.name) / "target"
        target.mkdir()
        (target / "marker").write_bytes(b"keep")
        self.output.parent.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular directory"):
            FixtureConverter(atomic=True).convert(str(self.output), force=True)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual((target / "marker").read_bytes(), b"keep")
        self.assertEqual(list(self.output.parent.glob(".model.*")), [])
