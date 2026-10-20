"""Regression checks for shared conversion output handling."""

import os
import stat
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

from ctranslate2.converters.converter import Converter


class FixtureConverter(Converter):
    def __init__(self):
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

    @unittest.skipIf(os.name == "nt", "POSIX directory permissions")
    def test_output_respects_umask(self):
        for mask in (0o022, 0o027, 0o077):
            with self.subTest(umask=oct(mask)):
                previous = os.umask(mask)
                try:
                    FixtureConverter().convert(str(self.output), force=True)
                    self.assertEqual(
                        stat.S_IMODE(self.output.stat().st_mode), 0o777 & ~mask
                    )
                    self.assertEqual(os.umask(mask), mask)
                finally:
                    os.umask(previous)
