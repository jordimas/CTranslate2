"""End-to-end checks of the standard model-spec conversion path."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from ctranslate2.converters import drama3 as c
from ctranslate2.converters import Converter, TransformersConverter


def read_model(path):
    with path.open("rb") as stream:
        def integer(size):
            return int.from_bytes(stream.read(size), "little")

        def string():
            return stream.read(integer(2))[:-1].decode()

        assert integer(4) == 6 and string() == "Drama3Spec" and integer(4) == 1
        values = {}
        for _ in range(integer(4)):
            name = string()
            shape = tuple(integer(4) for _ in range(integer(1)))
            dtype, size = integer(1), integer(4)
            values[name] = np.frombuffer(stream.read(size), dtype={0: "<f4", 1: "i1"}[dtype]).reshape(shape)
        for _ in range(integer(4)):
            alias, target = string(), string()
            values[alias] = values[target]
        return values


class StandardConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.temp.name) / "converted"
        for name in c.ASSETS + ("codec.pth",):
            (self.source / name).write_text("fixture\n")
        config = {"model_type": "fish_omni",
                  "audio_decoder_config": {"n_head": 1, "n_local_heads": 1, "head_dim": 2},
                  "quantization_config": {"weight_block_size": [2, 2]}}
        (self.source / "config.json").write_text(json.dumps(config))
        self.expert = "model.language_model.layers.0.mlp.experts.0.up_proj.weight"
        self.sensitive = np.array([[1.0001, -.33331], [.12345, -.76543]], np.float32)
        self.convolution = np.arange(6, dtype=np.float32).reshape(2, 1, 3) / np.float32(7)
        tensors = {
            self.expert: (np.array([[0x38, 0xb8], [0x30, 0x28]], np.uint8), "F8_E4M3FN"),
            self.expert + "_scale_inv": (np.array([[1.0001]], np.float32), "F32"),
            "lm_head.weight": (self.sensitive, "F32"),
            "model.language_model.layers.0.linear_attn.conv1d.weight": (self.convolution, "F32"),
        }
        data, header = b"", {}
        for name, (value, dtype) in tensors.items():
            payload = value.tobytes()
            header[name] = {"shape": list(value.shape), "dtype": dtype,
                            "data_offsets": [len(data), len(data) + len(payload)]}
            data += payload
        encoded = json.dumps(header).encode()
        shard = "model-00001-of-00001.safetensors"
        (self.source / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard for name in tensors}}))
        codec = c.codec
        manifest = {"tensors": [{"required_for_inference": True,
                                "source_name": "codec_weights/decoder.weight",
                                "expected_logical_shape": [1, 2]}]}
        self.codec = np.array([[.234567, -.345678]], np.float32)
        self.enterContext(patch.object(codec, "load_manifest", return_value=manifest))
        self.enterContext(patch.object(codec, "load_checkpoint", return_value={"decoder.weight": self.codec}))

    def test_standard_quantization_and_serialization_preserve_sensitive_weights(self):
        converter = TransformersConverter(self.source)
        converter.convert(str(self.output))
        result = converter.conversion_report
        values = read_model(self.output / "model.bin")
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        original = np.array([[1, -1], [.5, .25]], np.float32) * np.float32(1.0001)
        original = torch.from_numpy(original).bfloat16().float().numpy()
        scales = np.float32(127) / np.max(np.abs(original), axis=1)
        expected = np.rint(original * scales[:, None]).astype(np.int8)
        np.testing.assert_array_equal(values[expert], expected)
        np.testing.assert_array_equal(values[expert + "_scale"], scales)
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        np.testing.assert_array_equal(values["codec/decoder/weight"], self.codec)
        np.testing.assert_array_equal(values["backbone/layers_0/linear_attn/conv1d/weight"], self.convolution)
        self.assertEqual(result["quantization_policy"], "experts_int8_sensitive_float32")
        self.assertEqual(json.loads((self.output / "config.json").read_text())["matrix_dtype"], "mixed")
        self.assertEqual((self.output / "tokenizer.json").read_bytes(), (self.source / "tokenizer.json").read_bytes())

    def test_experts_are_stored_in_bf16_before_optimization(self):
        spec = TransformersConverter(self.source)._load()
        values = spec.variables()
        expert = values["backbone/layers_0/mlp/experts_0/up_proj/weight"]
        self.assertEqual(expert.dtype, torch.bfloat16)
        self.assertEqual(expert.numel() * expert.element_size(), 8)
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        report = TransformersConverter(self.source).inspect()
        self.assertEqual(report["estimated_float32_loading_bytes"] - report["estimated_loading_bytes"], 8)

    def test_float32_conversion_preserves_scaled_fp8_without_quantization(self):
        TransformersConverter(self.source).convert(str(self.output), quantization="float32")
        values = read_model(self.output / "model.bin")
        expected = np.array([[1, -1], [.5, .25]], np.float32) * np.float32(1.0001)
        np.testing.assert_array_equal(values["backbone/layers_0/mlp/experts_0/up_proj/weight"], expected)
        self.assertFalse(any(name.endswith("_scale") for name in values))

    def test_invalid_codec_is_rejected_before_loading_backbone(self):
        with patch.object(c.codec, "conversion_plan", side_effect=ValueError("invalid codec")), \
                patch.object(c, "load_model_tensors") as load:
            with self.assertRaisesRegex(ValueError, "invalid codec"):
                TransformersConverter(self.source).convert(str(self.output))
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failure_removes_partial_package(self):
        with patch.object(c.Drama3Spec, "save", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                TransformersConverter(self.source).convert(str(self.output))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".converted.*")), [])


    def test_standard_api_and_force(self):
        converter = TransformersConverter(self.source)
        self.assertIsInstance(converter, Converter)
        report = converter.inspect()
        self.assertTrue(report['dry_run'])
        self.assertFalse(self.output.exists())
        self.assertEqual(converter.convert(str(self.output)), str(self.output))
        values = read_model(self.output / 'model.bin')
        self.assertEqual(values['backbone/layers_0/mlp/experts_0/up_proj/weight'].dtype, 'int8')
        with self.assertRaises(FileExistsError):
            converter.convert(str(self.output))
        converter.convert(str(self.output), quantization='float32', force=True)
        values = read_model(self.output / 'model.bin')
        self.assertFalse(any(key.endswith('_scale') for key in values))

    def test_force_failure_preserves_old_output(self):
        converter = TransformersConverter(self.source)
        converter.convert(str(self.output))
        previous = (self.output / 'model.bin').read_bytes()
        with patch.object(c.Drama3Spec, 'save', side_effect=RuntimeError('write failed')):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                converter.convert(str(self.output), force=True)
        self.assertEqual((self.output / 'model.bin').read_bytes(), previous)

    def test_reject_unsupported_quantization_before_loading(self):
        with patch.object(TransformersConverter, '_load') as load:
            with self.assertRaisesRegex(ValueError, 'quantization'):
                TransformersConverter(self.source).convert(str(self.output), quantization='float16')
        load.assert_not_called()

    def test_transformers_converter_dispatches_local_drama3(self):
        from ctranslate2.converters import TransformersConverter
        converter = TransformersConverter(str(self.source))
        self.assertTrue(converter.inspect()["dry_run"])
        converter.convert(str(self.output), quantization="int8_float32")
        values = read_model(self.output / "model.bin")
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        self.assertEqual(converter.conversion_report["model_name"], "Drama3Spec")

    def test_repeated_conversion_restores_default_quantization(self):
        converter = TransformersConverter(self.source)
        converter.convert(str(self.output), quantization="float32")
        converter.convert(str(self.output), force=True)
        values = read_model(self.output / "model.bin")
        self.assertEqual(values["backbone/layers_0/mlp/experts_0/up_proj/weight"].dtype, "int8")
        self.assertEqual(converter.conversion_report["quantization"], "int8_float32")

    def test_transformers_cli_conversion_writes_report(self):
        from ctranslate2.converters.transformers import main
        report = self.output.parent / "report.json"
        argv = ["ct2-transformers-converter", "--model", str(self.source),
                "--output_dir", str(self.output), "--quantization", "float32",
                "--report", str(report)]
        with patch("sys.argv", argv):
            main()
        values = read_model(self.output / "model.bin")
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        self.assertEqual(json.loads(report.read_text()),
                         json.loads((self.output / "conversion.json").read_text()))

    def test_transformers_cli_dry_run_writes_report_without_loading(self):
        from ctranslate2.converters.transformers import main
        report = self.output.parent / "report.json"
        argv = ["ct2-transformers-converter", "--model", str(self.source),
                "--output_dir", str(self.output), "--dry-run", "--report", str(report)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(c, "load_model_tensors") as load:
            main()
        load.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertTrue(json.loads(report.read_text())["dry_run"])

    def test_transformers_other_models_keep_standard_conversion(self):
        from ctranslate2.converters import TransformersConverter
        (self.source / "config.json").write_text('{"model_type": "llama"}')
        with patch.object(Converter, "convert", return_value=str(self.output)) as convert:
            result = TransformersConverter(str(self.source)).convert(
                str(self.output), quantization="float16", force=True)
        self.assertEqual(result, str(self.output))
        convert.assert_called_once_with(str(self.output), vmap=None,
                                        quantization="float16", force=True)

if __name__ == "__main__":
    unittest.main()
