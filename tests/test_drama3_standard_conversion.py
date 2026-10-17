"""End-to-end checks of the standard model-spec conversion path."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("standard_conversion", ROOT / "tts/dev/convert_drama3_model.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


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
        codec = c.helper("convert_drama3_codec", development=True)
        manifest = {"tensors": [{"required_for_inference": True,
                                "source_name": "codec_weights/decoder.weight",
                                "expected_logical_shape": [1, 2]}]}
        self.codec = np.array([[.234567, -.345678]], np.float32)
        self.enterContext(patch.object(codec, "load_manifest", return_value=manifest))
        self.enterContext(patch.object(codec, "load_checkpoint", return_value={"decoder.weight": self.codec}))
        self.enterContext(patch.object(c, "helper", return_value=codec))

    def test_standard_quantization_and_serialization_preserve_sensitive_weights(self):
        result = c.convert(self.source, self.output)
        values = read_model(self.output / "model.bin")
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        original = np.array([[1, -1], [.5, .25]], np.float32) * np.float32(1.0001)
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

    def test_float32_conversion_preserves_scaled_fp8_without_quantization(self):
        c.convert(self.source, self.output, quantization="float32")
        values = read_model(self.output / "model.bin")
        expected = np.array([[1, -1], [.5, .25]], np.float32) * np.float32(1.0001)
        np.testing.assert_array_equal(values["backbone/layers_0/mlp/experts_0/up_proj/weight"], expected)
        self.assertFalse(any(name.endswith("_scale") for name in values))

    def test_insufficient_ram_fails_before_loading_or_creating_output(self):
        with patch.object(c, "available_memory", return_value=1), \
                patch.object(c, "load_model_tensors") as load:
            with self.assertRaisesRegex(MemoryError, "No weights were loaded"):
                c.convert(self.source, self.output)
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failure_removes_partial_package(self):
        with patch.object(c.Drama3Spec, "save", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                c.convert(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".converted.*")), [])


if __name__ == "__main__":
    unittest.main()
