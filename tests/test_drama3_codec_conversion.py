"""Run: python3 -m unittest discover -s tests -p test_drama3_codec_conversion.py"""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "codec_converter", Path(__file__).resolve().parents[1] / "tts/dev/convert_drama3_codec.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def manifest(name="decoder.conv.weight", shape=(2, 1, 2)):
    return {"tensors": [{"required_for_inference": True, "source_name": "codec_weights/" + name,
                         "expected_logical_shape": list(shape)}]}


class CodecConversionTest(unittest.TestCase):
    def test_independent_scalar_weight_norm(self):
        direction = np.array([[[3, 4]], [[0, -5]]], np.float32)
        magnitude = np.array([[[10]], [[2]]], np.float32)
        folded = converter.fold_axis0_chunk(direction, magnitude)
        np.testing.assert_allclose(folded, [[[6, 8]], [[0, -2]]], atol=1e-7)
        self.assertEqual(folded.dtype, np.float32)

    def test_transpose_weight_norm_reduces_all_axes_except_input_channel(self):
        # Transpose weight is [Cin,Cout,K]; normalize one full Cin slice,
        # never independently normalize Cout or kernel positions.
        direction = np.array([[[1, 2], [2, 4]], [[0, 0], [3, 4]]], np.float32)
        magnitude = np.array([[[5]], [[10]]], np.float32)
        result = converter.fold_axis0_chunk(direction, magnitude)
        np.testing.assert_allclose(result, [[[1, 2], [2, 4]], [[0, 0], [6, 8]]])

    def test_legacy_new_and_direct_conversion(self):
        direction = np.array([[[3, 4]], [[0, -5]]], np.float32)
        g = np.array([[[10]], [[2]]], np.float32)
        for style in ("direct", "legacy", "new"):
            with self.subTest(style=style):
                if style == "direct":
                    state = {"decoder.conv.weight": direction}
                    expected = direction
                else:
                    names = ("decoder.conv.weight_g", "decoder.conv.weight_v") if style == "legacy" else (
                        "decoder.conv.parametrizations.weight.original0",
                        "decoder.conv.parametrizations.weight.original1")
                    state = {names[0]: g, names[1]: direction}
                    expected = np.array([[[6, 8]], [[0, -2]]], np.float32)
                tensors = converter.load_codec_tensors(state, manifest(), chunk_rows=1)
                self.assertEqual(list(tensors), ["codec/decoder/conv/weight"])
                actual = tensors["codec/decoder/conv/weight"]
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, expected)

    def test_shape_and_dtype_audit_guards(self):
        for state in ({"decoder.conv.weight": np.ones((2, 2), np.float32)},
                      {"decoder.conv.weight": np.ones((2, 1, 2), np.int8)},
                      {"decoder.conv.weight_g": np.ones((2, 1, 1), np.float32)},
                      {"decoder.conv.weight_g": np.ones((1, 1, 1), np.float32),
                       "decoder.conv.weight_v": np.ones((2, 1, 2), np.float32)}):
            with self.assertRaises(ValueError):
                converter.conversion_plan(converter.state_schema(state), manifest())

    def test_ambiguous_materialized_and_normalized_weights_rejected(self):
        state = {"decoder.conv.weight": np.ones((2, 1, 2), np.float32),
                 "decoder.conv.weight_g": np.ones((2, 1, 1), np.float32),
                 "decoder.conv.weight_v": np.ones((2, 1, 2), np.float32)}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            converter.conversion_plan(converter.state_schema(state), manifest())

    def test_nonfinite_zero_norm_and_overflow_rejected(self):
        for direction, magnitude in (
                (np.zeros((1, 1, 2)), np.ones((1, 1, 1))),
                (np.array([[[np.nan, 1]]]), np.ones((1, 1, 1))),
                (np.ones((1, 1, 2)), np.array([[[np.inf]]]))):
            with self.assertRaises(ValueError):
                converter.fold_axis0_chunk(direction, magnitude)
        with self.assertRaises(ValueError):
            converter.numpy_float_chunk(np.array([1e100]))

    def test_derived_buffers_excluded_and_alpha_not_transformed(self):
        m = manifest("decoder.alpha", (1, 2, 1))
        m["tensors"].append({"required_for_inference": False, "source_name": "ignored"})
        state = {"decoder.alpha": np.array([[[.2], [-.3]]], np.float32)}
        tensors = converter.load_codec_tensors(state, m)
        np.testing.assert_array_equal(tensors["codec/decoder/alpha"], state["decoder.alpha"])
        self.assertEqual(len(tensors), 1)

    def test_ct2_binary_v6_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            state = {"decoder.conv.weight": np.array([[[3, 4]], [[5, 6]]], np.float32)}
            model = converter.Drama3Spec(converter.load_codec_tensors(state, manifest(), chunk_rows=1),
                                         model_name="Drama3CodecFixture")
            model.validate()
            model.optimize(quantization="float32")
            model.save(directory)
            data = memoryview(path.read_bytes())
            self.assertEqual(struct.unpack_from("<I", data)[0], 6)
            pos = 4
            name_len = struct.unpack_from("<H", data, pos)[0]
            pos += 2 + name_len
            revision, count = struct.unpack_from("<II", data, pos)
            self.assertEqual((revision, count), (1, 1))
            pos += 8
            name_len = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            self.assertEqual(bytes(data[pos:pos + name_len]), b"codec/decoder/conv/weight\0")
            pos += name_len
            rank = data[pos]
            self.assertEqual(rank, 3)
            pos += 1
            self.assertEqual(struct.unpack_from("<III", data, pos), (2, 1, 2))
            pos += 12
            dtype, size = struct.unpack_from("<BI", data, pos)
            self.assertEqual((dtype, size), (0, 16))
            np.testing.assert_array_equal(np.frombuffer(data[pos + 5:pos + 5 + size], dtype="<f4"), [3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
