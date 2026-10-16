"""Standalone tests for CT2-compatible row quantization and streaming output."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np


def load_helper(name):
    path = Path(__file__).resolve().parents[1] / "python/ctranslate2/converters" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quantization = load_helper("drama3_quantization")


class Reader:
    def __init__(self, matrix):
        self.matrix = matrix
        self.tensors = {"explicit_source": {"shape": matrix.shape}}
        self.calls = []

    def iter_matrix_rows(self, name, **options):
        self.calls.append((name, options))
        for start in range(0, len(self.matrix), options["chunk_rows"]):
            yield self.matrix[start:start + options["chunk_rows"]].copy()


class Writer:
    def __init__(self):
        self.variables = {}
        self.chunk_shapes = []

    def write_tensor(self, name, shape, dtype, chunks):
        values = []
        for chunk in chunks:
            self.chunk_shapes.append(chunk.shape)
            values.append(chunk.copy())
        self.variables[name] = (shape, dtype, np.concatenate(values))


class Drama3QuantizationTest(unittest.TestCase):
    def test_actual_reader_writer_and_failure_cleanup(self):
        readers = load_helper("drama3_weights")
        writers = load_helper("drama3_writer")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, output_path = root / "source.safetensors", root / "model.bin"
            matrix = np.array([[0., 0.], [1., -2.], [3., -4.]], np.float32)
            header = {"matrix": {"shape": [3, 2], "dtype": "F32", "data_offsets": [0, matrix.nbytes]}}
            encoded = json.dumps(header).encode()
            source_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + matrix.tobytes())
            reader = readers.SafeTensorMatrixReader(source_path)
            with writers.StreamingModelWriter(output_path, "Drama3MatrixFixture", 1, 2) as output:
                quantization.write_int8_matrix(output, reader, "matrix", "expert", chunk_rows=2)
                output.finish()
            self.assertTrue(output_path.is_file())
            self.assertIn(b"expert/weight\0", output_path.read_bytes())
            self.assertIn(b"expert/weight_scale\0", output_path.read_bytes())
            output_path.unlink()
            matrix[2, 0] = np.nan
            source_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + matrix.tobytes())
            reader = readers.SafeTensorMatrixReader(source_path)
            with writers.StreamingModelWriter(output_path, "Drama3MatrixFixture", 1, 2) as output:
                with self.assertRaises(ValueError):
                    quantization.write_int8_matrix(output, reader, "matrix", "expert", chunk_rows=2)
            self.assertEqual(list(root.iterdir()), [source_path])

    def test_zero_rows_signed_values_and_ties_to_even(self):
        matrix = np.array([[0, -0., 0, 0], [-127, -2.5, 2.5, 126.5],
                           [-2, -1, 1, 2]], dtype=np.float32)
        result, scales = quantization.quantize_int8_rows(matrix)
        np.testing.assert_array_equal(scales, [1, 1, 63.5])
        np.testing.assert_array_equal(result, [[0, 0, 0, 0], [-127, -2, 2, 126], [-127, -64, 64, 127]])
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float32)

    def test_chunk_boundaries_two_pass_and_explicit_source_options(self):
        matrix = np.array([[0, 0, 0], [1, 2, 3], [-4, 5, -6], [7, 8, 9], [10, -11, 12]], np.float32)
        reader, writer = Reader(matrix), Writer()
        scale_reader = object()
        quantization.write_int8_matrix(writer, reader, "explicit_source", "layer.expert.gate",
                                        chunk_rows=2, inverse_scale="explicit_scale", scale_reader=scale_reader)
        self.assertEqual(len(reader.calls), 2)
        self.assertEqual(reader.calls[0][1]["inverse_scale"], "explicit_scale")
        self.assertIs(reader.calls[0][1]["scale_reader"], scale_reader)
        self.assertEqual(writer.chunk_shapes, [(2, 3), (2, 3), (1, 3), (2,), (2,), (1,)])
        amax = np.max(np.abs(matrix), axis=1)
        amax[amax == 0] = 127
        expected_scales = np.float32(127) / amax
        expected_weights = np.rint(matrix * expected_scales[:, None]).astype(np.int8)
        np.testing.assert_array_equal(writer.variables["layer.expert.gate/weight"][2], expected_weights)
        np.testing.assert_array_equal(writer.variables["layer.expert.gate/weight_scale"][2], expected_scales)

    def test_large_finite_values(self):
        maximum = np.finfo(np.float32).max
        matrix = np.array([[maximum, -maximum, 0]], np.float32)
        result, scales = quantization.quantize_int8_rows(matrix)
        np.testing.assert_array_equal(result, [[127, -127, 0]])
        self.assertTrue(np.all(np.isfinite(scales)))

    def test_rejects_nonfinite_and_unrepresentable_tiny_scales(self):
        for value in (np.nan, np.inf, -np.inf, np.nextafter(np.float32(0), np.float32(1))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantization.quantize_int8_rows(np.array([[value]], np.float32))

    def test_rejects_wrong_types_shapes_and_scope(self):
        for value in (np.zeros((1, 1), np.float64), np.zeros(3, np.float32), np.zeros((2, 0), np.float32)):
            with self.assertRaises(ValueError):
                quantization.quantize_int8_rows(value)
        with self.assertRaises(ValueError):
            quantization.write_int8_matrix(Writer(), Reader(np.ones((1, 1), np.float32)), "explicit_source", "")


if __name__ == "__main__":
    unittest.main()
