"""Independent binary-v6 parser tests for the bounded streaming writer."""
import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "drama3_writer", Path(__file__).resolve().parents[1]
    / "python/ctranslate2/converters/drama3_writer.py")
writer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(writer)


def parse_model(path):
    """Test-only parser independent of writer helpers and dtype tables."""
    stream = io.BytesIO(path.read_bytes())

    def number(fmt):
        return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]

    def string():
        payload = stream.read(number("H"))
        assert payload[-1:] == b"\0"
        return payload[:-1].decode("utf-8")

    version, name, revision, count = number("I"), string(), number("I"), number("I")
    variables = {}
    for _ in range(count):
        variable = string()
        shape = tuple(number("I") for _ in range(number("B")))
        dtype, nbytes = number("B"), number("I")
        variables[variable] = (shape, dtype, stream.read(nbytes))
    aliases = [(string(), string()) for _ in range(number("I"))]
    assert stream.read() == b""
    return version, name, revision, variables, aliases


class Drama3WriterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "model.bin"

    def assert_aborted(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_binary_contract_chunked_arrays_bytes_scalars_and_aliases(self):
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 3) as output:
            output.write_tensor("expert.weight", (2, 2), "int8",
                                [np.array([[1, -2]], np.int8), bytes([3, 4])])
            output.write_tensor("expert.weight_scale", (2,), "float32",
                                [np.array([2, 3], np.float32)])
            output.write_tensor("scalar", (), "int32", [struct.pack("<i", -17)])
            self.assertFalse(self.path.exists())
            output.finish({"copy.weight": "expert.weight"})
        version, name, revision, variables, aliases = parse_model(self.path)
        self.assertEqual((version, name, revision), (6, "Drama3", 1))
        self.assertEqual(variables["expert.weight"], ((2, 2), 1, bytes([1, 254, 3, 4])))
        self.assertEqual(variables["expert.weight_scale"], ((2,), 0, struct.pack("<ff", 2, 3)))
        self.assertEqual(variables["scalar"], ((), 3, struct.pack("<i", -17)))
        self.assertEqual(aliases, [("copy.weight", "expert.weight")])
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_dtype_ids_and_bfloat16_bits(self):
        cases = [("float32", np.array([1.5], "<f4"), 0), ("int8", np.array([-1], "i1"), 1),
                 ("int16", np.array([-2], "<i2"), 2), ("int32", np.array([-3], "<i4"), 3),
                 ("float16", np.array([2.5], "<f2"), 4), ("bfloat16", np.array([0x3f80], "<u2"), 5)]
        with writer.StreamingModelWriter(self.path, "Drama3", 1, len(cases)) as output:
            for name, array, _ in cases:
                output.write_tensor(name, (1,), name, [array])
            output.finish()
        variables = parse_model(self.path)[3]
        for name, array, type_id in cases:
            self.assertEqual(variables[name], ((1,), type_id, array.tobytes()))

    def test_generator_chunks_are_consumed_incrementally(self):
        def chunks():
            for value in range(20):
                yield np.full((1, 3), value, dtype=np.float32)
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 1) as output:
            output.write_tensor("weight", (20, 3), "float32", chunks())
            output.finish()
        np.testing.assert_array_equal(np.frombuffer(parse_model(self.path)[3]["weight"][2], "<f4").reshape(20, 3),
                                      np.repeat(np.arange(20)[:, None], 3, axis=1))

    def test_missing_finish_or_exception_removes_temp(self):
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 1):
            pass
        self.assert_aborted()
        with self.assertRaises(RuntimeError):
            with writer.StreamingModelWriter(self.path, "Drama3", 1, 1):
                raise RuntimeError("conversion failed")
        self.assert_aborted()

    def test_payload_mismatch_dtype_and_noncontiguous_abort(self):
        for chunks in ([b""], [b"12345"], [np.array([1], np.int32)],
                       [np.ones((2, 2), np.float32)[:, ::2]], ["bad"]):
            with self.subTest(chunks=chunks):
                with writer.StreamingModelWriter(self.path, "Drama3", 1, 1) as output:
                    with self.assertRaises(ValueError):
                        output.write_tensor("weight", (1,), "float32", chunks)
                    with self.assertRaises(RuntimeError):
                        output.finish()
                self.assert_aborted()

    def test_shape_name_and_payload_size_guards(self):
        cases = [("", (1,), "float32"), ("bad\0name", (1,), "float32"),
                 ("w", (0,), "float32"), ("w", (-1,), "float32"),
                 ("w", (True,), "float32"), ("w", (1,) * 256, "float32"),
                 ("w", (1 << 32,), "float32"), ("w", (1 << 30,), "float32"),
                 ("w", (1,), "float64")]
        for name, shape, dtype in cases:
            with self.subTest(shape=shape), writer.StreamingModelWriter(self.path, "Drama3", 1, 1) as output:
                with self.assertRaises(ValueError):
                    output.write_tensor(name, shape, dtype, [])
            self.assert_aborted()

    def test_alias_targets_duplicates_and_implicit_scale_collisions(self):
        for aliases in ([('a', 'missing')], [('w', 'w')], [('a', 'w'), ('a', 'w')],
                        [('a', 'w'), ('b', 'a')], [('a', 'w'), ('a_scale', 'w')]):
            with self.subTest(aliases=aliases), writer.StreamingModelWriter(self.path, "Drama3", 1, 2) as output:
                output.write_tensor("w", (1,), "int8", [b"1"])
                output.write_tensor("w_scale", (1,), "float32", [bytes(4)])
                with self.assertRaises(ValueError):
                    output.finish(aliases)
            self.assert_aborted()

    def test_count_and_duplicate_tensor_guards(self):
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 1) as output:
            with self.assertRaises(ValueError):
                output.finish()
        self.assert_aborted()
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 2) as output:
            output.write_tensor("w", (1,), "int8", [b"1"])
            with self.assertRaises(ValueError):
                output.write_tensor("w", (1,), "int8", [b"1"])
        self.assert_aborted()

    def test_existing_or_racing_destination_is_preserved(self):
        self.path.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            writer.StreamingModelWriter(self.path, "Drama3", 1, 0)
        self.assertEqual(self.path.read_bytes(), b"keep")
        self.path.unlink()
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 0) as output:
            self.path.write_bytes(b"racing writer")
            with self.assertRaises(FileExistsError):
                output.finish()
        self.assertEqual(self.path.read_bytes(), b"racing writer")
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_utf8_name_lengths_are_bytes(self):
        with writer.StreamingModelWriter(self.path, "Drama3", 1, 1) as output:
            output.write_tensor("áudio", (1,), "int8", [b"x"])
            output.finish()
        self.assertIn("áudio", parse_model(self.path)[3])


if __name__ == "__main__":
    unittest.main()
