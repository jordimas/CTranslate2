"""Run: python3 -m unittest discover -s tests -p test_drama3_model_conversion.py"""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "model_converter", Path(__file__).resolve().parents[1] / "tts/dev/convert_drama3_model.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def config():
    return {"audio_decoder_config": {"n_head": 2, "n_local_heads": 1, "head_dim": 2},
            "quantization_config": {"weight_block_size": [2, 2]}}


def shard(path, tensors):
    header, data = {}, b""
    for name, (array, dtype) in tensors.items():
        payload = array.tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class Recorder:
    def __init__(self):
        self.values = {}
        self.max_rows = 0

    def write_tensor(self, name, shape, dtype, chunks):
        arrays = []
        for chunk in chunks:
            arrays.append(chunk.copy().reshape(-1))
            self.max_rows = max(self.max_rows, chunk.shape[0])
        self.values[name] = np.concatenate(arrays).reshape(shape)


class DramaModelConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_backbone_and_tower_names(self):
        cases = {
            "model.language_model.layers.3.mlp.gate.weight": "backbone/layers_3/mlp/router/weight",
            "model.language_model.layers.0.mlp.experts.12.up_proj.weight": "backbone/layers_0/mlp/experts_12/up_proj/weight",
            "model.language_model.embed_tokens.weight": "backbone/embeddings/weight",
            "model.language_model.norm.weight": "backbone/norm/weight",
            "lm_head.weight": "backbone/lm_head/weight",
            "audio_decoder.codebook_embeddings.weight": "tower/codebook_embeddings/weight",
            "audio_decoder_mtp_shared.layers.0.transformer_layer.attention.wo.weight": "tower/shared_layers_0/transformer_layer/attention/o_proj/weight",
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w1.weight": "tower/residual_layers_2/transformer_layer/feed_forward/gate_proj/weight",
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w3.weight": "tower/residual_layers_2/transformer_layer/feed_forward/up_proj/weight",
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w2.weight": "tower/residual_layers_2/transformer_layer/feed_forward/down_proj/weight",
            "audio_decoder_mtp.heads.8.weight": "tower/heads_8/weight"}
        for source, target in cases.items():
            self.assertEqual(converter.canonical_name(source), target)
        with self.assertRaises(ValueError):
            converter.canonical_name("unknown.weight")

    def test_bf16_tower_row_split_across_chunk_boundaries(self):
        name = "audio_decoder_mtp_shared.layers.0.transformer_layer.attention.wqkv.weight"
        original = np.arange(36, dtype=np.float32).reshape(12, 3)
        bf16 = (original.view(np.uint32) >> 16).astype("<u2")
        path = self.root / "tower.safetensors"
        shard(path, {name: (bf16, "BF16")})
        reader = converter.READERS.SafeTensorMatrixReader(path)
        for kind, rows in (("q_proj", [0, 1, 4, 5, 2, 3, 6, 7]),
                           ("k_proj", [8, 9]), ("v_proj", [10, 11])):
            adapter = converter.TowerMatrixReader(reader, name, kind, 2, 1, 2)
            chunks = list(adapter.iter_matrix_rows(name, chunk_rows=3))
            self.assertTrue(all(len(chunk) <= 3 for chunk in chunks))
            np.testing.assert_array_equal(np.concatenate(chunks), original[rows])
        metadata = {name: {**reader.tensors[name], "shard": path.name}}
        plan = converter.build_plan(metadata, config())
        self.assertEqual([item["shape"] for item in plan], [[8, 3], [2, 3], [2, 3]])
        output = Recorder()
        converter.write_model_tensors(output, plan, {path.name: reader}, config(), chunk_rows=3)
        prefix = "tower/shared_layers_0/transformer_layer/attention/"
        quantized, scales = converter.QUANTIZATION.quantize_int8_rows(original[[0, 1, 4, 5, 2, 3, 6, 7]])
        np.testing.assert_array_equal(output.values[prefix + "q_proj/weight"], quantized)
        np.testing.assert_array_equal(output.values[prefix + "q_proj/weight_scale"], scales)

    def test_cross_shard_fp8_scales_are_explicit_and_applied_before_int8(self):
        name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        scale_name = name + "_scale_inv"
        weight_path, scale_path = self.root / "weights.safetensors", self.root / "scales.safetensors"
        shard(weight_path, {name: (np.full((4, 4), 0x38, np.uint8), "F8_E4M3FN")})
        scales = np.array([[.5, 1], [2, 4]], np.float32)
        shard(scale_path, {scale_name: (scales, "F32")})
        readers = {p.name: converter.READERS.SafeTensorMatrixReader(p) for p in (weight_path, scale_path)}
        tensors = {n: {**t, "shard": p} for p, r in readers.items() for n, t in r.tensors.items()}
        plan = converter.build_plan(tensors, config())
        self.assertEqual(plan[0]["scale_shard"], scale_path.name)
        output = Recorder()
        converter.write_model_tensors(output, plan, readers, config(), chunk_rows=1)
        expected = np.repeat(np.repeat(scales, 2, axis=0), 2, axis=1)
        quantized, target_scales = converter.QUANTIZATION.quantize_int8_rows(expected)
        target = "backbone/layers_0/mlp/experts_0/gate_proj/weight"
        np.testing.assert_array_equal(output.values[target], quantized)
        np.testing.assert_array_equal(output.values[target + "_scale"], target_scales)

    def test_missing_or_wrong_source_scale_rejected(self):
        name = "lm_head.weight"
        tensors = {name: {"shape": (4, 4), "dtype": "F8_E4M3FN", "shard": "a"}}
        with self.assertRaisesRegex(ValueError, "explicit inverse scale"):
            converter.build_plan(tensors, config())
        tensors[name + "_scale_inv"] = {"shape": (1, 1), "dtype": "F32", "shard": "b"}
        with self.assertRaisesRegex(ValueError, "shape/type"):
            converter.build_plan(tensors, config())
        tensors[name]["dtype"] = "BF16"
        with self.assertRaisesRegex(ValueError, "non-FP8"):
            converter.build_plan(tensors, config())

    def test_vectors_and_convolutions_keep_float32_and_normalization_offsets(self):
        name = "model.language_model.layers.0.linear_attn.norm.weight"
        path = self.root / "vectors.safetensors"
        values = np.array([-.5, .3], dtype="<f4")
        shard(path, {name: (values, "F32")})
        reader = converter.READERS.SafeTensorMatrixReader(path)
        metadata = {name: {**reader.tensors[name], "shard": path.name}}
        output = Recorder()
        plan = converter.build_plan(metadata, config())
        self.assertEqual(plan[0]["dtype"], "float32")
        converter.write_model_tensors(output, plan, {path.name: reader}, config())
        np.testing.assert_array_equal(output.values["backbone/layers_0/linear_attn/norm/weight"], values)

    def test_index_exact_coverage_and_path_guards(self):
        name = "lm_head.weight"
        p = self.root / "model-00001-of-00001.safetensors"
        shard(p, {name: (np.ones((2, 2), np.float32), "F32")})
        index_path = self.root / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": {name: p.name}}))
        readers, tensors = converter.inspect_shards(self.root)
        self.assertEqual(len(readers), 1)
        self.assertEqual(set(tensors), {name})
        for bad in ("../outside", "/tmp/outside", "sub/weights", "..\\outside"):
            with self.assertRaises(ValueError):
                converter.source_file(self.root, bad)
        index_path.write_text(json.dumps({"weight_map": {"missing": p.name}}))
        with self.assertRaises(ValueError):
            converter.inspect_shards(self.root)

    def test_atomic_directory_publish_never_replaces_even_empty_directory(self):
        staging, destination = self.root / "staging", self.root / "destination"
        staging.mkdir()
        (staging / "marker").write_text("complete")
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            converter.publish_directory(staging, destination)
        self.assertTrue((staging / "marker").exists())
        destination.rmdir()
        converter.publish_directory(staging, destination)
        self.assertEqual((destination / "marker").read_text(), "complete")
        self.assertFalse(staging.exists())

    def test_orphan_scales_and_nonfinite_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, "orphan"):
            converter.build_plan({"unused.weight_scale_inv": {"shape": (1, 1), "dtype": "F32", "shard": "a"}}, config())
        path = self.root / "nonfinite.safetensors"
        name = "model.language_model.norm.weight"
        shard(path, {name: (np.array([np.nan], np.float32), "F32")})
        reader = converter.READERS.SafeTensorMatrixReader(path)
        with self.assertRaises(ValueError):
            list(converter.float_chunks(reader, name))

    def test_tiny_row_threshold_inclusive_and_above_threshold_preserved(self):
        threshold = converter.TINY_ROW_LIMIT
        above = np.nextafter(threshold, np.float32(np.inf))
        original = np.array([[threshold, -threshold], [above, -above], [0, 0]], np.float32)

        class Rows:
            tensors = {"w": {"shape": original.shape}}

            def iter_matrix_rows(self, *args, **kwargs):
                yield original.copy()

        wrapped = converter.TinyRowReader(Rows())
        transformed = np.concatenate(list(wrapped.iter_matrix_rows("w")))
        np.testing.assert_array_equal(transformed[0], [0, 0])
        np.testing.assert_array_equal(transformed[1:], original[1:])
        self.assertLessEqual(float(np.max(np.abs(original - transformed))), float(threshold))
        self.assertEqual(wrapped.zeroed_rows, 1)
        list(wrapped.iter_matrix_rows("w"))
        self.assertEqual(wrapped.zeroed_rows, 1)  # second quantizer pass is not double-counted
        quantized, scales = converter.QUANTIZATION.quantize_int8_rows(transformed)
        self.assertTrue(np.isfinite(scales).all())
        np.testing.assert_array_equal(quantized[0], [0, 0])


if __name__ == "__main__":
    unittest.main()
