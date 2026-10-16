"""Run with python3 -m unittest discover -s tests -p test_drama3_checkpoint_inspector.py."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "drama3_inspector", Path(__file__).resolve().parents[1] / "tts/dev/inspect_drama3_checkpoint.py")
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


def config():
    return {"model_type": "fish_omni", "text_config": {
        "hidden_size": 8, "num_hidden_layers": 2, "num_experts": 4,
        "num_experts_per_tok": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "moe_intermediate_size": 2,
        "layer_types": ["linear_attention", "full_attention"], "mlp_layer_types": ["moe", "moe"]},
        "audio_decoder_config": {"dim": 8, "n_layer": 4, "num_codebooks": 10,
                                 "shared_enrichment_layers": 4, "decoder_type": "codebook_mtp_shared"}}


def write_shard(path, header, payload=b"\x00" * 8):
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class CheckpointInspectorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(json.dumps(config()))
        self.header = {"text.experts.weight": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]},
                       "text.experts.weight_scale_inv": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
        self.shard = self.root / "model-00001.safetensors"
        write_shard(self.shard, self.header)

    def test_inventory_index_scales_and_codec_discovery(self):
        index = {"weight_map": {name: self.shard.name for name in self.header}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        (self.root / "codec_config.json").write_text("{}")
        (self.root / "codec.pth").write_bytes(b"never deserialize")
        result = inspector.inspect_checkpoint(self.root)
        self.assertEqual(result["raw_bytes_total"], 8)
        self.assertEqual(result["raw_bytes_by_component"], {"backbone_experts": 8})
        self.assertEqual(result["contracts"]["linear_attention_layers"], 1)
        self.assertEqual(result["codec_config_candidates"], ["codec_config.json"])
        self.assertEqual(result["uninspected_codec_candidates"], ["codec.pth"])
        self.assertEqual(result["tensors"][0]["quantization_role"], "fp8_weight")
        self.assertEqual(result["tensors"][1]["quantization_role"], "scale_candidate")

    def test_scalar_and_empty_tensor_spans(self):
        write_shard(self.shard, {"scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
                                 "empty": {"dtype": "F32", "shape": [0, 9], "data_offsets": [4, 4]}},
                    b"\0" * 4)
        self.assertEqual(inspector.inspect_checkpoint(self.root)["raw_bytes_total"], 4)

    def test_rejects_bad_shapes_dtypes_offsets_and_holes(self):
        bad_entries = [
            {"shape": [-1]}, {"shape": [True]}, {"shape": "4"},
            {"dtype": "unknown"}, {"data_offsets": [-1, 3]},
            {"data_offsets": [0, 9]}, {"data_offsets": [0, 3]},
            {"data_offsets": [True, 4]}, {"data_offsets": [2, 6]},
        ]
        for update in bad_entries:
            with self.subTest(update=update):
                header = copy.deepcopy(self.header)
                header["text.experts.weight"].update(update)
                write_shard(self.shard, header)
                with self.assertRaises(ValueError):
                    inspector.inspect_safetensors(self.shard)

    def test_rejects_overlap_and_trailing_payload(self):
        header = copy.deepcopy(self.header)
        header["text.experts.weight_scale_inv"]["data_offsets"] = [0, 4]
        write_shard(self.shard, header)
        with self.assertRaises(ValueError):
            inspector.inspect_safetensors(self.shard)
        write_shard(self.shard, self.header, b"\0" * 9)
        with self.assertRaises(ValueError):
            inspector.inspect_safetensors(self.shard)

    def test_rejects_short_oversized_duplicate_and_nonobject_headers(self):
        for data in (b"bad", struct.pack("<Q", inspector.MAX_HEADER_BYTES + 1),
                     struct.pack("<Q", 2) + b"[]",
                     struct.pack("<Q", 13) + b'{"x":1,"x":2}'):
            with self.subTest(data=data):
                self.shard.write_bytes(data)
                with self.assertRaises(ValueError):
                    inspector.inspect_safetensors(self.shard)

    def test_rejects_incomplete_or_escaping_indexes(self):
        for mapping in ({"missing": self.shard.name},
                        {"text.experts.weight": self.shard.name},
                        {"text.experts.weight": "../outside.safetensors"}):
            with self.subTest(mapping=mapping):
                (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
                with self.assertRaises(ValueError):
                    inspector.inspect_checkpoint(self.root)

    def test_rejects_invalid_model_contracts(self):
        for field, value in (("num_experts_per_tok", 5), ("num_hidden_layers", 3),
                             ("num_attention_heads", 3), ("num_experts", True)):
            cfg = config()
            cfg["text_config"][field] = value
            if field == "num_attention_heads":
                cfg["text_config"]["num_key_value_heads"] = 2
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspector.config_contract(cfg)
        cfg = config()
        cfg["model_type"] = "different"
        with self.assertRaises(ValueError):
            inspector.config_contract(cfg)


if __name__ == "__main__":
    unittest.main()
