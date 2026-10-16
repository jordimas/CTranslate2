#!/usr/bin/env python3
"""Inventory an official local Drama 3 checkpoint without reading tensor payloads."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct


MAX_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "F8_E4M3FN": 1, "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def inspect_safetensors(path):
    """Read only the length prefix and bounded header; validate against file size."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: missing safetensors length prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
            raise ValueError(f"{path}: invalid or oversized safetensors header")
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError(f"{path}: metadata must contain string values")
    data_start = 8 + length
    data_size = size - data_start
    tensors = {}
    spans = []
    for name, item in header.items():
        if not isinstance(item, dict):
            raise ValueError(f"{path}: malformed tensor {name}")
        dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"{path}: unsupported dtype for {name}: {dtype}")
        if not isinstance(shape, list) or any(not integer(d) or d < 0 for d in shape):
            raise ValueError(f"{path}: invalid shape for {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not integer(d) for d in offsets)):
            raise ValueError(f"{path}: invalid offsets for {name}")
        start, end = offsets
        elements = 1
        for dim in shape:
            elements *= dim
        if not 0 <= start <= end <= data_size or end - start != elements * DTYPE_BYTES[dtype]:
            raise ValueError(f"{path}: invalid byte span for {name}")
        spans.append((start, end, name))
        tensors[name] = {
            "shape": shape, "dtype": dtype, "data_offsets": offsets,
            "absolute_offsets": [data_start + start, data_start + end],
            "raw_bytes": end - start,
        }
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError(f"{path}: overlapping or noncontiguous span for {name}")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"{path}: tensor spans do not cover data section")
    return tensors


def config_contract(config):
    if not isinstance(config, dict) or config.get("model_type") != "fish_omni":
        raise ValueError("config.json must describe model_type fish_omni")
    text = config.get("text_config", {})
    audio = config.get("audio_decoder_config", {})
    for block, fields in ((text, ("hidden_size", "num_hidden_layers", "num_experts",
                                  "num_experts_per_tok", "num_attention_heads",
                                  "num_key_value_heads", "head_dim", "moe_intermediate_size")),
                          (audio, ("dim", "n_layer", "num_codebooks", "shared_enrichment_layers"))):
        if not isinstance(block, dict):
            raise ValueError("missing model subconfiguration")
        for field in fields:
            if not integer(block.get(field)) or block[field] <= 0:
                raise ValueError(f"missing/invalid configuration field: {field}")
    count = text["num_hidden_layers"]
    types = text.get("layer_types")
    if (not isinstance(types, list) or len(types) != count
            or any(t not in ("linear_attention", "full_attention") for t in types)):
        raise ValueError("layer_types must enumerate every backbone layer")
    if text.get("mlp_layer_types") != ["moe"] * count:
        raise ValueError("expected a MoE MLP for every backbone layer")
    if text["num_experts_per_tok"] > text["num_experts"]:
        raise ValueError("router top-k exceeds expert count")
    if text["num_attention_heads"] % text["num_key_value_heads"]:
        raise ValueError("attention heads must be divisible by KV heads")
    if audio.get("decoder_type") != "codebook_mtp_shared":
        raise ValueError("unsupported audio decoder_type")
    return {
        "backbone_layers": count,
        "full_attention_layers": types.count("full_attention"),
        "linear_attention_layers": types.count("linear_attention"),
        "routed_experts_per_layer": text["num_experts"],
        "routed_experts_per_token": text["num_experts_per_tok"],
        "shared_enrichment_layers": audio["shared_enrichment_layers"],
        "declared_audio_n_layer": audio["n_layer"],
        "num_codebooks": audio["num_codebooks"],
        "warning": "Configuration contracts do not prove tensor completeness or numerical parity.",
    }


def component(name, relative_file):
    value = (relative_file + "/" + name).lower()
    if any(key in value for key in ("codec", "dac", "vocoder")):
        return "codec"
    if "audio_decoder" in value or "audio_tower" in value:
        return "audio_tower"
    if "expert" in value:
        return "backbone_experts"
    return "backbone_or_other"


def inspect_checkpoint(directory):
    root = Path(directory).resolve()
    contracts = config_contract(read_json(root / "config.json"))
    records, totals, files = [], defaultdict(int), {}
    for path in sorted(root.rglob("*.safetensors")):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"checkpoint shard escapes root: {path}")
        relative = path.relative_to(root).as_posix()
        tensors = inspect_safetensors(path)
        files[relative] = tensors
        for name, record in tensors.items():
            group = component(name, relative)
            is_scale = "scale" in name.lower()
            totals[group] += record["raw_bytes"]
            records.append(dict(record, name=name, file=relative, component=group,
                                quantization_role="scale_candidate" if is_scale else
                                ("fp8_weight" if record["dtype"].startswith("F8_") else "other")))
    if not files:
        raise ValueError("no safetensors files found; no payloads were loaded")
    indexes = []
    for path in sorted(root.rglob("*.safetensors.index.json")):
        index = read_json(path)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{path}: missing/nonempty weight_map")
        expected = defaultdict(set)
        for name, filename in mapping.items():
            if not isinstance(filename, str):
                raise ValueError(f"{path}: non-string shard path")
            target = (path.parent / filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"{path}: shard path escapes checkpoint")
            relative = target.relative_to(root).as_posix()
            if relative not in files or name not in files[relative]:
                raise ValueError(f"{path}: indexed tensor {name} missing from {filename}")
            expected[relative].add(name)
        for filename, names in expected.items():
            if names != set(files[filename]):
                raise ValueError(f"{path}: shard contains tensors absent from index: {filename}")
        indexes.append(path.relative_to(root).as_posix())
    fp8_names = {r["name"] for r in records if r["quantization_role"] == "fp8_weight"}
    scale_groups = []
    for record in records:
        if record["quantization_role"] != "scale_candidate":
            continue
        name = record["name"]
        candidates = {name.removesuffix(suffix) for suffix in
                      ("_scale_inv", "_scale", ".scale_inv", ".scale") if name.endswith(suffix)}
        # Record only literal name matches; do not guess block broadcasting semantics.
        scale_groups.append({"scale": name, "file": record["file"],
                             "fp8_weight_candidates": sorted(candidates & fp8_names)})
    codec_configs = [p.relative_to(root).as_posix() for p in sorted(root.rglob("*.json"))
                     if any(k in p.relative_to(root).as_posix().lower() for k in ("codec", "dac", "vocoder"))]
    other_codec_files = [p.relative_to(root).as_posix() for p in sorted(root.rglob("*"))
                         if p.is_file() and p.suffix in (".pth", ".pt", ".bin", ".yaml", ".yml")
                         and any(k in p.name.lower() for k in ("codec", "dac", "vocoder"))]
    return {"schema_version": 1, "checkpoint": str(root), "contracts": contracts,
            "indexes_validated": indexes, "raw_bytes_by_component": dict(totals),
            "raw_bytes_total": sum(totals.values()), "tensors": records,
            "fp8_scale_groups": scale_groups,
            "codec_config_candidates": codec_configs,
            "uninspected_codec_candidates": other_codec_files,
            "limitations": ["Component and scale classifications are name-based candidates.",
                            "Pickle/PyTorch codec artifacts are listed but never loaded.",
                            "No tensor payload was read; no numerical or synthesis validation performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f"Checkpoint inspection failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
