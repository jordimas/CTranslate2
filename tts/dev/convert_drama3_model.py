#!/usr/bin/env python3
"""Bounded official Drama 3 -> CT2 binary-v6 converter (CPU layout version 1).

By default matrix weights become per-row INT8 plus multiplicative CT2 scales; vectors,
depthwise convolutions and the decoder-only codec are float32. Source FP8 scales
are explicit named tensors, independent of destination INT8 scales. Torch is
used only for safe mmap conversion of codec.pth, never inference.
The optional experts_int8_sensitive_float32 diagnostic policy keeps all matrices
except routed experts in float32 directly from the official checkpoint.

Use --dry-run --report plan.json to inspect header mappings without reading
weight payloads. A completed package is published atomically without replacing
an existing destination. Linux renameat2 or Windows no-replace rename is needed.
"""
import argparse
import ctypes
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
TINY_ROW_LIMIT = np.float32(np.float64(127) / np.finfo(np.float32).max)
QUANTIZATION_POLICIES = ("all_matrices_int8", "experts_int8_sensitive_float32")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "chat_template.jinja", "LICENSE", "README.md")


def helper(name, development=False):
    path = ROOT / ("tts/dev" if development else "python/ctranslate2/converters") / (name + ".py")
    spec = importlib.util.spec_from_file_location("ct2_converter_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READERS = helper("drama3_weights")
QUANTIZATION = helper("drama3_quantization")


def unique_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=READERS._unique_object)


def source_file(root, name):
    if not isinstance(name, str) or Path(name).name != name or "\\" in name or name in ("", ".", ".."):
        raise ValueError("checkpoint asset must be a plain filename")
    path = root / name
    if path.resolve().parent != root.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("missing or unsafe checkpoint asset: " + name)
    return path


def inspect_shards(root):
    root = Path(root)
    index = unique_json(source_file(root, "model.safetensors.index.json"))
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("checkpoint index needs a nonempty weight_map")
    readers, tensors = {}, {}
    for name, shard in mapping.items():
        if not isinstance(name, str) or not isinstance(shard, str) or not re.fullmatch(r"model-\d+-of-\d+\.safetensors", shard):
            raise ValueError("invalid checkpoint index name/shard")
        if shard not in readers:
            readers[shard] = READERS.SafeTensorMatrixReader(source_file(root, shard))
    actual_shards = {p.name for p in root.glob("*.safetensors")}
    if actual_shards != set(readers):
        raise ValueError("checkpoint index and actual shard set differ")
    for shard, reader in readers.items():
        for name, tensor in reader.tensors.items():
            if name in tensors or mapping.get(name) != shard:
                raise ValueError("duplicate tensor or shard/index mismatch: " + name)
            tensors[name] = {**tensor, "shard": shard}
    if set(tensors) != set(mapping):
        raise ValueError("checkpoint index names absent from shard headers")
    return readers, tensors


def canonical_name(source):
    if source == "model.language_model.embed_tokens.weight":
        return "backbone/embeddings/weight"
    if source == "model.language_model.norm.weight":
        return "backbone/norm/weight"
    if source == "lm_head.weight":
        return "backbone/lm_head/weight"
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", source)
    if match:
        tail = re.sub(r"^mlp\.gate\.", "mlp.router.", match[2])
        tail = re.sub(r"\.experts\.(\d+)\.", r".experts_\1.", tail)
        return "backbone/layers_" + match[1] + "/" + tail.replace(".", "/")
    if source == "audio_decoder.codebook_embeddings.weight":
        return "tower/codebook_embeddings/weight"
    match = re.fullmatch(r"audio_decoder_mtp(_shared)?\.layers\.(\d+)\.(.+)", source)
    if match:
        tail = match[3].replace("attention.wo.", "attention.o_proj.")
        tail = tail.replace("feed_forward.w1.", "feed_forward.gate_proj.")
        tail = tail.replace("feed_forward.w3.", "feed_forward.up_proj.")
        tail = tail.replace("feed_forward.w2.", "feed_forward.down_proj.")
        kind = "shared_layers_" if match[1] else "residual_layers_"
        return "tower/" + kind + match[2] + "/" + tail.replace(".", "/")
    match = re.fullmatch(r"audio_decoder_mtp\.heads\.(\d+)\.weight", source)
    if match:
        return "tower/heads_" + match[1] + "/weight"
    raise ValueError("unmapped official checkpoint tensor: " + source)


def tower_row_indices(kind, start, stop, heads, kv_heads, head_dim):
    """Fused [Qflat,Gateflat,Kflat,Vflat] -> per-head [Qd,Gated] Q rows."""
    rows = np.arange(start, stop, dtype=np.int64)
    width, kv_width = heads * head_dim, kv_heads * head_dim
    if kind == "q_proj":
        within = rows % (2 * head_dim)
        return (rows // (2 * head_dim)) * head_dim + within % head_dim + (within >= head_dim) * width
    if kind == "k_proj":
        return 2 * width + rows
    if kind == "v_proj":
        return 2 * width + kv_width + rows
    raise ValueError("invalid tower attention projection")


class TowerMatrixReader:
    def __init__(self, reader, source_name, kind, heads, kv_heads, head_dim):
        self.reader, self.source_name, self.kind = reader, source_name, kind
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        original = reader.tensors[source_name]
        expected = 2 * (heads + kv_heads) * head_dim
        if original["dtype"] not in ("F32", "F16", "BF16") or len(original["shape"]) != 2 or original["shape"][0] != expected:
            raise ValueError("unsupported tower fused Q/G/K/V weight layout")
        rows = 2 * heads * head_dim if kind == "q_proj" else kv_heads * head_dim
        if kind not in ("q_proj", "k_proj", "v_proj"):
            raise ValueError("invalid tower matrix slice")
        self.tensors = {source_name: {"shape": (rows, original["shape"][1]), "dtype": original["dtype"]}}

    def iter_matrix_rows(self, name, chunk_rows=128, expert_index=None,
                         inverse_scale=None, scale_reader=None, block_size=(128, 128)):
        if name != self.source_name or type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("invalid tower sliced-matrix request")
        if expert_index is not None or inverse_scale is not None or scale_reader is not None:
            raise ValueError("tower slicing only supports explicitly unquantized source matrices")
        raw = self.reader._map(name)
        rows = self.tensors[name]["shape"][0]
        for start in range(0, rows, chunk_rows):
            indices = tower_row_indices(self.kind, start, min(rows, start + chunk_rows),
                                        self.heads, self.kv_heads, self.head_dim)
            yield READERS._float32(raw[indices], self.tensors[name]["dtype"])


def validate_quantization_policy(policy):
    if not isinstance(policy, str) or policy not in QUANTIZATION_POLICIES:
        raise ValueError("unknown Drama3 quantization policy: " + repr(policy))
    return policy


def routed_expert_matrix(target):
    return re.fullmatch(r"backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight", target) is not None


def build_plan(tensors, config, quantization_policy="all_matrices_int8"):
    validate_quantization_policy(quantization_policy)
    audio = config["audio_decoder_config"]
    heads, kv_heads, head_dim = audio["n_head"], audio["n_local_heads"], audio["head_dim"]
    if any(type(d) is not int or d <= 0 for d in (heads, kv_heads, head_dim)) or heads % kv_heads:
        raise ValueError("invalid tower attention dimensions")
    block_size = config["quantization_config"]["weight_block_size"]
    if not isinstance(block_size, list) or len(block_size) != 2 or any(type(d) is not int or d <= 0 for d in block_size):
        raise ValueError("invalid explicit FP8 block dimensions")
    result, destinations, consumed_scales = [], set(), set()
    for source, tensor in sorted(tensors.items()):
        if source.endswith(".weight_scale_inv"):
            continue
        shape, dtype = tensor["shape"], tensor["dtype"]
        if not shape or any(d <= 0 for d in shape):
            raise ValueError("model tensors must have positive dimensions")
        target = canonical_name(source)
        variants = [(target, None, tuple(shape))]
        if source.endswith(".attention.wqkv.weight") and source.startswith("audio_decoder_mtp"):
            if len(shape) != 2 or shape[0] != 2 * (heads + kv_heads) * head_dim:
                raise ValueError("tower fused projection shape mismatch")
            if dtype not in ("BF16", "F16", "F32"):
                raise ValueError("FP8 fused tower source slicing is unsupported")
            prefix = target[:-len("wqkv/weight")]
            variants = [(prefix + kind + "/weight", kind,
                         (2 * heads * head_dim if kind == "q_proj" else kv_heads * head_dim, shape[1]))
                        for kind in ("q_proj", "k_proj", "v_proj")]
        inverse_scale = None
        if dtype.startswith("F8_"):
            if len(shape) != 2:
                raise ValueError("FP8 source must be an explicitly scaled matrix")
            inverse_scale = source + "_scale_inv"
            if inverse_scale not in tensors:
                raise ValueError("FP8 matrix has no explicit inverse scale: " + source)
            scale = tensors[inverse_scale]
            expected = tuple((size + block - 1) // block for size, block in zip(shape, block_size))
            if scale["dtype"] not in ("F32", "F16", "BF16") or tuple(scale["shape"]) != expected:
                raise ValueError("invalid source FP8 inverse-scale shape/type: " + source)
            consumed_scales.add(inverse_scale)
        elif source + "_scale_inv" in tensors:
            raise ValueError("inverse scale accompanies a non-FP8 matrix")
        for target, kind, output_shape in variants:
            matrix = len(output_shape) == 2
            if matrix and not target.endswith("/weight"):
                raise ValueError("matrix target must use canonical /weight naming")
            if not matrix and (len(output_shape) not in (1, 3) or dtype.startswith("F8_")):
                raise ValueError("unsupported non-matrix model tensor")
            quantized = matrix and (quantization_policy == "all_matrices_int8" or routed_expert_matrix(target))
            emitted = [target, target + "_scale"] if quantized else [target]
            if destinations.intersection(emitted):
                raise ValueError("duplicate converted tensor target")
            destinations.update(emitted)
            result.append({"source": source, "shard": tensor["shard"], "target": target,
                           "shape": list(output_shape), "dtype": "int8" if quantized else "float32",
                           "inverse_scale": inverse_scale,
                           "scale_shard": tensors[inverse_scale]["shard"] if inverse_scale else None,
                           "tower_slice": kind})
    if {name for name in tensors if name.endswith(".weight_scale_inv")} != consumed_scales:
        raise ValueError("orphan source inverse-scale tensors")
    return result


def float_chunks(reader, name, chunk_elements=1048576):
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    tensor = reader.tensors[name]
    if tensor["dtype"].startswith("F8_"):
        raise ValueError("unscaled FP8 raw tensor conversion is forbidden")
    raw = reader._map(name).reshape(-1)
    for start in range(0, len(raw), chunk_elements):
        values = READERS._float32(raw[start:start + chunk_elements], tensor["dtype"])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite source vector/convolution")
        yield np.ascontiguousarray(values, dtype=np.float32)


def finite_matrix_chunks(chunks):
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            raise ValueError("nonfinite source matrix")
        yield np.ascontiguousarray(chunk, dtype=np.float32)


class TinyRowReader:
    """Zero unrepresentably tiny nonzero rows before strict INT8 conversion.

    Some official BF16 GDN rows have magnitude around 7e-38. Their 127/amax
    scales overflow float32. Replacing such rows by zero bounds each changed
    weight by TINY_ROW_LIMIT and avoids invalid dequantization scales. Count only
    the first of the quantizer's two source passes, never keep matrix contents.
    """
    def __init__(self, reader):
        self.reader, self.tensors = reader, reader.tensors
        self.passes = 0
        self.zeroed_rows = 0
        self.max_abs_perturbation = 0.

    def iter_matrix_rows(self, *args, **kwargs):
        self.passes += 1
        count = self.passes == 1
        for chunk in self.reader.iter_matrix_rows(*args, **kwargs):
            maximum = np.max(np.abs(chunk), axis=1)
            mask = (maximum > 0) & (maximum <= TINY_ROW_LIMIT)
            if mask.any():
                if count:
                    self.zeroed_rows += int(mask.sum())
                    self.max_abs_perturbation = max(self.max_abs_perturbation, float(maximum[mask].max()))
                chunk = chunk.copy()
                chunk[mask] = 0
            yield chunk


def write_model_tensors(writer, plan, readers, config, chunk_rows=128):
    audio = config["audio_decoder_config"]
    tiny_rows = []
    for index, item in enumerate(plan):
        source = readers[item["shard"]]
        if item["tower_slice"]:
            source = TowerMatrixReader(source, item["source"], item["tower_slice"],
                                       audio["n_head"], audio["n_local_heads"], audio["head_dim"])
        if item["dtype"] == "int8":
            source = TinyRowReader(source)
            try:
                QUANTIZATION.write_int8_matrix(
                    writer, source, item["source"], item["target"][:-len("/weight")],
                    chunk_rows=chunk_rows, inverse_scale=item["inverse_scale"],
                    scale_reader=readers[item["scale_shard"]] if item["scale_shard"] else None,
                    block_size=config["quantization_config"]["weight_block_size"])
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"{item['source']} -> {item['target']}: {error}") from error
            if source.zeroed_rows:
                tiny_rows.append({"source": item["source"], "target": item["target"],
                                  "zeroed_rows": source.zeroed_rows,
                                  "max_abs_perturbation": source.max_abs_perturbation})
        elif len(item["shape"]) == 2:
            # Read original BF16/F32 or explicitly dequantized official FP8.
            # Never reconstruct sensitive weights from the existing INT8 model.
            chunks = source.iter_matrix_rows(
                item["source"], chunk_rows=chunk_rows, inverse_scale=item["inverse_scale"],
                scale_reader=readers[item["scale_shard"]] if item["scale_shard"] else None,
                block_size=config["quantization_config"]["weight_block_size"])
            writer.write_tensor(item["target"], item["shape"], "float32", finite_matrix_chunks(chunks))
        else:
            writer.write_tensor(item["target"], item["shape"], "float32", float_chunks(source, item["source"]))
        if (index + 1) % 1000 == 0:
            print(f"Converted {index + 1}/{len(plan)} model tensor mappings", flush=True)
    return {"zeroed_rows": sum(item["zeroed_rows"] for item in tiny_rows),
            "threshold_inclusive": float(TINY_ROW_LIMIT),
            "max_abs_perturbation": max((item["max_abs_perturbation"] for item in tiny_rows), default=0.),
            "sources": tiny_rows}


def publish_directory(staging, destination):
    """Atomic no-replace publication, including the existing-empty-dir case."""
    if os.name == "nt":
        os.rename(staging, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("atomic no-replace directory publication requires renameat2 on this platform")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(destination))


def convert(source, output, report=None, dry_run=False, chunk_rows=128,
            source_revision=SOURCE_REVISION, quantization_policy="all_matrices_int8"):
    validate_quantization_policy(quantization_policy)
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    source, output = Path(source), Path(output)
    config = unique_json(source_file(source, "config.json"))
    if config.get("model_type") != "fish_omni":
        raise ValueError("expected official Fish Omni checkpoint")
    for name in ASSETS:
        source_file(source, name)
    source_file(source, "codec.pth")
    readers, tensors = inspect_shards(source)
    plan = build_plan(tensors, config, quantization_policy)
    codec = helper("convert_drama3_codec", development=True)
    codec_manifest = codec.load_manifest()
    codec_count = sum(t["required_for_inference"] for t in codec_manifest["tensors"])
    count = sum(2 if item["dtype"] == "int8" else 1 for item in plan) + codec_count
    payload_bytes = sum(math.prod(item["shape"]) * (1 if item["dtype"] == "int8" else 4)
                        + (4 * item["shape"][0] if item["dtype"] == "int8" else 0) for item in plan)
    payload_bytes += sum(math.prod(t["expected_logical_shape"]) * 4
                         for t in codec_manifest["tensors"] if t["required_for_inference"])
    result = {"model_name": "Drama3Spec", "ct2_layout_version": 1,
              "source_revision": source_revision, "source_shards": len(readers),
              "source_tensor_count": len(tensors), "model_mapping_count": len(plan),
              "codec_parameter_count": codec_count, "variable_count": count,
              "estimated_payload_bytes": payload_bytes, "dry_run": dry_run, "mappings": plan,
              "quantization_policy": quantization_policy,
              "matrix_counts": {dtype: sum(len(item["shape"]) == 2 and item["dtype"] == dtype for item in plan)
                                for dtype in ("int8", "float32")}}
    if dry_run:
        if report:
            Path(report).write_text(json.dumps(result, indent=2) + "\n")
        return result
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        state = codec.load_checkpoint(source / "codec.pth")
        codec.conversion_plan(codec.state_schema(state), codec_manifest)
        Writer = helper("drama3_writer").StreamingModelWriter
        with Writer(staging / "model.bin", "Drama3Spec", 1, count) as writer:
            result["tiny_row_zeroing"] = write_model_tensors(writer, plan, readers, config, chunk_rows)
            codec.write_codec_tensors(state, writer, codec_manifest, chunk_rows=min(chunk_rows, 32))
            writer.finish()
        package_config = {"model_type": "drama3", "ct2_layout_version": 1,
                          "source_config": config, "source_revision": source_revision,
                          "activation_dtype": "float32",
                          "matrix_dtype": "int8" if quantization_policy == "all_matrices_int8" else "mixed",
                          "quantization_policy": quantization_policy}
        (staging / "config.json").write_text(json.dumps(package_config, indent=2) + "\n")
        shutil.copy2(source / "config.json", staging / "source_config.json")
        for name in ASSETS:
            shutil.copy2(source_file(source, name), staging / name)
        result.update(output=str(output), output_bytes=(staging / "model.bin").stat().st_size)
        (staging / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
        publish_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if report:
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--quantization-policy", choices=QUANTIZATION_POLICIES,
                        default="all_matrices_int8",
                        help="optional sensitivity diagnostic: preserve non-routed matrices from original source as float32")
    args = parser.parse_args()
    try:
        result = convert(args.source, args.output, args.report, args.dry_run,
                         args.chunk_rows, args.source_revision, args.quantization_policy)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Drama 3 conversion failed: {error}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "mappings"}, indent=2))


if __name__ == "__main__":
    main()
