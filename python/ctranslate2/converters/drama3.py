#!/usr/bin/env python3
"""Convert Drama3 using standard in-memory CTranslate2 model specifications.

Source weights are decoded in F32; quantized routed experts are retained in
BF16 until standard optimization. Sensitive tensors and codec weights stay F32.
"""
import ctypes
import errno
import importlib
import json
import math
import os
import re
import shutil
import tempfile

from pathlib import Path

import numpy as np

from ctranslate2.specs import model_spec
from ctranslate2.specs.drama3_spec import Drama3Spec

SOURCE_REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
TINY_ROW_LIMIT = np.float32(np.float64(127) / np.finfo(np.float32).max)
QUANTIZATION_POLICIES = ("all_matrices_int8", "experts_int8_sensitive_float32")
ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
)


def helper(name, development=False):
    # Keep the development helper names available to existing diagnostics.
    if name == "convert_drama3_codec":
        name = "drama3_codec"
    return importlib.import_module("ctranslate2.converters." + name)


READERS = helper("drama3_weights")


def unique_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=READERS._unique_object
    )


def source_file(root, name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or "\\" in name
        or name in ("", ".", "..")
    ):
        raise ValueError("checkpoint asset must be a plain filename")
    path = root / name
    if (
        path.resolve().parent != root.resolve()
        or not path.is_file()
        or path.is_symlink()
    ):
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
        if (
            not isinstance(name, str)
            or not isinstance(shard, str)
            or not re.fullmatch(r"model-\d+-of-\d+\.safetensors", shard)
        ):
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
        return (
            (rows // (2 * head_dim)) * head_dim
            + within % head_dim
            + (within >= head_dim) * width
        )
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
        if (
            original["dtype"] not in ("F32", "F16", "BF16")
            or len(original["shape"]) != 2
            or original["shape"][0] != expected
        ):
            raise ValueError("unsupported tower fused Q/G/K/V weight layout")
        rows = 2 * heads * head_dim if kind == "q_proj" else kv_heads * head_dim
        if kind not in ("q_proj", "k_proj", "v_proj"):
            raise ValueError("invalid tower matrix slice")
        self.tensors = {
            source_name: {
                "shape": (rows, original["shape"][1]),
                "dtype": original["dtype"],
            }
        }

    def iter_matrix_rows(
        self,
        name,
        chunk_rows=128,
        expert_index=None,
        inverse_scale=None,
        scale_reader=None,
        block_size=(128, 128),
    ):
        if name != self.source_name or type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("invalid tower sliced-matrix request")
        if (
            expert_index is not None
            or inverse_scale is not None
            or scale_reader is not None
        ):
            raise ValueError(
                "tower slicing only supports explicitly unquantized source matrices"
            )
        raw = self.reader._map(name)
        rows = self.tensors[name]["shape"][0]
        for start in range(0, rows, chunk_rows):
            indices = tower_row_indices(
                self.kind,
                start,
                min(rows, start + chunk_rows),
                self.heads,
                self.kv_heads,
                self.head_dim,
            )
            yield READERS._float32(raw[indices], self.tensors[name]["dtype"])


def validate_quantization_policy(policy):
    if not isinstance(policy, str) or policy not in QUANTIZATION_POLICIES:
        raise ValueError("unknown Drama3 quantization policy: " + repr(policy))
    return policy


def routed_expert_matrix(target):
    return (
        re.fullmatch(
            r"backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight",
            target,
        )
        is not None
    )


def build_plan(tensors, config, quantization_policy="experts_int8_sensitive_float32"):
    validate_quantization_policy(quantization_policy)
    audio = config["audio_decoder_config"]
    heads, kv_heads, head_dim = (
        audio["n_head"],
        audio["n_local_heads"],
        audio["head_dim"],
    )
    if (
        any(type(d) is not int or d <= 0 for d in (heads, kv_heads, head_dim))
        or heads % kv_heads
    ):
        raise ValueError("invalid tower attention dimensions")
    block_size = config["quantization_config"]["weight_block_size"]
    if (
        not isinstance(block_size, list)
        or len(block_size) != 2
        or any(type(d) is not int or d <= 0 for d in block_size)
    ):
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
        if source.endswith(".attention.wqkv.weight") and source.startswith(
            "audio_decoder_mtp"
        ):
            if len(shape) != 2 or shape[0] != 2 * (heads + kv_heads) * head_dim:
                raise ValueError("tower fused projection shape mismatch")
            if dtype not in ("BF16", "F16", "F32"):
                raise ValueError("FP8 fused tower source slicing is unsupported")
            prefix = target[: -len("wqkv/weight")]
            variants = [
                (
                    prefix + kind + "/weight",
                    kind,
                    (
                        2 * heads * head_dim
                        if kind == "q_proj"
                        else kv_heads * head_dim,
                        shape[1],
                    ),
                )
                for kind in ("q_proj", "k_proj", "v_proj")
            ]
        inverse_scale = None
        if dtype.startswith("F8_"):
            if len(shape) != 2:
                raise ValueError("FP8 source must be an explicitly scaled matrix")
            inverse_scale = source + "_scale_inv"
            if inverse_scale not in tensors:
                raise ValueError("FP8 matrix has no explicit inverse scale: " + source)
            scale = tensors[inverse_scale]
            expected = tuple(
                (size + block - 1) // block for size, block in zip(shape, block_size)
            )
            if (
                scale["dtype"] not in ("F32", "F16", "BF16")
                or tuple(scale["shape"]) != expected
            ):
                raise ValueError(
                    "invalid source FP8 inverse-scale shape/type: " + source
                )
            consumed_scales.add(inverse_scale)
        elif source + "_scale_inv" in tensors:
            raise ValueError("inverse scale accompanies a non-FP8 matrix")
        for target, kind, output_shape in variants:
            matrix = len(output_shape) == 2
            if matrix and not target.endswith("/weight"):
                raise ValueError("matrix target must use canonical /weight naming")
            if not matrix and (
                len(output_shape) not in (1, 3) or dtype.startswith("F8_")
            ):
                raise ValueError("unsupported non-matrix model tensor")
            quantized = matrix and (
                quantization_policy == "all_matrices_int8"
                or routed_expert_matrix(target)
            )
            emitted = [target, target + "_scale"] if quantized else [target]
            if destinations.intersection(emitted):
                raise ValueError("duplicate converted tensor target")
            destinations.update(emitted)
            result.append(
                {
                    "source": source,
                    "shard": tensor["shard"],
                    "target": target,
                    "shape": list(output_shape),
                    "dtype": "int8" if quantized else "float32",
                    "inverse_scale": inverse_scale,
                    "scale_shard": tensors[inverse_scale]["shard"]
                    if inverse_scale
                    else None,
                    "tower_slice": kind,
                }
            )
    if {
        name for name in tensors if name.endswith(".weight_scale_inv")
    } != consumed_scales:
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
        values = READERS._float32(raw[start : start + chunk_elements], tensor["dtype"])
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
        self.max_abs_perturbation = 0.0

    def iter_matrix_rows(self, *args, **kwargs):
        self.passes += 1
        count = self.passes == 1
        for chunk in self.reader.iter_matrix_rows(*args, **kwargs):
            maximum = np.max(np.abs(chunk), axis=1)
            mask = (maximum > 0) & (maximum <= TINY_ROW_LIMIT)
            if mask.any():
                if count:
                    self.zeroed_rows += int(mask.sum())
                    self.max_abs_perturbation = max(
                        self.max_abs_perturbation, float(maximum[mask].max())
                    )
                chunk = chunk.copy()
                chunk[mask] = 0
            yield chunk


def load_model_tensors(plan, readers, config, chunk_rows=128):
    """Stage quantized experts in BF16, keeping other parameters in F32."""
    import torch

    audio = config["audio_decoder_config"]
    variables, tiny_rows = {}, []
    for index, item in enumerate(plan):
        source = readers[item["shard"]]
        if item["tower_slice"]:
            source = TowerMatrixReader(
                source,
                item["source"],
                item["tower_slice"],
                audio["n_head"],
                audio["n_local_heads"],
                audio["head_dim"],
            )
        if item["dtype"] == "int8":
            source = TinyRowReader(source)
            variables[item["target"] + "_scale"] = model_spec.OPTIONAL
        if len(item["shape"]) == 2:
            chunks = source.iter_matrix_rows(
                item["source"],
                chunk_rows=chunk_rows,
                inverse_scale=item["inverse_scale"],
                scale_reader=readers[item["scale_shard"]]
                if item["scale_shard"]
                else None,
                block_size=config["quantization_config"]["weight_block_size"],
            )
            chunks = finite_matrix_chunks(chunks)
        else:
            chunks = float_chunks(source, item["source"])
        bf16 = item["dtype"] == "int8" and "/mlp/experts_" in item["target"]
        value = (
            torch.empty(item["shape"], dtype=torch.bfloat16)
            if bf16
            else np.empty(item["shape"], dtype=np.float32)
        )
        flat = value.reshape(-1)
        offset = 0
        for chunk in chunks:
            if bf16:
                flat[offset : offset + chunk.size].copy_(
                    torch.from_numpy(chunk.reshape(-1))
                )
            else:
                flat[offset : offset + chunk.size] = chunk.reshape(-1)
            offset += chunk.size
        variables[item["target"]] = value
        if item["dtype"] == "int8" and source.zeroed_rows:
            tiny_rows.append(
                {
                    "source": item["source"],
                    "target": item["target"],
                    "zeroed_rows": source.zeroed_rows,
                    "max_abs_perturbation": source.max_abs_perturbation,
                }
            )
        if (index + 1) % 1000 == 0:
            print(f"Loaded {index + 1}/{len(plan)} model tensors", flush=True)
    return variables, {
        "zeroed_rows": sum(item["zeroed_rows"] for item in tiny_rows),
        "threshold_inclusive": float(TINY_ROW_LIMIT),
        "max_abs_perturbation": max(
            (item["max_abs_perturbation"] for item in tiny_rows), default=0.0
        ),
        "sources": tiny_rows,
    }


def available_memory():
    """Linux preflight; other platforms rely on their normal allocation errors."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def publish_directory(staging, destination):
    """Atomic no-replace publication, including the existing-empty-dir case."""
    if os.name == "nt":
        os.rename(staging, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError(
            "atomic no-replace directory publication requires renameat2 on this platform"
        )
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(destination))


def inspect_checkpoint(
    source,
    chunk_rows=128,
    source_revision=SOURCE_REVISION,
    quantization_policy="experts_int8_sensitive_float32",
    quantization="int8_float32",
):
    validate_quantization_policy(quantization_policy)
    if quantization not in ("int8_float32", "float32"):
        raise ValueError("quantization must be int8_float32 or float32")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    source = Path(source)
    config = unique_json(source_file(source, "config.json"))
    if config.get("model_type") != "fish_omni":
        raise ValueError("expected official Fish Omni checkpoint")
    for name in ASSETS:
        source_file(source, name)
    source_file(source, "codec.pth")
    readers, tensors = inspect_shards(source)
    plan = build_plan(tensors, config, quantization_policy)
    if quantization == "float32":
        for item in plan:
            item["dtype"] = "float32"
    codec = helper("convert_drama3_codec", development=True)
    codec_manifest = codec.load_manifest()
    codec_count = sum(t["required_for_inference"] for t in codec_manifest["tensors"])
    count = sum(2 if item["dtype"] == "int8" else 1 for item in plan) + codec_count
    payload_bytes = sum(
        math.prod(item["shape"]) * (1 if item["dtype"] == "int8" else 4)
        + (4 * item["shape"][0] if item["dtype"] == "int8" else 0)
        for item in plan
    )
    payload_bytes += sum(
        math.prod(t["expected_logical_shape"]) * 4
        for t in codec_manifest["tensors"]
        if t["required_for_inference"]
    )
    float32_bytes = sum(math.prod(item["shape"]) * 4 for item in plan)
    float32_bytes += sum(
        math.prod(t["expected_logical_shape"]) * 4
        for t in codec_manifest["tensors"]
        if t["required_for_inference"]
    )
    loading_bytes = float32_bytes - sum(
        math.prod(item["shape"]) * 2
        for item in plan
        if item["dtype"] == "int8" and "/mlp/experts_" in item["target"]
    )
    result = {
        "estimated_float32_loading_bytes": float32_bytes,
        "estimated_loading_bytes": loading_bytes,
        "quantization": quantization,
        "model_name": "Drama3Spec",
        "ct2_layout_version": 1,
        "source_revision": source_revision,
        "source_shards": len(readers),
        "source_tensor_count": len(tensors),
        "model_mapping_count": len(plan),
        "codec_parameter_count": codec_count,
        "variable_count": count,
        "estimated_payload_bytes": payload_bytes,
        "dry_run": True,
        "mappings": plan,
        "quantization_policy": quantization_policy,
        "matrix_counts": {
            dtype: sum(
                len(item["shape"]) == 2 and item["dtype"] == dtype for item in plan
            )
            for dtype in ("int8", "float32")
        },
    }
    return config, readers, plan, codec, codec_manifest, result


def load_spec(
    model_path,
    quantization="int8_float32",
    chunk_rows=128,
    source_revision=SOURCE_REVISION,
    quantization_policy="experts_int8_sensitive_float32",
):
    """Load Drama3 weights into the standard model specification."""
    model_path = Path(model_path)
    config, readers, plan, codec, manifest, result = inspect_checkpoint(
        model_path, chunk_rows, source_revision, quantization_policy, quantization
    )
    state = codec.load_checkpoint(model_path / "codec.pth")
    codec.conversion_plan(codec.state_schema(state), manifest)
    variables, result["tiny_row_zeroing"] = load_model_tensors(
        plan, readers, config, chunk_rows
    )
    variables.update(
        codec.load_codec_tensors(state, manifest, chunk_rows=min(chunk_rows, 32))
    )
    package_config = {
        "model_type": "drama3",
        "ct2_layout_version": 1,
        "source_config": config,
        "source_revision": source_revision,
        "activation_dtype": "float32",
        "matrix_dtype": (
            "float32"
            if quantization == "float32"
            else ("int8" if quantization_policy == "all_matrices_int8" else "mixed")
        ),
        "quantization_policy": quantization_policy,
    }
    spec = Drama3Spec(variables, package_config)
    spec.register_file(model_path / "config.json", "source_config.json")
    for name in ASSETS:
        spec.register_file(source_file(model_path, name))
    return spec, dict(result, dry_run=False)


def convert_model(
    converter, standard_convert, output_dir, vmap=None, quantization=None, force=False
):
    """Publish a complete Drama3 package using the base Converter conversion flow."""
    # Use Converter for validation, optimization and saving; publish only a
    # complete package so failed conversions leave no partial destination.
    if vmap is not None:
        raise ValueError("Drama 3 does not support vocabulary mapping")
    quantization = quantization or "int8_float32"
    if quantization not in ("int8_float32", "float32"):
        raise ValueError("quantization must be int8_float32 or float32")
    converter._drama3_quantization = quantization
    output = Path(output_dir)
    if os.path.lexists(output) and not force:
        raise FileExistsError(f"{output} already exists; use --force to overwrite")
    if Path(converter._model_name_or_path).resolve().is_relative_to(output.resolve()):
        raise ValueError("output directory must not contain the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        standard_convert(str(staging), quantization=quantization, force=True)
        converter.conversion_report.update(
            output=str(output), output_bytes=(staging / "model.bin").stat().st_size
        )
        (staging / "conversion.json").write_text(
            json.dumps(converter.conversion_report, indent=2) + "\n"
        )
        if force and os.path.lexists(output):
            if output.is_symlink() or not output.is_dir():
                raise ValueError("--force requires an existing regular directory")
            shutil.rmtree(output)
        publish_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return str(output_dir)


def convert(
    source,
    output,
    report=None,
    dry_run=False,
    chunk_rows=128,
    source_revision=SOURCE_REVISION,
    quantization_policy="experts_int8_sensitive_float32",
    quantization="int8_float32",
):
    """Compatibility interface for development diagnostics."""
    from ctranslate2.converters.transformers import TransformersConverter

    validate_quantization_policy(quantization_policy)
    converter = TransformersConverter(str(source))
    converter._drama3_options = dict(
        quantization_policy=quantization_policy,
        chunk_rows=chunk_rows,
        source_revision=source_revision,
    )
    if dry_run:
        result = converter.inspect(quantization)
    else:
        converter.convert(str(output), quantization=quantization)
        result = converter.conversion_report
    if report:
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
    return result
