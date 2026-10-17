#!/usr/bin/env python3
"""Convert Drama3 using standard in-memory CTranslate2 model specifications.

Routed experts retain their original FP8 bytes and block scales by default.
Sensitive and codec tensors stay F32.
"""

import json
import math
import re

from pathlib import Path

import numpy as np

from ctranslate2.converters import drama3_codec as codec
from ctranslate2.converters import drama3_weights as READERS
from ctranslate2.specs.drama3_spec import Drama3Spec
from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES, OPTIONAL

SOURCE_REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
FP8_POLICY = "experts_fp8_sensitive_float32"
STANDARD_POLICY = "standard"
ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
)


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
    if policy not in (FP8_POLICY, STANDARD_POLICY):
        raise ValueError("unknown Drama3 quantization policy: " + repr(policy))
    return policy


def resolve_quantization(policy, quantization=None):
    validate_quantization_policy(policy)
    quantization = quantization or "float32"
    if quantization not in ACCEPTED_MODEL_TYPES:
        raise ValueError("Invalid Drama3 quantization: " + repr(quantization))
    return quantization


def routed_expert_matrix(target):
    return (
        re.fullmatch(
            r"backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight",
            target,
        )
        is not None
    )


def build_plan(tensors, config, quantization_policy=FP8_POLICY):
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
    if block_size != [128, 128] or config["quantization_config"].get(
        "quant_method"
    ) not in ("fp8", "bsgemm-moe"):
        raise ValueError(
            "lossless FP8 requires a supported quant_method and 128x128 blocks"
        )
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
                        (
                            2 * heads * head_dim
                            if kind == "q_proj"
                            else kv_heads * head_dim
                        ),
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
            fp8 = quantization_policy == FP8_POLICY and routed_expert_matrix(target)
            if fp8 and (
                dtype not in ("F8_E4M3", "F8_E4M3FN") or scale["dtype"] != "F32"
            ):
                raise ValueError(
                    "lossless FP8 experts require E4M3FN weights and F32 inverse scales"
                )
            emitted = [target, target + "_scale_inv"] if fp8 else [target]
            if destinations.intersection(emitted):
                raise ValueError("duplicate converted tensor target")
            destinations.update(emitted)
            result.append(
                {
                    "source": source,
                    "shard": tensor["shard"],
                    "target": target,
                    "shape": list(output_shape),
                    "dtype": "fp8" if fp8 else "float32",
                    "inverse_scale": inverse_scale,
                    "scale_shard": (
                        tensors[inverse_scale]["shard"] if inverse_scale else None
                    ),
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


def load_model_tensors(plan, readers, config, chunk_rows=128):
    """Preserve source FP8 experts and load other weights as float32."""
    audio = config["audio_decoder_config"]
    variables = {}
    for index, item in enumerate(plan):
        source = readers[item["shard"]]
        if item["dtype"] == "fp8":
            raw = source._map(item["source"])
            if np.any((raw & 0x7F) == 0x7F):
                raise ValueError("nonfinite E4M3FN source weight: " + item["source"])
            scale = np.array(
                readers[item["scale_shard"]]._map(item["inverse_scale"]), copy=True
            )
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("FP8 inverse scales must be finite and positive")
            variables[item["target"]] = np.array(raw, copy=True).view(np.int8)
            variables[item["target"] + "_scale_inv"] = scale
            continue
        if item["tower_slice"]:
            source = TowerMatrixReader(
                source,
                item["source"],
                item["tower_slice"],
                audio["n_head"],
                audio["n_local_heads"],
                audio["head_dim"],
            )
        if len(item["shape"]) == 2:
            chunks = source.iter_matrix_rows(
                item["source"],
                chunk_rows=chunk_rows,
                inverse_scale=item["inverse_scale"],
                scale_reader=(
                    readers[item["scale_shard"]] if item["scale_shard"] else None
                ),
                block_size=config["quantization_config"]["weight_block_size"],
            )
            chunks = finite_matrix_chunks(chunks)
        else:
            chunks = float_chunks(source, item["source"])
        value = np.empty(item["shape"], dtype=np.float32)
        flat = value.reshape(-1)
        offset = 0
        for chunk in chunks:
            flat[offset : offset + chunk.size] = chunk.reshape(-1)
            offset += chunk.size
        variables[item["target"]] = value
        if (index + 1) % 1000 == 0:
            print(f"Loaded {index + 1}/{len(plan)} model tensors", flush=True)
    return variables


def inspect_checkpoint(
    source,
    chunk_rows=128,
    source_revision=SOURCE_REVISION,
    quantization_policy=FP8_POLICY,
    quantization=None,
):
    quantization = resolve_quantization(quantization_policy, quantization)
    if quantization != "float32":
        quantization_policy = STANDARD_POLICY
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
    codec_plan = codec.conversion_plan(
        codec.state_schema(codec.load_checkpoint(source / "codec.pth"))
    )
    codec_count = len(codec_plan)
    codec_bytes = sum(math.prod(t["shape"]) * 4 for t in codec_plan)
    count = sum(2 if item["dtype"] == "fp8" else 1 for item in plan) + codec_count
    payload_bytes = (
        sum(
            math.prod(item["shape"]) * (1 if item["dtype"] == "fp8" else 4)
            + (
                4 * math.prod(tensors[item["inverse_scale"]]["shape"])
                if item["dtype"] == "fp8"
                else 0
            )
            for item in plan
        )
        + codec_bytes
    )
    float32_bytes = sum(math.prod(item["shape"]) * 4 for item in plan) + codec_bytes
    loading_bytes = payload_bytes
    if quantization_policy == STANDARD_POLICY:
        payload_bytes = codec_bytes
        for item in plan:
            elements = math.prod(item["shape"])
            if len(item["shape"]) != 2:
                payload_bytes += elements * 4
            elif quantization.startswith("int8"):
                payload_bytes += elements + item["shape"][0] * 4
                count += 1
            elif quantization == "int16":
                payload_bytes += elements * 2 + 4
                count += 1
            else:
                payload_bytes += elements * (4 if quantization == "float32" else 2)
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
            for dtype in ("fp8", "float32")
        },
    }
    if quantization_policy == FP8_POLICY:
        result["fp8_format"] = "e4m3fn_block128"
    return config, readers, plan, codec_plan, result


def load_spec(
    model_path,
    quantization=None,
    chunk_rows=128,
    source_revision=SOURCE_REVISION,
    quantization_policy=FP8_POLICY,
):
    """Load Drama3 weights into the standard model specification."""
    quantization = resolve_quantization(quantization_policy, quantization)
    model_path = Path(model_path)
    config, readers, plan, codec_plan, result = inspect_checkpoint(
        model_path, chunk_rows, source_revision, quantization_policy, quantization
    )
    state = codec.load_checkpoint(model_path / "codec.pth")
    codec_variables = codec.load_codec_tensors(
        state, plan=codec_plan, chunk_rows=min(chunk_rows, 32)
    )
    variables = load_model_tensors(plan, readers, config, chunk_rows)
    variables.update(codec_variables)
    quantization_policy = result["quantization_policy"]
    for item in plan:
        if len(item["shape"]) == 2 and item["dtype"] != "fp8":
            variables[item["target"] + "_scale"] = OPTIONAL
    package_config = {
        "model_type": "drama3",
        "ct2_layout_version": 1,
        "source_config": config,
        "source_revision": source_revision,
        "activation_dtype": "float32",
        "matrix_dtype": "fp8_float32" if quantization_policy == FP8_POLICY else quantization,
        "quantization_policy": quantization_policy,
    }
    if quantization_policy == FP8_POLICY:
        package_config["fp8_format"] = "e4m3fn_block128"
    spec = Drama3Spec(variables, package_config)
    spec.register_file(model_path / "config.json", "source_config.json")
    for name in ASSETS:
        spec.register_file(source_file(model_path, name))
    return spec, dict(result, dry_run=False)
