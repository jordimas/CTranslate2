#!/usr/bin/env python3
"""Convert official decoder-only weights into a CT2-v6 variable fixture.

PyTorch is a conversion-only optional dependency. Safe loading requires
weights_only=True and mmap=True; there is no unsafe pickle fallback. Weights are
folded in row chunks and retained as FLOAT32 arrays in memory. This utility supplies tensor
conversion, not a runnable model graph. audio.cpp and GGML are never imported.

Mapping: strip codec_weights/, replace dots by slashes, prepend --scope (codec).
Example: decoder.model.0.conv.weight -> codec/decoder/model/0/conv/weight.
load_codec_tensors supplies arrays for standard ModelSpec conversion.
"""
import argparse
import json
import math

from collections import Counter
from pathlib import Path

import numpy as np

from ctranslate2.specs.drama3_spec import Drama3Spec

DEFAULT_MANIFEST = Path(__file__).with_name("drama3_codec_manifest.json")
FLOAT_TYPES = {
    "float32",
    "float16",
    "float64",
    "torch.float32",
    "torch.float16",
    "torch.float64",
    "torch.bfloat16",
}


def load_manifest(path=DEFAULT_MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conversion_plan(schema, manifest, scope="codec"):
    """Resolve direct/legacy/parametrized tensors using only names and shapes."""
    keys = schema.get("keys", schema)
    if not isinstance(scope, str) or not scope or any(c in scope for c in ("\0", ".")):
        raise ValueError("scope must be a nonempty CT2 slash-separated name")
    plan, targets = [], set()
    for tensor in manifest["tensors"]:
        if not tensor["required_for_inference"]:
            continue
        source = tensor["source_name"]
        if not source.startswith("codec_weights/"):
            raise ValueError("manifest tensor is outside codec_weights/")
        source = source[len("codec_weights/") :]
        shape = tensor["expected_logical_shape"]
        if not shape or any(
            type(d) is not int or d <= 0 or d > 2**32 - 1 for d in shape
        ):
            raise ValueError("invalid manifest shape: " + source)
        if math.prod(shape) * 4 > 2**32 - 1:
            raise ValueError("codec tensor exceeds CT2-v6 byte limit")
        candidates = []
        if source in keys:
            candidates.append(("direct", source, None))
        if source.endswith(".weight"):
            stem = source[: -len(".weight")]
            for style, g, v in (
                (
                    "parametrized_axis0",
                    stem + ".parametrizations.weight.original0",
                    stem + ".parametrizations.weight.original1",
                ),
                ("legacy_axis0", stem + ".weight_g", stem + ".weight_v"),
            ):
                if g in keys or v in keys:
                    if g not in keys or v not in keys:
                        raise ValueError(
                            "incomplete weight-normalization pair: " + source
                        )
                    candidates.append((style, v, g))
        if len(candidates) != 1:
            raise ValueError("missing or ambiguous official tensor: " + source)
        style, value_key, magnitude_key = candidates[0]
        if list(keys[value_key]["shape"]) != shape:
            raise ValueError("official shape differs from codec manifest: " + source)
        selected = [value_key]
        if magnitude_key:
            if list(keys[magnitude_key]["shape"]) != [shape[0]] + [1] * (
                len(shape) - 1
            ):
                raise ValueError(
                    "weight-normalization magnitude must retain axis0: " + source
                )
            selected.append(magnitude_key)
        if any(keys[key]["dtype"] not in FLOAT_TYPES for key in selected):
            raise ValueError("non-floating codec tensor: " + source)
        target = scope.rstrip("/") + "/" + source.replace(".", "/")
        if target in targets:
            raise ValueError("duplicate codec target: " + target)
        targets.add(target)
        plan.append(
            {
                "source_name": source,
                "target": target,
                "shape": shape,
                "style": style,
                "value_key": value_key,
                "magnitude_key": magnitude_key,
            }
        )
    return plan


def audit_schema(schema, manifest, scope="codec"):
    plan = conversion_plan(schema, manifest, scope)
    return {
        "format": "ct2-drama3-official-codec-audit-v1",
        "matched": True,
        "checkpoint_revision": schema.get("revision"),
        "parameter_count": len(plan),
        "style_counts": dict(Counter(p["style"] for p in plan)),
        "output_dtype": "float32",
        "discrepancies": [],
        "tensors": plan,
    }


def numpy_float_chunk(value):
    if isinstance(value, np.ndarray):
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(value, dtype=np.float32, order="C")
    else:
        # Only the supplied slice is converted, never a whole checkpoint bank.
        result = (
            value.detach()
            .to(device="cpu", dtype=__import__("torch").float32)
            .contiguous()
            .numpy()
        )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite codec weight or float32 conversion overflow")
    return result


def fold_axis0_chunk(direction, magnitude):
    """Independent axis-0 WN: each first-axis row has one magnitude."""
    direction = numpy_float_chunk(direction)
    magnitude = numpy_float_chunk(magnitude)
    if direction.ndim < 2 or magnitude.shape != (direction.shape[0],) + (1,) * (
        direction.ndim - 1
    ):
        raise ValueError("incompatible axis0 weight-normalization shapes")
    axes = tuple(range(1, direction.ndim))
    # Float64 norm avoids squaring overflow; product is checked again in F32.
    norm = np.sqrt(np.sum(direction.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.isfinite(norm).all() or (norm == 0).any():
        raise ValueError("zero or nonfinite weight-normalization norm")
    folded = np.asarray(
        direction * (magnitude.astype(np.float64) / norm), dtype=np.float32, order="C"
    )
    if not np.isfinite(folded).all():
        raise ValueError("weight-normalization output overflows float32")
    return folded


def tensor_chunks(state, item, chunk_rows):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    value = state[item["value_key"]]
    magnitude = state[item["magnitude_key"]] if item["magnitude_key"] else None
    for start in range(0, item["shape"][0], chunk_rows):
        end = min(start + chunk_rows, item["shape"][0])
        if magnitude is None:
            yield numpy_float_chunk(value[start:end])
        else:
            yield fold_axis0_chunk(value[start:end], magnitude[start:end])


def state_schema(state):
    return {
        "keys": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        }
    }


def load_codec_tensors(state, manifest=None, scope="codec", chunk_rows=32):
    """Load decoder weights as whole FLOAT32 arrays for standard ModelSpec."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    manifest = load_manifest() if manifest is None else manifest
    plan = conversion_plan(state_schema(state), manifest, scope)
    tensors = {}
    for item in plan:
        value = np.empty(item["shape"], dtype=np.float32)
        start = 0
        for chunk in tensor_chunks(state, item, chunk_rows):
            value[start : start + len(chunk)] = chunk
            start += len(chunk)
        tensors[item["target"]] = value
    return tensors


def load_checkpoint(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("expected a flat tensor-only codec state dictionary")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--schema", type=Path, help="Header-only schema audit; no checkpoint load"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output", type=Path, help="New CT2-v6 model.bin path; requires --checkpoint"
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--scope", default="codec")
    parser.add_argument("--model-name", default="Drama3CodecFixture")
    parser.add_argument("--chunk-rows", type=int, default=32)
    args = parser.parse_args()
    if (
        (args.checkpoint is None) == (args.schema is None)
        or args.output
        and not args.checkpoint
    ):
        parser.error(
            "choose --checkpoint or --schema; writing weights requires --checkpoint"
        )
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    try:
        manifest = load_manifest(args.manifest)
        state = load_checkpoint(args.checkpoint) if args.checkpoint else None
        schema = (
            state_schema(state)
            if state is not None
            else json.loads(args.schema.read_text())
        )
        result = audit_schema(schema, manifest, args.scope)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model = Drama3Spec(
                load_codec_tensors(state, manifest, args.scope, args.chunk_rows),
                model_name=args.model_name,
            )
            model.validate()
            model.optimize(quantization="float32")
            model._serialize(str(args.output))
            result.update(
                output=str(args.output),
                model_name=args.model_name,
                output_bytes=args.output.stat().st_size,
                limitation="Variable fixture only; native codec graph still required",
            )
        args.audit_output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Codec conversion failed: {error}\n")
    print(f"Validated {result['parameter_count']} decoder tensors")


if __name__ == "__main__":
    main()
