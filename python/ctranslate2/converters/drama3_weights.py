"""Bounded-memory safetensors matrix reads for an independent Drama 3 converter.

Rank-two matrices and explicitly selected rank-three expert matrices are supported.
No tensor-name conventions or automatic quantization-scale matching are assumed.
Files must remain immutable throughout reading. No PyTorch or native CT2 import
is needed by this module.
"""

import json
import math
from pathlib import Path
import struct

import numpy as np


MAX_HEADER_BYTES = 64 * 1024 * 1024
STORAGE_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2",
                  "F8_E4M3FN": "u1", "F8_E4M3": "u1"}


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate safetensors header key: {name}")
        result[name] = value
    return result


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def decode_e4m3fn(raw):
    """Decode E4M3 finite-numbers FP8 (bias 7); 0x7f/0xff are NaNs."""
    raw = np.asarray(raw, dtype=np.uint8)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    magnitude = np.where(exponent == 0,
                         mantissa.astype(np.float32) / np.float32(512),
                         np.ldexp(np.float32(1) + mantissa.astype(np.float32) / 8,
                                  exponent.astype(np.int32) - 7)).astype(np.float32)
    magnitude = np.where((exponent == 15) & (mantissa == 7), np.float32(np.nan), magnitude)
    return np.copysign(magnitude, np.where(raw & 128, np.float32(-1), np.float32(1)))


def _float32(raw, dtype):
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dtype.startswith("F8_"):
        return decode_e4m3fn(raw)
    # Copy even F32 slices: yielded chunks remain valid independently of mappings.
    return np.array(raw, dtype=np.float32, copy=True)


class SafeTensorMatrixReader:
    """Inspect one immutable safetensors shard, mapping only requested tensors.

    Conversion buffers scale with chunk_rows * matrix columns plus relevant
    block-scale rows. A memmap reserves virtual address space, not a full dense
    tensor buffer. Consumers must process/discard chunks rather than collect them
    if bounded working memory is required.
    """

    def __init__(self, path):
        self.path = Path(path)
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors length prefix")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > stat.st_size - 8:
                raise ValueError("invalid or oversized safetensors header")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
            raise ValueError("safetensors metadata must contain strings")
        self.data_start = 8 + length
        self.tensors = {}
        intervals = []
        for name, tensor in header.items():
            if not isinstance(tensor, dict):
                raise ValueError(f"invalid tensor description: {name}")
            dtype, shape, span = tensor.get("dtype"), tensor.get("shape"), tensor.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in STORAGE_DTYPES:
                raise ValueError(f"unsupported safetensors dtype for {name}: {dtype}")
            if not isinstance(shape, list) or any(not _is_int(d) or d < 0 for d in shape):
                raise ValueError(f"invalid shape: {name}")
            if not isinstance(span, list) or len(span) != 2 or any(not _is_int(d) for d in span):
                raise ValueError(f"invalid offsets: {name}")
            start, end = span
            expected_bytes = math.prod(shape) * np.dtype(STORAGE_DTYPES[dtype]).itemsize
            if not 0 <= start <= end <= stat.st_size - self.data_start or end - start != expected_bytes:
                raise ValueError(f"invalid byte span: {name}")
            intervals.append((start, end))
            self.tensors[name] = {"dtype": dtype, "shape": tuple(shape), "data_offsets": (start, end)}
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("safetensors spans overlap or have holes")
            cursor = end
        if cursor != stat.st_size - self.data_start:
            raise ValueError("safetensors spans do not cover payload")

    def _map(self, name):
        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != self._identity:
            raise ValueError("checkpoint shard changed after header inspection")
        tensor = self.tensors[name]
        if math.prod(tensor["shape"]) == 0:
            return np.empty(tensor["shape"], dtype=STORAGE_DTYPES[tensor["dtype"]])
        return np.memmap(self.path, mode="r", dtype=STORAGE_DTYPES[tensor["dtype"]],
                         offset=self.data_start + tensor["data_offsets"][0], shape=tensor["shape"])

    def iter_dequantized_rows(self, name, chunk_rows=128, inverse_scale=None,
                              block_size=(128, 128)):
        """Yield independent float32 [rows, columns] chunks in row order.

        inverse_scale is an explicitly supplied tensor name in the same shard.
        For FP8, W = decode(fp8) * inverse_scale[row//Br, column//Bc].
        Scale tensors must be F32/F16/BF16 and strictly positive and finite.
        This compatibility entry point accepts rank-two matrices only. Use
        iter_matrix_rows for explicit expert selection or cross-shard scales.
        """
        yield from self.iter_matrix_rows(name, chunk_rows=chunk_rows,
                                         inverse_scale=inverse_scale,
                                         block_size=block_size)

    def iter_matrix_rows(self, name, chunk_rows=128, expert_index=None,
                         inverse_scale=None, block_size=(128, 128), scale_reader=None):
        """Yield bounded FP32 row chunks of a matrix or one selected expert.

        Rank-three banks have [experts, output_rows, input_columns] layout and
        require expert_index. Their scales must have the same expert axis followed
        by the two block-grid axes. Only the selected expert's data is decoded.
        scale_reader, when supplied, explicitly locates inverse_scale in another
        safetensors shard; scale names and layouts are never inferred.
        """
        if not _is_int(chunk_rows) or chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if (not isinstance(block_size, (tuple, list)) or len(block_size) != 2
                or any(not _is_int(d) or d <= 0 for d in block_size)):
            raise ValueError("block_size must contain two positive integers")
        tensor = self.tensors[name]
        rank = len(tensor["shape"])
        if rank not in (2, 3):
            raise ValueError("row conversion requires a matrix or rank-three expert bank")
        if rank == 3:
            if (not _is_int(expert_index) or not 0 <= expert_index < tensor["shape"][0]):
                raise ValueError("rank-three bank requires a valid explicit expert_index")
        elif expert_index is not None:
            raise ValueError("expert_index cannot be supplied for a rank-two matrix")
        rows, cols = tensor["shape"][-2:]
        if scale_reader is not None and not isinstance(scale_reader, SafeTensorMatrixReader):
            raise ValueError("scale_reader must be a SafeTensorMatrixReader")
        if scale_reader is not None and inverse_scale is None:
            raise ValueError("scale_reader requires an explicit inverse_scale tensor name")
        scale_source = self if scale_reader is None else scale_reader
        br, bc = block_size
        scale_meta = None
        scales = None
        if inverse_scale is not None:
            if not tensor["dtype"].startswith("F8_"):
                raise ValueError("inverse_scale is supported only for FP8 matrices")
            scale_meta = scale_source.tensors[inverse_scale]
            expected_shape = ((rows + br - 1) // br, (cols + bc - 1) // bc)
            if rank == 3:
                expected_shape = (tensor["shape"][0],) + expected_shape
            if scale_meta["shape"] != expected_shape:
                raise ValueError("inverse_scale shape does not match block grid")
            if scale_meta["dtype"] not in ("F32", "F16", "BF16"):
                raise ValueError("inverse_scale must be F32/F16/BF16")
            scales = scale_source._map(inverse_scale)
            if rank == 3:
                scales = scales[expert_index]
        matrix = self._map(name)
        if rank == 3:
            matrix = matrix[expert_index]
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            chunk = _float32(matrix[start:stop], tensor["dtype"])
            if scales is not None:
                first_block = start // br
                scale_rows = _float32(scales[first_block:(stop - 1) // br + 1], scale_meta["dtype"])
                if not np.all(np.isfinite(scale_rows)) or np.any(scale_rows <= 0):
                    raise ValueError("inverse_scale values must be positive and finite")
                # Avoid expanding an entire rows*columns scale matrix.
                for row in range(stop - start):
                    sr = scale_rows[(start + row) // br - first_block]
                    for col_block in range(len(sr)):
                        chunk[row, col_block * bc:min((col_block + 1) * bc, cols)] *= sr[col_block]
            yield chunk
