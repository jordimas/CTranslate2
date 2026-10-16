"""Streaming per-row INT8 conversion using CT2 model_spec.py conventions.

This is a matrix-conversion building block, not a complete Drama 3 converter.
Sources provide explicit iter_matrix_rows; no checkpoint names are inferred.
"""

import numpy as np


def int8_row_scales(matrix):
    """Return CT2 multiplicative scales: 127/amax, with all-zero rows using 1.

    Reject nonfinite inputs and scales not representable as positive float32.
    Silently overflowing a scale for extremely tiny values would corrupt weights.
    """
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("INT8 conversion requires nonempty-column float32 matrices")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cannot quantize nonfinite weights")
    amax = np.max(np.abs(matrix), axis=1)
    amax[amax == 0] = np.float32(127)
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        scales = np.float32(127) / amax
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("per-row INT8 scale is not representable as positive float32")
    return scales.astype(np.float32, copy=False)


def quantize_int8_rows(matrix):
    """Round ties to even, matching np.rint in CT2's converter."""
    scales = int8_row_scales(matrix)
    values = np.rint(matrix * scales[:, None])
    # With finite float32 scale and amax this cannot exceed the int8 interval;
    # make that invariant explicit instead of allowing integer wraparound.
    if np.any(values < -127) or np.any(values > 127):
        raise ValueError("INT8 quantization produced an out-of-range value")
    return values.astype(np.int8), scales


def write_int8_matrix(
    writer,
    source_reader,
    source_name,
    output_scope,
    chunk_rows=128,
    expert_index=None,
    inverse_scale=None,
    scale_reader=None,
    block_size=(128, 128),
):
    """Write scope/weight then scope/weight_scale with two bounded source passes.

    FP8 inverse scales are source dequantization multipliers. Output CT2 row
    scales are quantization multipliers and must not be confused with them.
    Reader and writer objects are injected, avoiding native package imports.
    A rank-three source requires explicit expert_index. Memory is O(chunk_rows
    * columns), never the full bank; no row-scale vector for the bank is retained.
    The source must remain immutable across both passes.
    """
    if not isinstance(output_scope, str) or not output_scope or "\0" in output_scope:
        raise ValueError("output_scope must be nonempty and contain no NUL")
    shape = source_reader.tensors[source_name]["shape"]
    if len(shape) not in (2, 3) or any(dim <= 0 for dim in shape[-2:]):
        raise ValueError("source must contain a nonempty matrix or expert bank")
    rows, cols = shape[-2:]
    options = dict(
        chunk_rows=chunk_rows,
        expert_index=expert_index,
        inverse_scale=inverse_scale,
        scale_reader=scale_reader,
        block_size=block_size,
    )

    def quantized_chunks():
        for chunk in source_reader.iter_matrix_rows(source_name, **options):
            yield quantize_int8_rows(chunk)[0]

    def scale_chunks():
        for chunk in source_reader.iter_matrix_rows(source_name, **options):
            yield int8_row_scales(chunk)

    weight_name = output_scope + "/weight"
    writer.write_tensor(weight_name, (rows, cols), "int8", quantized_chunks())
    writer.write_tensor(weight_name + "_scale", (rows,), "float32", scale_chunks())
    return {
        "weight": weight_name,
        "scale": weight_name + "_scale",
        "shape": [rows, cols],
    }
