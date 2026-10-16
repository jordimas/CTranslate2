"""Bounded-memory CT2 binary-v6 writer, independent of the native extension.

Layout follows specs/model_spec.py ModelSpec._serialize and DataType IDs in
include/ctranslate2/types.h. Bytes are explicitly little-endian (the supported
deployment hosts). This serializes variables, not a complete runnable model.
Each tensor is limited by v6's uint32 payload byte count; split expert banks into
individual expert variables rather than emitting a multi-gigabyte bank.
"""

import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


DTYPES = {"float32": (0, "<f4"), "int8": (1, "i1"), "int16": (2, "<i2"),
          "int32": (3, "<i4"), "float16": (4, "<f2"), "bfloat16": (5, "<u2")}
UINT32_MAX = (1 << 32) - 1


def _integer(value, maximum=UINT32_MAX, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"expected integer in {minimum}..{maximum}")
    return value


def _string(value):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("model and variable names must be nonempty strings without NUL")
    encoded = value.encode("utf-8")
    if len(encoded) + 1 > 65535:
        raise ValueError("encoded name exceeds uint16 string length")
    return struct.pack("<H", len(encoded) + 1) + encoded + b"\0"


class StreamingModelWriter:
    """Write into a sibling temporary file, publishing only on explicit finish.

    Existing destination files are never overwritten. Use as a context manager;
    exiting without finish aborts and deletes the temporary file. All write or
    validation failures abort the transaction, even if callers catch the error.
    chunks must contain contiguous arrays of exactly the requested dtype, or raw
    contiguous bytes-like payloads in little-endian format. For bfloat16, arrays
    must contain uint16 BF16 bit patterns; no implicit numerical conversion occurs.
    """

    def __init__(self, path, model_name, revision, variable_count):
        self.path = Path(path)
        self._file = None
        self._temp = None
        self._finished = False
        self._names = set()
        name = _string(model_name)
        _integer(revision)
        self._count = _integer(variable_count)
        if self.path.exists():
            raise FileExistsError(f"destination already exists: {self.path}")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        self._temp = Path(temporary)
        self._file = os.fdopen(descriptor, "wb")
        try:
            self._file.write(struct.pack("<I", 6) + name + struct.pack("<II", revision, variable_count))
        except BaseException:
            self.abort()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self._finished:
            self.abort()

    def abort(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._temp is not None:
            self._temp.unlink(missing_ok=True)
            self._temp = None

    def _require_open(self):
        if self._file is None:
            raise RuntimeError("writer is closed or already finished")

    def write_tensor(self, name, shape, dtype, chunks):
        self._require_open()
        try:
            encoded_name = _string(name)
            if name in self._names or len(self._names) >= self._count:
                raise ValueError("duplicate tensor or declared variable count exceeded")
            if not isinstance(dtype, str) or dtype not in DTYPES:
                raise ValueError(f"unsupported CT2 dtype: {dtype}")
            shape = tuple(shape)
            _integer(len(shape), maximum=255)
            for dim in shape:
                _integer(dim, minimum=1)
            type_id, numpy_dtype = DTYPES[dtype]
            numpy_dtype = np.dtype(numpy_dtype)
            expected = math.prod(shape) * numpy_dtype.itemsize
            _integer(expected)
            self._file.write(encoded_name + struct.pack("<B", len(shape)))
            for dim in shape:
                self._file.write(struct.pack("<I", dim))
            self._file.write(struct.pack("<BI", type_id, expected))
            written = 0
            for chunk in chunks:
                if isinstance(chunk, np.ndarray):
                    if chunk.dtype != numpy_dtype or not chunk.flags.c_contiguous:
                        raise ValueError("array chunks must have exact dtype and C-contiguous layout")
                try:
                    payload = memoryview(chunk)
                    if not payload.c_contiguous:
                        raise ValueError("byte chunks must be contiguous")
                    payload = payload.cast("B")
                except TypeError as exc:
                    raise ValueError("chunks must be contiguous arrays or bytes-like objects") from exc
                if written + len(payload) > expected:
                    raise ValueError("tensor payload exceeds declared shape")
                self._file.write(payload)
                written += len(payload)
            if written != expected:
                raise ValueError("tensor payload shorter than declared shape")
            self._names.add(name)
        except BaseException:
            self.abort()
            raise

    def finish(self, aliases=()):
        self._require_open()
        try:
            if len(self._names) != self._count:
                raise ValueError("written variable count differs from declared count")
            pairs = list(aliases.items() if isinstance(aliases, dict) else aliases)
            _integer(len(pairs))
            names = set(self._names)
            encoded = []
            for alias, target in pairs:
                a, t = _string(alias), _string(target)
                if alias in names or target not in self._names:
                    raise ValueError("alias duplicates a name or does not target a concrete tensor")
                names.add(alias)
                encoded.append((a, t))
            # The native loader implicitly aliases associated scales and zeros.
            for alias, target in pairs:
                for suffix in ("_scale", "_zero"):
                    if target + suffix in self._names:
                        if alias + suffix in names:
                            raise ValueError("alias collides with an implicit scale/zero alias")
                        names.add(alias + suffix)
            self._file.write(struct.pack("<I", len(pairs)))
            for alias, target in encoded:
                self._file.write(alias + target)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
            # Atomic, no-clobber publication on the same filesystem. A destination
            # created after __init__ causes failure without damaging either file.
            os.link(self._temp, self.path)
            self._temp.unlink()
            self._temp = None
            self._finished = True
            return self.path
        except BaseException:
            self.abort()
            raise
