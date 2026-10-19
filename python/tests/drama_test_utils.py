"""Independent checkpoint writer shared by Drama3 conversion tests."""
import ctypes
import json
import os
import struct
import unittest

requires_atomic_publish = unittest.skipUnless(
    os.name == "nt" or hasattr(ctypes.CDLL(None), "renameat2"),
    "Atomic directory publication requires Windows rename or renameat2",
)


def shard(path, tensors):
    header, data = {}, b""
    for name, (array, dtype) in tensors.items():
        payload = array.tobytes(order="C")
        header[name] = {
            "dtype": "F8_E4M3" if dtype == "F8_E4M3FN" else dtype,
            "shape": list(array.shape),
            "data_offsets": [len(data), len(data) + len(payload)],
        }
        data += payload
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
