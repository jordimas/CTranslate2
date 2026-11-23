"""Verify saved final encoder tensors, including signed-zero bit patterns."""
import json
from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parent
rows = []
for compute in ("int8_float32", "float32"):
    reference = np.load(f"/tmp/ct2-encoder-bench-{compute}-regular.npy")
    for variant in ("regular", "head", "workspace", "tile192", "tile240"):
        actual = np.load(f"/tmp/ct2-encoder-bench-{compute}-{variant}.npy")
        assert reference.shape == actual.shape and reference.dtype == actual.dtype
        changed = int(np.count_nonzero(reference.view(np.uint32) != actual.view(np.uint32)))
        rows.append(dict(compute=compute, variant=variant, shape=list(actual.shape),
                         changed_float_bits=changed, bitwise_identical=(changed == 0)))
(out / "encoder-bitwise.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
assert all(row["bitwise_identical"] for row in rows)
