#!/usr/bin/env python3
"""Generate a standard ModelSpec fixture for the native reader test."""

import argparse
from pathlib import Path

import numpy as np
from ctranslate2.specs.drama3_spec import Drama3Spec
from ctranslate2.specs.model_spec import OPTIONAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    spec = Drama3Spec(
        {
            "projection/weight": np.array([[-3, 2], [0, 7], [8, -9]], np.int8),
            "projection/weight_scale": np.array([1, 2, 4], np.float32),
            "tied/weight": np.array([[-3, 2], [0, 7], [8, -9]], np.int8),
            "tied/weight_scale": OPTIONAL,
            "marker": np.array([17, -8], np.int32),
        },
        model_name="Drama3SerializationFixture",
    )
    spec.validate()
    spec.optimize()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    spec.save(args.output_dir)


if __name__ == "__main__":
    main()
