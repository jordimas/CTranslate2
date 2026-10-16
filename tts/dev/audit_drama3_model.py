#!/usr/bin/env python3
"""Independently audit a Drama3 CT2-v6 binary in bounded chunks (NumPy only).

Compare conversion.json and the external decoder tensor contract against binary
headers, stream every float32 value, and seek over integer weight payloads.
This checks structure/finite values, not numerical fidelity to source weights.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEC_MANIFEST = ROOT / 'tts/results/codec-tensor-contract.json'
POLICIES = ('all_matrices_int8', 'experts_int8_sensitive_float32')
ROUTED = re.compile(r'backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight\Z')


def unique_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def positive_shape(shape):
    if (not isinstance(shape, list) or not shape or len(shape) > 255
            or any(type(d) is not int or not 1 <= d <= 0xffffffff for d in shape)):
        raise ValueError('invalid tensor shape')
    return tuple(shape)


def expected_variables(conversion, codec_contract, config):
    policy = conversion.get('quantization_policy', 'all_matrices_int8')
    if policy not in POLICIES or config.get('quantization_policy', 'all_matrices_int8') != policy:
        raise ValueError('unknown or mismatched quantization policy')
    if conversion.get('model_name') != 'Drama3Spec' or conversion.get('ct2_layout_version') != 1:
        raise ValueError('unsupported converted model identity/layout')
    if config.get('model_type') != 'drama3' or config.get('ct2_layout_version') != 1:
        raise ValueError('unsupported package configuration')
    if config.get('source_revision') != conversion.get('source_revision'):
        raise ValueError('source revision mismatch')
    mappings = conversion.get('mappings')
    if not isinstance(mappings, list) or len(mappings) != conversion.get('model_mapping_count'):
        raise ValueError('mapping count mismatch')
    expected, matrices = {}, Counter({'int8': 0, 'float32': 0})

    def add(name, shape, dtype, scale=False):
        if not isinstance(name, str) or not name or '\0' in name or name in expected:
            raise ValueError('invalid or duplicate expected variable: '+repr(name))
        shape = positive_shape(list(shape))
        size = math.prod(shape) * (1 if dtype == 'int8' else 4)
        if size > 0xffffffff:
            raise ValueError('tensor exceeds CT2-v6 size limit: '+name)
        expected[name] = {'shape': shape, 'dtype': dtype, 'bytes': size, 'scale': scale}

    for item in mappings:
        name = item['target']; shape = positive_shape(item['shape'])
        if not isinstance(name, str):
            raise ValueError('mapping target must be a string')
        if name.startswith('codec/'):
            raise ValueError('codec tensor must come from independent codec contract')
        quantized = len(shape) == 2 and (policy == 'all_matrices_int8' or ROUTED.fullmatch(name))
        dtype = 'int8' if quantized else 'float32'
        if item.get('dtype') != dtype:
            raise ValueError('mapping dtype violates quantization policy: '+name)
        if len(shape) == 2:
            matrices[dtype] += 1
        add(name, shape, dtype)
        if quantized:
            add(name+'_scale', (shape[0],), 'float32', scale=True)
    codec_count = 0
    for tensor in codec_contract['tensors']:
        if not tensor['required_for_inference']:
            continue
        name = tensor['ct2_target']
        if not isinstance(name, str) or not name.startswith('codec/'):
            raise ValueError('non-codec name in decoder contract')
        add(name, tensor['expected_logical_shape'], 'float32')
        codec_count += 1
    if codec_count != conversion.get('codec_parameter_count'):
        raise ValueError('codec count mismatch')
    if len(expected) != conversion.get('variable_count'):
        raise ValueError('expected variable count mismatch')
    payload = sum(t['bytes'] for t in expected.values())
    if payload != conversion.get('estimated_payload_bytes'):
        raise ValueError('expected payload byte count mismatch')
    if 'matrix_counts' in conversion and dict(matrices) != conversion['matrix_counts']:
        raise ValueError('matrix classification count mismatch')
    return expected, policy, dict(matrices), payload


def audit(model, codec_manifest=DEFAULT_CODEC_MANIFEST, chunk_bytes=4*1024*1024):
    if type(chunk_bytes) is not int or chunk_bytes < 4 or chunk_bytes % 4:
        raise ValueError('chunk_bytes must be a positive multiple of four')
    model = Path(model)
    conversion = unique_json(model/'conversion.json')
    expected, policy, matrices, payload = expected_variables(
        conversion, unique_json(codec_manifest), unique_json(model/'config.json'))
    binary = model/'model.bin'; size = binary.stat().st_size
    if 'output_bytes' in conversion and size != conversion['output_bytes']:
        raise ValueError('binary size differs from conversion report')
    finite_values = 0; scales = 0; seen = set(); actual_payload = 0
    started = time.monotonic()
    with binary.open('rb') as stream:
        def read(count):
            if count < 0 or count > size-stream.tell():
                raise ValueError('binary range outside file')
            value = stream.read(count)
            if len(value) != count:
                raise ValueError('truncated binary')
            return value
        def integer(count):
            return int.from_bytes(read(count), 'little')
        def string():
            length = integer(2)
            if length < 2:
                raise ValueError('invalid binary string length')
            raw = read(length)
            if raw[-1:] != b'\0' or b'\0' in raw[:-1]:
                raise ValueError('invalid binary string terminator')
            return raw[:-1].decode('utf-8')
        if integer(4) != 6 or string() != 'Drama3Spec' or integer(4) != 1:
            raise ValueError('unsupported binary version/spec/revision')
        count = integer(4)
        if count != len(expected):
            raise ValueError('binary variable count mismatch')
        for _ in range(count):
            name = string()
            if name in seen or name not in expected:
                raise ValueError('duplicate or unexpected binary variable: '+name)
            seen.add(name); spec = expected[name]
            rank = integer(1); shape = tuple(integer(4) for _ in range(rank))
            dtype = integer(1); byte_count = integer(4)
            if shape != spec['shape'] or dtype != (1 if spec['dtype'] == 'int8' else 0):
                raise ValueError('binary shape/dtype mismatch: '+name)
            if byte_count != spec['bytes'] or byte_count > size-stream.tell():
                raise ValueError('binary payload size/range mismatch: '+name)
            actual_payload += byte_count
            if dtype == 0:
                remaining = byte_count
                while remaining:
                    take = min(remaining, chunk_bytes)
                    values = np.frombuffer(read(take), dtype='<f4')
                    if not np.isfinite(values).all() or (spec['scale'] and not (values > 0).all()):
                        raise ValueError('nonfinite float32 or nonpositive INT8 scale: '+name)
                    finite_values += values.size; remaining -= take
                scales += int(spec['scale'])
            else:
                stream.seek(byte_count, 1)
        if integer(4) != 0:
            raise ValueError('unexpected aliases: converter contract declares none')
        if stream.tell() != size:
            raise ValueError('trailing bytes after alias table')
    if actual_payload != payload or seen != set(expected):
        raise ValueError('incomplete payload/name coverage')
    return {'status':'pass', 'model_name':'Drama3Spec', 'binary_version':6, 'spec_revision':1,
            'quantization_policy':policy, 'matrix_counts':matrices,
            'source_revision':conversion.get('source_revision'),
            'variable_count':len(expected), 'payload_bytes':payload, 'file_bytes':size,
            'int8_scale_tensors':scales, 'finite_float32_values_checked':finite_values,
            'all_shapes_and_dtypes_match_manifest':True, 'all_payload_ranges_valid':True,
            'all_variable_names_unique':True,
            'all_int8_scales_positive_finite':True, 'no_trailing_bytes':True,
            'codec_manifest':str(Path(codec_manifest)), 'chunk_bytes':chunk_bytes,
            'audit_seconds':time.monotonic()-started,
            'limitations':['Structure/finite-value audit; no source-weight numerical equivalence or speech quality claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--codec-manifest', type=Path, default=DEFAULT_CODEC_MANIFEST)
    args = parser.parse_args()
    try:
        result = audit(args.model, args.codec_manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status':'fail', 'model':str(args.model), 'error':str(error)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
