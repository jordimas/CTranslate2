"""Original-source sensitive F32 conversion and routed-expert policy tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('base_conversion_tests', Path(__file__).with_name('test_drama3_model_conversion.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
c = base.converter
POLICY = 'experts_int8_sensitive_float32'


class MixedPrecisionConversionTest(unittest.TestCase):
    def test_only_exact_routed_expert_matrices_are_quantized(self):
        names = ['model.language_model.layers.0.mlp.experts.3.gate_proj.weight',
                 'model.language_model.layers.0.mlp.gate.weight',
                 'model.language_model.layers.0.mlp.shared_expert.up_proj.weight',
                 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
                 'model.language_model.embed_tokens.weight', 'lm_head.weight',
                 'audio_decoder_mtp.heads.0.weight']
        tensors = {name: {'shape': (2, 2), 'dtype': 'F32', 'shard': 'a'} for name in names}
        default = c.build_plan(tensors, base.config(), "all_matrices_int8")
        mixed = c.build_plan(tensors, base.config(), POLICY)
        self.assertEqual(c.build_plan(tensors, base.config()), mixed)
        self.assertTrue(all(x['dtype'] == 'int8' for x in default))
        for item in mixed:
            self.assertEqual(item['dtype'], 'int8' if '.experts.3.' in item['source'] else 'float32')
        self.assertFalse(c.routed_expert_matrix('tower/experts_3/gate_proj/weight'))
        self.assertFalse(c.routed_expert_matrix('backbone/layers_0/mlp/experts_3/not_a_projection/weight'))

    def test_sensitive_float32_preserves_bf16_tiny_rows_and_tower_qgate_layout(self):
        with base.tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'a.safetensors'
            qkv = 'audio_decoder_mtp_shared.layers.0.transformer_layer.attention.wqkv.weight'
            projection = 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight'
            original = np.arange(36, dtype=np.float32).reshape(12, 3)
            tiny = np.array([[np.float32(2**-123), -np.float32(2**-123)], [.125, .25]], np.float32)
            bf16 = lambda x: (x.view(np.uint32) >> 16).astype('<u2')
            base.shard(path, {qkv: (bf16(original), 'BF16'), projection: (bf16(tiny), 'BF16')})
            reader = c.READERS.SafeTensorMatrixReader(path)
            tensors = {name: {**metadata, 'shard': path.name} for name, metadata in reader.tensors.items()}
            plan = c.build_plan(tensors, base.config(), POLICY)
            loaded, stats = c.load_model_tensors(plan, {path.name: reader}, base.config(), chunk_rows=3)
            output = base.optimized_variables(loaded)
            prefix = 'tower/shared_layers_0/transformer_layer/attention/'
            np.testing.assert_array_equal(output[prefix+'q_proj/weight'], original[[0,1,4,5,2,3,6,7]])
            np.testing.assert_array_equal(output[prefix+'k_proj/weight'], original[[8,9]])
            np.testing.assert_array_equal(output[prefix+'v_proj/weight'], original[[10,11]])
            np.testing.assert_array_equal(output['backbone/layers_0/linear_attn/in_proj_qkv/weight'], tiny)
            self.assertFalse(any(name.endswith('_scale') for name in output))
            self.assertEqual(stats['zeroed_rows'], 0)

    def test_fp8_sensitive_matrix_uses_original_cross_shard_block_scales(self):
        with base.tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = 'lm_head.weight'; scale_name = name+'_scale_inv'
            weights, scales = root/'weights.safetensors', root/'scales.safetensors'
            base.shard(weights, {name: (np.full((4,4), 0x38, np.uint8), 'F8_E4M3FN')})
            inverse = np.array([[.500123,1.000321],[2.00321,4.00213]], np.float32)
            base.shard(scales, {scale_name: (inverse, 'F32')})
            readers = {p.name:c.READERS.SafeTensorMatrixReader(p) for p in (weights,scales)}
            tensors = {n:{**t,'shard':p} for p,r in readers.items() for n,t in r.tensors.items()}
            plan = c.build_plan(tensors, base.config(), POLICY)
            loaded, _ = c.load_model_tensors(plan, readers, base.config(), chunk_rows=1)
            output = base.optimized_variables(loaded)
            np.testing.assert_array_equal(output['backbone/lm_head/weight'], np.repeat(np.repeat(inverse,2,0),2,1))
            self.assertEqual(len(output),1)

    def test_rejects_unknown_policy_before_reading_checkpoint_and_nonfinite_matrices(self):
        for policy in ('unsupported',None,False):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError,'quantization policy'):
                c.convert('/absent','/absent',quantization_policy=policy)
        with self.assertRaisesRegex(ValueError,'nonfinite source matrix'):
            list(c.finite_matrix_chunks([np.array([[np.nan]],np.float32)]))


if __name__ == '__main__':
    unittest.main()
