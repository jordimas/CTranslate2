"""Small independent binary fixtures exercise streaming model-audit failures."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('binary_audit',Path(__file__).resolve().parents[1]/'tts/dev/audit_drama3_model.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class BinaryAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,policy='all_matrices_int8',duplicate=False,quantization='int8_float32',aliases=()):
        mixed=policy=='experts_int8_sensitive_float32'
        expert='backbone/layers_0/mlp/experts_0/gate_proj/weight';router='backbone/layers_0/mlp/router/weight';norm='backbone/norm/weight'
        mappings=[{'target':expert,'shape':[2,2],'dtype':'int8'},
                  {'target':router,'shape':[2,2],'dtype':'float32' if mixed else 'int8'},
                  {'target':norm,'shape':[2],'dtype':'float32'}]
        tensors=[(expert,[2,2],1,struct.pack('4b',1,2,3,4)),(expert+'_scale',[2],0,struct.pack('<2f',2,4))]
        tensors += [(router,[2,2],0,struct.pack('<4f',.1,.2,.3,.4))] if mixed else [(router,[2,2],1,struct.pack('4b',1,2,3,4)),(router+'_scale',[2],0,struct.pack('<2f',2,4))]
        tensors += [(norm,[2],0,struct.pack('<2f',.25,-.5)),('codec/example/weight',[1],0,struct.pack('<f',.5))]
        if quantization == 'float32':
            tensors = [(name, shape, 0, struct.pack('<'+'f'*len(payload),
                        *struct.unpack('<'+'b'*len(payload), payload)) if dtype == 1 else payload)
                       for name, shape, dtype, payload in tensors if not name.endswith('_scale')]
            for item in mappings:
                item['dtype'] = 'float32'
        codec={'tensors':[{'required_for_inference':True,'ct2_target':'codec/example/weight','expected_logical_shape':[1]}]}
        (self.root/'codec.json').write_text(json.dumps(codec))
        conversion={'model_name':'Drama3Spec','ct2_layout_version':1,'source_revision':'test','quantization_policy':policy,
                    'model_mapping_count':len(mappings),'codec_parameter_count':1,'variable_count':len(tensors),
                    'estimated_payload_bytes':sum(len(t[3]) for t in tensors),'mappings':mappings,
                    'matrix_counts':{'int8':0 if quantization == 'float32' else (1 if mixed else 2),
                                     'float32':2 if quantization == 'float32' else (1 if mixed else 0)}}
        if quantization != 'int8_float32':
            conversion['quantization'] = quantization
        (self.root/'conversion.json').write_text(json.dumps(conversion))
        (self.root/'config.json').write_text(json.dumps({'model_type':'drama3','ct2_layout_version':1,'source_revision':'test','quantization_policy':policy}))
        string=lambda s:struct.pack('<H',len(s.encode())+1)+s.encode()+b'\0'
        tensors = [tensor for tensor in tensors if tensor[0] not in dict(aliases)]
        data=bytearray(struct.pack('<I',6)+string('Drama3Spec')+struct.pack('<II',1,len(tensors)));self.positions={}
        for i,(name,shape,dtype,payload) in enumerate(tensors):
            data+=string(expert if duplicate and i==1 else name);data+=struct.pack('B',len(shape))+struct.pack('<'+'I'*len(shape),*shape)
            dtype_pos=len(data);data+=struct.pack('<BI',dtype,len(payload));self.positions[name]={'dtype':dtype_pos,'length':dtype_pos+1,'payload':len(data)};data+=payload
        data+=struct.pack('<I',len(aliases))
        for name, target in aliases:
            data+=string(name)+string(target)
        (self.root/'model.bin').write_bytes(data)
        return self.root/'codec.json'

    def test_both_policies_and_small_chunk_boundaries(self):
        for policy in audit.POLICIES:
            codec=self.fixture(policy)
            report=audit.audit(self.root,codec,chunk_bytes=4)
            self.assertEqual(report['status'],'pass');self.assertTrue(report['no_trailing_bytes'])
            self.assertEqual(report['int8_scale_tensors'],1 if policy.startswith('experts_') else 2)

    def test_nan_and_zero_scale_are_rejected(self):
        for name,value in [('backbone/norm/weight',float('nan')),
                           ('backbone/layers_0/mlp/experts_0/gate_proj/weight_scale',0.)]:
            codec=self.fixture();p=self.root/'model.bin';data=bytearray(p.read_bytes());offset=self.positions[name]['payload'];data[offset:offset+4]=struct.pack('<f',value);p.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'nonfinite|nonpositive'):audit.audit(self.root,codec,chunk_bytes=4)

    def test_duplicate_truncation_and_trailing_bytes_are_rejected(self):
        codec=self.fixture(duplicate=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):audit.audit(self.root,codec)
        for change in (lambda b:b[:-1],lambda b:b+b'junk'):
            codec=self.fixture();p=self.root/'model.bin';p.write_bytes(change(p.read_bytes()))
            with self.assertRaisesRegex(ValueError,'range|truncated|trailing'):audit.audit(self.root,codec)

    def test_payload_size_and_dtype_mismatch_rejected(self):
        for field,fmt,value in [('length','<I',0xffffffff),('dtype','B',4)]:
            codec=self.fixture();p=self.root/'model.bin';data=bytearray(p.read_bytes());offset=self.positions['backbone/norm/weight'][field];data[offset:offset+struct.calcsize(fmt)]=struct.pack(fmt,value);p.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'payload|dtype'):audit.audit(self.root,codec)

    def test_policy_rejects_manifest_that_would_requantize_sensitive_weights(self):
        codec=self.fixture('experts_int8_sensitive_float32');p=self.root/'conversion.json';data=json.loads(p.read_text());data['mappings'][1]['dtype']='int8';p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'violates quantization policy'):audit.audit(self.root,codec)
        self.fixture();p=self.root/'config.json';data=json.loads(p.read_text());data['quantization_policy']='typo';p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'quantization policy'):audit.audit(self.root,codec)

    def test_standard_aliases_preserve_logical_coverage(self):
        expert='backbone/layers_0/mlp/experts_0/gate_proj/weight'
        router='backbone/layers_0/mlp/router/weight'
        codec=self.fixture(aliases=[(router,expert),(router+'_scale',expert+'_scale')])
        report=audit.audit(self.root,codec,chunk_bytes=4)
        self.assertEqual(report['alias_count'],2)
        self.assertEqual(report['int8_scale_tensors'],2)
        self.assertEqual(report['variable_count'],report['physical_variable_count']+2)
        self.assertLess(report['physical_payload_bytes'],report['payload_bytes'])

    def test_alias_target_shape_dtype_and_scale_validation(self):
        expert='backbone/layers_0/mlp/experts_0/gate_proj/weight'
        router='backbone/layers_0/mlp/router/weight'
        norm='backbone/norm/weight'
        cases=[('all_matrices_int8',router,'missing','target'),
               ('all_matrices_int8',norm,'codec/example/weight','shape/dtype'),
               ('experts_int8_sensitive_float32',router,expert,'shape/dtype'),
               ('all_matrices_int8',expert+'_scale',norm,'nonpositive')]
        for policy,name,target,message in cases:
            with self.subTest(name=name,target=target):
                codec=self.fixture(policy,aliases=[(name,target)])
                with self.assertRaisesRegex(ValueError,message):audit.audit(self.root,codec)
        # A scale may validly alias a positive non-scale tensor.
        codec=self.fixture(aliases=[(expert+'_scale',norm)])
        p=self.root/'model.bin';data=bytearray(p.read_bytes());offset=self.positions[norm]['payload']
        data[offset:offset+8]=struct.pack('<2f',.25,.5);p.write_bytes(data)
        self.assertEqual(audit.audit(self.root,codec)['status'],'pass')

    def test_real_standard_serializer_implicit_scale_alias(self):
        import numpy as np
        module_spec=importlib.util.spec_from_file_location(
            'standard_spec',Path(__file__).resolve().parents[1]/'python/ctranslate2/specs/model_spec.py')
        standard=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(standard)

        class Matrix(standard.LayerSpec):
            def __init__(self):
                self.weight=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
                self.weight_scale=standard.OPTIONAL

        class Model(standard.ModelSpec):
            def __init__(self):
                super().__init__()
                self.a=Matrix();self.b=Matrix()
            @property
            def name(self):return 'Drama3Spec'

        model=Model();model.validate();model.optimize('int8_float32')
        model._serialize(self.root/'model.bin')
        variables=model.variables()
        self.assertEqual(variables['b/weight'],'a/weight')
        self.assertNotIn('b/weight_scale',variables)
        mappings=[{'target':name+'/weight','shape':[2,2],'dtype':'int8'} for name in ('a','b')]
        conversion={'model_name':'Drama3Spec','ct2_layout_version':1,'source_revision':'test',
                    'model_mapping_count':2,'codec_parameter_count':0,'variable_count':4,
                    'estimated_payload_bytes':24,'mappings':mappings}
        (self.root/'conversion.json').write_text(json.dumps(conversion))
        (self.root/'config.json').write_text(json.dumps(
            {'model_type':'drama3','ct2_layout_version':1,'source_revision':'test'}))
        codec=self.root/'codec.json';codec.write_text(json.dumps({'tensors':[]}))
        report=audit.audit(self.root,codec)
        self.assertEqual(report['physical_variable_count'],2)
        self.assertEqual(report['alias_count'],1)
        self.assertEqual(report['implicit_alias_count'],1)
        self.assertEqual(report['int8_scale_tensors'],2)

    def test_float32_conversion_policy(self):
        for policy in audit.POLICIES:
            codec=self.fixture(policy,quantization='float32')
            report=audit.audit(self.root,codec)
            self.assertEqual(report['matrix_counts'],{'int8':0,'float32':2})
            self.assertEqual(report['int8_scale_tensors'],0)
            self.assertEqual(report['quantization'],'float32')


if __name__=='__main__':unittest.main()
