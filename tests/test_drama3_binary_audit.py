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

    def fixture(self,policy='all_matrices_int8',duplicate=False):
        mixed=policy=='experts_int8_sensitive_float32'
        expert='backbone/layers_0/mlp/experts_0/gate_proj/weight';router='backbone/layers_0/mlp/router/weight';norm='backbone/norm/weight'
        mappings=[{'target':expert,'shape':[2,2],'dtype':'int8'},
                  {'target':router,'shape':[2,2],'dtype':'float32' if mixed else 'int8'},
                  {'target':norm,'shape':[2],'dtype':'float32'}]
        tensors=[(expert,[2,2],1,struct.pack('4b',1,2,3,4)),(expert+'_scale',[2],0,struct.pack('<2f',2,4))]
        tensors += [(router,[2,2],0,struct.pack('<4f',.1,.2,.3,.4))] if mixed else [(router,[2,2],1,struct.pack('4b',1,2,3,4)),(router+'_scale',[2],0,struct.pack('<2f',2,4))]
        tensors += [(norm,[2],0,struct.pack('<2f',.25,-.5)),('codec/example/weight',[1],0,struct.pack('<f',.5))]
        codec={'tensors':[{'required_for_inference':True,'ct2_target':'codec/example/weight','expected_logical_shape':[1]}]}
        (self.root/'codec.json').write_text(json.dumps(codec))
        conversion={'model_name':'Drama3Spec','ct2_layout_version':1,'source_revision':'test','quantization_policy':policy,
                    'model_mapping_count':len(mappings),'codec_parameter_count':1,'variable_count':len(tensors),
                    'estimated_payload_bytes':sum(len(t[3]) for t in tensors),'mappings':mappings,
                    'matrix_counts':{'int8':1 if mixed else 2,'float32':1 if mixed else 0}}
        (self.root/'conversion.json').write_text(json.dumps(conversion))
        (self.root/'config.json').write_text(json.dumps({'model_type':'drama3','ct2_layout_version':1,'source_revision':'test','quantization_policy':policy}))
        string=lambda s:struct.pack('<H',len(s.encode())+1)+s.encode()+b'\0'
        data=bytearray(struct.pack('<I',6)+string('Drama3Spec')+struct.pack('<II',1,len(tensors)));self.positions={}
        for i,(name,shape,dtype,payload) in enumerate(tensors):
            data+=string(expert if duplicate and i==1 else name);data+=struct.pack('B',len(shape))+struct.pack('<'+'I'*len(shape),*shape)
            dtype_pos=len(data);data+=struct.pack('<BI',dtype,len(payload));self.positions[name]={'dtype':dtype_pos,'length':dtype_pos+1,'payload':len(data)};data+=payload
        data+=struct.pack('<I',0);(self.root/'model.bin').write_bytes(data)
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


if __name__=='__main__':unittest.main()
