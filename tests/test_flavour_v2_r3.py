"""R3 interventions preserve event and optimizer-budget contracts."""

from types import SimpleNamespace
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.r2.train import OffsetSampler
from analysis.flavour.v2.r3.sampling import GlobalBatchSampler
from analysis.flavour.v2.r1.models import build_model as reference_model
from analysis.flavour.v2.r3.models import build_model, relation_features, PrelocalRelations
from analysis.flavour.v2.r3.relations import candidate_edges
from analysis.flavour.v2.r3.data import collate
from analysis.flavour.v2.r3.collate_audit import reference_collate
from analysis.flavour.v2.r3.phase import phase_forward
from analysis.flavour.v2.r3.accept import gate_matches
from analysis.flavour.v2.r3.evaluate import configuration
from analysis.flavour.v2.r3.sensor_rows import sensor_row_geometry, sensor_projection
from analysis.flavour.v2.r1.data import projection
from analysis.flavour.v2.r3.mapped_inputs import MappedItems,write_mapped_items
from analysis.flavour.v2.r3.scientific_review import promotion_checks
from analysis.flavour.v2.r3.probes import ReadableFeatures
from analysis.flavour.v2.r3.handoff import validate_release


class GlobalOrderTests(unittest.TestCase):
    def data(self):
        sizes=np.array([8,8,9,9,10])
        boundaries=np.r_[0,sizes.cumsum()]
        return SimpleNamespace(paths=list(range(len(sizes))),indexes=np.arange(boundaries[-1]),
            native=SimpleNamespace(boundaries=boundaries),config={'cache_shards':2,'bucket_size':0})

    def test_global_order_preserves_every_update_size_and_event_each_pass(self):
        data=self.data()
        native=SpatialBatchSampler(data,8,42)
        global_order=GlobalBatchSampler(data,8,42)
        for epoch in range(4):
            native.epoch=global_order.epoch=epoch
            a,b=list(native),list(global_order)
            self.assertEqual([len(x) for x in a],[len(x) for x in b])
            np.testing.assert_array_equal(np.sort(np.concatenate(b)),np.arange(len(data.indexes)))
            self.assertFalse(np.array_equal(np.concatenate(a),np.concatenate(b)))

    def test_order_is_deterministic_epoch_dependent_and_label_independent(self):
        data=self.data()
        a,b=GlobalBatchSampler(data,8,42),GlobalBatchSampler(data,8,42)
        self.assertEqual(list(a),list(b))
        first=list(a)
        a.epoch=1
        self.assertNotEqual(first,list(a))
        # The fixture exposes no labels: constructing/iterating cannot consult truth.
        self.assertFalse(hasattr(data,'metadata'))

    def test_resume_replays_exact_remaining_global_order(self):
        sampler=GlobalBatchSampler(self.data(),8,42)
        sampler.epoch=2
        whole=list(sampler)
        resumed=OffsetSampler(sampler)
        resumed.offset=3
        self.assertEqual(list(resumed),whole[3:])


class LocalRetentionTests(unittest.TestCase):
    def test_default_adapter_is_exact_and_stride_control_keeps_parameters(self):
        torch.set_num_threads(1)
        config={'family':'B1','task':'T4','image':{'height':128,'width':96},'cnn_width':16}
        torch.manual_seed(42)
        original=reference_model(config).eval()
        torch.manual_seed(42)
        unchanged=build_model(config).eval()
        torch.manual_seed(42)
        finer=build_model({**config,'cnn_first_stride':2}).eval()
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,unchanged.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(value,finer.state_dict()[name],rtol=0,atol=0)
        image=torch.rand(2,4,128,96)
        self.assertEqual(original.backbone.stem(image).shape[-2:],(32,24))
        self.assertEqual(finer.backbone.stem(image).shape[-2:],(64,48))
        batch={'label':torch.tensor([0,1]),'projection_event':torch.tensor([0,0,1,1]),
            'projection_indices':torch.tensor([0,123,321,543]),'projection_values':torch.ones(4),
            'tracker':torch.zeros(2,1,5),'tracker_mask':torch.zeros(2,1,dtype=torch.bool),
            'presence':torch.tensor([[1.,0.,0.],[1.,0.,0.]])}
        torch.testing.assert_close(original(batch),unchanged(batch),rtol=0,atol=0)
        reconstructed,_,_=phase_forward(unchanged,batch)
        torch.testing.assert_close(original(batch),reconstructed,rtol=0,atol=0)


class NativeRelationTests(unittest.TestCase):
    def nodes(self, xyz, measured=None):
        nodes=np.zeros((len(xyz),18),dtype='float32')
        nodes[:,:3]=np.asarray(xyz)/[210.,210.,3000.]
        nodes[:,3:5]=1 if measured is None else measured
        nodes[:,12]=nodes[:,15]=1
        return nodes

    def event(self, nodes, groups, index=0):
        edges,_=candidate_edges(nodes,np.asarray(groups))
        return dict(nodes=nodes,groups=np.asarray(groups),relation_edges=edges,
            family='B2',label=index,index=index,tracker=np.empty((0,5),dtype='float32'),
            presence=np.array([1.,0.,0.],dtype='float32'))

    def test_bounds_cross_window_and_skip_visible_plane(self):
        nodes=self.nodes([[0,0,0],[1,0,0],[2,0,0],[0,0,10],[0,0,30],[0,0,70]])
        groups=np.array([0,0,1,2,3,4])
        edges,summary=candidate_edges(nodes,groups)
        self.assertTrue(((edges>=0)&(edges<len(nodes))).all())
        self.assertEqual(np.unique(edges,axis=1).shape[1],edges.shape[1])
        pairs=set(map(tuple,edges.T))
        self.assertNotIn((0,1),pairs)
        self.assertIn((0,2),pairs)
        self.assertIn((0,5),pairs)
        self.assertLessEqual(summary['maximum_out_degree'],12)
        empty,_=candidate_edges(np.zeros((1,18),dtype='float32'),np.zeros(1,dtype='int64'))
        self.assertEqual(empty.shape,(2,0))

    def test_bar_interval_does_not_invent_second_measured_axis(self):
        nodes=self.nodes([[0,150,0],[.1,0,10],[200,0,10]],[[1,0],[1,1],[0,1]])
        nodes[0,12]=0;nodes[0,13]=1;nodes[0,8:10]=[1/420,400/420]
        edges,_=candidate_edges(nodes,np.arange(3))
        pairs=set(map(tuple,edges.T))
        self.assertIn((0,1),pairs)
        self.assertNotIn((0,2),pairs)
        a=torch.from_numpy(nodes[[0]])
        b=torch.from_numpy(nodes[[1]])
        f=relation_features(a,b,(.0208,.0228))
        self.assertEqual(float(f[0,1]),0.)
        self.assertEqual(float(f[0,4]),0.)
        reverse=relation_features(b,a,(.0208,.0228))
        torch.testing.assert_close(f[:,:3],-reverse[:,:3])
        self.assertGreater(float(f[0,3]),0.)
        torch.testing.assert_close(f[:,3:5],reverse[:,3:5])

    def test_collation_preserves_node_indices_and_event_separation(self):
        a=self.event(self.nodes([[0,0,0],[1,0,0],[0,0,10]]),[0,0,1])
        b=self.event(self.nodes([[0,0,0],[0,0,10]]),[0,1],1)
        batch=collate([a,b])
        source,target=batch['relation_source'],batch['relation_target']
        self.assertTrue(batch['local_mask'].flatten()[source].all())
        self.assertTrue(batch['local_mask'].flatten()[target].all())
        self.assertTrue(torch.equal(batch['event_index'][source//32],batch['event_index'][target//32]))
        self.assertTrue((source>=64).any())

    def test_vectorized_padding_matches_accepted_inputs_with_unsorted_and_empty_events(self):
        rng=np.random.default_rng(20261115)
        groups=np.repeat(np.arange(4),[32,1,17,9])
        nodes=self.nodes(rng.normal(size=(len(groups),3)))
        order=rng.permutation(len(groups))
        a=self.event(nodes[order],groups[order])
        a['tracker']=rng.normal(size=(7,5)).astype('float32')
        b=self.event(np.zeros((1,18),dtype='float32'),[0],1)
        for relations in [True,False]:
            items=[dict(a),dict(b)]
            if not relations:
                for item in items:item.pop('relation_edges')
            old,new=reference_collate(items),collate(items)
            self.assertEqual(set(old),set(new))
            for name in old:
                self.assertEqual(old[name].dtype,new[name].dtype)
                self.assertTrue(torch.equal(old[name],new[name]),name)
        bad={**a,'groups':np.zeros(len(groups),dtype='int64')}
        with self.assertRaises(ValueError):collate([bad])

    def test_modality_planes_bridge_interleaved_bar_planes(self):
        nodes=self.nodes([[0,0,0],[0,0,100]]+[[0,0,z] for z in range(10,100,10)])
        nodes[2:,12]=0;nodes[2:,13]=1
        groups=np.arange(len(nodes))
        old,_=candidate_edges(nodes,groups)
        new,summary=candidate_edges(nodes,groups,plane_policy='modality_visible')
        self.assertNotIn((0,1),set(map(tuple,old.T)))
        self.assertIn((0,1),set(map(tuple,new.T)))
        self.assertLessEqual(summary['maximum_out_degree'],14)

    def test_nominal_material_depth_uses_real_slab_centers(self):
        geometry={k:[0.,30.] for k in ['tungsten_Zpos','pixel_Zpos','scint_Zpos']}
        geometry.update(tungsten_thickness=5.,silicon_thickness=10.,scint_thickness=5.)
        layer=PrelocalRelations(16,(.0208,.0228),geometry)
        self.assertTrue(all(torch.isfinite(v).all() for v in layer.state_dict().values()))
        depths=layer.nominal_depths(torch.tensor([-10.,0.,10.,30.,50.]))
        torch.testing.assert_close(depths[:,0],torch.tensor([0.,2.5,5.,7.5,10.]))
        torch.testing.assert_close(depths[:,1],torch.tensor([0.,.005,.010,.015,.020]),rtol=2e-4,atol=1e-6)

    def test_native_control_is_exact_and_relations_receive_gradient(self):
        torch.set_num_threads(1)
        config={'family':'B2','task':'T4','dim':16,'seed':42}
        batch=collate([self.event(self.nodes([[0,0,0],[1,0,0],[0,0,10]]),[0,1,2])])
        torch.manual_seed(42);original=reference_model(config).eval()
        torch.manual_seed(42);control=build_model({**config,'native_relation_adapter':True}).eval()
        torch.manual_seed(42);related=build_model({**config,'native_relation_adapter':True,'native_relations':True}).eval()
        for k,v in original.backbone.state_dict().items():
            torch.testing.assert_close(v,control.backbone.encoder.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(v,related.backbone.encoder.state_dict()[k],rtol=0,atol=0)
        torch.testing.assert_close(original(batch),control(batch),rtol=0,atol=0)
        related.train();logits=related(batch);logits.square().mean().backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(float(related.backbone.relations.message[0].weight.grad.norm()),0.)
        self.assertGreater(float(related.backbone.relations.gate.grad.abs()),0.)

    def test_retrained_ablation_readout_has_no_invented_regional_stage(self):
        config={'family':'B2','task':'T4','dim':16,'seed':42,
            'native_relation_adapter':True,'native_relations':True,'native_global_ablation':True}
        batch=collate([self.event(self.nodes([[0,0,0],[1,0,0],[0,0,10]]),[0,1,2])])
        model=build_model(config).eval();hooks=ReadableFeatures(model)
        with torch.inference_mode():
            output=model(batch)
            features=hooks.collect(batch)
        hooks.close()
        self.assertEqual(set(features),{'local','global','tracker'})
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(len(value)==1 and np.isfinite(value).all() for value in features.values()))


class DeliveryContractTests(unittest.TestCase):
    def test_handoff_rejects_live_previous_writers_and_wrong_claim(self):
        release={'allocation_id':'replacement','models':['a','b'],
            'previous_allocations':[{'id':'1.0','models':['a','b'],'JobStatus':3,'terminal_confirmed':True}],
            'previous_writers_terminal':True,'completion_time_decision':'measured saving including preparation'}
        validate_release(release,'replacement',['a','b'])
        with self.assertRaises(ValueError):validate_release(release,'other',['a','b'])
        with self.assertRaises(ValueError):validate_release(release,'replacement',['a'])
        release['previous_allocations'][0]['models']=['a']
        with self.assertRaises(ValueError):validate_release(release,'replacement',['a','b'])
        release['previous_allocations'][0]['models']=['a','b']
        release['previous_allocations'][0]['JobStatus']=2
        with self.assertRaises(ValueError):validate_release(release,'replacement',['a','b'])

    def test_compensating_recalls_and_tau_common_task_loss_do_not_pass_promotion(self):
        def metric(point,low,high):return {'difference_a_minus_b':point,'95pct':[low,high]}
        values={'BA3':metric(.025,.021,.029),'NLL3':metric(-.02,-.03,-.01),
            'recall_e':metric(.025,.020,.030),'recall_mu':metric(.025,.020,.030),
            'recall_NC':metric(.025,.020,.030),'tau_eff_1pct':metric(.015,.011,.019),
            'NLL4':metric(-.01,-.02,0)}
        good=promotion_checks(values)
        self.assertTrue(good['common_three_class']['passed'])
        self.assertTrue(good['tau']['passed'])
        # Positive aggregate BA does not compensate for a class-quality loss.
        values['recall_e']=metric(-.0342,-.0403,-.0280)
        self.assertFalse(promotion_checks(values)['common_three_class']['passed'])
        # A positive tau working point cannot waive the common-task guard.
        values['BA3']=metric(-.002,-.006,.002)
        self.assertFalse(promotion_checks(values)['tau']['passed'])
        values['BA3']=metric(.025,None,.03)
        self.assertFalse(promotion_checks(values)['common_three_class']['passed'])

    def test_learning_gate_rejects_different_stride_and_relation_geometry(self):
        base={'family':'B1','task':'T4','precision':'BF16_encoder_FP32_head','cnn_width':16,
            'image':{'height':32,'width':32}}
        gate={'status':'complete','config':base}
        self.assertTrue(gate_matches(gate,base))
        self.assertFalse(gate_matches(gate,{**base,'cnn_first_stride':2}))
        native={'family':'B2','task':'T4','precision':'BF16_encoder_FP32_head','dim':64,
            'native_relation_adapter':True,'native_relations':True,'relation_plane_policy':'modality_visible'}
        self.assertFalse(gate_matches({'status':'complete','config':native},{**native,'relation_material_context':True}))

    def test_evaluation_requires_unique_candidates_and_resolved_pair_names(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);experiment=root/'candidate';experiment.mkdir()
            (experiment/'config.json').write_text(json.dumps(dict(family='B1',task='T4',seed=42,design='G1')))
            contract=root/'contract.json'
            value={'experiments':[{'id':'a','path':str(experiment)}],
                'contrasts':[dict(a='a',b='missing',domains=['all'],reason='negative_fixture')]}
            contract.write_text(json.dumps(value))
            with self.assertRaises(ValueError):configuration(root,contract)
            value['contrasts'][0]['b']='a';contract.write_text(json.dumps(value))
            self.assertEqual(len(configuration(root,contract)['models']),2)
            value['experiments']*=2;contract.write_text(json.dumps(value))
            with self.assertRaises(ValueError):configuration(root,contract)


class SensorRowTests(unittest.TestCase):
    def test_row_mapping_preserves_each_measurement_and_physical_spacing(self):
        geometry={'pixel_Xpos':[-1.,0.,1.],'pixel_Ypos':[-1.,0.,1.],
            'pixel_Zpos':[0.,100.], 'scint_Zpos':[10.,30.],
            'scint_bar_Xcenter':[-1.,1.],'scint_bar_Ycenter':[-1.,1.],
            'scint_bar_width':1.,'scint_bar_height':1.}
        observation={'pixel':{'xyz_mm':np.array([[0.,0.,0.],[1.,1.,100.]])},
            'bar':{'bounds_mm':np.array([[-1.5,-.5,-2.,2.,10.],[-2.,2.,.5,1.5,30.]]),
                'horizontal':np.array([False,True]),'edep_proxy_MeV':np.array([2.,3.]),
                'pe_proxy':np.array([20.,30.])}}
        image=sensor_row_geometry(geometry)
        old,values=projection(observation,geometry,image['physical_grid'])
        compact,new_values=sensor_projection(observation,geometry,image)
        w=image['width'];h=image['height'];physical_h=image['physical_grid']['height']
        restored=((compact//(h*w))*physical_h+np.asarray(image['physical_row'])[(compact//w)%h])*w+compact%w
        np.testing.assert_array_equal(old,restored)
        np.testing.assert_array_equal(values,new_values)
        self.assertEqual(image['row_z_mm'],[0.,10.,30.,100.])
        config={'family':'B1','task':'T4','image':image,'cnn_width':16,
            'sensor_row_image':True,'cnn_first_stride':2}
        model=build_model(config)
        self.assertEqual(model.backbone.stem[0].in_channels,9)
        self.assertEqual(model.backbone.stem[0].stride,(2,2))


class MappedInputTests(unittest.TestCase):
    def test_ragged_views_and_spawn_serialization_preserve_arrays_without_copying_payload(self):
        items=[{'index':0,'label':1,'nodes':np.arange(720,dtype='float32').reshape(40,18),
            'relation_edges':np.array([[0,3],[3,1]],dtype='int64'),'tracker':np.empty((0,5),dtype='float32')},
            {'index':1,'label':2,'nodes':np.zeros((1,18),dtype='float32'),
            'relation_edges':np.empty((2,0),dtype='int64'),'tracker':np.ones((3,5),dtype='float32')}]
        with tempfile.TemporaryDirectory() as scratch:
            cache=write_mapped_items([[items[0]],[items[1]]],Path(scratch)/'inputs')
            for i,item in enumerate(items):
                for key,value in item.items():np.testing.assert_array_equal(cache[i][key],value)
            self.assertIsInstance(cache.arrays['nodes'],np.memmap)
            with self.assertRaises(ValueError):cache[0]['nodes'][0,0]=10.
            payload=pickle.dumps(cache,protocol=5)
            self.assertLess(len(payload),2000)
            restored=pickle.loads(payload)
            self.assertIsInstance(restored,MappedItems)
            self.assertEqual(restored.arrays,{})
            np.testing.assert_array_equal(restored[1]['nodes'],items[1]['nodes'])


if __name__=='__main__':
    unittest.main()
