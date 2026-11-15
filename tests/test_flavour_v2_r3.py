"""R3 interventions preserve event and optimizer-budget contracts."""

from types import SimpleNamespace
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
from analysis.flavour.v2.r3.phase import phase_forward


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
        depths=layer.nominal_depths(torch.tensor([-10.,0.,10.,30.,50.]))
        torch.testing.assert_close(depths[:,0],torch.tensor([0.,2.5,5.,7.5,10.]))
        torch.testing.assert_close(depths[:,1],torch.tensor([0.,5.,10.,15.,20.]))

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


if __name__=='__main__':
    unittest.main()
