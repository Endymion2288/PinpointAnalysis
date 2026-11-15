"""R3 interventions preserve event and optimizer-budget contracts."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.r2.train import OffsetSampler
from analysis.flavour.v2.r3.sampling import GlobalBatchSampler
from analysis.flavour.v2.r1.models import build_model as reference_model
from analysis.flavour.v2.r3.models import build_model


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


if __name__=='__main__':
    unittest.main()
