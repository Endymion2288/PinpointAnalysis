"""S2 tests: observable information loss, complete coverage and model gradients."""

import unittest

import numpy as np
import torch

from analysis.flavour.s2.dataset import collate_spatial
from analysis.flavour.s2.models import PointClassifier
from analysis.flavour.s2.representation import pixel_view, token_inputs


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry={'pixel_Xpos':np.arange(100)*.02,'pixel_Ypos':np.arange(100)*.02,
                       'pixel_Zpos':[0.,10.]}

    def pixel(self,xy):
        xy=np.asarray(xy,dtype='int64').reshape(-1,2)
        return {'channel_id':xy[:,1]*100+xy[:,0],
                'xyz_mm':np.column_stack([xy*.02,np.zeros(len(xy))])}

    def test_projection_erases_pairing_and_joint_keeps_it(self):
        a=self.pixel([[10,10],[20,20]])
        b=self.pixel([[10,20],[20,10]])
        for pitch in [0.,1.]:
            va=pixel_view(a,self.geometry,pitch,True)
            vb=pixel_view(b,self.geometry,pitch,True)
            for key in ['xyz_mm','count','measured','view']:
                np.testing.assert_array_equal(va[key],vb[key])
        self.assertFalse(np.array_equal(pixel_view(a,self.geometry)['xyz_mm'],pixel_view(b,self.geometry)['xyz_mm']))

    def test_coarse_count_OR_and_no_native_residual(self):
        a=self.pixel([[10,10],[20,20]])
        count=pixel_view(a,self.geometry,1.,False,'count')
        binary=pixel_view(a,self.geometry,1.,False,'or')
        self.assertEqual(count['count'].tolist(),[2])
        self.assertEqual(binary['count'].tolist(),[1])
        np.testing.assert_allclose(count['xyz_mm'][0,:2],[.49,.49])
        shifted=self.pixel([[11,12],[21,24]])
        np.testing.assert_array_equal(count['xyz_mm'],pixel_view(shifted,self.geometry,1.)['xyz_mm'])
        self.assertEqual(count['native_to_token'].tolist(),[[0],[0]])

    def test_all_points_survive_blocking_shuffle_and_empty(self):
        pixel=self.pixel([[i%10,i//10] for i in range(100)])
        obs={'pixel':pixel}
        a,g,_=token_inputs(obs,self.geometry,modalities=['pixel'])
        shuffled={'pixel':{k:v[::-1] for k,v in pixel.items()}}
        b,h,_=token_inputs(shuffled,self.geometry,modalities=['pixel'])
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(g,h)
        self.assertEqual(len(a),100)
        self.assertEqual(np.bincount(g).tolist(),[32,32,32,4])
        empty,group,_=token_inputs({'pixel':self.pixel([])},self.geometry,modalities=['pixel'])
        self.assertEqual(empty.shape,(1,18))
        self.assertEqual(empty.sum(),0)
        batch=collate_spatial([{'nodes':a,'groups':g,'label':0,'index':0},
                               {'nodes':empty,'groups':group,'label':3,'index':1}])
        self.assertEqual(batch['local_mask'].sum().item(),101)
        self.assertEqual(batch['event_mask'].sum().item(),5)

    def test_models_empty_batch_gradients_and_block_permutation(self):
        torch.manual_seed(17)
        nodes,groups,_=token_inputs({'pixel':self.pixel([[i,i] for i in range(40)])},self.geometry,modalities=['pixel'])
        empty=np.zeros((1,18),dtype='float32')
        batch=collate_spatial([{'nodes':nodes,'groups':groups,'label':0,'index':0},
                               {'nodes':empty,'groups':np.array([0]),'label':3,'index':1}])
        for family in ['part','particlenet']:
            model=PointClassifier(family,dim=16,k=4).eval()
            output=model(batch)
            self.assertEqual(output.shape,(2,4))
            self.assertTrue(torch.isfinite(output).all())
            torch.nn.functional.cross_entropy(output,batch['label']).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            order=torch.tensor([1,0,2])
            permuted={k:v[order] if k in ['nodes','local_mask','event_index','block_index'] else v
                      for k,v in batch.items()}
            torch.testing.assert_close(model(batch),model(permuted),atol=1e-6,rtol=1e-5)
            model.train()
            model.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(batch),batch['label'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':
    unittest.main()
