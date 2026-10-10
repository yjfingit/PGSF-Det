"""Training behavior that must hold before unattended full-data runs."""
import unittest
import numpy as np
from pgsf.data.uavcb import PairAugment
from pgsf.engine.trainer import EarlyStopping,DistributedEvalSampler


class TrainingControls(unittest.TestCase):
    def test_patience_and_resume_preserve_exact_stop_epoch(self):
        early=EarlyStopping(30,1e-4)
        self.assertEqual(early.update(.2),(True,False))
        for _ in range(12):self.assertFalse(early.update(.2)[1])
        restored=EarlyStopping(30,1e-4);restored.load_state_dict(early.state_dict())
        for _ in range(17):self.assertFalse(restored.update(.2)[1])
        self.assertTrue(restored.update(.2)[1])
        self.assertEqual(restored.update(.21),(True,False))

    def test_validation_partition_covers_odd_dataset_once(self):
        parts=[list(DistributedEvalSampler(range(7),r,2)) for r in range(2)]
        self.assertEqual(sorted(parts[0]+parts[1]),list(range(7)))
        self.assertFalse(set(parts[0])&set(parts[1]))

    def test_canvas_zoom_changes_size_and_synchronizes_geometry(self):
        a=PairAugment((1.1,1.1),1.,0.,0.,True,zoom_prob=1.)
        box=np.array([.4,.45,.08,.1],np.float32)
        rgb=np.zeros((80,120,3),np.uint8);tir=np.zeros((40,60),np.uint8)
        _,_,r,t,meta=a(rgb,tir,box,box,(120,80),(60,40),(80,120),(40,60),True)
        for b,k in [(r,'rgb'),(t,'tir')]:
            np.testing.assert_allclose(meta[k]@np.r_[box[:2],1],np.r_[b[:2],1],atol=1e-6)
            np.testing.assert_allclose(b[2:],box[2:]*1.1,atol=1e-6)

    def test_mild_mode_keeps_both_sensors_and_disables_shift(self):
        a=PairAugment((1.,1.),0.,0.,0.,True,modality_dropout=1.,translation_prob=1.,
                      translation_range=.1,degrade_prob=1.)
        a.set_mild_mode(True)
        box=np.array([.5,.5,.08,.1],np.float32)
        rgb=np.full((80,120,3),128,np.uint8);tir=np.full((40,60),128,np.uint8)
        _,_,r,t,meta=a(rgb,tir,box,box,(120,80),(60,40),(80,120),(40,60),True)
        np.testing.assert_array_equal(meta['available'],[1.,1.])
        np.testing.assert_allclose(r,box,atol=1e-6);np.testing.assert_allclose(t,box,atol=1e-6)
        self.assertEqual(float(a._mild),1.)

    def test_clipped_detection_box_is_excluded_from_geometric_nll(self):
        from pgsf.models.detector_v2 import paired_geometry_mask
        import torch
        a=PairAugment((1.2,1.2),0.,0.,0.,True,zoom_prob=1.)
        box=np.array([.91,.5,.12,.1],np.float32)
        _,_,r,t,meta=a(np.zeros((80,120,3),np.uint8),np.zeros((40,60),np.uint8),
                      box,box,(120,80),(60,40),(80,120),(40,60),True)
        self.assertIsNotNone(r);self.assertIsNotNone(t)
        self.assertEqual(float(meta['geometry_valid'].sum()),0.)
        b={'mask':torch.ones(1,1),'geometry_valid':torch.tensor(meta['geometry_valid'])[None]}
        self.assertEqual(float(paired_geometry_mask(b).sum()),0.)


if __name__=='__main__':unittest.main()
