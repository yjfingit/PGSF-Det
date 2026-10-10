"""Regression tests for data, geometry, association, metrics and stage transfer.

Run: python tests/test_regressions.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from pgsf.data import Frame, PairAugment, UAVCBDataset, collate
from pgsf.engine.trainer import PGSFTrainer
from pgsf.losses import correspondence_mask, supervised_association_loss, draw_gaussian, gaussian_focal_loss
from pgsf.models.candidate import CandidateGenerator, sample_points
from pgsf.models.detector import PGSFDet, decode_detections, param_groups
from pgsf.models.fusion import MidFusion, scatter_points
from pgsf.models.geometry import invert_affine, augment_affine
from pgsf.models.late_fusion import LateFusion
from pgsf.utils.metrics import average_precision, eval_detection, match_accuracy, filtered_ground_truth
import test_units


def tiny_config(stage=2):
    cfg = test_units.TestAblations()._tiny_cfg()
    cfg.stage = stage
    return cfg


class DataRegression(unittest.TestCase):
    def test_config_cli_size_values_stay_numeric(self):
        from pgsf.config import load_config
        for override in ("data.rgb_size=768,1280", "data.rgb_size=[768,1280]"):
            cfg = load_config("configs/pgsf_stage2.yaml", [override])
            self.assertEqual(list(cfg.data.rgb_size), [768,1280])

    def test_size_labels_and_flip(self):
        rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        tir = np.zeros((40, 60), dtype=np.uint8)
        box = np.array([.8, .4, .1, .2], dtype=np.float32)
        augment = PairAugment(scale_range=(1.25, 1.25), hflip=1., color_jitter=0., tir_gain_jitter=0.)
        r, t, rb, tb, transform = augment(rgb, tir, box, box, (120,80), (60,40), (32,64), (32,32))
        self.assertEqual(r.shape, (3,32,64))
        self.assertEqual(t.shape, (1,32,32))
        np.testing.assert_allclose(rb, [.2,.4,.1,.2], atol=1e-6)
        np.testing.assert_allclose(tb, rb)
        np.testing.assert_allclose(transform @ np.r_[box[:2], 1], [.2,.4,1.], atol=1e-6)

    def test_native_pixel_decode_matches_gt(self):
        with tempfile.TemporaryDirectory() as d:
            rp, tp = str(Path(d)/'r.png'), str(Path(d)/'t.png')
            Image.fromarray(np.zeros((80,120,3), np.uint8)).save(rp)
            Image.fromarray(np.zeros((40,60), np.uint8)).save(tp)
            box = np.array([.8,.4,.1,.2], np.float32)
            f = Frame('building','1','1',rp,tp,(120,80),(60,40),box,box.copy())
            batch = collate([UAVCBDataset([f],(32,64),(32,32),train=False)[0]])
            self.assertEqual(batch['rgb'].shape, (1,3,32,64))
            for name in ('rgb','tir'):
                raw = torch.zeros(1,1,5)
                raw[...,2:4] = torch.logit(torch.tensor([.1,.2])/.5)
                raw[...,4] = 10.
                pred = decode_detections(raw,batch['centers_'+name][:,:1],batch[name+'_hw'])
                gt = filtered_ground_truth(batch,name)
                self.assertEqual(len(gt[0]),1)
                self.assertAlmostEqual(eval_detection(pred,gt)['AP50:95'],1.)

    def test_injected_candidates_exclude_padding_and_empty_images(self):
        gen = CandidateGenerator(8)
        pts, scores = torch.rand(2,64,2), torch.rand(2,64)
        centers, mask = torch.zeros(2,50,2), torch.zeros(2,50)
        centers[0,0] = torch.tensor([.7,.6]); mask[0,0] = 1
        out, _, flags = gen._inject_gt(pts,scores,centers,mask)
        self.assertEqual(int(flags[0].sum()),16)
        self.assertEqual(int(flags[1].sum()),0)
        self.assertTrue(torch.equal(out[1],pts[1]))
        self.assertLess(float((out[0][flags[0]>0]-centers[0,0]).norm(dim=-1).max()),.06)
        self.assertTrue(((out[0][flags[0]>0]-centers[0,0]).norm(dim=-1)==0).any())

    def test_subpixel_gaussian_has_positive_gradient(self):
        logits = torch.zeros(1,1,19,27,requires_grad=True)
        gt = draw_gaussian(torch.zeros_like(logits),torch.tensor([[[.423,.567]]]),
                           torch.tensor([[[.02,.03]]]),torch.ones(1,1))
        self.assertEqual(int((gt==1).sum()),1)
        gaussian_focal_loss(logits,gt).backward()
        self.assertLess(float(logits.grad[gt==1].item()),0.)

    def test_sampling_and_scatter_share_coordinates(self):
        feat = torch.arange(8*10,dtype=torch.float32).reshape(1,1,8,10)
        pt = torch.tensor([[[3/9,2/7]]])
        value = sample_points(feat,pt)
        splat, coverage = scatter_points(pt,value,8,10)
        self.assertAlmostEqual(float(value.item()),23.,places=5)
        self.assertAlmostEqual(float(splat[0,0,2,3]),23.,places=5)
        self.assertEqual(int(coverage.sum()),1)

    def test_partition_indices_on_nondivisible_width(self):
        heat = torch.rand(1,1,11,13)
        gen = CandidateGenerator(8)
        idx, scores = gen._partition_topk(heat,1)
        actual = heat.flatten(1).gather(1,idx)
        self.assertTrue(torch.allclose(actual[scores>=0],scores[scores>=0]))


class MetricsRegression(unittest.TestCase):
    def test_perfect_prediction_after_padding_filter(self):
        gt = torch.tensor([[[10.,10.,20.,20.]]])
        batch = {'boxes_rgb_px':torch.cat([gt,torch.zeros(1,49,4)],1),
                 'mask':torch.cat([torch.ones(1,1),torch.zeros(1,49)],1)}
        pred = [{'boxes':gt[0],'scores':torch.ones(1)}]
        res = eval_detection(pred,filtered_ground_truth(batch,'rgb'))
        self.assertEqual(res['n_gt'],1)
        self.assertEqual(res['AP50'],1.)

    def test_score_order_and_empty_gt(self):
        g = torch.tensor([[10.,10.,20.,20.]])
        pred = {'boxes':torch.tensor([[10.,10.,21.,21.],[10.,10.,20.,20.]]),
                'scores':torch.tensor([.9,.1])}
        self.assertEqual(eval_detection([pred],[g])['AP50'],1.)
        # An FP on a negative image must count, and empty GT must not crash.
        fp = {'boxes':g.clone(),'scores':torch.tensor([.95])}
        self.assertAlmostEqual(eval_detection([fp,pred],[torch.zeros(0,4),g])['AP50'],.5)

    def test_actual_101_point_interpolation(self):
        self.assertAlmostEqual(average_precision(np.array([.5]),np.array([1.])),51/101)

    def test_nonregistered_perfect_pair(self):
        gr, gt = torch.tensor([[100.,100.,120.,120.]]),torch.tensor([[35.,35.,45.,45.]])
        pr, pt = [{'boxes':gr,'scores':torch.ones(1)}],[{'boxes':gt,'scores':torch.ones(1)}]
        self.assertEqual(match_accuracy(pr,pt,[gr],[gt])['match_acc'],1.)

    def test_trainer_and_overfit_evaluate_perfect_boxes(self):
        cfg = tiny_config()
        batch = test_units.TestAblations()._batch(cfg)
        for name in ('rgb','tir'):
            box = torch.tensor([100.,100.,120.,120.]) if name=='rgb' else torch.tensor([35.,35.,45.,45.])
            batch['boxes_'+name+'_px'] = torch.zeros(1,50,4)
            batch['boxes_'+name+'_px'][0,0] = box
        class Perfect:
            def eval(self): return self
            def train(self): return self
            def inference(self,batch,**kwargs):
                return {n:[{'boxes':b,'scores':torch.ones(len(b))} for b in filtered_ground_truth(batch,n)]
                        for n in ('rgb','tir')},None
        tr = PGSFTrainer.__new__(PGSFTrainer)
        tr.cfg=cfg; tr.device=torch.device('cpu');tr.model=Perfect();tr.ema=None
        tr.val_loader=[batch];tr.test_loader=[batch];tr.loader=[batch]
        result=tr.evaluate()
        self.assertEqual(result['rgb']['n_gt'],1)
        self.assertEqual(result['rgb']['AP50:95'],1.)
        self.assertEqual(result['tir']['AP50:95'],1.)
        from scripts.overfit_test import evaluate
        res=evaluate(tr.model,[batch],tr.device,cfg)
        self.assertEqual(res['rgb_AP50'],1.)
        self.assertEqual(res['tir_AP50'],1.)

    def test_score_threshold_filters_output(self):
        raw = torch.zeros(1,2,5);raw[...,4]=-5.
        out = decode_detections(raw,torch.rand(1,2,2),torch.tensor([[100.,100.]]),score_thr=.5)
        self.assertEqual(len(out[0]['scores']),0)


class FusionRegression(unittest.TestCase):
    def test_bidirectional_geometry_and_augmented_prior(self):
        A=torch.tensor([[.9,.04,.08],[-.03,1.1,-.06]])
        q=torch.tensor([[[.4,.6]]]);f=torch.randn(1,1,16)
        mapped=torch.cat([q,torch.ones_like(q[...,:1])],-1)@A.T
        mid=MidFusion(16,4,n_local=2,n_global=2)
        mid.geo.set_affine(A,A[:,2])
        r=mid(f,q,torch.randn(1,16,8,8),torch.randn(1,2,16),'r2t')
        t=mid(f,mapped,torch.randn(1,16,8,8),torch.randn(1,2,16),'t2r')
        self.assertTrue(torch.allclose(r['stats']['mu'],mapped,atol=1e-6))
        self.assertTrue(torch.allclose(t['stats']['mu'],q,atol=1e-6))
        T=torch.tensor([[[-1.,0.,1.],[0.,1.,0.],[0.,0.,1.]]])
        Af=augment_affine(A,T)
        qf=q.clone();qf[...,0]=1-q[...,0]
        tf=mapped.clone();tf[...,0]=1-mapped[...,0]
        actual=torch.cat([qf,torch.ones_like(qf[...,:1])],-1)@Af.transpose(-1,-2)
        self.assertTrue(torch.allclose(actual,tf,atol=1e-6))
        late=LateFusion(16,4)
        boxes_q=torch.cat([q,q.new_full((1,1,2),.1)],-1)
        boxes_t=torch.cat([mapped,mapped.new_full((1,1,2),.1)],-1)
        seen=[]
        handle=late.box_prior.register_forward_pre_hook(lambda mod,args: seen.append(args[2].clone()))
        late(f,boxes_t,f,boxes_q,A,A[:,2],'t2r')
        handle.remove()
        self.assertTrue(torch.allclose(seen[0],invert_affine(A),atol=1e-6))

    def test_zero_residual_maps_and_own_query_coordinates(self):
        mid=MidFusion(16,4,n_local=2,n_global=2)
        feature=torch.randn(1,2,16);pts=torch.tensor([[[.2,.3],[.7,.8]]])
        fm=torch.randn(1,16,8,8)
        initial=mid(feature,pts,fm,torch.randn(1,2,16),maps={'P3':fm})
        self.assertTrue(torch.equal(initial['maps']['P3'],fm))
        with torch.no_grad():mid.attn_r2t.w_o.bias.fill_(.25)
        changed=mid(feature,pts,fm,torch.randn(1,2,16),maps={'P3':fm})
        residual=changed['feats']-feature
        expected,cov=scatter_points(pts,residual,8,8)
        self.assertTrue(torch.allclose(changed['maps']['P3']-fm,expected*cov,atol=1e-6))

    def test_association_is_gt_identity_and_order_independent(self):
        gq=torch.tensor([[[.2,.2,.04,.04],[.8,.8,.04,.04]]])
        gk=torch.tensor([[[.4,.3,.04,.04],[.6,.7,.04,.04]]])
        q=gq[...,:2];k=gk[:,[1,0],:2]
        mask=torch.ones(1,2);valid=torch.ones(1)
        matches=correspondence_mask(q,k,gq,gk,mask,valid)
        self.assertTrue(torch.equal(matches,torch.tensor([[[False,True],[True,False]]])))
        attn=torch.tensor([[[.1,.8,.1],[.8,.1,.1]]],requires_grad=True)
        loss=supervised_association_loss(attn,matches)
        loss.backward()
        self.assertAlmostEqual(float(loss),-np.log(.8),places=6)
        self.assertTrue((attn.grad[...,:-1][matches]<0).all())
        perm=torch.tensor([1,0])
        perm_match=correspondence_mask(q,k[:,perm],gq,gk,mask,valid)
        perm_attn=torch.cat([attn.detach()[...,:-1][...,perm],attn.detach()[...,-1:]],-1)
        self.assertAlmostEqual(float(supervised_association_loss(perm_attn,perm_match)),float(loss),places=6)
        # An absent corresponding key must supervise null, independently per query.
        only_first=k[:,1:]
        missing=correspondence_mask(q,only_first,gq,gk,mask,valid)
        self.assertFalse(bool(missing[0,1].any()))

    def test_losses_are_not_divided_by_batch_twice(self):
        cfg=tiny_config();model=PGSFDet(cfg).eval()
        batch=test_units.TestAblations()._batch(cfg)
        duplicate={k:v.repeat(2,*([1]*(v.dim()-1))) for k,v in batch.items()}
        with torch.no_grad():
            one,_=model.compute_losses(model(batch))
            two,_=model.compute_losses(model(duplicate))
        self.assertTrue(torch.allclose(one,two,rtol=1e-4,atol=1e-4), (float(one),float(two)))

    def test_inference_needs_no_gt(self):
        cfg=tiny_config();model=PGSFDet(cfg).eval()
        batch={'rgb':torch.rand(1,3,*cfg.data.rgb_size),'tir':torch.rand(1,1,*cfg.data.tir_size),
               'rgb_hw':torch.tensor([[1080.,1920.]]),'tir_hw':torch.tensor([[512.,640.]])}
        with torch.no_grad():result,_=model.inference(batch)
        self.assertIn('rgb',result);self.assertIn('tir',result)

    def test_stage_transfer_is_identity_and_strict_for_branch_weights(self):
        cfg1=tiny_config(1);stage1=PGSFDet(cfg1).eval()
        A=torch.tensor([[.95,.02,.03],[-.04,1.1,-.02]])
        stage1.set_affine(A,A[:,2])
        stage2=PGSFDet(tiny_config()).eval()
        stage2.load_checkpoint_weights(stage1.state_dict(),stage1_transfer=True)
        self.assertTrue(torch.equal(stage2.affine[0],A))
        batch=test_units.TestAblations()._batch(cfg1)
        with torch.no_grad():a,b=stage1(batch),stage2(batch)
        for name in ('rgb','tir'):
            self.assertTrue(torch.allclose(a['pred_'+name],b['pred_'+name],atol=1e-6))
        broken=dict(stage1.state_dict());del broken['rgb_branch.det.out.weight']
        with self.assertRaises(RuntimeError):stage2.load_checkpoint_weights(broken,stage1_transfer=True)

    def test_checkpoint_resume_restores_optimizer(self):
        cfg=tiny_config();model=PGSFDet(cfg)
        tr=PGSFTrainer.__new__(PGSFTrainer)
        tr.cfg=cfg;tr.model=model;tr.ema=None
        tr.optimizer=torch.optim.AdamW(param_groups(model,cfg))
        tr.scaler=torch.amp.GradScaler('cuda',enabled=False)
        tr.history=[{'epoch':1}];tr.current_epoch=1;tr.start_epoch=0
        tr.best_metrics={'AP50:95':.2}
        p=model.rgb_branch.det.out.weight;p.grad=torch.ones_like(p)
        tr.optimizer.step();tr.optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as d:
            tr.out_dir=d;os.mkdir(Path(d)/'ckpt')
            path=tr.save('last.pt')
            saved=p.clone();p.data.zero_()
            tr.optimizer=torch.optim.AdamW(param_groups(model,cfg))
            tr.resume(path)
            self.assertTrue(torch.equal(saved,p))
            self.assertEqual(tr.start_epoch,1)
            self.assertEqual(int(tr.optimizer.state[p]['step']),1)
            self.assertEqual(tr.best_metrics['AP50:95'],.2)


if __name__ == '__main__':
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                             for c in (DataRegression,MetricsRegression,FusionRegression))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
