"""Behavioral and gradient regressions for non-registered fusion and v2 heads."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pgsf.config import PGSFConfig
from pgsf.models.detector import PGSFDet
from pgsf.models.detector_v2 import pair_affines,global_candidates
from pgsf.models.proposal import proposal_losses,decode_coarse
from pgsf.models.fusion_v2 import ConditionalGeoPrior,CovarianceMidFusion,gaussian_log_prob
from pgsf.models.geometry import ProbGeoPrior
from pgsf.models.head import decode_pred
from pgsf.data.uavcb import PairAugment
from scripts.check_forward import make_fake_batch


def config(stage=2):
    c=PGSFConfig();c.stage=stage
    c.data.rgb_size=(128,160);c.data.tir_size=(96,128)
    m=c.model;m.backbone_name='lightcsp';m.architecture_version=2
    m.backbone_depths=(1,1,1,1);m.width_mult=.5
    m.neck_dim=32;m.neck_out=32;m.neck_repeats=1;m.context_dim=32
    m.proposal_width=16;m.msce_width=16;m.head_width=32;m.geo_jitter_prob=0
    return c


class V2Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(4)

    def test_quality_focal_has_optimum_at_target_quality(self):
        from pgsf.losses import quality_focal_loss
        target=torch.tensor([.2,.5,.8])
        logits=torch.logit(target).requires_grad_()
        loss=quality_focal_loss(logits,target);loss.backward()
        self.assertLess(float(loss),1e-12)
        self.assertLess(float(logits.grad.abs().sum()),1e-7)

    def test_proposals_cover_both_scales_and_have_direct_gradients(self):
        c=config(1);m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),2)
        out=m(b)
        for name in ('rgb','tir'):
            cand=out['cand_'+name]
            self.assertEqual(cand['pts'].shape,(2,64,2))
            self.assertTrue((cand['levels']==0).any());self.assertTrue((cand['levels']==1).any())
            self.assertTrue((cand['boxes'][...,2:]>0).all())
        pre=proposal_losses(out['cand_rgb'],b['boxes_rgb'],b['mask'])
        pre['box'].backward(retain_graph=True)
        for layer in (m.rgb_branch.heat.p2[-1],m.rgb_branch.heat.p3[-1]):
            self.assertGreater(float(layer.weight.grad[1:5].abs().sum()),0)
        m.zero_grad(set_to_none=True)
        pre['quality'].backward()
        self.assertGreater(float(m.rgb_branch.heat.p2[-1].weight.grad[5].abs().sum()),0)

    def test_final_box_loss_reaches_region_features(self):
        c=config(1);m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),2);out=m(b)
        sel=out['sel_rgb']
        losses=m.det_loss_rgb(out['pred_rgb'],sel['pts'],b['boxes_rgb'],b['mask'],sel['boxes'],True)
        losses['loss'].backward()
        self.assertGreater(float(m.rgb_branch.refine.merge[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(m.rgb_branch.roi_project[-1].weight.grad.abs().sum()),0)

    def test_conv_bn_fusion_preserves_outputs(self):
        from pgsf.utils.deploy import fuse_for_inference
        c=config(1);m=PGSFDet(c).eval();f=fuse_for_inference(m)
        b=make_fake_batch(c,torch.device('cpu'),1)
        with torch.no_grad():a,z=m(b),f(b)
        self.assertTrue(torch.allclose(a['pred_rgb'],z['pred_rgb'],atol=2e-5))
        self.assertTrue(torch.allclose(a['pred_tir'],z['pred_tir'],atol=2e-5))
        self.assertIsNot(m.rgb_branch.neck.lat2[0],f.rgb_branch.neck.lat2[0])

    def test_multi_target_budget_can_expand_to_fifty(self):
        c=config();c.model.n_cand_late=50;m=PGSFDet(c).eval()
        b=make_fake_batch(c,torch.device('cpu'),1)
        with torch.no_grad():out=m(b)
        self.assertEqual(out['pred_rgb'].shape,(1,50,5))
        self.assertEqual(out['pred_tir'].shape,(1,50,5))

    def test_relative_head_starts_from_coarse_box(self):
        base=torch.tensor([[[.3,.4,.02,.03]]]);raw=torch.zeros(1,1,5)
        decoded,_=decode_pred(raw,base[...,:2],base_boxes=base)
        self.assertTrue(torch.equal(decoded,base))
        raw[...,:2]=1
        decoded,_=decode_pred(raw,base[...,:2],base_boxes=base)
        self.assertTrue(((decoded[...,:2]-base[...,:2]).abs()<base[...,2:]/2).all())

    def test_injection_anneals_to_zero_and_globals_exclude_gt(self):
        c=config();m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),2)
        cand=m(b)['cand_rgb'];gf,gp,gq=global_candidates(cand,16)
        predicted=cand['pts'][0][cand['injected'][0]<.5]
        self.assertTrue(((gp[0][:,None]-predicted[None]).abs().amax(-1).min(-1).values<1e-6).all())
        m.set_training_progress(.8)
        self.assertEqual(float(m(b)['cand_rgb']['injected'].sum()),0.)

    def test_conditional_context_changes_mean_and_covariance_is_spd(self):
        g=ConditionalGeoPrior(16,8)
        with torch.no_grad():g.residual[-1].weight.fill_(.1)
        p=torch.rand(2,3,2);f=torch.randn(2,3,16)
        mu,L,_=g(p,f,context=torch.zeros(2,8))
        other,_,_=g(p,f,context=torch.ones(2,8))
        self.assertFalse(torch.allclose(mu,other))
        self.assertTrue((torch.linalg.eigvalsh(L@L.transpose(-1,-2))>0).all())

    def mid_fixture(self):
        mid=CovarianceMidFusion(16,4,context_dim=8)
        q=torch.randn(1,3,16);p=torch.full((1,3,2),.5)
        maps={'P2':torch.randn(1,16,24,32),'P3':torch.randn(1,16,12,16)}
        glob=(torch.randn(1,16,16),torch.rand(1,16,2),torch.ones(1,16))
        return mid,q,p,maps,glob

    def test_covariance_controls_distinct_sampling_points(self):
        mid,q,p,maps,glob=self.mid_fixture()
        first=mid(q,p,maps,glob)
        points=first['stats']['local_pts'][0,0]
        self.assertEqual(len(torch.unique(points,dim=0)),8)
        distance=(points-first['stats']['mu'][0,0]).norm(dim=-1).mean()
        with torch.no_grad():mid.geo.unc[-1].bias[[0,2]]=1.
        wider=mid(q,p,maps,glob)
        other=(wider['stats']['local_pts'][0,0]-wider['stats']['mu'][0,0]).norm(dim=-1).mean()
        self.assertGreater(float(other),float(distance))

    def test_failed_geometry_has_global_null_fallback(self):
        mid,q,p,maps,glob=self.mid_fixture()
        out=mid(q,p,maps,glob,prior_jitter=torch.full((1,1,2),3.))
        attn=out['stats']['attn']
        self.assertLess(float(attn[...,:8].sum()),1e-6)
        self.assertTrue(torch.allclose(attn[...,8:].sum(-1),torch.ones(1,3)))
        self.assertTrue(torch.isfinite(out['feats']).all())

    def test_direction_specific_geometry(self):
        c=config();m=PGSFDet(c)
        A=torch.tensor([[.9,0.,.1],[0.,1.1,-.05]])
        m.set_affine(A,A[:,2])
        self.assertFalse(torch.equal(m.mid.geo.A,m.mid.geo_t2r.A))
        with torch.no_grad():m.mid.geo.residual[-1].bias.fill_(.05)
        self.assertTrue(torch.equal(m.mid.geo_t2r.residual[-1].bias,torch.zeros(2)))

    def test_analytic_gaussian_matches_reference(self):
        g=ProbGeoPrior(8);mu,L,_=g(torch.rand(2,4,2),torch.randn(2,4,8));p=torch.rand_like(mu)
        self.assertTrue(torch.allclose(gaussian_log_prob(p,mu,L),g.log_prob(p,mu,L),atol=1e-4))

    def test_independent_translation_updates_boxes_and_affine(self):
        rgb=np.zeros((80,120,3),np.uint8);tir=np.zeros((40,60),np.uint8)
        box=np.array([.5,.5,.05,.05],np.float32)
        aug=PairAugment((1.,1.),0.,0.,0.,True,translation_prob=1.,translation_range=.1)
        r,t,br,bt,meta=aug(rgb,tir,box,box,(120,80),(60,40),(80,120),(40,60),True)
        np.testing.assert_allclose(meta['rgb']@np.r_[box[:2],1],np.r_[br[:2],1],atol=1e-6)
        np.testing.assert_allclose(meta['tir']@np.r_[box[:2],1],np.r_[bt[:2],1],atol=1e-6)
        c=config();model=PGSFDet(c)
        b={f'geo_transform_{name}':torch.tensor(meta[key])[None] for name,key in [('rgb','rgb'),('tir','tir')]}
        b.update({f'geo_inverse_{name}':torch.linalg.inv(b[f'geo_transform_{name}']) for name in ('rgb','tir')})
        A,Ar=pair_affines(model,b)
        actual=torch.cat((torch.tensor(br[:2])[None,None],torch.ones(1,1,1)),-1)@A.transpose(-1,-2)
        self.assertTrue(torch.allclose(actual[0,0],torch.tensor(bt[:2]),atol=1e-6))

    def test_missing_modality_has_null_supervision(self):
        from pgsf.models.detector_v2 import association,mask_for
        c=config();m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),1)
        b['modality_available']=torch.tensor([[1.,0.]])
        pts=b['centers_rgb'][:,:1]
        attn=torch.tensor([[[.1,.9]]],requires_grad=True)
        loss=association(m,attn,pts,pts,b,'rgb','tir',torch.ones(1,1))
        self.assertAlmostEqual(float(loss),-np.log(.9),places=6)
        self.assertEqual(float(mask_for(b,'tir').sum()),0.)
        loss.backward();self.assertLess(float(attn.grad[...,-1]),0.)

    def test_dropped_sensor_does_not_receive_background_detection_supervision(self):
        c=config();m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),1)
        b['modality_available']=torch.tensor([[1.,0.]])
        out=m(b);loss,terms=m.compute_losses(out)
        for key in ('loss_cls','loss_box','pre_box','pre_quality','heat'):
            self.assertEqual(float(terms['tir_'+key]),0.)
        self.assertGreater(float(terms['rgb_heat']),0.)
        self.assertGreater(float(terms['assoc_mid']),0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(m.tir_branch.det.out.weight.grad)

    def test_augmentation_drops_only_one_modality(self):
        a=PairAugment((1.,1.),0.,0.,0.,True,modality_dropout=1.,rgb_normalization='unit')
        _,_,_,_,meta=a(np.full((32,40,3),120,np.uint8),np.full((32,40),120,np.uint8),
                        None,None,(40,32),(40,32),(32,40),(32,40),True)
        self.assertEqual(float(meta['available'].sum()),1.)

    def test_v2_stage_transfer_preserves_initial_detection(self):
        c1=config(1);c2=config(2)
        m1=PGSFDet(c1).eval();m2=PGSFDet(c2).eval()
        m2.load_checkpoint_weights(m1.state_dict(),True)
        b=make_fake_batch(c1,torch.device('cpu'),1)
        with torch.no_grad():a,z=m1(b),m2(b)
        for name in ('rgb','tir'):
            self.assertTrue(torch.allclose(a['pred_'+name],z['pred_'+name],atol=1e-6))

    def test_inference_does_not_require_annotations(self):
        c=config();m=PGSFDet(c).eval();b=make_fake_batch(c,torch.device('cpu'),1)
        data={k:b[k] for k in ('rgb','tir','rgb_hw','tir_hw')}
        with torch.no_grad():result,_=m.inference(data)
        self.assertIn('rgb',result);self.assertIn('tir',result)

    def test_no_null_and_no_geometry_ablation_runs(self):
        c=config();c.model.use_null=False;c.model.use_geo_prior=False;c.model.use_gate=False
        m=PGSFDet(c);b=make_fake_batch(c,torch.device('cpu'),2)
        loss,_=m.compute_losses(m(b));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(hasattr(m.mid,'null_token'));self.assertFalse(hasattr(m.mid,'geo'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_graph_replays_new_images_and_rejects_wrong_shapes(self):
        from pgsf.utils.deploy import fuse_for_inference, CudaGraphForward
        c=config();model=fuse_for_inference(PGSFDet(c).cuda().eval())
        b=make_fake_batch(c,torch.device('cuda'),1)
        images={k:b[k] for k in ('rgb','tir','rgb_hw','tir_hw')}
        with torch.inference_mode(), torch.amp.autocast('cuda',dtype=torch.float16):
            graph=CudaGraphForward(model,images,decode=True)
            for factor in (1.,.7):
                updated={k:(v*factor if k in ('rgb','tir') else v) for k,v in images.items()}
                expected=model(updated)
                actual=graph(updated)
                for name in ('rgb','tir'):
                    self.assertTrue(torch.allclose(expected['pred_'+name],actual['pred_'+name],atol=1e-3))
                    from pgsf.models.detector import decode_boxes_scores
                    sel=expected['sel_'+name]
                    boxes,scores=decode_boxes_scores(expected['pred_'+name],sel['pts'],updated[name+'_hw'],
                                                      base_boxes=sel['boxes'],proposal_scores=sel['objectness'])
                    self.assertTrue(torch.allclose(boxes,actual['_decoded_'+name][0],atol=1e-3))
                    self.assertTrue(torch.allclose(scores,actual['_decoded_'+name][1],atol=1e-4))
            wrong=dict(images);wrong['rgb']=wrong['rgb'][...,:-1]
            with self.assertRaises(ValueError):graph(wrong)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_gpu7_amp_forward_backward(self):
        c=config();m=PGSFDet(c).cuda();b=make_fake_batch(c,torch.device('cuda'),2)
        with torch.amp.autocast('cuda'):
            out=m(b);loss,_=m.compute_losses(out)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main(verbosity=2)
