"""Numeric semantics, not just tensor shapes or existence of gradients."""
import unittest
import torch
from pgsf.losses import ciou_loss,geometry_nll,correspondence_mask,supervised_association_loss
from pgsf.models.head import MultiScaleBoxRefine
from pgsf.models.geometry import ProbGeoPrior
from pgsf.models.candidate import local_nms_topk
from pgsf.models.fusion_v2 import CovarianceMidFusion


class AuditRegressions(unittest.TestCase):
    def test_ciou_is_symmetric_under_vertical_reflection_and_axis_swap(self):
        target=torch.tensor([[.5,.5,.02,.02]])
        up=target.clone();up[:,1]-=.01
        down=target.clone();down[:,1]+=.01
        a=ciou_loss(up,target);b=ciou_loss(down,target)
        self.assertTrue(torch.allclose(a,b,atol=1e-5))
        self.assertTrue(torch.allclose(b,ciou_loss(down[:,[1,0,3,2]],target[:,[1,0,3,2]]),atol=1e-5))
        self.assertGreater(float(b),.66)
        self.assertLess(float(ciou_loss(target,target)),1e-5)

    def test_ciou_half_precision_inputs_are_computed_in_fp32(self):
        b=torch.tensor([[.5,.5,.001,.001]],dtype=torch.float16)
        loss=ciou_loss(b,b)
        self.assertEqual(loss.dtype,torch.float32)
        self.assertLess(float(loss),1e-4)

    def test_roi_numeric_channels_and_candidate_order(self):
        module=MultiScaleBoxRefine(4,1,8).eval()
        y,x=torch.meshgrid(torch.linspace(0,1,12),torch.linspace(0,1,16),indexing='ij')
        maps={'P2':torch.stack((x,y,x*y,x.square()+y.square()))[None]}
        pts=torch.tensor([[[.2,.3],[.7,.8]]]);boxes=torch.cat((pts,torch.full_like(pts,.1)),-1)
        with torch.no_grad():
            a=module(maps,pts,boxes)
            permuted=module(maps,pts.flip(1),boxes.flip(1)).flip(1)
            individual=torch.cat([module(maps,pts[:,i:i+1],boxes[:,i:i+1]) for i in range(2)],1)
        self.assertTrue(torch.allclose(a,permuted,atol=1e-6))
        self.assertTrue(torch.allclose(a,individual,atol=1e-6))

    def test_invalid_local_key_selects_null_and_has_gradient(self):
        pts=torch.tensor([[[.005,.5]]]);key=torch.tensor([[[-.001,.5]]])
        gt=torch.tensor([[[.005,.5,.02,.02]]])
        matches=correspondence_mask(pts,key,gt,gt,torch.ones(1,1),torch.ones(1))
        self.assertFalse(bool(matches.any()))
        logits=torch.tensor([[[0.,1.]]],requires_grad=True)
        loss=supervised_association_loss(logits.softmax(-1),matches);loss.backward()
        self.assertLess(float(logits.grad[...,-1]),0.)

    def test_covariance_floor_bounds_eigenvalues_with_large_correlation(self):
        module=ProbGeoPrior(8)
        with torch.no_grad():module.unc[-1].bias[:]=torch.tensor([-3.,2.,-3.])
        _,L,_=module(torch.rand(1,1,2),torch.zeros(1,1,8))
        eig=torch.linalg.eigvalsh(L@L.transpose(-1,-2))
        self.assertGreaterEqual(float(eig.min()),module.sigma_floor**2-1e-6)

    def test_geometry_shift_is_nonnegative_and_preserves_gradients(self):
        mu=torch.tensor([[[.5,.5]]],requires_grad=True);target=mu.detach()+.01
        sigma=torch.eye(2)[None,None]*(.03**2);sigma.requires_grad_()
        valid=torch.ones(1,1)
        raw=geometry_nll(mu,sigma,target,valid)
        shifted=geometry_nll(mu,sigma,target,valid,minimum_variance=.02**2)
        a=torch.autograd.grad(raw,(mu,sigma),retain_graph=True)
        b=torch.autograd.grad(shifted,(mu,sigma))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assertGreaterEqual(float(shifted),0.)
        zero=geometry_nll(mu.detach(),torch.eye(2)[None,None]*.02**2,mu.detach(),valid,minimum_variance=.02**2)
        self.assertLess(abs(float(zero)),1e-6)

    def test_peak_fill_does_not_discard_lower_isolated_peak(self):
        heat=torch.linspace(.001,.01,256).reshape(1,1,16,16)
        heat[0,0,8,8]=.9;heat[0,0,8,9]=.8;heat[0,0,3,3]=.5
        idx,scores=local_nms_topk(heat,64,5)
        self.assertIn(3*16+3,idx[0,:3].tolist())
        self.assertNotIn(8*16+9,idx[0,:3].tolist())
        self.assertEqual(len(torch.unique(idx)),64)
        self.assertTrue((scores[:,:-1]>=scores[:,1:]).all())

    def test_proposal_ranking_does_not_gate_center_recall_by_quality(self):
        from pgsf.models.proposal import MultiScaleCandidates
        module=MultiScaleCandidates(4,n_cand=64)
        dense={}; maps={}
        for name,h,w in [('P2',8,8),('P3',4,4)]:
            raw=torch.zeros(1,6,h,w);raw[:,0]=-2.;raw[:,5]=-8.
            raw[0,0,h//2,w//2]=5.;raw[0,5,h//2,w//2]=-8.
            dense[name]=raw;maps[name]=torch.randn(1,4,h,w)
        out=module(dense,maps)
        self.assertGreater(float(out['objectness'].max()),.9)
        # The quality logit is near zero, but the center candidate remains rankable.
        self.assertGreater(float(out['scores'].max()),.9)
        # Reversing the quality ordering must not alter top-k/partition positions.
        altered={name:raw.clone() for name,raw in dense.items()}
        for raw in altered.values():raw[:,5]=torch.randn_like(raw[:,5])*10
        changed=module(altered,maps)
        self.assertTrue(torch.equal(out['pts'],changed['pts']))
        self.assertTrue(torch.equal(out['scores'],changed['scores']))

    def test_search_jitter_does_not_change_supervised_mean(self):
        module=CovarianceMidFusion(16,4,context_dim=8)
        q=torch.randn(1,1,16);pts=torch.full((1,1,2),.5)
        maps={'P2':torch.randn(1,16,16,16),'P3':torch.randn(1,16,8,8)}
        globals=(torch.randn(1,16,16),torch.rand(1,16,2),torch.ones(1,16))
        plain=module(q,pts,maps,globals);shifted=module(q,pts,maps,globals,prior_jitter=torch.ones(1,1,2)*.1)
        self.assertTrue(torch.equal(plain['stats']['mu'],shifted['stats']['mu']))
        self.assertTrue(torch.allclose(shifted['stats']['local_pts']-plain['stats']['local_pts'],torch.full((1,1,8,2),.1),atol=1e-6))

    def test_fusion_terms_have_zero_lower_bound_and_late_only_geometry_trains(self):
        from test_v2 import config
        from scripts.check_forward import make_fake_batch
        from pgsf.models.detector import PGSFDet
        for mid in (True,False):
            c=config();c.model.use_mid_fusion=mid
            model=PGSFDet(c);batch=make_fake_batch(c,torch.device('cpu'),2)
            loss,terms=model.compute_losses(model(batch));loss.backward()
            for key in ('geo','size_geo','assoc_mid','assoc_late'):
                if key in terms:self.assertGreaterEqual(float(terms[key]),-1e-6)
            if not mid:
                self.assertGreater(float(model.late.geo.unc[-1].weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
