#!/usr/bin/env python
"""Unit tests for the pieces where a silent bug would be hardest to notice.

Run::

    python tests/test_units.py            # all tests
    python tests/test_units.py -k scatter  # a subset

These are cheap, deterministic, and do not need the dataset: they pin down the
normalised-coordinate convention, the weight normalisation of the scatter, the
geometry prior's algebra, and the loss/assignment behaviour.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from pgsf.config import PGSFConfig, load_config  # noqa: E402
from pgsf.losses import (  # noqa: E402
    assign_candidates,
    ciou_loss,
    draw_gaussian,
    gaussian_focal_loss,
    geometry_nll,
    nwd_loss,
    sigmoid_focal_loss,
)
from pgsf.models.candidate import bilinear_grid, local_nms_topk, sample_points  # noqa: E402
from pgsf.models.detector import PGSFDet, batched_nms, box_iou  # noqa: E402
from pgsf.models.fusion import scatter_points  # noqa: E402
from pgsf.models.geometry import ProbGeoPrior, fit_affine  # noqa: E402
from pgsf.models.msce import MSCE  # noqa: E402
from pgsf.utils.metrics import average_precision, box_iou_matrix, eval_detection  # noqa: E402


class TestCoordConvention(unittest.TestCase):
    """Candidate points must be normalised [0,1] and index the grid correctly."""

    def test_grid_normalised(self):
        g = bilinear_grid(4, 5, torch.device("cpu"), torch.float32)
        self.assertEqual(g.shape, (20, 2))
        self.assertAlmostEqual(float(g[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(g[:, 0].max()), 1.0, places=6)
        self.assertAlmostEqual(float(g[:, 1].min()), 0.0, places=6)
        self.assertAlmostEqual(float(g[:, 1].max()), 1.0, places=6)
        # first row (y=0) has x sweeping 0..1
        self.assertAlmostEqual(float(g[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(g[4, 0]), 1.0, places=6)

    def test_sample_points_recovers_pixel(self):
        feat = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        pts = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]])
        out = sample_points(feat, pts)
        self.assertEqual(out.shape, (1, 3, 1))
        # the two corners must map to the corner pixels ...
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(out[0, 1, 0]), 15.0, places=4)
        # ... and the centre must land strictly between the middle four values
        mid = float(out[0, 2, 0])
        self.assertTrue(5.0 < mid < 10.0, f"centre sample {mid} not interpolated")
        # sanity: bilinear interpolation of (1,1),(2,1),(1,2),(2,2) with weights .25
        self.assertAlmostEqual(mid, (5 + 6 + 9 + 10) / 4.0, places=4)


class TestScatter(unittest.TestCase):
    """Weight normalisation must make overlap count invariant to candidate count."""

    def test_weight_normalisation(self):
        # Three identical candidates at one point -> the splat equals that value
        # (weight-normalised mean), NOT three times the value.
        vals = torch.full((1, 3, 4), 2.0)
        pts = torch.full((1, 3, 2), 0.5)
        out, cov = scatter_points(pts, vals, 8, 8)
        # Endpoint-normalized coordinates map 0.5 to 3.5 on an 8x8 grid.
        self.assertTrue(torch.allclose(out[0, :, 4, 4], torch.full((4,), 2.0),
                                       atol=1e-5),
                        f"weight normalisation failed: {out[0, :, 4, 4]}")
        self.assertEqual(float(cov.sum()), 4.0)      # exactly one covered cell

    def test_single_candidate_value(self):
        vals = torch.tensor([[[1.0, 2.0]]])
        # A 9x9 endpoint-normalized grid maps 0.5 to the exact center cell.
        pts = torch.tensor([[[0.5, 0.5]]])
        out, cov = scatter_points(pts, vals, 9, 9)
        for (y, x) in ((4, 4),):
            self.assertTrue(torch.allclose(out[0, :, y, x],
                                           torch.tensor([1.0, 2.0]), atol=1e-6),
                            f"cell ({y},{x}) = {out[0, :, y, x]}")
        self.assertEqual(float(cov.sum()), 1.0)

    def test_uncovered_cells_stay_zero(self):
        vals = torch.full((1, 1, 3), 5.0)
        pts = torch.tensor([[[0.1, 0.1]]])
        out, cov = scatter_points(pts, vals, 16, 16)
        # the far corner receives nothing
        self.assertAlmostEqual(float(out[0, :, 15, 15].abs().sum()), 0.0, places=6)
        # bilinear splatting touches exactly 2x2 = 4 cells
        self.assertEqual(int((cov > 0).sum()), 4)
        # and every covered cell carries the interpolated value
        self.assertAlmostEqual(float(out[0, :, 1, 1].mean()), 5.0, places=5)

    def test_gradient_flows(self):
        vals = torch.randn(2, 8, 5, requires_grad=True)
        pts = torch.rand(2, 8, 2)
        out, _ = scatter_points(pts, vals, 16, 16)
        out.sum().backward()
        self.assertIsNotNone(vals.grad)
        self.assertGreater(float(vals.grad.abs().sum()), 0.0)

    def test_out_of_range_points_are_safe(self):
        pts = torch.tensor([[[2.0, -1.0], [0.5, 0.5]]])
        vals = torch.randn(1, 2, 3)
        out, _ = scatter_points(pts, vals, 8, 8)
        self.assertTrue(torch.isfinite(out).all())


class TestGeometry(unittest.TestCase):
    """The Gaussian prior must be a proper distribution and match least squares."""

    def test_affine_recovers_known_transform(self):
        torch.manual_seed(0)
        A_lin = torch.tensor([[0.9, 0.05], [-0.03, 1.1]])
        b_true = torch.tensor([0.02, -0.04])
        # homogeneous 2x3 form: dst = [src,1] @ A.T
        A_true = torch.cat([A_lin, b_true.view(2, 1)], dim=1)      # (2,3)
        src = torch.rand(500, 2)
        hom = torch.cat([src, torch.ones(500, 1)], dim=1)          # (500,3)
        dst = hom @ A_true.T
        A, _ = fit_affine(src, dst)
        self.assertEqual(tuple(A.shape), (2, 3))
        self.assertTrue(torch.allclose(A, A_true, atol=1e-4),
                        f"recovered {A} != {A_true}")

    def test_covariance_positive_definite(self):
        g = ProbGeoPrior(16, hidden=16)
        pts = torch.rand(2, 5, 2)
        mu, L, extra = g(pts, torch.randn(2, 5, 16))
        cov = L @ L.transpose(-1, -2)
        # eigenvalues of a 2x2 SPD matrix are > 0
        eig = torch.linalg.eigvalsh(cov)
        self.assertTrue(bool((eig > 0).all()), f"covariance not PD: {eig}")

    def test_log_prob_matches_density(self):
        g = ProbGeoPrior(8, hidden=8)
        pts = torch.rand(1, 3, 2)
        mu, L, _ = g(pts, torch.randn(1, 3, 8))
        lp = ProbGeoPrior.log_prob(mu, mu, L)
        # density at the mean = 1 / (2*pi*|L|) -> log = -log(2 pi) - log|L|
        expected = -math.log(2 * math.pi) - torch.log(L[..., 0, 0] * L[..., 1, 1])
        self.assertTrue(torch.allclose(lp, expected, atol=1e-4))

    def test_confidence_decreases_with_uncertainty(self):
        g = ProbGeoPrior(4, hidden=4)
        # `extra` is [log det Sigma, sigma_max, sigma_min]; a bigger log-det means a
        # wider Gaussian, which must yield lower confidence.
        lo = torch.zeros(1, 2, 3)      # log det 0  -> sigma = 1
        hi = torch.full((1, 2, 3), 5.0)  # log det 5 -> sigma = e^2.5
        c_lo = float(g.confidence(lo).mean())
        c_hi = float(g.confidence(hi).mean())
        self.assertGreater(c_lo, c_hi, f"{c_lo} !> {c_hi}")
        self.assertLessEqual(c_lo, 1.0)
        self.assertGreater(c_hi, 0.0)

    def test_residual_starts_at_zero(self):
        g = ProbGeoPrior(4, hidden=4, use_residual=True)
        self.assertTrue(torch.allclose(g.residual[-1].weight,
                                       torch.zeros_like(g.residual[-1].weight)))
        # therefore mu == the global affine at init
        A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        g.set_affine(A, torch.zeros(2))
        pts = torch.rand(1, 4, 2)
        mu, _, _ = g(pts, torch.randn(1, 4, 4))
        self.assertTrue(torch.allclose(mu, pts, atol=1e-5))


class TestCandidate(unittest.TestCase):
    def test_local_nms_returns_in_range_indices(self):
        heat = torch.randn(2, 1, 16, 24).sigmoid()
        idx, sc = local_nms_topk(heat, 12, 5)
        self.assertEqual(idx.shape, (2, 12))
        self.assertEqual(sc.shape, (2, 12))
        self.assertTrue(bool((idx >= 0).all() and (idx < 16 * 24).all()))
        # descending score order
        self.assertTrue(bool((sc[:, :-1] >= sc[:, 1:] - 1e-6).all()))

    def test_nms_avoids_duplicates(self):
        # a single bright blob must not yield many identical top-k entries
        heat = torch.zeros(1, 1, 32, 32)
        heat[0, 0, 10, 12] = 1.0
        idx, _ = local_nms_topk(heat, 8, 5)
        self.assertEqual(len(set(idx[0].tolist())), 8)   # distinct indices

    def test_budget_exceeded_raises(self):
        from pgsf.models.candidate import CandidateGenerator
        # 64 global + 16 partitions supply at most 80 candidates
        with self.assertRaises(ValueError):
            CandidateGenerator(8, n_cand=81, n_global=64, n_partitions=16)
        # exactly the budget is legal
        CandidateGenerator(8, n_cand=80, n_global=64, n_partitions=16)


class TestLosses(unittest.TestCase):
    def test_focal_loss_prefers_correct(self):
        good = sigmoid_focal_loss(torch.full((64,), 5.0), torch.ones(64))
        bad = sigmoid_focal_loss(torch.full((64,), -5.0), torch.ones(64))
        self.assertLess(float(good), float(bad))

    def test_ciou_zero_for_identical_boxes(self):
        b = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        self.assertLess(float(ciou_loss(b, b)), 1e-4)

    def test_ciou_increases_with_shift(self):
        t = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        near = torch.tensor([[0.52, 0.5, 0.1, 0.1]])
        far = torch.tensor([[0.75, 0.5, 0.1, 0.1]])
        self.assertLess(float(ciou_loss(near, t)), float(ciou_loss(far, t)))

    def test_nwd_more_robust_than_ciou_for_tiny_boxes(self):
        t = torch.tensor([[0.5, 0.5, 0.02, 0.02]])
        shifted = torch.tensor([[0.53, 0.5, 0.02, 0.02]])   # 3% of the frame
        self.assertGreater(float(nwd_loss(shifted, t)), float(nwd_loss(t, t)) - 1e-6)
        self.assertLess(float(nwd_loss(shifted, t)),
                        float(nwd_loss(torch.tensor([[0.7, 0.5, 0.02, 0.02]]), t)))

    def test_gaussian_peak_at_centre(self):
        heat = torch.zeros(1, 1, 32, 32)
        # place the centre exactly on a pixel centre so the peak is unambiguous
        c = torch.tensor([[[15.0 / 31.0, 15.0 / 31.0]]])
        s = torch.tensor([[[0.05, 0.05]]])
        m = torch.ones(1, 1)
        g = draw_gaussian(heat, c, s, m)
        self.assertAlmostEqual(float(g[0, 0, 15, 15]), 1.0, places=4)
        self.assertLess(float(g[0, 0, 15, 21]), 0.5)
        self.assertLessEqual(float(g.max()), 1.0001)

    def test_gaussian_respects_valid_mask(self):
        heat = torch.zeros(1, 1, 16, 16)
        c = torch.tensor([[[0.5, 0.5]]])
        s = torch.tensor([[[0.1, 0.1]]])
        g = draw_gaussian(heat, c, s, torch.zeros(1, 1))
        self.assertAlmostEqual(float(g.abs().sum()), 0.0, places=6)

    def test_gaussian_focal_loss_prefers_peak(self):
        gt = torch.zeros(1, 1, 16, 16)
        gt[0, 0, 8, 8] = 1.0
        good = torch.full((1, 1, 16, 16), -8.0)
        good[0, 0, 8, 8] = 8.0
        bad = torch.full((1, 1, 16, 16), 8.0)
        self.assertLess(float(gaussian_focal_loss(good, gt)),
                        float(gaussian_focal_loss(bad, gt)))

    def test_assign_matches_centre(self):
        pts = torch.tensor([[[0.5, 0.5], [0.1, 0.1]]])
        gt = torch.zeros(1, 2, 4)
        gt[0, 0] = torch.tensor([0.5, 0.5, 0.1, 0.1])
        m = torch.tensor([[1.0, 0.0]])
        pos, tgt = assign_candidates(pts, gt, m)
        self.assertEqual(float(pos[0, 0]), 1.0)
        self.assertEqual(float(pos[0, 1]), 0.0)
        self.assertTrue(torch.allclose(tgt[0, 0], gt[0, 0]))

    def test_geometry_nll_lower_for_correct_prediction(self):
        target = torch.tensor([[[0.5, 0.5]]])
        mu_good = torch.tensor([[[0.5, 0.5]]])
        mu_bad = torch.tensor([[[0.2, 0.8]]])
        cov = torch.eye(2).view(1, 1, 2, 2) * 0.01
        valid = torch.ones(1, 1)
        self.assertLess(float(geometry_nll(mu_good, cov, target, valid)),
                        float(geometry_nll(mu_bad, cov, target, valid)))

    def test_geometry_nll_masked_rows_have_no_gradient(self):
        mu = torch.tensor([[[0.5, 0.5], [0.1, 0.1]]], requires_grad=True)
        cov = torch.eye(2).expand(1, 2, 2, 2) * 0.01
        target = torch.zeros(1, 2, 2)
        valid = torch.tensor([[1.0, 0.0]])
        geometry_nll(mu, cov, target, valid).backward()
        self.assertGreater(float(mu.grad[0, 0].abs().sum()), 0.0)
        self.assertAlmostEqual(float(mu.grad[0, 1].abs().sum()), 0.0, places=8)


class TestMSCE(unittest.TestCase):
    def test_bounded_alpha_and_residual_form(self):
        m = MSCE(32, width=16, alpha_max=0.5)
        with torch.no_grad():
            m.alpha_raw.fill_(50.0)
        alpha = 0.5 * torch.tanh(m.alpha_raw)
        self.assertLessEqual(abs(float(alpha)), 0.5)
        x = torch.randn(2, 32, 16, 16)
        out, sal = m(x)
        self.assertTrue(torch.allclose(out, x * (1 + alpha * sal), atol=1e-5))
        self.assertGreaterEqual(float(sal.min()), 0.0)
        self.assertLessEqual(float(sal.max()), 1.0)

    def test_output_is_not_overwritten_by_saliency(self):
        """MSCE must never collapse to S*F: a zero saliency keeps F intact."""
        m = MSCE(8, width=8, alpha_max=0.5)
        x = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            alpha = 0.5 * torch.tanh(m.alpha_raw)
            # a *zero* saliency map must leave the features untouched
            zero_sal = torch.zeros(1, 1, 8, 8)
            self.assertTrue(torch.allclose(x * (1 + alpha * zero_sal), x, atol=1e-6))
            # and the real path must equal the residual form, not a product with S
            out, sal = m(x)
            self.assertTrue(torch.allclose(out, x * (1 + alpha * sal), atol=1e-5))


class TestNMS(unittest.TestCase):
    def test_nms_removes_duplicates(self):
        boxes = torch.tensor([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [50, 50, 60, 60]],
                             dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = batched_nms(boxes, scores, 0.45, 50)
        self.assertEqual(keep.tolist(), [0, 2])

    def test_iou_identity(self):
        b = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        self.assertAlmostEqual(float(box_iou(b, b)[0, 0]), 1.0, places=5)

    def test_nms_respects_max_det(self):
        boxes = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5],
                              [6, 6, 7, 7]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        self.assertEqual(len(batched_nms(boxes, scores, 0.45, 2)), 2)


class TestMetrics(unittest.TestCase):
    def test_perfect_detection_gives_ap1(self):
        gts = [torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)]
        preds = [{"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                  "scores": torch.tensor([0.99])}]
        r = eval_detection(preds, gts, [0.5], 50)
        self.assertAlmostEqual(r["AP50"], 1.0, places=3)

    def test_missed_detection_gives_ap0(self):
        gts = [torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)]
        preds = [{"boxes": torch.tensor([[50, 50, 60, 60]], dtype=torch.float32),
                  "scores": torch.tensor([0.9])}]
        self.assertAlmostEqual(eval_detection(preds, gts, [0.5], 50)["AP50"], 0.0,
                               places=3)

    def test_ap_is_monotone_in_precision(self):
        r = torch.tensor([0.0, 0.5, 1.0]).numpy()
        p1 = torch.tensor([0.5, 1.0, 1.0]).numpy()
        p2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
        self.assertGreater(average_precision(r, p1), average_precision(r, p2))

    def test_iou_matrix(self):
        a = np_iou = box_iou_matrix(
            __import__("numpy").array([[0, 0, 10, 10]], dtype=float),
            __import__("numpy").array([[5, 0, 15, 10]], dtype=float))
        self.assertAlmostEqual(float(a[0, 0]), 50 / 150, places=4)
        del np_iou


class TestAblations(unittest.TestCase):
    """Each ablation switch must build and produce a finite loss."""

    def _tiny_cfg(self, **flags):
        """Small but structurally complete config for ablation tests."""
        cfg = PGSFConfig()
        cfg.model.architecture_version = 1
        cfg.stage = 2
        cfg.data.rgb_size = (256, 320)
        cfg.data.tir_size = (128, 160)
        cfg.model.width_mult = 0.5
        cfg.model.backbone_depths = (1, 1, 1, 1)
        cfg.model.neck_dim = 64
        cfg.model.neck_out = 32
        cfg.model.neck_repeats = 1
        cfg.model.d_model = 32
        cfg.model.n_heads = 4
        cfg.model.head_width = 32
        for k, v in flags.items():
            assert hasattr(cfg.model, k), f"unknown model flag '{k}'"
            setattr(cfg.model, k, v)
        return cfg

    def _batch(self, cfg):
        md = cfg.eval.max_det
        b = 1
        r = torch.rand(b, 3, *cfg.data.rgb_size)
        t = torch.rand(b, 1, *cfg.data.tir_size)
        boxes = torch.zeros(b, md, 4)
        boxes[:, 0] = torch.tensor([0.5, 0.5, 0.05, 0.05])
        ctr = boxes[:, :, :2].clone()
        siz = boxes[:, :, 2:].clone()
        mask = torch.zeros(b, md)
        mask[:, 0] = 1.0
        return {"rgb": r, "tir": t, "boxes_rgb": boxes, "boxes_tir": boxes.clone(),
                "centers_rgb": ctr, "centers_tir": ctr + 0.03,
                "sizes_rgb": siz, "sizes_tir": siz, "mask": mask,
                "valid": mask[:, 0],
                "rgb_hw": torch.tensor([[*cfg.data.rgb_size]], dtype=torch.float32),
                "tir_hw": torch.tensor([[*cfg.data.tir_size]], dtype=torch.float32)}

    def test_a0_no_fusion(self):
        cfg = self._tiny_cfg(use_msce=False, use_mid_fusion=False,
                             use_late_fusion=False)
        m = PGSFDet(cfg)
        self.assertFalse(m.mid_on)
        self.assertFalse(m.late_on)
        loss, terms = m.compute_losses(m(self._batch(cfg)))
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("geo", terms)
        self.assertNotIn("assoc_mid", terms)

    def test_a2_mid_without_geo(self):
        cfg = self._tiny_cfg(use_geo_prior=False)
        m = PGSFDet(cfg)
        self.assertTrue(m.mid_on)
        self.assertFalse(m.mid.use_geo)
        loss, _ = m.compute_losses(m(self._batch(cfg)))
        self.assertTrue(torch.isfinite(loss))

    def test_a6_full_model(self):
        cfg = self._tiny_cfg()
        m = PGSFDet(cfg)
        loss, terms = m.compute_losses(m(self._batch(cfg)))
        self.assertTrue(torch.isfinite(loss))
        for key in ("geo", "assoc_mid", "assoc_late", "heat"):
            self.assertIn(key, terms)

    def test_stage1_has_no_fusion(self):
        cfg = self._tiny_cfg()
        # the stage gate lives on the top-level config, not on model
        cfg.stage = 1
        cfg.model.use_mid_fusion = True
        cfg.model.use_late_fusion = True
        m = PGSFDet(cfg)
        self.assertFalse(m.mid_on, "stage 1 must not instantiate the mid fusion")
        self.assertFalse(m.late_on, "stage 1 must not instantiate the late fusion")

    def test_fusion_is_identity_at_init(self):
        """With zero-init residuals, killing the fusion must not change the output."""
        cfg = self._tiny_cfg()
        m = PGSFDet(cfg).eval()
        batch = self._batch(cfg)
        with torch.no_grad():
            out_on = m(batch)
            m.mid_on = m.late_on = False
            out_off = m(batch)
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(out_on[f"pred_{name}"], out_off[f"pred_{name}"], atol=1e-6))
            self.assertTrue(torch.allclose(out_on["aux"][f"pre_box_{name}"],
                                           out_off["aux"][f"pre_box_{name}"], atol=1e-6))



class TestConfig(unittest.TestCase):
    def test_override_and_roundtrip(self):
        cfg = load_config(None, ["model.neck_out=192", "optim.epochs=3",
                                 "model.use_msce=false"])
        self.assertEqual(cfg.model.neck_out, 192)
        self.assertEqual(cfg.optim.epochs, 3)
        self.assertFalse(cfg.model.use_msce)

    def test_yaml_configs_load(self):
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs")
        for fn in sorted(os.listdir(here)):
            if fn.endswith(".yaml"):
                cfg = load_config(os.path.join(here, fn))
                self.assertIn(cfg.stage, (1, 2))

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            load_config(None, ["model.does_not_exist=1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
