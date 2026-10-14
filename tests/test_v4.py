#!/usr/bin/env python
"""v4 behaviour: CFPM matching, TD-Fusion asymmetry, and the v1/v2/v3 compatibility.

Each test pins a *property* that would otherwise only fail after a long run:

* K candidates with a per-level quota, RoI descriptor shapes;
* different resolutions, multi-object, unequal counts, multi-class;
* missing modality, missing instance correspondence;
* GT-free inference (no injection, no affine read);
* the localization branch receives a *channel modulation* of its own RoI only;
* CFPM offsets scale with the anchor box (no camera constant);
* RoIAlign reference agreement and level assignment;
* JS divergence bounds; matching CE;
* v1 / v2 / v3 still build and train in the v4 project;
* (with ``V4_DEEP=1``) a gradient audit over every trainable tensor.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from pgsf.config import PGSFConfig
from pgsf.models.detector import PGSFDet
from pgsf.models.fusion_v4 import AsymmetricFusion
from pgsf.models.head_v4 import (class_probabilities, decode_boxes_scores_v4, decode_pred_v4,
                                 js_divergence)
from pgsf.models.matching_v4 import CoarseToFineMatching, identity_matches
from pgsf.models.proposal_v4 import (assign_levels, roi_align, _roi_align_grid_sample)
from scripts.check_forward import make_fake_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
def config(stage=2, **flags):
    """Small but structurally complete v4 config."""
    c = PGSFConfig()
    c.stage = stage
    c.data.rgb_size = (128, 160)
    c.data.tir_size = (96, 128)
    m = c.model
    m.architecture_version = 4
    m.backbone_name = "lightcsp"
    m.backbone_depths = (1, 1, 1, 1)
    m.width_mult = 0.5
    m.neck_dim = 32
    m.neck_out = 32
    m.neck_repeats = 1
    m.context_dim = 32
    m.proposal_width = 16
    m.msce_width = 16
    m.head_width = 32
    m.n_cand = 64
    m.n_cand_late = 64
    m.cand_min_per_level = 4
    m.roi_size = 3
    m.retrieval_dim = 16
    m.td_top_l = 2
    m.td_samples = 4
    m.td_heads = 4
    m.td_ff_mult = 2
    m.use_mid_fusion = False
    m.use_late_fusion = False
    m.use_geo_prior = False
    m.use_v3_fusion = False
    for k, v in flags.items():
        if not hasattr(m, k):
            raise KeyError(f"unknown model flag '{k}'")
        setattr(m, k, v)
    return c


def multi_batch(cfg, device=DEVICE, rgb=((.5, .5, .06, .06),),
                tir=((.5, .47, .06, .06),), corr=((0, 0),),
                labels_rgb=None, labels_tir=None, available=(1.0, 1.0)):
    """One image with an arbitrary, possibly unequal, number of objects per modality."""
    md = cfg.eval.max_det
    b = 1

    def pack(boxes, labels):
        bb = torch.zeros(b, md, 4)
        mask = torch.zeros(b, md)
        lab = torch.full((b, md), -1, dtype=torch.long)
        for i, box in enumerate(boxes):
            bb[0, i] = torch.tensor(box, dtype=torch.float32)
            mask[0, i] = 1.0
            lab[0, i] = 0 if labels is None else labels[i]
        return bb, mask, lab

    br, mr, lr = pack(rgb, labels_rgb)
    bt, mt, lt = pack(tir, labels_tir)
    out = {
        "rgb": torch.rand(b, 3, *cfg.data.rgb_size, device=device),
        "tir": torch.rand(b, 1, *cfg.data.tir_size, device=device),
        "boxes_rgb": br.to(device), "boxes_tir": bt.to(device),
        "centers_rgb": br[..., :2].clone().to(device),
        "centers_tir": bt[..., :2].clone().to(device),
        "sizes_rgb": br[..., 2:].clone().to(device),
        "sizes_tir": bt[..., 2:].clone().to(device),
        "mask": torch.maximum(mr, mt).to(device),
        "mask_rgb": mr.to(device), "mask_tir": mt.to(device),
        "labels_rgb": lr.to(device), "labels_tir": lt.to(device),
        "valid": torch.tensor([1.0 if (len(rgb) and len(tir)) else 0.0], device=device),
        "rgb_hw": torch.tensor([list(cfg.data.rgb_size)], dtype=torch.float32, device=device),
        "tir_hw": torch.tensor([list(cfg.data.tir_size)], dtype=torch.float32, device=device),
        "modality_available": torch.tensor([list(available)], device=device),
    }
    if corr is not None:
        pairs = torch.zeros(b, md, md)
        for i, j in corr:
            pairs[0, i, j] = 1.0
        out["corr_pairs"] = pairs.to(device)
    return out


# --------------------------------------------------------------------------- #
class ArchitectureContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_one_interaction_only_no_v2_or_v3_modules(self):
        model = PGSFDet(config())
        self.assertTrue(model.v4)
        self.assertFalse(model.v3)
        self.assertFalse(model.v2)
        self.assertFalse(hasattr(model, "mid"))
        self.assertFalse(hasattr(model, "late"))
        self.assertFalse(model._affine_A.requires_grad)     # no fitted affine prior
        self.assertTrue(hasattr(model, "match") and hasattr(model, "td"))
        self.assertTrue(model.v4_on)

    def test_budget_shapes_and_roi_descriptor(self):
        cfg = config(n_cand=128, cand_min_per_level=8)
        model = PGSFDet(cfg).to(DEVICE)
        out = model(multi_batch(cfg))
        for name in ("rgb", "tir"):
            cand, sel = out["cand_" + name], out["sel_" + name]
            self.assertEqual(tuple(cand["pts"].shape), (1, 128, 2))
            self.assertEqual(tuple(cand["boxes"].shape), (1, 128, 4))
            self.assertEqual(tuple(out["roi_" + name].shape), (1, 128, cfg.model.neck_out, 3, 3))
            self.assertEqual(tuple(out["token_" + name].shape), (1, 128, cfg.model.neck_out))
            self.assertEqual(tuple(sel["boxes"].shape), (1, 128, 4))
            self.assertEqual(out["pred_" + name].shape[-1], 4 + cfg.model.num_classes)
            self.assertTrue((cand["boxes"][..., 2:] > 0).all())
            # every level contributes at least the reserved quota when possible
            counts = torch.bincount(cand["levels"][0], minlength=len(cfg.model.cand_levels))
            self.assertGreaterEqual(int(counts.min()), 1)
            self.assertEqual(int(counts.sum()), 128)

    def test_affine_prior_has_no_effect_on_v4(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            before = model(batch)
            model._affine_A.copy_(torch.tensor([[1.6, .35, .2], [-.25, .8, -.1]]))
            after = model(batch)
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(before["pred_" + name], after["pred_" + name],
                                           atol=1e-7))

    def test_different_resolutions_run_end_to_end(self):
        for rgb_size, tir_size in (((128, 160), (96, 128)), ((192, 256), (128, 160)),
                                   ((64, 64), (48, 48))):
            cfg = config()
            cfg.data.rgb_size = rgb_size
            cfg.data.tir_size = tir_size
            model = PGSFDet(cfg).to(DEVICE)
            loss, terms = model.compute_losses(model(multi_batch(cfg)))
            self.assertTrue(torch.isfinite(loss), f"{rgb_size}/{tir_size}: {loss}")
            for key in ("match", "cons", "rgb_loss", "tir_loss", "rgb_self", "rgb_prop"):
                self.assertIn(key, terms)
                self.assertTrue(torch.isfinite(terms[key]), key)


class FusionAsymmetry(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_localization_receives_only_a_channel_modulation(self):
        """``F_reg`` must be ``S`` scaled *per channel* -- no additive cross-modal map."""
        # (a) the operator itself, on synthetic inputs where the gain is macroscopic
        fuse = AsymmetricFusion(dim=8, alpha_init=0.5)
        roi = torch.randn(2, 3, 8, 3, 3)
        r = torch.rand(2, 3)
        gamma = torch.randn(2, 3, 8)
        modulated = fuse.modulate(roi, r, gamma)
        expected = roi * (1.0 + 0.5 * r[..., None] * gamma)[..., None, None]
        self.assertTrue(torch.allclose(modulated, expected, atol=1e-6))
        self.assertGreater(float((modulated - roi).abs().max()), 1e-2)
        # rank-1 per channel: exactly one scalar per (batch, candidate, channel)
        flat_roi = roi.reshape(2, 3, 8, -1)
        flat_out = modulated.reshape(2, 3, 8, -1)
        idx = flat_roi.abs().argmax(-1, keepdim=True)
        gain = flat_out.gather(-1, idx) / flat_roi.gather(-1, idx)
        self.assertTrue(torch.allclose(flat_out, flat_roi * gain, rtol=1e-5, atol=1e-6),
                        "localization gained a cross-modal spatial pattern")
        # (b) end to end: the fused box head consumed exactly that modulated RoI
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            out = model(batch)
            for name, key in (("rgb", "r2t"), ("tir", "t2r")):
                mod = model.td[key].modulate(out["roi_" + name], out["aux"]["r_" + name],
                                             out["aux"]["gamma_" + name])
                box = getattr(model, name + "_branch").reg(mod, out["sel_" + name]["boxes"])
                self.assertTrue(torch.equal(box, out["pred_" + name][..., :4]))

    def test_gate_and_modulation_are_identity_at_initialisation(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            fused = model(batch)
            model.v4_on = False
            independent = model(batch)
            model.v4_on = True
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(fused["pred_" + name], independent["pred_" + name],
                                           atol=1e-6))
            self.assertTrue(torch.allclose(fused["pred_" + name],
                                           fused["self_pred_" + name], atol=1e-6))

    def test_missing_other_modality_behaves_like_a_single_modal_detector(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg, available=(1.0, 0.0))
        with torch.no_grad():
            out = model(batch)
        self.assertTrue(torch.allclose(out["pred_rgb"], out["self_pred_rgb"], atol=1e-6))
        self.assertEqual(float(out["aux"]["r_rgb"].abs().max()), 0.0)

    def test_classification_does_change_with_the_other_modality(self):
        """The fused classifier must actually use the cross-modal semantics."""
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        with torch.no_grad():
            for d in ("r2t", "t2r"):
                model.td[d].w_cls.weight.normal_(0, .2)
                model.td[d].w_cls.bias.normal_(0, .1)
                model.td[d].gate[-1].bias.fill_(2.0)
        a = multi_batch(cfg)
        with torch.no_grad():
            fused = model(a)
            model.v4_on = False
            independent = model(a)
            model.v4_on = True
        self.assertFalse(torch.allclose(fused["pred_rgb"][..., 4:],
                                        independent["pred_rgb"][..., 4:], atol=1e-6))

    def test_cfpm_offsets_scale_with_the_anchor_box(self):
        """Sampling offsets follow the anchor box size, so no camera constant exists."""
        matcher = CoarseToFineMatching(dim=16, retrieval_dim=8, top_l=2, samples=4,
                                       heads=4, ff_mult=2)
        with torch.no_grad():
            matcher.offset[-1].bias.normal_(0, .5)
        z = torch.randn(1, 4, 16)
        boxes = torch.tensor([[[.3, .3, .1, .1], [.7, .7, .1, .1],
                               [.4, .6, .1, .1], [.6, .4, .1, .1]]])
        maps = {n: torch.randn(1, 16, 8, 8) for n in ("P2", "P3", "P4", "P5")}
        _, s1 = matcher(z, z, boxes, maps)
        centre = boxes[..., :2]
        scaled = torch.cat((centre, boxes[..., 2:] * 2.0), -1)
        _, s2 = matcher(z, z, scaled, maps)
        anchor = s1["anchor_idx"]
        for b in range(1):
            for q in range(anchor.shape[1]):
                a = anchor[b, q, 0]
                d1 = (s1["local_pts"][b, q, 0] - boxes[b, a, :2]).abs().mean()
                d2 = (s2["local_pts"][b, q, 0] - boxes[b, a, :2]).abs().mean()
                if float(d1) > 1e-6:
                    self.assertAlmostEqual(float(d2 / d1), 2.0, places=4)

    def test_retrieval_is_appearance_only(self):
        """The Top-L anchors must not depend on any coordinate."""
        matcher = CoarseToFineMatching(dim=16, retrieval_dim=8, top_l=2, samples=4,
                                       heads=4, ff_mult=2)
        z = torch.randn(1, 4, 16)
        boxes = torch.tensor([[[.3, .3, .1, .1], [.7, .7, .1, .1],
                               [.4, .6, .1, .1], [.6, .4, .1, .1]]])
        maps = {n: torch.randn(1, 16, 8, 8) for n in ("P2", "P3", "P4", "P5")}
        _, first = matcher(z, z, boxes, maps)
        _, moved = matcher(z, z, boxes + 0.37, maps)
        self.assertTrue(torch.equal(first["anchor_idx"], moved["anchor_idx"]))
        sim = matcher.similarity(z, z)
        self.assertEqual(tuple(sim.shape), (1, 4, 4))
        self.assertGreater(float(sim.std()), 0.0)


class MatchingAndConsistency(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_identity_matches_do_not_assume_the_same_index(self):
        q_boxes = torch.tensor([[[.2, .2, .1, .1], [.8, .8, .1, .1], [.5, .9, .1, .1]]])
        k_boxes = torch.tensor([[[.6, .3, .1, .1], [.25, .25, .1, .1]]])
        pairs = torch.zeros(1, 3, 2)
        pairs[0, 0, 1] = 1
        pairs[0, 1, 0] = 1
        match = identity_matches(q_boxes[..., :2], q_boxes, torch.ones(1, 3),
                                 k_boxes[..., :2], k_boxes, torch.ones(1, 2), pairs)
        self.assertEqual(int(match.sum()), 2)
        self.assertFalse(bool(match[0, 2].any()))
        perm = torch.tensor([1, 0])
        swapped = identity_matches(q_boxes[..., :2], q_boxes, torch.ones(1, 3),
                                   k_boxes[..., :2][:, perm], k_boxes[:, perm],
                                   torch.ones(1, 2), pairs[:, :, perm])
        self.assertTrue(torch.equal(swapped, match[:, :, perm]))

    def test_matching_ce_rewards_the_correspondent(self):
        from pgsf.models.detector_v4 import matching_loss
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        matches = torch.zeros(1, 4, 4, dtype=torch.bool)
        matches[0, 0, 2] = True
        matches[0, 1, 0] = True
        good = torch.zeros(1, 4, 4)
        good[0, 0, 2] = 5.0
        good[0, 1, 0] = 5.0
        bad = torch.zeros(1, 4, 4)
        bad[0, 0, 3] = 5.0
        bad[0, 1, 1] = 5.0
        self.assertLess(float(matching_loss(model, good, matches)),
                        float(matching_loss(model, bad, matches)))
        self.assertEqual(float(matching_loss(model, good, torch.zeros_like(matches))), 0.0)

    def test_js_divergence_bounds_and_symmetry(self):
        p = torch.tensor([[.7, .3]])
        q = torch.tensor([[.2, .8]])
        self.assertAlmostEqual(float(js_divergence(p, p)), 0.0, places=6)
        v1, v2 = float(js_divergence(p, q)), float(js_divergence(q, p))
        self.assertAlmostEqual(v1, v2, places=6)
        self.assertGreater(v1, 0.0)
        self.assertLessEqual(v1, math.log(2) + 1e-6)

    def test_class_probabilities_single_and_multi(self):
        one = class_probabilities(torch.tensor([[[0.0]]]), 1)
        self.assertTrue(torch.allclose(one.sum(-1), torch.ones(1, 1)))
        self.assertAlmostEqual(float(one[0, 0, 0]), 0.5, places=6)
        three = class_probabilities(torch.zeros(1, 2, 3), 3)
        self.assertTrue(torch.allclose(three.sum(-1), torch.ones(1, 2), atol=1e-6))

    def test_no_correspondence_labels_means_no_match_loss(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg, rgb=((.2, .2, .1, .1), (.8, .25, .1, .1)),
                            tir=((.5, .5, .1, .1),), corr=None)
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms["match"]), 0.0)
        self.assertEqual(float(terms["cons"]), 0.0)
        self.assertGreater(float(terms["rgb_loss"]), 0.0)

    def test_switches_turn_the_extra_losses_off(self):
        cfg = config(use_match_loss=False, use_cons_loss=False)
        model = PGSFDet(cfg).to(DEVICE)
        loss, terms = model.compute_losses(model(multi_batch(cfg)))
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("match", terms)
        self.assertNotIn("cons", terms)

    def test_consistency_uses_mutual_selection_after_warmup(self):
        """Two mutually-selected candidates are paired; before warmup nothing is."""
        from pgsf.models.detector_v4 import consistency_loss
        logits = torch.randn(1, 2, 1)
        sim = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
        cold = PGSFDet(config(cons_start_fraction=0.5))
        self.assertEqual(float(consistency_loss(cold, logits, logits, sim, sim, None, 0.0)), 0.0)
        warm = PGSFDet(config(cons_start_fraction=0.0, cons_margin=-1e9))
        loss = consistency_loss(warm, logits, logits, sim, sim, None, 0.9)
        self.assertGreater(float(loss), 0.0)


class ObjectsAndClasses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_multi_object_unequal_counts(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg,
                            rgb=((.2, .2, .1, .1), (.8, .25, .1, .1), (.5, .85, .1, .1)),
                            tir=((.6, .3, .1, .1), (.25, .3, .1, .1)),
                            corr=((0, 1), (1, 0)))
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms["match"]), 0.0)
        for name in ("rgb", "tir"):
            self.assertGreater(float(terms[name + "_prop"]), 0.0)
            self.assertGreater(float(terms[name + "_self"]), 0.0)

    def test_multi_class_head_and_decode(self):
        cfg = config(num_classes=3)
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg, rgb=((.3, .3, .1, .1), (.7, .7, .1, .1)),
                            tir=((.3, .3, .1, .1), (.7, .7, .1, .1)),
                            labels_rgb=(0, 2), labels_tir=(1, 2))
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(out["pred_rgb"].shape[-1], 7)
        boxes, scores, labels = decode_boxes_scores_v4(
            out["pred_rgb"], out["sel_rgb"]["boxes"], batch["rgb_hw"],
            out["sel_rgb"]["objectness"], cfg.model.score_fusion_power, cfg.model.max_wh)
        self.assertEqual(tuple(boxes.shape), (1, cfg.model.n_cand, 4))
        self.assertTrue(((labels >= 0) & (labels < 3)).all())
        model.train()
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms["rgb_qfl"]), 0.0)

    def test_missing_modality_supervises_only_the_present_one(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg, available=(1.0, 0.0))
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        for key in ("loss", "qfl", "iou", "l1", "self", "prop", "heat"):
            self.assertEqual(float(terms["tir_" + key]), 0.0, key)
        self.assertGreater(float(terms["rgb_loss"]), 0.0)
        self.assertGreater(float(terms["rgb_prop"]), 0.0)

    def test_decoder_matches_the_coarse_box_at_init(self):
        base = torch.tensor([[[.3, .4, .02, .03]]])
        raw = torch.zeros(1, 1, 5)
        boxes, logits = decode_pred_v4(raw, base)
        self.assertTrue(torch.equal(boxes, base))
        raw[..., :2] = 1.0
        boxes, _ = decode_pred_v4(raw, base)
        self.assertTrue(((boxes[..., :2] - base[..., :2]).abs() < base[..., 2:] / 2).all())


class RoIAlignContract(unittest.TestCase):
    def test_both_roi_align_paths_reproduce_a_linear_map(self):
        """Bilinear RoIAlign is exact on a linear ramp, so it has a closed-form answer.

        The ramp is defined in *continuous* pixel coordinates (pixel ``i`` covers
        ``[i, i+1)``), which is the convention ``roi_align(aligned=True)`` uses, so
        the value at a bin centre ``(x, y)`` is exactly ``0.5 x + 0.25 y``.
        """
        torch.manual_seed(3)
        h, w = 12, 16
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        ramp = (0.5 * (xs + 0.5) + 0.25 * (ys + 0.5)).float()
        feat = ramp.expand(2, 4, h, w).contiguous()
        rois = torch.tensor([[0., 1.5, 2.0, 9.0, 8.0],
                             [1., 3.0, 1.0, 12.0, 10.0]])
        out = 3
        step = (torch.arange(out, dtype=torch.float32) + 0.5) / out
        expected = []
        for roi in rois:
            x0, y0, x1, y1 = roi[1:]
            px = x0 + (x1 - x0) * step
            py = y0 + (y1 - y0) * step
            expected.append(0.5 * px[None, :] + 0.25 * py[:, None])
        expected = torch.stack(expected)[:, None]                            # (N,1,out,out)
        ref = roi_align(feat, rois, out, 2)
        alt = _roi_align_grid_sample(feat, rois, out)
        self.assertTrue(torch.allclose(ref, expected.expand_as(ref), atol=1e-4),
                        f"reference path max diff {float((ref - expected).abs().max())}")
        self.assertTrue(torch.allclose(alt, expected.expand_as(alt), atol=1e-4),
                        f"fallback path max diff {float((alt - expected).abs().max())}")
        self.assertTrue(torch.isfinite(ref).all() and torch.isfinite(alt).all())

    def test_level_assignment_follows_box_area(self):
        shapes = [(32, 40), (16, 20), (8, 10), (4, 5)]
        tiny = assign_levels(torch.tensor([[[.5, .5, .01, .01]]]), shapes)[0, 0]
        mid = assign_levels(torch.tensor([[[.5, .5, .30, .30]]]), shapes)[0, 0]
        huge = assign_levels(torch.tensor([[[.5, .5, .80, .80]]]), shapes)[0, 0]
        self.assertEqual(int(tiny), 0)
        self.assertLessEqual(int(tiny), int(mid))
        self.assertLessEqual(int(mid), int(huge))
        self.assertEqual(int(huge), 3)


class InferenceContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_inference_reads_no_ground_truth(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = {"rgb": torch.rand(1, 3, *cfg.data.rgb_size, device=DEVICE),
                 "tir": torch.rand(1, 1, *cfg.data.tir_size, device=DEVICE),
                 "rgb_hw": torch.tensor([[1080., 1920.]], device=DEVICE),
                 "tir_hw": torch.tensor([[512., 640.]], device=DEVICE)}
        with torch.no_grad():
            detections, out = model.inference(batch)
        for name in ("rgb", "tir"):
            self.assertIn("boxes", detections[name][0])
            self.assertIn("scores", detections[name][0])
            self.assertIn("labels", detections[name][0])
            if detections[name][0]["boxes"].numel():
                limit = float(batch[name + "_hw"].max())
                self.assertLessEqual(float(detections[name][0]["boxes"].max()), limit + 1e-3)
            # no GT injection is possible without annotations
            self.assertEqual(float(out["cand_" + name]["injected"].abs().sum()), 0.0)

    def test_inference_is_invariant_to_the_frozen_affine(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = {"rgb": torch.rand(1, 3, *cfg.data.rgb_size, device=DEVICE),
                 "tir": torch.rand(1, 1, *cfg.data.tir_size, device=DEVICE),
                 "rgb_hw": torch.tensor([[1080., 1920.]], device=DEVICE),
                 "tir_hw": torch.tensor([[512., 640.]], device=DEVICE)}
        with torch.no_grad():
            first = model(batch)["pred_rgb"]
            model._affine_A.copy_(torch.tensor([[2.0, .5, .3], [-.4, 1.3, -.2]]))
            second = model(batch)["pred_rgb"]
        self.assertTrue(torch.equal(first, second))

    def test_v4_stage_transfer_preserves_initial_detection(self):
        cfg1, cfg2 = config(1), config(2)
        m1 = PGSFDet(cfg1).to(DEVICE).eval()
        m2 = PGSFDet(cfg2).to(DEVICE).eval()
        m2.load_checkpoint_weights(m1.state_dict(), stage1_transfer=True)
        batch = multi_batch(cfg1)
        with torch.no_grad():
            a, z = m1(batch), m2(batch)
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(a["pred_" + name], z["pred_" + name], atol=1e-6))


class LegacyArchitectures(unittest.TestCase):
    """v1 / v2 / v3 must keep working inside the v4 project."""

    def _tiny(self, arch):
        c = config()
        c.model.architecture_version = arch
        c.model.use_v3_fusion = arch >= 3
        c.model.use_td_fusion = arch >= 4
        if arch == 2:
            c.model.use_mid_fusion = True
            c.model.use_late_fusion = True
            c.model.use_geo_prior = True
        return c

    def test_all_versions_build_and_train(self):
        torch.manual_seed(29)
        for arch in (1, 2, 3, 4):
            cfg = self._tiny(arch)
            model = PGSFDet(cfg).to(DEVICE)
            self.assertEqual(model.arch, arch)
            batch = make_fake_batch(cfg, DEVICE, 1)
            loss, terms = model.compute_losses(model(batch))
            self.assertTrue(torch.isfinite(loss), f"arch={arch}")
            model.inference({k: batch[k] for k in ("rgb", "tir", "rgb_hw", "tir_hw")})


class DeepGradientAudit(unittest.TestCase):
    """Every trainable v4 tensor must receive a gradient (opt-in: slow).

    The optimizer is AdamW -- the one the v4 configs actually use.  That matters:
    TD-Fusion's projection gates are zero-initialised so the model starts as two
    independent detectors, and under plain SGD the resulting products are so small
    that the deep local-probing branch needs tens of steps to open.  Adam's
    per-parameter normalisation opens it immediately, which is the behaviour the
    real training relies on; a parameter that stays dead here is a wiring bug.
    """

    @unittest.skipUnless(os.environ.get("V4_DEEP") == "1",
                         "set V4_DEEP=1 to run the full gradient audit")
    def test_every_parameter_receives_a_gradient(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        def dead_now():
            return [n for n, p in model.named_parameters()
                    if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]

        history = []
        for step in range(6):
            opt.zero_grad(set_to_none=True)
            loss, _ = model.compute_losses(model(batch))
            loss.backward()
            history.append(len(dead_now()))
            opt.step()
        opt.zero_grad(set_to_none=True)
        loss, _ = model.compute_losses(model(batch))
        loss.backward()
        dead = dead_now()
        print(f"  dead parameters per step: {history} -> {len(dead)}")
        self.assertEqual(dead, [], f"parameters without gradient: {dead}")
        self.assertEqual(history[-1], 0, f"the audit expects a fully open graph by the "
                                        f"last step, reached {history[-1]}")


class CudaPath(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_amp_forward_backward(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg)
        with torch.amp.autocast("cuda", enabled=True):
            out = model(batch)
            loss, terms = model.compute_losses(out)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        alive = [n for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(alive, "no parameter received a gradient under AMP")
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), n)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_full_resolution_memory_budget(self):
        cfg = config()
        cfg.data.rgb_size = (800, 1280)
        cfg.data.tir_size = (512, 640)
        cfg.model.n_cand = 128
        cfg.model.cand_min_per_level = 8
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = {"rgb": torch.rand(1, 3, *cfg.data.rgb_size, device=DEVICE),
                 "tir": torch.rand(1, 1, *cfg.data.tir_size, device=DEVICE),
                 "rgb_hw": torch.tensor([[1080., 1920.]], device=DEVICE),
                 "tir_hw": torch.tensor([[512., 640.]], device=DEVICE)}
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model.inference(batch)
        peak = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 20
        self.assertLess(peak, 2048.0, f"inference peak {peak:.0f} MiB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
