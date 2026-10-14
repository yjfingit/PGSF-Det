#!/usr/bin/env python
"""v3 behaviour: 128-candidate object-level fusion, rejection, and data contract.

The tests target the failure modes that would otherwise show up only after a long
training run, so each one pins down a *property*, not a shape:

* resolution independence, multi-object, multi-class, unequal object counts;
* missing modality and missing instance correspondence;
* GT-free inference;
* positive-definite anchor covariance;
* rejection is the identity at initialisation, and the fitted affine is gone;
* v2 still runs (compatibility);
* (with ``--deep``) a gradient audit over every trainable tensor.
"""

from __future__ import annotations

import json
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

from pgsf.config import PGSFConfig
from pgsf.data.uavcb import Frame, PairAugment, UAVCBDataset, collate, list_frames
from pgsf.models.correspondence_v3 import (BidirectionalCorrespondence,
                                           GaussianAnchorHead, identity_matches)
from pgsf.models.detector import PGSFDet
from pgsf.models.head_v3 import decode_boxes_scores_v3, decode_pred_v3
from pgsf.models.proposal_v3 import GridTokens
from scripts.check_forward import make_fake_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
def config(stage=2, **flags):
    """Small but structurally complete v3 config."""
    c = PGSFConfig()
    c.stage = stage
    c.data.rgb_size = (128, 160)
    c.data.tir_size = (96, 128)
    m = c.model
    m.architecture_version = 3
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
    m.n_cand = 128
    m.n_cand_late = 128
    m.retrieval_dim = 16
    m.roi_width = 16
    m.corr_heads = 4
    m.corr_ff_mult = 2
    m.v3_grid = 8
    m.v3_grid_levels = ("P3", "P4")
    m.use_mid_fusion = False
    m.use_late_fusion = False
    m.use_geo_prior = False
    m.geo_jitter_prob = 0.0
    for k, v in flags.items():
        if not hasattr(m, k):
            raise KeyError(f"unknown model flag '{k}'")
        setattr(m, k, v)
    return c


def multi_batch(cfg, device=DEVICE, rgb=((0.5, 0.5, 0.06, 0.06),),
                tir=((0.5, 0.47, 0.06, 0.06),), corr=((0, 0),),
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
class V3Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_one_fusion_only_and_no_v2_modules(self):
        model = PGSFDet(config())
        self.assertTrue(model.v3)
        self.assertFalse(model.v2)
        # the v2 fixed-affine prior is frozen and never optimised
        self.assertFalse(model._affine_A.requires_grad)
        # the two-stage v2 machinery is gone from the v3 graph
        self.assertFalse(hasattr(model, "mid"))
        self.assertFalse(hasattr(model, "late"))
        self.assertTrue(hasattr(model, "corr"))
        self.assertTrue(model.v3_on)

    def test_candidate_and_token_budget(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        out = model(multi_batch(cfg))
        for name in ("rgb", "tir"):
            self.assertEqual(tuple(out["cand_" + name]["pts"].shape), (1, 128, 2))
            self.assertEqual(tuple(out["sel_" + name]["feats"].shape), (1, 128, cfg.model.neck_out))
            self.assertEqual(tuple(out["sel_" + name]["roi"].shape), (1, 128, cfg.model.roi_width))
            self.assertEqual(tuple(out["grid_" + name][0].shape), (1, 64, cfg.model.neck_out))
            self.assertEqual(tuple(out["grid_" + name][1].shape), (64, 2))
            # 4 box channels + num_classes logits
            self.assertEqual(out["pred_" + name].shape[-1], 4 + cfg.model.num_classes)

    def test_grid_tokens_are_resolution_independent(self):
        grid = GridTokens(8, 4, ("P3", "P4"))
        a, pts_a = grid({"P3": torch.randn(1, 8, 10, 12), "P4": torch.randn(1, 8, 5, 6)})
        b, pts_b = grid({"P3": torch.randn(1, 8, 20, 24), "P4": torch.randn(1, 8, 10, 12)})
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(pts_a, pts_b))
        self.assertTrue(((pts_a > 0) & (pts_a < 1)).all())

    def test_different_resolutions_run_end_to_end(self):
        for rgb_size, tir_size in (((128, 160), (96, 128)), ((192, 256), (128, 160)),
                                   ((64, 64), (48, 48))):
            cfg = config()
            cfg.data.rgb_size = rgb_size
            cfg.data.tir_size = tir_size
            model = PGSFDet(cfg).to(DEVICE)
            batch = multi_batch(cfg)
            loss, terms = model.compute_losses(model(batch))
            self.assertTrue(torch.isfinite(loss), f"{rgb_size}/{tir_size}: {loss}")
            self.assertEqual(tuple(batch["rgb"].shape), (1, 3, *rgb_size))
            self.assertEqual(tuple(batch["tir"].shape), (1, 1, *tir_size))

    def test_rejection_fusion_is_identity_at_initialisation(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            fused = model(batch)
            model.v3_on = False
            independent = model(batch)
            model.v3_on = True
        for name in ("rgb", "tir"):
            delta = (fused["pred_" + name] - independent["pred_" + name]).abs().max()
            self.assertLess(float(delta), 1e-6)

    def test_fitted_affine_has_no_effect_on_v3_predictions(self):
        """The v2 prior is *removed*, not merely disabled: changing A changes nothing."""
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            before = model(batch)
            model._affine_A.copy_(torch.tensor([[1.7, 0.4, 0.21], [-0.3, 0.8, -0.11]]))
            after = model(batch)
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(before["pred_" + name], after["pred_" + name],
                                           atol=1e-7))

    def test_heads_read_only_their_own_modality(self):
        """Structural proof that a box cannot be copied across modalities.

        (a) each head is a pure function of its own modality's tensors;
        (b) the fusion only *enhances tokens* -- the coarse box used as the
            regression base stay exactly the modality's own proposal, so no
            counterpart coordinate can leak into the box equation;
        (c) the decoded box lives in its own pixel frame.
        """
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg)
        with torch.no_grad():
            out = model(batch)
            for name in ("rgb", "tir"):
                sel = out["sel_" + name]
                direct = getattr(model, name + "_branch").det(
                    sel["feats"], sel["roi"], sel["boxes"])
                self.assertTrue(torch.equal(direct, out["pred_" + name]),
                                f"{name} head reads something outside its own modality")
            fused_boxes = {n: out["sel_" + n]["boxes"].clone() for n in ("rgb", "tir")}
            model.v3_on = False
            plain = model(batch)
            model.v3_on = True
        for name in ("rgb", "tir"):
            self.assertTrue(torch.equal(fused_boxes[name], plain["sel_" + name]["boxes"]),
                            f"{name} coarse box changed because of the other modality")
        with torch.no_grad():
            boxes, scores, labels = decode_boxes_scores_v3(
                out["pred_tir"], out["sel_tir"]["boxes"], batch["tir_hw"],
                out["sel_tir"]["objectness"], cfg.model.score_fusion_power, cfg.model.max_wh)
            self.assertEqual(tuple(boxes.shape), (1, 128, 4))
            self.assertEqual(tuple(scores.shape), (1, 128))
            self.assertTrue(torch.isfinite(boxes).all())
            # the two modality streams stay independent: the fused tokens differ
            self.assertFalse(torch.allclose(out["sel_rgb"]["feats"],
                                            out["sel_tir"]["feats"]))


class CorrespondenceModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.corr = BidirectionalCorrespondence(dim=16, retrieval_dim=8, top_k=2,
                                                samples=8, heads=4, ff_mult=2)
        self.q = torch.randn(1, 3, 16)
        self.qp = torch.rand(1, 3, 2)
        self.of = torch.randn(1, 5, 16)
        self.op = torch.rand(1, 5, 2)
        self.gf = torch.randn(1, 4, 16)
        self.gp = torch.rand(1, 4, 2)
        self.maps = {n: torch.randn(1, 16, 8, 8) for n in ("P2", "P3", "P4", "P5")}

    def run_corr(self, **kw):
        return self.corr(self.q, self.qp, kw.pop("of", self.of), kw.pop("op", self.op),
                         kw.pop("gf", self.gf), kw.pop("gp", self.gp),
                         kw.pop("maps", self.maps), **kw)

    def test_topk_anchors_are_retrieved_by_appearance_only(self):
        """Moving the anchors' coordinates must not change which keys were retrieved."""
        first = self.run_corr()
        moved = self.run_corr(op=self.op + 0.3, gp=self.gp + 0.3)
        self.assertTrue(torch.equal(first["stats"]["anchor_idx"],
                                    moved["stats"]["anchor_idx"]))
        # ... and no coordinate is read anywhere in the retrieval: its logits are
        # a pure function of the four feature tensors.
        lp = first["stats"]["retrieval_logits"]
        self.assertEqual(lp.shape, (1, 3, 5 + 4 + 1))

    def test_identity_at_init_and_gate_can_reject(self):
        out = self.run_corr()
        self.assertTrue(torch.equal(out["feats"], self.q))
        with torch.no_grad():
            for d in ("r2t", "t2r"):
                self.corr.mix[d].weight.normal_(0, .5)
                self.corr.mix[d].bias.normal_(0, .5)
            self.corr.gate["r2t"][-1].bias.fill_(5.0)
        changed = self.run_corr()["feats"]
        self.assertFalse(torch.allclose(changed, self.q, atol=1e-6))
        # p_null -> 1 must switch the fusion off completely, message or not
        with torch.no_grad():
            self.corr.null_head["r2t"][-1].bias.fill_(30.0)
        rejected = self.run_corr()["feats"]
        self.assertTrue(torch.allclose(rejected, self.q, atol=1e-6))

    def test_covariance_spread_drives_the_local_sampling(self):
        tight = self.run_corr()["stats"]
        with torch.no_grad():
            self.corr.anchor["r2t"].unc[-1].bias[[0, 2]] = 5.0
        wide = self.run_corr()["stats"]
        spread = lambda s: float((s["local_pts"][0, 0] - s["mu"][0, 0, :, None]).norm(dim=-1).mean())
        self.assertGreater(spread(wide), spread(tight))
        # 2 anchors x 8 samples spread over P2-P5 = 16 distinct probe points
        self.assertEqual(tight["local_pts"].shape[-3:], (2, 2 * 4, 2))

    def test_covariance_is_positive_definite_for_extreme_outputs(self):
        head = GaussianAnchorHead(8, 8, sigma_floor=0.02, init_sigma=0.05)
        for raw in (torch.tensor([10., 50., 10.]), torch.tensor([-10., -50., -10.]),
                    torch.tensor([0., 0., 0.])):
            with torch.no_grad():
                head.unc[-1].bias[:] = raw
            mu, L = head(torch.randn(1, 2, 8), torch.rand(1, 2, 3, 2),
                         torch.randn(1, 2, 3, 8))
            cov = L @ L.transpose(-1, -2)
            eig = torch.linalg.eigvalsh(cov)
            self.assertTrue(bool((eig > 0).all()), f"not PD for raw={raw.tolist()}: {eig}")
            self.assertGreaterEqual(float(eig.min()), head.sigma_floor ** 2 - 1e-6)
            self.assertTrue(torch.isfinite(mu).all())

    def test_identity_matches_handle_unequal_counts_and_reordering(self):
        boxes_q = torch.tensor([[[.2, .2, .1, .1], [.8, .8, .1, .1], [.5, .9, .1, .1]]])
        boxes_k = torch.tensor([[[.6, .3, .1, .1], [.25, .25, .1, .1]]])
        mask_q = torch.ones(1, 3)
        mask_k = torch.ones(1, 2)
        pairs = torch.zeros(1, 3, 2)
        pairs[0, 0, 1] = 1        # RGB object 0 <-> TIR object 1
        pairs[0, 1, 0] = 1        # RGB object 1 <-> TIR object 0
        # object 2 has no counterpart at all
        q_pts = boxes_q[..., :2]
        match = identity_matches(q_pts, boxes_q, mask_q, boxes_k[..., :2], boxes_k,
                                 mask_k, pairs)
        self.assertEqual(match.shape, (1, 3, 2))
        self.assertEqual(int(match.sum()), 2)
        self.assertFalse(bool(match[0, 2].any()))
        # reordering the keys must permute the columns, not the matching
        perm = torch.tensor([1, 0])
        swapped = identity_matches(q_pts, boxes_q, mask_q, boxes_k[..., :2][:, perm],
                                   boxes_k[:, perm], mask_k, pairs[:, :, perm])
        self.assertTrue(torch.equal(swapped, match[:, :, perm]))


class MultiObjectAndClasses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_multi_object_unequal_counts_and_partial_pairing(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(
            cfg,
            rgb=((.2, .2, .1, .1), (.8, .25, .1, .1), (.5, .85, .1, .1)),
            tir=((.6, .3, .1, .1), (.25, .3, .1, .1)),
            corr=((0, 1), (1, 0)))            # RGB 2 has no counterpart -> null
        out = model(batch)
        loss, terms = model.compute_losses(out)
        self.assertTrue(torch.isfinite(loss))
        for key in ("corr", "corr_null", "corr_retr", "corr_geo"):
            self.assertIn(key, terms)
            self.assertTrue(torch.isfinite(terms[key]), key)
        self.assertGreater(float(terms["corr_null"]), 0.0)
        self.assertGreater(float(terms["corr_retr"]), 0.0)

    def test_no_instance_correspondence_means_detection_losses_only(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg, rgb=((.2, .2, .1, .1), (.8, .25, .1, .1)),
                            tir=((.5, .5, .1, .1),), corr=None)
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms["corr"]), 0.0)
        self.assertEqual(float(terms["corr_geo"]), 0.0)
        # the detection terms are still alive
        self.assertGreater(float(terms["rgb_heat"]), 0.0)

    def test_multi_class_head_and_decode(self):
        cfg = config(num_classes=3)
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = multi_batch(cfg, rgb=((.3, .3, .1, .1), (.7, .7, .1, .1)),
                            tir=((.3, .3, .1, .1), (.7, .7, .1, .1)),
                            labels_rgb=(0, 2), labels_tir=(1, 2))
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(out["pred_rgb"].shape[-1], 7)
        boxes, scores, labels = decode_boxes_scores_v3(
            out["pred_rgb"], out["sel_rgb"]["boxes"], batch["rgb_hw"],
            out["sel_rgb"]["objectness"], cfg.model.score_fusion_power, cfg.model.max_wh)
        self.assertEqual(tuple(boxes.shape[:2]), (1, 128))
        self.assertTrue(((labels >= 0) & (labels < 3)).all())
        model.train()
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms["rgb_loss_cls"]), 0.0)

    def test_missing_modality_supervises_null_and_skips_its_detection(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg, available=(1.0, 0.0))
        out = model(batch)
        loss, terms = model.compute_losses(out)
        self.assertTrue(torch.isfinite(loss))
        # a dropped sensor is not a background-only image
        for key in ("loss_cls", "loss_box", "loss", "pre_box", "heat"):
            self.assertEqual(float(terms["tir_" + key]), 0.0, key)
        self.assertGreater(float(terms["rgb_heat"]), 0.0)
        stats = out["aux"]["corr_stats_rgb"]
        self.assertTrue("matches_key" in stats)
        self.assertFalse(bool(stats["matches_key"].any()))     # nothing to attend to

    def test_multiclass_labels_reach_the_loss(self):
        """Two classes survive listing -> dataset -> collate -> detection loss."""
        with tempfile.TemporaryDirectory() as d:
            vdir = Path(d) / "building" / "visible"
            tdir = Path(d) / "building" / "thermal"
            vdir.mkdir(parents=True); tdir.mkdir(parents=True)
            Image.fromarray(np.zeros((80, 120, 3), np.uint8)).save(
                vdir / "building_visible_1_1.jpg")
            Image.fromarray(np.zeros((40, 60), np.uint8)).save(
                tdir / "building_thermal_1_1.jpg")
            shapes = [("drone", 10, 10, 20, 20), ("bird", 60, 30, 80, 50)]
            for path, size in ((vdir / "building_visible_1_1.json", (120, 80)),
                               (tdir / "building_thermal_1_1.json", (60, 40))):
                data = {"imageWidth": size[0], "imageHeight": size[1],
                        "shapes": [{"label": lab, "shape_type": "rectangle",
                                    "points": [[x0, y0], [x1, y1]]}
                                   for lab, x0, y0, x1, y1 in shapes]}
                path.write_text(json.dumps(data))
            frames = list_frames(d, ["building"], class_names=("drone", "bird"))
            self.assertEqual(frames[0].rgb_labels.tolist(), [0, 1])

            cfg = config(num_classes=2)
            cfg.data.class_names = ("drone", "bird")
            ds = UAVCBDataset(frames, cfg.data.rgb_size, cfg.data.tir_size, train=False,
                              class_names=cfg.data.class_names, pair_by_index=True)
            batch = collate([ds[0]])
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            self.assertEqual(int(batch["labels_rgb"][0, 0]), 0)
            self.assertEqual(int(batch["labels_rgb"][0, 1]), 1)
            model = PGSFDet(cfg).to(DEVICE)
            loss, terms = model.compute_losses(model(batch))
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(terms["rgb_loss_cls"]), 0.0)


class MissingLabelsAndInference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_inference_needs_no_ground_truth(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = {"rgb": torch.rand(1, 3, *cfg.data.rgb_size, device=DEVICE),
                 "tir": torch.rand(1, 1, *cfg.data.tir_size, device=DEVICE),
                 "rgb_hw": torch.tensor([[1080., 1920.]], device=DEVICE),
                 "tir_hw": torch.tensor([[512., 640.]], device=DEVICE)}
        with torch.no_grad():
            detections, out = model.inference(batch)
        self.assertIn("rgb", detections)
        self.assertIn("tir", detections)
        for name in ("rgb", "tir"):
            self.assertIn("boxes", detections[name][0])
            self.assertIn("scores", detections[name][0])
            self.assertIn("labels", detections[name][0])     # v3 always reports a class
            if detections[name][0]["boxes"].numel():
                limit = float(batch[name + "_hw"].max())
                self.assertLessEqual(float(detections[name][0]["boxes"].max()), limit + 1e-3)

    def test_v2_architecture_still_runs(self):
        """Compatibility: the v2 graph is untouched by the v3 dispatch."""
        cfg = config()
        cfg.model.architecture_version = 2
        cfg.model.use_mid_fusion = True
        cfg.model.use_late_fusion = True
        cfg.model.use_geo_prior = True
        model = PGSFDet(cfg).to(DEVICE)
        self.assertTrue(model.v2)
        self.assertFalse(model.v3)
        self.assertTrue(hasattr(model, "mid") and hasattr(model, "late"))
        batch = make_fake_batch(cfg, DEVICE, 1)
        loss, terms = model.compute_losses(model(batch))
        self.assertTrue(torch.isfinite(loss))
        for key in ("geo", "assoc_mid", "assoc_late"):
            self.assertIn(key, terms)

    def test_v3_stage_transfer_preserves_initial_detection(self):
        cfg1, cfg2 = config(1), config(2)
        m1 = PGSFDet(cfg1).to(DEVICE).eval()
        m2 = PGSFDet(cfg2).to(DEVICE).eval()
        m2.load_checkpoint_weights(m1.state_dict(), stage1_transfer=True)
        batch = multi_batch(cfg1)
        with torch.no_grad():
            a, z = m1(batch), m2(batch)
        for name in ("rgb", "tir"):
            self.assertTrue(torch.allclose(a["pred_" + name], z["pred_" + name], atol=1e-6))

    def test_decoder_matches_head_convention(self):
        raw = torch.zeros(1, 1, 5)
        base = torch.tensor([[[.3, .4, .02, .03]]])
        boxes, logits = decode_pred_v3(raw, base)
        self.assertTrue(torch.equal(boxes, base))          # zero-init -> the coarse box
        raw[..., :2] = 1.0
        boxes, _ = decode_pred_v3(raw, base)
        self.assertTrue(((boxes[..., :2] - base[..., :2]).abs() < base[..., 2:] / 2).all())


class DatasetContract(unittest.TestCase):
    """The data pipeline must carry multi-object / multi-class / mixed sizes."""

    def _write_frame(self, root, scene, stem, rgb_size, tir_size, rgb_shapes, tir_shapes):
        vdir = Path(root) / scene / "visible"
        tdir = Path(root) / scene / "thermal"
        vdir.mkdir(parents=True, exist_ok=True)
        tdir.mkdir(parents=True, exist_ok=True)
        rgb_stem = stem.replace("_thermal_", "_visible_")
        tir_stem = stem
        Image.fromarray(np.zeros((rgb_size[1], rgb_size[0], 3), np.uint8)).save(
            vdir / f"{rgb_stem}.jpg")
        Image.fromarray(np.zeros((tir_size[1], tir_size[0]), np.uint8)).save(
            tdir / f"{tir_stem}.jpg")
        for path, size, shapes in ((vdir / f"{rgb_stem}.json", rgb_size, rgb_shapes),
                                   (tdir / f"{tir_stem}.json", tir_size, tir_shapes)):
            data = {"imageWidth": size[0], "imageHeight": size[1], "shapes": []}
            for label, x0, y0, x1, y1 in shapes:
                data["shapes"].append({"label": label, "shape_type": "rectangle",
                                       "points": [[x0, y0], [x1, y1]]})
            path.write_text(json.dumps(data))
        return rgb_stem, tir_stem

    def test_multi_object_labels_masks_and_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_frame(d, "building", "building_thermal_1_1",
                              (120, 80), (60, 40),
                              [("drone", 10, 10, 20, 20), ("drone", 60, 30, 80, 50)],
                              [("drone", 5, 5, 15, 15)])
            frames = list_frames(d, ["building"])
            self.assertEqual(len(frames), 1)
            f = frames[0]
            self.assertEqual(f.rgb_boxes.shape, (2, 4))
            self.assertEqual(f.tir_boxes.shape, (1, 4))
            self.assertTrue(np.array_equal(f.rgb_labels, [0, 0]))

            ds = UAVCBDataset(frames, (32, 64), (32, 32), train=False)
            item = ds[0]
            self.assertEqual(float(item["mask_rgb"].sum()), 2.0)
            self.assertEqual(float(item["mask_tir"].sum()), 1.0)
            self.assertEqual(int(item["n_objects_rgb"]), 2)
            self.assertEqual(int(item["n_objects_tir"]), 1)
            self.assertEqual(int(item["labels_rgb"][0]), 0)
            self.assertEqual(int(item["labels_rgb"][1]), 0)
            self.assertEqual(int(item["labels_rgb"][2]), -1)      # padding
            # unequal counts, no explicit pairing -> no correspondence labels
            self.assertNotIn("corr_pairs", item)
            batch = collate([item])
            self.assertEqual(batch["labels_rgb"].dtype, torch.long)
            self.assertEqual(tuple(batch["boxes_rgb"].shape), (1, 50, 4))

            paired = UAVCBDataset(frames, (32, 64), (32, 32), train=False,
                                  pair_by_index=True)
            item2 = paired[0]
            self.assertIn("corr_pairs", item2)
            self.assertEqual(float(item2["corr_pairs"][0, 0]), 1.0)
            self.assertEqual(float(item2["corr_pairs"][1, 1]), 0.0)   # no TIR slot 1

    def test_single_object_frames_pair_unambiguously(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_frame(d, "cloud", "cloud_thermal_2_7", (120, 80), (60, 40),
                              [("drone", 30, 20, 50, 40)], [("drone", 15, 10, 25, 20)])
            frames = list_frames(d, ["cloud"])
            item = UAVCBDataset(frames, (32, 64), (32, 32), train=False)[0]
            self.assertIn("corr_pairs", item)                    # unambiguous 1<->1
            self.assertEqual(float(item["corr_pairs"][0, 0]), 1.0)

    def test_unknown_labels_are_dropped_not_remapped(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_frame(d, "ground", "ground_thermal_1_3", (120, 80), (60, 40),
                              [("drone", 10, 10, 20, 20), ("bird", 40, 40, 50, 50)],
                              [("drone", 5, 5, 15, 15)])
            f = list_frames(d, ["ground"])[0]
            self.assertEqual(f.rgb_boxes.shape, (1, 4))           # 'bird' is dropped
            self.assertEqual(f.rgb_labels.tolist(), [0])

    def test_mixed_native_resolutions_keep_pixel_boxes(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_frame(d, "powerline", "powerline_thermal_1_1", (200, 120), (100, 50),
                              [("drone", 20, 12, 60, 52)], [("drone", 10, 6, 30, 26)])
            frames = list_frames(d, ["powerline"])
            item = UAVCBDataset(frames, (32, 64), (32, 32), train=False)[0]
            # native sizes are the frame's own, not the network input
            self.assertTrue(torch.equal(item["rgb_hw"], torch.tensor([120., 200.])))
            self.assertTrue(torch.equal(item["tir_hw"], torch.tensor([50., 100.])))
            gx = item["boxes_rgb_px"][0]
            self.assertAlmostEqual(float(gx[0]), 20.0, places=4)
            self.assertAlmostEqual(float(gx[2]), 60.0, places=4)
            gy = item["boxes_tir_px"][0]
            self.assertAlmostEqual(float(gy[3]), 26.0, places=4)

    def test_augmentation_filters_boxes_and_labels_together(self):
        aug = PairAugment((1.0, 1.0), 0.0, 0.0, 0.0, True)
        rgb = np.zeros((80, 120, 3), np.uint8)
        tir = np.zeros((40, 60), np.uint8)
        boxes = np.array([[.2, .2, .1, .1], [1.03, .5, .1, .1]], np.float32)
        labels = np.array([0, 2], np.int64)
        _, _, rb, _, rlab, _, _ = aug.augment_boxes(rgb, tir, boxes, boxes,
                                                    (120, 80), (60, 40),
                                                    (32, 64), (32, 32), labels, labels)
        # the second box is almost entirely outside the frame -> dropped WITH its label
        self.assertEqual(len(rb), 1)
        self.assertEqual(rlab.tolist(), [0])
        self.assertAlmostEqual(float(rb[0, 0]), 0.2, places=5)


class DeepGradientAudit(unittest.TestCase):
    """Every trainable v3 tensor must receive a gradient (opt-in: slow on CPU)."""

    @unittest.skipUnless(os.environ.get("V3_DEEP") == "1",
                         "set V3_DEEP=1 to run the full gradient audit")
    def test_every_parameter_receives_a_gradient(self):
        cfg = config()
        model = PGSFDet(cfg).to(DEVICE)
        batch = multi_batch(cfg)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss, _ = model.compute_losses(model(batch))
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        loss, _ = model.compute_losses(model(batch))
        loss.backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
        self.assertEqual(dead, [], f"parameters without gradient: {dead}")


class CudaPath(unittest.TestCase):
    """AMP is the default training precision, so it must be exercised on GPU."""

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
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_inference_and_memory_budget(self):
        cfg = config()
        cfg.data.rgb_size = (800, 1280)
        cfg.data.tir_size = (512, 640)
        model = PGSFDet(cfg).to(DEVICE).eval()
        batch = {"rgb": torch.rand(1, 3, *cfg.data.rgb_size, device=DEVICE),
                 "tir": torch.rand(1, 1, *cfg.data.tir_size, device=DEVICE),
                 "rgb_hw": torch.tensor([[1080., 1920.]], device=DEVICE),
                 "tir_hw": torch.tensor([[512., 640.]], device=DEVICE)}
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            detections, _ = model.inference(batch)
        peak = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 20
        self.assertIn("rgb", detections)
        # a 10GB RTX 3080 must fit this with a large margin
        self.assertLess(peak, 2048.0, f"inference peak {peak:.0f} MiB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
