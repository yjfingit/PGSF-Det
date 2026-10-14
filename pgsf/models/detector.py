"""PGSF-Det: full detector.

Data flow (identical for both modalities, only the weights differ)::

    image -> LightCSPBackbone -> {P2,P3,P4,P5}
          -> MSCE on P2
          -> PAN-FPN  (all levels -> 128 ch)
          -> HeatmapHead(P2,P3) -> candidate heatmap
          -> CandidateGenerator (64 candidates, local NMS + partition fallback)
          -> [first fusion]  MidFusion  (RGB <-> TIR, probabilistic geometry prior)
          -> enhanced candidates written back into P3/P4 (weight-normalised splat)
          -> preliminary box head -> top-32 candidates
          -> [second fusion] LateFusion (object level, box prior, null token)
          -> DetHead  128 -> 128 -> 5      -> (cx,cy,w,h,score) for each modality

Ablation switches (see :class:`pgsf.config.ModelConfig`):

======  =========================================================
A0      backbone + neck + candidate + DetHead, no fusion
A1      + MSCE
A2      + MidFusion **without** the geometric prior
A3      + probabilistic geometric prior
A4      + reliability gate and null matching
A5      + LateFusion
A6      + NWD auxiliary localisation loss
======  =========================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import DualBackbone
from pgsf.models.candidate import CandidateGenerator, HeatmapHead, sample_points
from pgsf.models.fusion import MidFusion
from pgsf.models.geometry import ProbGeoPrior, invert_affine, augment_affine
from pgsf.models.head import BoxHead, DetHead, MultiScaleBoxRefine
from pgsf.models.late_fusion import LateFusion, select_topk_candidates
from pgsf.models.msce import MSCE
from pgsf.models.neck import DualNeck
from pgsf.losses import (
    DetectionLoss,
    assign_candidates,
    correspondence_mask,
    supervised_association_loss,
    draw_gaussian,
    gaussian_focal_loss,
    geometry_nll,
)


class ModalityBranch(nn.Module):
    """Everything that is per-modality: backbone, MSCE, neck, heads."""

    def __init__(self, in_ch: int, cfg):
        super().__init__()
        m = cfg.model
        self.backbone = LightBackboneCtor(in_ch, m)
        chans = self.backbone.out_channels
        self.msce_on = m.use_msce
        if self.msce_on:
            self.msce = MSCE(chans[0], width=64, alpha_max=m.msce_alpha_max)
        self.neck = NeckCtor(chans, m)
        self.heat = HeatmapHead(m.neck_out, hidden=64)
        self.cand = CandidateGenerator(
            m.neck_out, n_cand=m.n_cand, n_global=m.n_cand - 16,
            nms_kernel=m.nms_kernel, n_partitions=16,
            inject_gt_ratio=m.inject_gt_ratio)
        self.refine = MultiScaleBoxRefine(m.neck_out, n_levels=4, hidden=64)
        self.box_pre = BoxHead(self.refine.hidden, m.head_width)
        self.det = DetHead(m.neck_out, m.head_width, m.n_layers_head, max_wh=m.max_wh)
        self.neck_out = m.neck_out

    def features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        if self.msce_on:
            f["P2"], _ = self.msce(f["P2"])
        return self.neck(f)


def LightBackboneCtor(in_ch: int, m) -> nn.Module:
    from pgsf.models.backbone import LightCSPBackbone
    if m.backbone_name == "yolo11s":
        from pgsf.models.backbone import YOLO11Backbone
        return YOLO11Backbone(in_ch,m.backbone_weights)
    if m.backbone_name != "lightcsp":raise ValueError(f"unknown backbone {m.backbone_name}")
    return LightCSPBackbone(in_ch, m.width_mult, m.backbone_depths, m.use_dw_bottleneck)


def NeckCtor(chans, m) -> nn.Module:
    from pgsf.models.neck import PANFPN
    return PANFPN(chans, m.neck_dim, m.neck_out, m.neck_repeats)


class PGSFDet(nn.Module):
    """The full network.  ``forward`` returns a dict with detections and all
    intermediate tensors needed by the losses."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ``A`` is a *trainable* parameter, not a frozen buffer: the RGB->TIR
        # mapping is a property of the camera pair, so it must stay adjustable
        # when the model moves to another platform, instead of locking in the
        # value fitted on the training set. ``_affine_reverse`` is kept only for
        # checkpoint compatibility -- the forward pass recomputes the inverse
        # from ``A`` each step (see ``pair_affines`` in detector_v2).
        self._affine_A = nn.Parameter(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        self.register_buffer("_affine_b", torch.zeros(2))
        self.register_buffer("_affine_reverse", self._affine_A.detach().clone())
        # ``arch`` is the *only* place the architecture generation is decided.
        # ``v2`` stays strictly "the v2 graph" so v2-only assertions (e.g. the
        # gradient audit in scripts/check_forward.py) are not accidentally taken
        # by a v3 model, which has no ``mid``/``late`` submodules at all.
        self.arch = int(cfg.model.architecture_version)
        self.v4 = self.arch >= 4
        self.v3 = self.arch == 3
        self.v2 = self.arch == 2
        if self.v4:
            from pgsf.models.detector_v4 import build_modules as build_v4
            build_v4(self, cfg)
            self.mid_on = self.late_on = False
            return
        if self.v3:
            from pgsf.models.detector_v3 import build_modules as build_v3
            build_v3(self, cfg)
            self.mid_on = self.late_on = False
            return
        if self.v2:
            from pgsf.models.detector_v2 import build_modules
            build_modules(self,cfg)
            return
        m = cfg.model
        self.rgb_branch = ModalityBranch(3, cfg)
        self.tir_branch = ModalityBranch(1, cfg)
        # NB: no ``self.rgb_bbone = ...`` alias -- re-registering a submodule would
        # duplicate its parameters in ``model.parameters()``.

        self.mid_on = m.use_mid_fusion and cfg.stage >= 2
        if self.mid_on:
            self.mid = MidFusion(
                d_model=m.neck_out, n_heads=m.n_heads, n_local=m.n_local_samples,
                n_global=m.n_global, lambda_geo=m.lambda_geo,
                use_geo=m.use_geo_prior, use_gate=m.use_gate)
            # Feature-conditioned residual/uncertainty heads are shared; each
            # direction receives its own forward/inverse affine.
        self.late_on = m.use_late_fusion and cfg.stage >= 2
        if self.late_on:
            self.late = LateFusion(d_model=m.neck_out, n_heads=m.n_heads,
                                   lambda_box=m.lambda_box,
                                   use_gate=m.use_gate,
                                   use_box_prior=m.use_geo_prior)

        self.det_loss_rgb = DetectionLoss(m.w_cls, m.w_box, m.w_nwd, m.use_nwd,
                                          m.assign_pos_thr, m.assign_center_thr,
                                          m.max_wh)
        self.det_loss_tir = DetectionLoss(m.w_cls, m.w_box, m.w_nwd, m.use_nwd,
                                          m.assign_pos_thr, m.assign_center_thr,
                                          m.max_wh)
        self.w_heat, self.w_geo = m.w_heat, m.w_geo
        self.w_assoc_mid, self.w_assoc_late = m.w_assoc_mid, m.w_assoc_late
        self.n_cand_late = m.n_cand_late

    # ------------------------------------------------------------------ #
    # geometry prior control
    # ------------------------------------------------------------------ #
    @property
    def affine(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The fitted global affine (A is 2x3 homogeneous, b is its 3rd column)."""
        return self._affine_A, self._affine_b

    @torch.no_grad()
    def set_affine(self, A: torch.Tensor, b: torch.Tensor) -> None:
        """Seed the homogeneous RGB->TIR affine, which is now trainable.

        Called at start-up with the training-split fit; the optimiser keeps
        updating the value afterwards.

        v3 has no fixed affine prior by design -- correspondence comes from
        learned appearance retrieval -- so this is a documented no-op there and
        the value is never read by the v3 forward pass.  v4 removes the affine
        prior as well, so the same holds for it.
        """
        if self.v3 or self.v4:
            return
        A = A.detach().reshape(2, 3)
        self._affine_A.copy_(A.to(self._affine_A))
        self._affine_b.copy_(A[:, 2].to(self._affine_b))
        self._affine_reverse.copy_(invert_affine(self._affine_A))
        if self.mid_on:
            if self.v2:self.mid.set_affine(self._affine_A,self._affine_b,self._affine_reverse)
            else:self.mid.geo.set_affine(self._affine_A,self._affine_b)
        if self.v2 and self.late_on and hasattr(self.late,'set_affine'):
            self.late.set_affine(self._affine_A,self._affine_b,self._affine_reverse)

    def load_checkpoint_weights(self, state, stage1_transfer: bool = False):
        """Load weights strictly; stage transfer permits only new fusion modules.

        Older checkpoints stored affine buffers under fusion modules. Normalize
        those keys here without hiding missing backbone/head weights.
        """
        state = dict(state)
        A = state.get("_affine_A", state.get("mid.geo.A", state.get(
            "late.box_prior._A_affine", self._affine_A)))
        state.setdefault("_affine_A", A)
        state.setdefault("_affine_b", A[:, 2])
        state.setdefault("_affine_reverse",invert_affine(A))
        state.pop("late.box_prior._A_affine", None)
        state.pop("late.box_prior._b_affine", None)
        own = self.state_dict()
        if stage1_transfer:
            missing = set(own) - set(state)
            unexpected = set(state) - set(own)
            bad = [k for k in missing
                   if not k.startswith(("mid.", "late.", "corr.", "match.", "td.",
                                        "geometry_context."))]
            if bad or unexpected:
                raise RuntimeError(f"invalid stage-1 transfer: missing={bad}, unexpected={sorted(unexpected)}")
            for key in missing:
                state[key] = own[key]
        result = self.load_state_dict(state, strict=True)
        if not (self.v3 or self.v4):
            # v3/v4 never read the affine; re-seeding it would only create the
            # illusion that a fitted prior is in play.
            self.set_affine(self._affine_A, self._affine_b)
        return result

    def set_training_progress(self, fraction):
        self.progress_fraction = float(fraction)
        if self.arch < 2:return
        end=max(self.cfg.model.injection_end_fraction,1e-6)
        ratio=self.cfg.model.inject_gt_ratio*max(0.,1-fraction/end)
        for branch in (self.rgb_branch,self.tir_branch):branch.cand.inject_gt_ratio=ratio

    # ------------------------------------------------------------------ #
    def _branch_forward(self, branch: ModalityBranch, x: torch.Tensor,
                        centers: Optional[torch.Tensor], training: bool, mask=None):
        feats = branch.features(x)
        heat = branch.heat(feats["P2"], feats["P3"])
        cand = branch.cand(heat, feats["P3"], gt_centers=centers, training=training, gt_mask=mask)
        return feats, heat, cand

    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        if self.v4:
            from pgsf.models.detector_v4 import forward as forward_v4
            return forward_v4(self, batch)
        if self.v3:
            from pgsf.models.detector_v3 import forward as forward_v3
            return forward_v3(self,batch)
        if self.v2:
            from pgsf.models.detector_v2 import forward
            return forward(self,batch)
        rgb = batch["rgb"]
        tir = batch["tir"]
        training = self.training
        c_rgb, c_tir = batch.get("centers_rgb"), batch.get("centers_tir")
        A, b = self.affine
        A = augment_affine(A, batch.get("geo_transform"))
        A_reverse = invert_affine(A)

        f_rgb, heat_rgb, cand_rgb = self._branch_forward(
            self.rgb_branch, rgb, c_rgb, training, batch.get("mask"))
        f_tir, heat_tir, cand_tir = self._branch_forward(
            self.tir_branch, tir, c_tir, training, batch.get("mask"))

        out: Dict[str, object] = {}
        aux: Dict[str, torch.Tensor] = {}

        if self.mid_on:
            # ---- first fusion: mid-stage sparse cross-modal attention
            g_rgb, p_rgb = self._global_of(cand_rgb, self.n_global())
            g_tir, p_tir = self._global_of(cand_tir, self.n_global())
            res_rt = self.mid(cand_rgb["feats"], cand_rgb["pts"], f_tir["P3"],
                              g_tir, "r2t", affine=A, other_global_pts=p_tir,
                              maps={n: f_rgb[n] for n in ("P3", "P4")})
            res_tr = self.mid(cand_tir["feats"], cand_tir["pts"], f_rgb["P3"],
                              g_rgb, "t2r", affine=A_reverse, other_global_pts=p_rgb,
                              maps={n: f_tir[n] for n in ("P3", "P4")})
            # Each query's residual is written to its own modality and coordinates.
            f_rgb.update(res_rt["maps"])
            f_tir.update(res_tr["maps"])
            cand_rgb["feats"], cand_tir["feats"] = res_rt["feats"], res_tr["feats"]
            aux["mid_stats_rgb"], aux["mid_stats_tir"] = res_rt["stats"], res_tr["stats"]

        # ---- preliminary box head + top-k selection
        pre_rgb, sel_rgb = self._preliminary(self.rgb_branch, f_rgb, cand_rgb)
        pre_tir, sel_tir = self._preliminary(self.tir_branch, f_tir, cand_tir)
        aux["pre_box_rgb"], aux["pre_box_tir"] = pre_rgb, pre_tir

        if self.late_on:
            late_rt = self.late(sel_rgb["feats"], sel_rgb["boxes"],
                                sel_tir["feats"], sel_tir["boxes"], A, b, "r2t")
            late_tr = self.late(sel_tir["feats"], sel_tir["boxes"],
                                sel_rgb["feats"], sel_rgb["boxes"], A, b, "t2r")
            sel_rgb["feats"], sel_tir["feats"] = late_rt["feats"], late_tr["feats"]
            aux["late_rgb"], aux["late_tir"] = late_rt, late_tr

        # ---- detection heads
        pred_rgb = self.rgb_branch.det(sel_rgb["feats"])      # (B,K,5)
        pred_tir = self.tir_branch.det(sel_tir["feats"])

        out["pred_rgb"] = pred_rgb
        out["pred_tir"] = pred_tir
        out["cand_rgb"] = cand_rgb
        out["cand_tir"] = cand_tir
        out["sel_rgb"] = sel_rgb
        out["sel_tir"] = sel_tir
        out["heat_rgb"] = heat_rgb
        out["heat_tir"] = heat_tir
        out["aux"] = aux
        out["batch"] = batch
        return out

    # ------------------------------------------------------------------ #
    def _global_of(self, cand: Dict[str, torch.Tensor], n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-n *global* candidates (highest heatmap score) as cross-modal keys."""
        n = min(n, cand["feats"].shape[1])
        idx = cand["scores"].topk(n, dim=1).indices
        return (torch.gather(cand["feats"], 1, idx[..., None].expand(-1, -1, cand["feats"].shape[-1])),
                torch.gather(cand["pts"], 1, idx[..., None].expand(-1, -1, 2)))

    def n_global(self) -> int:
        return self.cfg.model.n_global

    def _preliminary(self, branch: ModalityBranch, feats: Dict[str, torch.Tensor],
                     cand: Dict[str, torch.Tensor]):
        """Coarse box for every candidate, then keep the top ``n_cand_late``.

        The centre is the candidate point itself (the heatmap already localises
        it well); the box head only has to estimate the *size*, which is much
        easier and is all the late-stage box prior needs.
        """
        pts = cand["pts"]
        # start from a neutral box so MultiScaleBoxRefine samples a sane pattern
        init = torch.cat([pts, pts.new_full((*pts.shape[:2], 2), 0.05)], dim=-1)
        ctx = branch.refine(feats, pts, init)
        size = branch.box_pre(ctx)[..., 2:].sigmoid() * self.cfg.model.max_wh
        box = torch.cat([pts, size], dim=-1)          # (B,K,4) normalised cx,cy,w,h
        sel = select_topk_candidates(cand, box, self.n_cand_late)
        return box, sel

    # ------------------------------------------------------------------ #
    def compute_losses(self, out: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full objective; returns (total_loss, per-term detached dict)."""
        if self.v4:
            from pgsf.models.detector_v4 import compute_losses as compute_losses_v4
            return compute_losses_v4(self, out)
        if self.v3:
            from pgsf.models.detector_v3 import compute_losses as compute_losses_v3
            return compute_losses_v3(self,out)
        if self.v2:
            from pgsf.models.detector_v2 import compute_losses
            return compute_losses(self,out)
        batch: Dict[str, torch.Tensor] = out["batch"]  # type: ignore[assignment]
        cand_rgb, cand_tir = out["cand_rgb"], out["cand_tir"]  # type: ignore[assignment]
        sel_rgb, sel_tir = out["sel_rgb"], out["sel_tir"]        # type: ignore[assignment]
        aux = out["aux"]                                        # type: ignore[assignment]
        terms: Dict[str, torch.Tensor] = {}
        device = batch["rgb"].device

        # ---- detection, both modalities
        lr = self.det_loss_rgb(out["pred_rgb"], sel_rgb["pts"],   # type: ignore[index]
                              batch["boxes_rgb"], batch["mask"])
        lt = self.det_loss_tir(out["pred_tir"], sel_tir["pts"],
                               batch["boxes_tir"], batch["mask"])
        for k, v in lr.items():
            terms[f"rgb_{k}"] = v
        for k, v in lt.items():
            terms[f"tir_{k}"] = v
        loss = self.cfg.model.w_det * (lr["loss"] + lt["loss"])

        # ---- heatmaps
        if self.cfg.model.w_heat > 0:
            h_r = draw_gaussian(torch.zeros_like(cand_rgb["heat"]),
                                batch["centers_rgb"], batch["sizes_rgb"],
                                batch["mask"])
            h_t = draw_gaussian(torch.zeros_like(cand_tir["heat"]),
                                batch["centers_tir"], batch["sizes_tir"],
                                batch["mask"])
            l_h = gaussian_focal_loss(out["heat_rgb"], h_r) + \
                gaussian_focal_loss(out["heat_tir"], h_t)           # type: ignore[index]
            terms["heat"] = l_h.detach()
            loss = loss + self.w_heat * l_h

        # ---- geometry + association
        if self.mid_on:
            loss = loss + self._mid_losses(aux, batch, cand_rgb, cand_tir, terms)
        if self.late_on:
            loss = loss + self._late_losses(aux, batch, sel_rgb, sel_tir, terms)
        # Each component is already averaged; do not divide by batch twice.
        terms["total"] = loss.detach()
        del device
        return loss, terms

    # ------------------------------------------------------------------ #
    def _mid_losses(self, aux, batch, cand_rgb, cand_tir, terms):
        cfg = self.cfg.model
        s_rgb, s_tir = aux["mid_stats_rgb"], aux["mid_stats_tir"]

        # geometry NLL: predict RGB->TIR and TIR->RGB correspondence of the *same*
        # target. The target for the RGB query is the TIR GT centre of the same
        # frame, even when the two centers are far apart.
        mu_rgb, L_rgb = s_rgb["mu"], s_rgb["sigma"]
        mu_tir, L_tir = s_tir["mu"], s_tir["sigma"]
        cov_rgb = L_rgb @ L_rgb.transpose(-1, -2)
        cov_tir = L_tir @ L_tir.transpose(-1, -2)

        # Which GT does each candidate belong to?  The geometry target must be the
        # *same object's* centre in the other modality, so it is gathered by the
        # matched index rather than broadcast from object 0 (which would be wrong
        # as soon as a frame has more than one target).
        pos_r, _ = assign_candidates(cand_rgb["pts"], batch["boxes_rgb"], batch["mask"])
        pos_t, _ = assign_candidates(cand_tir["pts"], batch["boxes_tir"], batch["mask"])
        idx_r = matched_gt_index(cand_rgb["pts"], batch["boxes_rgb"], batch["mask"])
        idx_t = matched_gt_index(cand_tir["pts"], batch["boxes_tir"], batch["mask"])
        tgt_rgb = gather_by_index(batch["centers_tir"], idx_r)   # (B,K,2)
        tgt_tir = gather_by_index(batch["centers_rgb"], idx_t)   # (B,K,2)

        # only supervise geometry on candidates that actually sit on a target
        vr = batch["valid"].unsqueeze(1) * pos_r
        vt = batch["valid"].unsqueeze(1) * pos_t
        l_geo = geometry_nll(mu_rgb, cov_rgb, tgt_rgb, vr) + \
            geometry_nll(mu_tir, cov_tir, tgt_tir, vt)
        terms["geo"] = l_geo.detach()
        loss = cfg.w_geo * l_geo

        if not cfg.use_geo_prior:
            loss = l_geo * 0.0  # A2 excludes geometric supervision as well as the prior
        l_assoc = self._association(s_rgb["attn"], cand_rgb["pts"], s_rgb["key_pts"],
                                    batch, "rgb", "tir") + self._association(
                                        s_tir["attn"], cand_tir["pts"], s_tir["key_pts"],
                                        batch, "tir", "rgb")
        terms["assoc_mid"] = l_assoc.detach()
        return loss + cfg.w_assoc_mid * l_assoc

    def _association(self, attn, query_pts, key_pts, batch, query_name, key_name):
        matches = correspondence_mask(
            query_pts, key_pts, batch[f"boxes_{query_name}"], batch[f"boxes_{key_name}"],
            batch["mask"], batch["valid"], self.cfg.model.assign_center_thr)
        return supervised_association_loss(attn, matches)

    def _late_losses(self, aux, batch, sel_rgb, sel_tir, terms):
        l = self._association(aux["late_rgb"]["attn"], sel_rgb["pts"], sel_tir["pts"],
                              batch, "rgb", "tir") + self._association(
                                  aux["late_tir"]["attn"], sel_tir["pts"], sel_rgb["pts"],
                                  batch, "tir", "rgb")
        terms["assoc_late"] = l.detach()
        return self.cfg.model.w_assoc_late * l

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def inference(self, batch: Dict[str, torch.Tensor], max_det: int = 50,
                  score_thr: float = 0.05, nms_thr: float = 0.45):
        """Greedy per-modality decoding with NMS, in modality-native pixels."""
        was_training = self.training
        if was_training:
            self.eval()
        out = self.forward(batch)
        res = {}
        for name, size_key in (("rgb", "rgb_hw"), ("tir", "tir_hw")):
            if '_decoded_' + name in out:
                boxes,scores=out['_decoded_'+name][:2]
                labels=out['_decoded_'+name][2] if len(out['_decoded_'+name])>2 else None
                res[name]=postprocess_detections(boxes,scores,max_det,score_thr,nms_thr,labels)
                continue
            boxes,scores,labels=decode_modality(out[f"pred_{name}"], out["sel_"+name],
                                                batch[size_key], self.cfg.model)
            res[name]=postprocess_detections(boxes,scores,max_det,score_thr,nms_thr,
                                             labels if self.arch>=3 else None)
        if was_training:
            self.train()
        return res, out


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def matched_gt_index(cand_pts: torch.Tensor, gt_boxes: torch.Tensor,
                     gt_mask: torch.Tensor) -> torch.Tensor:
    """For every candidate, the index of the GT box it is closest to (B,K).

    Unmatched / empty frames fall back to index 0; the accompanying validity mask
    zeroes their loss anyway, so the value itself is irrelevant.
    """
    b, k, _ = cand_pts.shape
    m = gt_boxes.shape[1]
    if m == 0:
        return torch.zeros(b, k, dtype=torch.long, device=cand_pts.device)
    _,target=assign_candidates(cand_pts,gt_boxes,gt_mask)
    d=(target.unsqueeze(2)-gt_boxes.unsqueeze(1)).abs().sum(-1)
    d=d.masked_fill((gt_mask<=.5).unsqueeze(1),float('inf'))
    return d.argmin(-1)



def gather_by_index(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """src (B,M,C), idx (B,K) -> (B,K,C)."""
    b, k = idx.shape
    c = src.shape[-1]
    return torch.gather(src, 1, idx[..., None].expand(b, k, c))


def decode_detections(raw: torch.Tensor, cand_pts: torch.Tensor, img_hw: torch.Tensor,
                      max_det: int = 50, score_thr: float = 0.05,
                      nms_thr: float = 0.45, max_wh: float = 0.5, base_boxes=None, proposal_scores=None, score_power=.5
                      ) -> List[Dict[str, torch.Tensor]]:
    """Head output -> per-image pixel boxes ``x0,y0,x1,y1`` + scores.

    Uses the same :func:`pgsf.models.head.decode_pred` as the loss, so training and
    inference can never disagree about what the head predicts.
    """
    boxes, scores = decode_boxes_scores(raw,cand_pts,img_hw,max_wh,base_boxes,proposal_scores,score_power)
    return postprocess_detections(boxes,scores,max_det,score_thr,nms_thr)


def decode_modality(raw, sel, img_hw, model_cfg):
    """One shared decoder for every architecture generation.

    Returns ``(boxes_xyxy (B,K,4), scores (B,K), labels (B,K) long)`` in the
    modality's **own** pixel space.  ``labels`` is all-zero for the single-class
    legacy head, so callers that ignore classes stay source compatible.
    """
    if int(model_cfg.architecture_version) >= 4:
        from pgsf.models.head_v4 import decode_boxes_scores_v4
        return decode_boxes_scores_v4(raw, sel["boxes"], img_hw, sel.get("objectness"),
                                      model_cfg.score_fusion_power, model_cfg.max_wh)
    if int(model_cfg.architecture_version) >= 3:
        from pgsf.models.head_v3 import decode_boxes_scores_v3
        return decode_boxes_scores_v3(raw, sel["boxes"], img_hw, sel.get("objectness"),
                                      model_cfg.score_fusion_power, model_cfg.max_wh)
    use_v2 = int(model_cfg.architecture_version) >= 2
    boxes, scores = decode_boxes_scores(
        raw, sel["pts"], img_hw, model_cfg.max_wh,
        sel["boxes"] if use_v2 else None,
        sel.get("objectness") if use_v2 else None, model_cfg.score_fusion_power)
    return boxes, scores, torch.zeros(boxes.shape[:2], dtype=torch.long, device=boxes.device)


def decode_boxes_scores(raw,cand_pts,img_hw,max_wh=.5,base_boxes=None,proposal_scores=None,score_power=.5):
    """Fixed-shape decode, suitable for CUDA graph capture; returns B,K,4 and B,K."""
    from pgsf.models.head import decode_pred
    boxes_n, logits = decode_pred(raw,cand_pts,max_wh,base_boxes)
    center, half_size = boxes_n[...,:2], boxes_n[...,2:] * .5
    scale=img_hw.flip(-1).repeat(1,2)
    boxes=torch.cat((center-half_size,center+half_size),-1)*scale[:,None]
    scores=logits.sigmoid()
    if proposal_scores is not None:
        scores=scores*proposal_scores.clamp_min(1e-8).pow(score_power)
    return boxes,scores


def postprocess_detections(boxes,scores,max_det=50,score_thr=.05,nms_thr=.45,labels=None):
    """Variable-size thresholding and NMS stay outside the captured graph."""
    result=[]
    for i in range(boxes.shape[0]):
        valid=torch.where(scores[i]>=score_thr)[0]
        keep=valid[batched_nms(boxes[i][valid],scores[i][valid],nms_thr,max_det)]
        det={'boxes':boxes[i][keep],'scores':scores[i][keep]}
        if labels is not None:det['labels']=labels[i][keep]
        result.append(det)
    return result


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-6)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float,
                max_det: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    try:
        from torchvision.ops import nms
        return nms(boxes.float(),scores.float(),iou_thr)[:max_det]
    except (ImportError,RuntimeError):
        pass  # CPU-only/minimal torchvision builds retain the reference path.
    keep: List[int] = []
    order = scores.argsort(descending=True)
    while order.numel() > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


# --------------------------------------------------------------------------- #
# parameter groups
# --------------------------------------------------------------------------- #
def param_groups(model: PGSFDet, cfg) -> List[Dict[str, object]]:
    """Backbone gets ``lr_backbone_mult`` * lr, newly initialised fusion modules
    get ``lr_new_mult`` * lr.  Everything else uses the base lr."""
    bb, new, geo, rest = [], [], [], []
    new_prefixes = ("mid.", "late.", "corr.", "match.", "td.", "geometry_context.")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("_affine_A"):
            # Trainable global cross-modal affine. It starts from the
            # training-split fit, so it gets the conservative backbone learning
            # rate and, crucially, **no weight decay**: AdamW would otherwise
            # shrink the affine towards zero.
            geo.append(p)
        elif n.startswith("rgb_bbone") or n.startswith("tir_bbone") or ".backbone." in n:
            bb.append(p)
        elif n.startswith(new_prefixes):
            new.append(p)
        else:
            rest.append(p)
    groups = [
        {"params": bb, "lr": cfg.optim.lr * cfg.optim.lr_backbone_mult, "name": "backbone"},
        {"params": rest, "lr": cfg.optim.lr, "name": "rest"},
    ]
    if geo:
        groups.append({"params": geo, "lr": cfg.optim.lr * cfg.optim.lr_backbone_mult,
                       "name": "geo_affine", "weight_decay": 0.0})
    if new:
        groups.append({"params": new, "lr": cfg.optim.lr * cfg.optim.lr_new_mult,
                       "name": "new_modules"})
    return groups


def load_pretrained_backbone(model: PGSFDet, path: str) -> Tuple[int, int]:
    """Load a *compatible* CNN checkpoint (any dict with matching shapes).

    Only tensors whose name and shape both match are copied, so a checkpoint
    trained on another architecture can still seed the overlapping stem/layers.
    Returns (n_loaded, n_total_backbone_params).
    """
    if not path:
        return 0, 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    own = model.state_dict()
    loaded, total = 0, 0
    for k, v in own.items():
        if "backbone" not in k:
            continue
        total += 1
        for cand in (k, k.replace("rgb_bbone.", "rgb_branch.backbone.")):
            if cand in state and state[cand].shape == v.shape:
                own[k] = state[cand].clone()
                loaded += 1
                break
    model.load_state_dict(own)
    return loaded, total
