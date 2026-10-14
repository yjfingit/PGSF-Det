"""PGSF-Det **v4**: object-level matching + asymmetric semantic/localization fusion.

Data flow (identical for both modalities, only the weights differ)::

    image -> backbone -> {P2..P5} (all 128 ch)
          -> P2-P5 class-quality + centre/size + quality proposals
          -> K candidates (per-level reserved quota + global ranking)
          -> multi-scale RoIAlign  S [B,K,128,r,r]   (localization input)
          -> GAP                   z [B,K,128]       (matching / classification input)
          -> [ ONE object-level interaction ]
                CFPM: 64-d semantic retrieval (Top-L) + local multi-level probing
                reliability gate r  + consistency token c
                TD-Fusion: f_cls = z + r * W_c(c)              (classification)
                           F_reg = S * (1 + alpha r tanh(W_r c)) (localization)
          -> per-modality class logits and boxes in the modality's OWN frame

What v4 deliberately does **not** contain:

* no train-fitted affine prior (``_affine_A`` is frozen and never read),
* no write-back into intermediate feature maps,
* no second late Transformer -- exactly one object-level interaction,
* no coordinate subtraction between the two modalities anywhere: the retrieval is
  pure appearance, and the fine probing offsets are scaled by the *other*
  modality's own box size.

The auxiliary ``L_self`` is computed with the *same* heads on the un-fused
``z``/``S``, so inference needs no extra branch.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from pgsf.models.detector import LightBackboneCtor
from pgsf.models.fusion_v4 import AsymmetricFusion
from pgsf.models.head_v4 import (ClsHeadV4, DetectionLossV4, RegHeadV4,
                                 class_probabilities, js_divergence)
from pgsf.models.matching_v4 import CoarseToFineMatching, identity_matches
from pgsf.models.msce import LiteMSCE
from pgsf.models.neck import PANFPN
from pgsf.models.proposal_v4 import (DenseProposalHeadV4, MultiScaleCandidatesV4,
                                     MultiScaleRoIAlign, candidate_losses_v4)


# --------------------------------------------------------------------------- #
# per-modality branch
# --------------------------------------------------------------------------- #
class RegionBranchV4(nn.Module):
    """Backbone, neck, proposals, RoI descriptor, and the two detection heads."""

    def __init__(self, in_ch: int, cfg):
        super().__init__()
        m = cfg.model
        self.backbone = LightBackboneCtor(in_ch, m)
        self.msce_on = bool(m.use_msce)
        if self.msce_on:
            self.msce = LiteMSCE(self.backbone.out_channels[0], m.msce_width, m.msce_alpha_max)
        self.neck = PANFPN(self.backbone.out_channels, m.neck_dim, m.neck_out, m.neck_repeats)
        self.prop = DenseProposalHeadV4(m.neck_out, m.proposal_width, m.num_classes,
                                        m.max_wh, m.cand_levels)
        self.cand = MultiScaleCandidatesV4(m.neck_out, m.n_cand, m.cand_levels,
                                           m.cand_min_per_level, m.nms_kernel,
                                           m.num_classes, m.inject_gt_ratio, m.max_wh)
        self.roi = MultiScaleRoIAlign(m.neck_out, m.cand_levels, m.roi_size,
                                      m.roi_sampling_ratio)
        self.cls = ClsHeadV4(m.neck_out, m.head_width, m.num_classes)
        self.reg = RegHeadV4(m.neck_out, m.head_width)

    def features(self, x: torch.Tensor):
        f = self.backbone(x)
        if self.msce_on:
            f["P2"], _ = self.msce(f["P2"])
        return self.neck(f)

    def objects(self, x, gt_centers=None, gt_mask=None, training=False):
        feats = self.features(x)
        dense = self.prop(feats)
        cand = self.cand(dense, feats, gt_centers, gt_mask, training)
        roi = self.roi(feats, cand["boxes"])
        token = roi.mean((-2, -1))
        return {"feats": feats, "dense": dense, "cand": cand, "roi": roi, "token": token}

    def predict(self, token, roi, coarse_box):
        """Concatenated head output ``[box_delta | class_logits]`` -> ``(B,K,4+C)``."""
        return torch.cat((self.reg(roi, coarse_box), self.cls(token)), -1)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_modules(model, cfg) -> None:
    m = cfg.model
    model.rgb_branch = RegionBranchV4(3, cfg)
    model.tir_branch = RegionBranchV4(1, cfg)
    model.v4_on = bool(m.use_td_fusion) and cfg.stage >= 2
    if model.v4_on:
        model.match = nn.ModuleDict({
            d: CoarseToFineMatching(m.neck_out, m.retrieval_dim, m.td_top_l, m.td_samples,
                                    m.cand_levels, m.td_heads, m.td_ff_mult)
            for d in ("r2t", "t2r")})
        model.td = nn.ModuleDict({
            d: AsymmetricFusion(m.neck_out, m.alpha_init) for d in ("r2t", "t2r")})
    model.det_loss_rgb = DetectionLossV4(m.num_classes, m.lambda_iou, m.lambda_l1,
                                        m.max_wh, m.assign_center_thr, m.assign_pos_thr)
    model.det_loss_tir = DetectionLossV4(m.num_classes, m.lambda_iou, m.lambda_l1,
                                        m.max_wh, m.assign_center_thr, m.assign_pos_thr)
    model.n_cand_late = m.n_cand
    if hasattr(model, "_affine_A"):
        model._affine_A.requires_grad_(False)      # v4 has no fitted affine prior


def mask_for(batch, name):
    mask = batch.get("mask_" + name, batch["mask"])
    available = batch.get("modality_available")
    if available is not None:
        mask = mask * available[:, 0 if name == "rgb" else 1, None]
    return mask


def _other_available(batch, name):
    """``(B,1,1)`` availability of the *other* modality (1 when not annotated)."""
    available = batch.get("modality_available")
    if available is None:
        return None
    idx = 1 if name == "rgb" else 0
    return available[:, idx].reshape(-1, 1, 1)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def forward(model, batch):
    cfg = model.cfg
    m = cfg.model
    per = {}
    for name in ("rgb", "tir"):
        branch = getattr(model, name + "_branch")
        mask = mask_for(batch, name) if "mask" in batch else None
        obj = branch.objects(batch[name], batch.get("centers_" + name), mask, model.training)
        obj["self_pred"] = branch.predict(obj["token"], obj["roi"], obj["cand"]["boxes"])
        per[name] = obj

    aux = {}
    if model.v4_on:
        for name, other, direction in (("rgb", "tir", "r2t"), ("tir", "rgb", "t2r")):
            self_obj, other_obj = per[name], per[other]
            message, stats = model.match[direction](
                self_obj["token"], other_obj["token"], other_obj["cand"]["boxes"],
                other_obj["feats"])
            fused = model.td[direction](
                self_obj["token"], message,
                quality=self_obj["cand"]["quality"],
                quality_other=other_obj["cand"]["quality"],
                margin=stats["margin"], other_available=_other_available(batch, name))
            # localization sees only a channel modulation of ITS OWN RoI descriptor
            modulated = model.td[direction].modulate(self_obj["roi"], fused["r"], fused["gamma"])
            per[name]["fused_pred"] = getattr(model, name + "_branch").predict(
                fused["f_cls"], modulated, self_obj["cand"]["boxes"])
            aux["sim_" + name] = stats["sim"]
            aux["margin_" + name] = stats["margin"]
            aux["anchor_" + name] = stats["anchor_idx"]
            aux["r_" + name] = fused["r"]
            aux["gamma_" + name] = fused["gamma"]
            aux["alpha_" + name] = fused["alpha"]

    out = {"batch": batch, "aux": aux}
    for name in ("rgb", "tir"):
        cand = per[name]["cand"]
        out["cand_" + name] = cand
        out["self_pred_" + name] = per[name]["self_pred"]
        out["pred_" + name] = per[name].get("fused_pred", per[name]["self_pred"])
        out["heat_" + name] = cand["dense"][m.cand_levels[0]][:, :1]
        out["dense_" + name] = cand["dense"]
        out["roi_" + name] = per[name]["roi"]
        out["token_" + name] = per[name]["token"]
        out["sel_" + name] = {
            "idx": torch.arange(cand["pts"].shape[1], device=cand["pts"].device),
            "pts": cand["pts"], "boxes": cand["boxes"],
            "objectness": cand["objectness"], "quality": cand["quality"],
            "levels": cand["levels"], "injected": cand["injected"],
            "cls_logits": cand["cls_logits"],
        }
        aux["pre_box_" + name] = cand["boxes"]
    return out


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def matching_loss(model, sim, matches):
    """``CE(A, target)`` over the query rows that actually have a correspondent.

    Rows without any true match carry no information under a "no-key" hypothesis
    that does not exist in a plain ``[B,K,K]`` similarity matrix, so they are
    excluded rather than assigned a fabricated target.
    """
    target = matches.to(sim.dtype)
    positive = matches.any(-1)
    if not bool(positive.any()):
        return sim.sum() * 0.0
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    logp = F.log_softmax(sim, -1)
    ce = -(target * logp).sum(-1)
    return ce[positive].mean()


def consistency_loss(model, logits_self, logits_other, sim_self, sim_other, matches,
                     progress: float):
    """``JS(p_self || p_other)`` on true pairs and on confident mutual selections."""
    m = model.cfg.model
    b, k, c = logits_self.shape
    idx = torch.arange(k, device=logits_self.device)[None].expand(b, -1)
    pairs = torch.full_like(idx, -1)

    if matches is not None:
        has = matches.any(-1)
        chosen = matches.to(torch.int64).argmax(-1)
        pairs = torch.where(has, chosen, pairs)

    if progress >= float(m.cons_start_fraction) and k > 1:
        # mutual top-1 selection: self picks j, j picks self back, with a clear margin
        top1 = sim_self.argmax(-1)
        back = sim_other.argmax(-1).gather(1, top1)
        best = sim_self.gather(2, top1[..., None]).squeeze(-1)
        second = sim_self.topk(2, -1).values[..., 1]
        mutual = (back == idx) & ((best - second) >= float(m.cons_margin)) & (pairs < 0)
        pairs = torch.where(mutual, top1, pairs)

    valid = pairs >= 0
    if not bool(valid.any()):
        return logits_self.sum() * 0.0
    p = class_probabilities(logits_self, m.num_classes)
    q = class_probabilities(logits_other, m.num_classes)
    q_sel = q.gather(1, pairs.clamp_min(0)[..., None].expand(-1, -1, q.shape[-1]))
    return js_divergence(p, q_sel)[valid].mean()


def compute_losses(model, out):
    m = model.cfg.model
    batch = out["batch"]
    terms = {}
    total = out["pred_rgb"].sum() * 0.0
    available = batch.get("modality_available")

    for name in ("rgb", "tir"):
        mask = mask_for(batch, name)
        sel = out["sel_" + name]
        cand = out["cand_" + name]
        boxes = batch["boxes_" + name]
        centers, sizes = batch["centers_" + name], batch["sizes_" + name]
        labels = batch.get("labels_" + name)
        final_raw, self_raw = out["pred_" + name], out["self_pred_" + name]

        if available is not None:
            active = available[:, 0 if name == "rgb" else 1] > 0.5
            if not active.any():
                # A dropped sensor is not a background-only image: no supervision at all.
                zero = final_raw.sum() * 0.0
                for key in ("loss", "qfl", "iou", "l1", "pos_ratio", "self", "prop", "heat"):
                    terms[name + "_" + key] = zero.detach()
                continue
            b_full = final_raw.shape[0]
            final_raw, self_raw = final_raw[active], self_raw[active]
            mask, boxes = mask[active], boxes[active]
            centers, sizes = centers[active], sizes[active]
            if labels is not None:
                labels = labels[active]
            sel = {k: (v[active] if isinstance(v, torch.Tensor) and v.dim() > 0
                       and v.shape[0] == b_full else v) for k, v in sel.items()}
            cand = {k: (v[active] if isinstance(v, torch.Tensor) and v.dim() > 0
                        and v.shape[0] == b_full else v) for k, v in cand.items()}
            cand["dense"] = {k: v[active] for k, v in out["dense_" + name].items()}

        loss_fn = getattr(model, "det_loss_" + name)
        final = loss_fn(final_raw, sel["pts"], sel["boxes"], boxes, mask, labels)
        total = total + final["loss"]
        for k, v in final.items():
            terms[name + "_" + k] = v.detach()

        self_term = loss_fn(self_raw, sel["pts"], sel["boxes"], boxes, mask, labels)
        total = total + m.w_self * self_term["loss"]
        terms[name + "_self"] = self_term["loss"].detach()

        prop = candidate_losses_v4(cand, boxes, mask, labels, m.num_classes, m.max_wh,
                                   m.lambda_pbox, m.lambda_quality, centers, sizes, m.w_heat)
        total = total + m.w_prop * prop["loss"]
        terms[name + "_prop"] = prop["loss"].detach()
        terms[name + "_heat"] = prop["heat"].detach()

    # ---- cross-modal terms (only with an explicit correspondence source) ---- #
    if model.v4_on:
        match_term = total.new_zeros(())
        cons_term = total.new_zeros(())
        progress = float(getattr(model, "progress_fraction", 0.0))
        for name, other in (("rgb", "tir"), ("tir", "rgb")):
            sim_self = out["aux"]["sim_" + name]
            sim_other = out["aux"]["sim_" + other]
            matches = None
            if batch.get("corr_pairs") is not None and "mask" in batch:
                matches = identity_matches(
                    out["sel_" + name]["pts"], batch["boxes_" + name], mask_for(batch, name),
                    out["sel_" + other]["pts"], batch["boxes_" + other], mask_for(batch, other),
                    batch["corr_pairs"], m.assign_center_thr, batch.get("valid"))
            if m.use_match_loss and matches is not None:
                match_term = match_term + matching_loss(model, sim_self, matches)
            if m.use_cons_loss:
                cons_term = cons_term + consistency_loss(
                    model, out["pred_" + name][..., 4:], out["pred_" + other][..., 4:],
                    sim_self, sim_other, matches, progress)
        if m.use_match_loss:
            total = total + m.lambda_match * match_term
            terms["match"] = match_term.detach()
        if m.use_cons_loss:
            total = total + m.w_cons * cons_term
            terms["cons"] = cons_term.detach()

    terms["total"] = total.detach()
    return total, terms


__all__ = ["RegionBranchV4", "build_modules", "forward", "compute_losses",
           "mask_for", "matching_loss", "consistency_loss"]
