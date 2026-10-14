"""PGSF-Det **v3**: independent detection -> object-level correspondence -> rejection fusion.

Data flow (identical for both modalities, only the weights differ)::

    image -> backbone -> {P2..P5}
          -> MSCE on P2 -> PAN-FPN (all levels 128ch)
          -> P2/P3 dense center + coarse box + quality
          -> 128 sparse candidates  (+ 8x8 grid tokens as extra retrieval keys)
          -> 128-d RoI object token per candidate (P2-P5 sampling around the coarse box)
          -> [ ONE object-level fusion ]  bidirectional correspondence + rejection gate
          -> classification on the enhanced token
             box regression on (enhanced token, own RoI descriptor, own coarse box)

Differences from v2, by design:

* **No fixed affine prior.**  Correspondence is retrieved by a learned 64-d Q/K dot
  product, and only the *uncertainty* of the resulting match is modelled
  (a per-anchor Gaussian), never a global RGB->TIR warp fitted on the training
  split.  ``_affine_A`` is retained for checkpoint compatibility but is frozen and
  never read by the forward pass -- ``tests/test_v3.py`` asserts exactly that.
* **No intermediate feature-map write-back.**  Only object tokens are enhanced.
* **One fusion, not two.**  The second (late) Transformer of v2 is gone.  What v2
  did in two stages now happens once, at object level.
* **Independent output coordinate systems.**  Each modality regresses its own box
  from its own coarse box; the counterpart box is not an input and cannot be copied.
* **Conditional correspondence supervision.**  Matching / retrieval / geometric
  terms are computed only when the batch actually carries instance correspondence
  (``corr_pairs``).  Without such labels the objective degenerates to the two
  per-modality detection losses, which is the honest thing to do when the pairing
  is unknown.
"""

from __future__ import annotations

import torch
from torch import nn

from pgsf.losses import (draw_gaussian, gaussian_focal_loss, geometry_nll,
                         supervised_association_loss)
from pgsf.models.correspondence_v3 import BidirectionalCorrespondence, match_target_center
from pgsf.models.head import MultiScaleBoxRefine
from pgsf.models.head_v3 import DetHeadV3, DetectionLossV3
from pgsf.models.late_fusion import select_topk_candidates
from pgsf.models.msce import LiteMSCE
from pgsf.models.neck import PANFPN
from pgsf.models.proposal_v3 import (CandidateV3, DenseProposalHeadV3, GridTokens,
                                    proposal_losses)


# --------------------------------------------------------------------------- #
# per-modality branch
# --------------------------------------------------------------------------- #
class RegionBranchV3(nn.Module):
    """Everything per-modality: backbone, MSCE, neck, proposals, RoI tokens, head."""

    def __init__(self, in_ch: int, cfg):
        super().__init__()
        from pgsf.models.detector import LightBackboneCtor
        m = cfg.model
        self.backbone = LightBackboneCtor(in_ch, m)
        self.msce_on = bool(m.use_msce)
        if self.msce_on:
            self.msce = LiteMSCE(self.backbone.out_channels[0], m.msce_width, m.msce_alpha_max)
        self.neck = PANFPN(self.backbone.out_channels, m.neck_dim, m.neck_out, m.neck_repeats)
        self.heat = DenseProposalHeadV3(m.neck_out, m.proposal_width, m.max_wh)
        self.cand = CandidateV3(m.neck_out, m.n_cand, m.inject_gt_ratio, m.max_wh, m.nms_kernel)
        self.grid = GridTokens(m.neck_out, m.v3_grid, m.v3_grid_levels)
        # The 128-d object token = candidate feature + RoI spatial context + box geometry.
        self.refine = MultiScaleBoxRefine(m.neck_out, n_levels=4, hidden=m.roi_width)
        self.roi_project = nn.Sequential(nn.LayerNorm(m.roi_width),
                                         nn.Linear(m.roi_width, m.neck_out))
        self.box_position = nn.Sequential(nn.Linear(4, m.neck_out), nn.SiLU(),
                                          nn.Linear(m.neck_out, m.neck_out))
        nn.init.normal_(self.roi_project[-1].weight, std=0.01)
        nn.init.zeros_(self.roi_project[-1].bias)
        nn.init.normal_(self.box_position[-1].weight, std=0.01)
        nn.init.zeros_(self.box_position[-1].bias)
        self.det = DetHeadV3(m.neck_out, m.roi_width, m.head_width, m.n_layers_head,
                             m.num_classes, m.max_wh)

    def features(self, x: torch.Tensor):
        f = self.backbone(x)
        if self.msce_on:
            f["P2"], _ = self.msce(f["P2"])
        return self.neck(f)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_modules(model, cfg) -> None:
    m = cfg.model
    model.rgb_branch = RegionBranchV3(3, cfg)
    model.tir_branch = RegionBranchV3(1, cfg)
    model.v3_on = bool(m.use_v3_fusion) and cfg.stage >= 2
    if model.v3_on:
        model.corr = BidirectionalCorrespondence(
            dim=m.neck_out, retrieval_dim=m.retrieval_dim, top_k=m.corr_top_k,
            samples=m.corr_samples, levels=m.corr_levels, heads=m.corr_heads,
            ff_mult=m.corr_ff_mult, sigma_floor=m.corr_sigma_floor,
            lambda_local=m.lambda_local, use_null=m.use_null, use_gate=m.use_gate)
    model.det_loss_rgb = DetectionLossV3(
        m.w_cls, m.w_box, m.w_nwd, m.use_nwd, m.assign_pos_thr, m.assign_center_thr,
        m.max_wh, m.num_classes)
    model.det_loss_tir = DetectionLossV3(
        m.w_cls, m.w_box, m.w_nwd, m.use_nwd, m.assign_pos_thr, m.assign_center_thr,
        m.max_wh, m.num_classes)
    model.n_cand_late = m.n_cand_late
    # The v2 fixed affine is not part of v3; freeze it so it never enters the optimiser.
    if hasattr(model, "_affine_A"):
        model._affine_A.requires_grad_(False)


def mask_for(batch, name):
    mask = batch.get("mask_" + name, batch["mask"])
    available = batch.get("modality_available")
    if available is not None:
        mask = mask * available[:, 0 if name == "rgb" else 1, None]
    return mask


def select_regions(branch, maps, cand, k):
    """Top-``k`` candidates -> 128-d RoI object tokens (own modality, own coordinates)."""
    selected = select_topk_candidates(cand, cand["boxes"], k)
    ctx = branch.refine(maps, selected["boxes"][..., :2], selected["boxes"])
    position = torch.cat((selected["boxes"][..., :2],
                          selected["boxes"][..., 2:].clamp_min(1e-5).log()), -1)
    selected["feats"] = (selected["feats"] + branch.roi_project(ctx)
                         + branch.box_position(position))
    selected["roi"] = ctx
    return selected


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def forward(model, batch):
    cfg = model.cfg
    m = cfg.model
    features, dense, candidates, grids = {}, {}, {}, {}
    for name in ("rgb", "tir"):
        branch = getattr(model, name + "_branch")
        features[name] = branch.features(batch[name])
        dense[name] = branch.heat(features[name]["P2"], features[name]["P3"])
        mask = mask_for(batch, name) if "mask" in batch else None
        candidates[name] = branch.cand(dense[name], features[name],
                                       batch.get("centers_" + name), mask, model.training)
        grids[name] = branch.grid(features[name])

    selected = {name: select_regions(getattr(model, name + "_branch"), features[name],
                                     candidates[name], m.n_cand_late)
                for name in ("rgb", "tir")}

    aux = {}
    if model.v3_on:
        # Both directions read the *unmodified* candidate tokens, so neither
        # direction sees the other's enhancement (no ordering side effects).
        results = {}
        for name, other, direction in (("rgb", "tir", "r2t"), ("tir", "rgb", "t2r")):
            sel = selected[name]
            have_mask = "mask" in batch
            results[name] = model.corr(
                sel["feats"], sel["pts"],
                candidates[other]["feats"], candidates[other]["pts"],
                grids[other][0], grids[other][1], features[other], direction=direction,
                other_dense=dense[other],
                other_boxes=batch.get("boxes_" + other),
                other_mask=mask_for(batch, other) if have_mask else None,
                q_boxes=batch.get("boxes_" + name),
                q_mask=mask_for(batch, name) if have_mask else None,
                corr_pairs=batch.get("corr_pairs"),
                frame_valid=batch.get("valid"),
                center_thr=m.assign_center_thr)
        for name in ("rgb", "tir"):
            selected[name]["feats"] = results[name]["feats"]
            aux["corr_stats_" + name] = results[name]["stats"]

    out = {"batch": batch, "aux": aux}
    for name in ("rgb", "tir"):
        branch = getattr(model, name + "_branch")
        out["pred_" + name] = branch.det(selected[name]["feats"], selected[name]["roi"],
                                         selected[name]["boxes"])
        out["cand_" + name] = candidates[name]
        out["sel_" + name] = selected[name]
        out["heat_" + name] = dense[name]["P2"][:, :1]
        out["dense_" + name] = dense[name]
        out["grid_" + name] = grids[name]
        aux["pre_box_" + name] = candidates[name]["boxes"]
    return out


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def corr_objectives(model, stats, batch, name, other, q_scores):
    """Matching / retrieval / geometric terms for one direction.

    All three are driven only by instance correspondence (``corr_pairs``).  When
    that key is absent the caller skips this function entirely, so a dataset
    without pairing annotations trains on the two detection losses alone.
    """
    m = model.cfg.model
    b = stats["enhanced"].shape[0]
    if "matches_key" not in stats:
        zero = stats["enhanced"].sum() * 0.0
        return {"null": zero, "retr": zero, "geo": zero}

    matches_key = stats["matches_key"]                       # (B,N,M) bool
    matches_anchor = stats["matches_anchor"]                 # (B,N,A) bool
    a, n_local = stats["n_anchor"], stats["n_local_key"]
    per = max(n_local // max(a, 1), 1)
    matches_local = (matches_anchor.unsqueeze(-1).expand(-1, -1, -1, per)
                     .reshape(b, matches_key.shape[1], n_local))
    matches_keys = torch.cat((matches_anchor, matches_local), -1)

    # --- (a) rejection: should this query attend a real key, or the null token? ---
    weights = 1.0 + m.corr_hard_negative_weight * q_scores.detach()
    null = supervised_association_loss(stats["attn"], matches_keys, m.use_null, weights)

    # --- (b) coarse retrieval: the correspondent must win the 64-d Q/K softmax ---
    logits = stats["retrieval_logits"]
    m_other = stats["n_cand_other"]
    positive = matches_key.any(-1)
    probability = logits.softmax(-1)
    mass = (probability[..., :m_other] * matches_key.to(probability.dtype)).sum(-1)
    if positive.any():
        retr = -mass[positive].clamp_min(1e-8).log().mean()
    else:
        retr = logits.sum() * 0.0

    # --- (c) geometry: the anchor Gaussian must put mass on the counterpart ---
    mu = stats["mu"][..., 0, :]                              # top-1 anchor
    L = stats["sigma"][..., 0, :, :]
    cov = L @ L.transpose(-1, -2)
    target = match_target_center(matches_key, stats["key_slot"], batch["centers_" + other])
    # ``positive`` already requires both sides to have a valid identity, so
    # clipped / unmatched queries never contribute a gradient here.
    geo = geometry_nll(mu, cov, target, positive,
                       minimum_variance=model.corr.sigma_floor ** 2)
    return {"null": null, "retr": retr, "geo": geo}


def compute_losses(model, out):
    m = model.cfg.model
    batch = out["batch"]
    terms = {}
    total = out["pred_rgb"].sum() * 0.0

    for name in ("rgb", "tir"):
        mask = mask_for(batch, name)
        sel, cand = out["sel_" + name], out["cand_" + name]
        raw, boxes = out["pred_" + name], batch["boxes_" + name]
        centers, sizes, dense = batch["centers_" + name], batch["sizes_" + name], out["dense_" + name]
        labels = batch.get("labels_" + name)
        available = batch.get("modality_available")
        if available is not None:
            active = available[:, 0 if name == "rgb" else 1] > 0.5
            if not active.any():
                # A dropped sensor is not a labelled background-only image.
                zero = raw.sum() * 0.0
                for key in ("loss_cls", "loss_box", "loss_nwd", "loss", "pos_ratio",
                            "pre_box", "pre_quality", "heat"):
                    terms[name + "_" + key] = zero.detach()
                continue
            raw, mask, boxes = raw[active], mask[active], boxes[active]
            centers, sizes = centers[active], sizes[active]
            if labels is not None:
                labels = labels[active]
            sel = {k: (v[active] if isinstance(v, torch.Tensor) else v) for k, v in sel.items()}
            dense = {k: v[active] for k, v in dense.items()}
            cand = {k: cand[k][active] for k in ("pts", "boxes", "quality_logits")}
            cand["dense"] = dense

        det = getattr(model, "det_loss_" + name)(raw, sel["pts"], boxes, mask,
                                                gt_labels=labels, base_boxes=sel["boxes"],
                                                quality_aware=True)
        total = total + m.w_det * det["loss"]
        for k, v in det.items():
            terms[name + "_" + k] = v.detach()

        pre = proposal_losses(cand, boxes, mask, m.use_nwd, m.max_wh)
        total = total + m.w_pre_box * pre["box"] + m.w_pre_quality * pre["quality"]
        terms[name + "_pre_box"] = pre["box"].detach()
        terms[name + "_pre_quality"] = pre["quality"].detach()

        heat = total.new_zeros(())
        for raw_dense in dense.values():
            target = draw_gaussian(torch.zeros_like(raw_dense[:, :1]), centers, sizes, mask)
            heat = heat + gaussian_focal_loss(raw_dense[:, :1], target)
        total = total + m.w_heat * heat
        terms[name + "_heat"] = heat.detach()

    if model.v3_on:
        corr = total.new_zeros(())
        terms_corr = {"null": total.new_zeros(()), "retr": total.new_zeros(()),
                      "geo": total.new_zeros(())}
        if batch.get("corr_pairs") is not None:
            for name, other in (("rgb", "tir"), ("tir", "rgb")):
                stats = out["aux"]["corr_stats_" + name]
                part = corr_objectives(model, stats, batch, name, other,
                                       out["sel_" + name]["scores"])
                for k, v in part.items():
                    terms_corr[k] = terms_corr[k] + v
            corr = terms_corr["null"] + terms_corr["retr"] + m.w_corr_geo * terms_corr["geo"]
        total = total + m.w_corr * corr
        terms["corr"] = corr.detach()
        for k, v in terms_corr.items():
            terms["corr_" + k] = v.detach()

    terms["total"] = total.detach()
    return total, terms


__all__ = ["RegionBranchV3", "build_modules", "forward", "compute_losses",
           "corr_objectives", "select_regions", "mask_for"]
