"""PGSF-Det v4 proposals: P2-P5 class-quality heads, K cross-scale candidates, multi-scale RoIAlign.

Three pieces, matching the v4 spec:

``DenseProposalHeadV4``
    Every level (P2-P5, all 128 channels) predicts
    ``[C class-quality logits | 2-d centre offset | 2-d width/height | 1-d proposal quality]``.

``MultiScaleCandidatesV4``
    Picks ``K`` candidates across the four levels: a small **reserved quota per
    level** guarantees spatial/scale coverage, and the remaining budget is filled
    by **global score ranking**.  Scores are the class-quality maximum, never the
    proposal-quality channel -- the v1/v2 lesson was that gating candidate recall
    by a poorly calibrated quality head hides true centres, after which they
    receive no positive supervision and the head collapses.

``MultiScaleRoIAlign``
    Turns each candidate into a ``[B,K,128,r,r]`` descriptor by RoIAlign on the
    FPN level the candidate's own box size selects.  ``torchvision.ops.roi_align``
    is the reference path; a vectorised ``grid_sample`` fallback keeps the module
    usable on minimal torchvision builds and is checked against the reference in
    ``tests/test_v4.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import DWConvBNAct
from pgsf.models.candidate import bilinear_grid, local_nms_topk, sample_points
from pgsf.losses import assign_candidates, ciou_loss, draw_gaussian, gaussian_focal_loss, quality_focal_loss
from pgsf.models.proposal import aligned_iou


# --------------------------------------------------------------------------- #
# dense head
# --------------------------------------------------------------------------- #
class DenseProposalHeadV4(nn.Module):
    """Per-level dense proposal head: ``C + 5`` channels.

    Channel layout (the single place it is defined; decode and loss both use it)::

        [0 : C]        class-quality logits
        [C : C+2]      centre offset inside the cell (tanh, half a cell)
        [C+2 : C+4]    width / height (sigmoid, scaled by ``max_wh``)
        [C+4]          proposal quality logit
    """

    def __init__(self, channels: int = 128, hidden: int = 32, num_classes: int = 1,
                 max_wh: float = 0.5, levels=("P2", "P3", "P4", "P5")):
        super().__init__()
        self.levels = tuple(levels)
        self.num_classes = int(num_classes)
        self.max_wh = max_wh
        self.out_channels = self.num_classes + 5
        self.in_ch = int(channels)
        self.hidden = int(hidden)
        self.heads = nn.ModuleDict({
            name: _level_head(channels, hidden, self.out_channels, self.num_classes, max_wh)
            for name in self.levels})

    def forward(self, maps):
        return {name: module(maps[name]) for name, module in self.heads.items()}


def _level_head(channels, hidden, out_channels, num_classes, max_wh):
    h = nn.Sequential(DWConvBNAct(channels, hidden), DWConvBNAct(hidden, hidden),
                      nn.Conv2d(hidden, out_channels, 1))
    nn.init.normal_(h[-1].weight, std=0.01)
    nn.init.zeros_(h[-1].bias)
    with torch.no_grad():
        h[-1].bias[:num_classes] = -4.0                   # low initial objectness
        h[-1].bias[num_classes + 2:num_classes + 4] = math.log(0.03 / max_wh / (1 - 0.03 / max_wh))
    return h


def decode_coarse_v4(raw: torch.Tensor, pts: torch.Tensor, step: torch.Tensor,
                     num_classes: int = 1, max_wh: float = 0.5) -> torch.Tensor:
    """Raw proposal output at ``pts`` -> coarse box ``(cx,cy,w,h)``."""
    center = pts + raw[..., num_classes:num_classes + 2].tanh() * step * .5
    size = raw[..., num_classes + 2:num_classes + 4].sigmoid() * max_wh
    return torch.cat((center, size), -1)


def objectness(raw: torch.Tensor, num_classes: int = 1) -> torch.Tensor:
    """Candidate score = max class-quality probability (never the quality channel).

    Works on both layouts: dense maps ``(B,C,H,W)`` (class dim 1) and per-candidate
    rows ``(B,K,C+5)`` (class dim last).  Slicing ``raw[..., :C]`` would silently
    take the *last* axis and produce a channel-shaped score, so the axis is chosen
    from the rank and sliced with ``narrow``.
    """
    dim = 1 if raw.dim() == 4 else raw.dim() - 1
    return raw.narrow(dim, 0, num_classes).sigmoid().amax(dim)


# --------------------------------------------------------------------------- #
# candidate selection
# --------------------------------------------------------------------------- #
class MultiScaleCandidatesV4(nn.Module):
    """``K`` candidates from P2-P5: reserved per-level quota + global ranking."""

    def __init__(self, channels: int = 128, k: int = 128, levels=("P2", "P3", "P4", "P5"),
                 min_per_level: int = 8, nms_kernel: int = 5, num_classes: int = 1,
                 inject_gt_ratio: float = 0.25, max_wh: float = 0.5):
        super().__init__()
        self.levels = tuple(levels)
        self.k = int(k)
        self.min_per_level = int(min_per_level)
        self.nms_kernel = int(nms_kernel)
        self.num_classes = int(num_classes)
        self.inject_gt_ratio = float(inject_gt_ratio)
        self.max_wh = max_wh
        if self.min_per_level * len(self.levels) > self.k:
            raise ValueError("per-level quota exceeds the candidate budget")
        self.register_buffer("_blank", torch.zeros(()))

    def forward(self, dense, maps, gt_centers=None, gt_mask=None, training=False):
        b = dense[self.levels[0]].shape[0]
        reserved_pts, reserved_score, reserved_level = [], [], []
        pool_pts, pool_score, pool_level = [], [], []

        for lv, name in enumerate(self.levels):
            raw = dense[name]
            hh, ww = raw.shape[-2:]
            score = objectness(raw, self.num_classes)                       # (B,H,W)
            peaks = max(1, min(self.k, hh * ww))
            idx, sc = local_nms_topk(score[:, None], peaks, self.nms_kernel)
            grid = bilinear_grid(hh, ww, raw.device, torch.float32)
            pts_all = grid[idx]                                             # (B,peaks,2)
            keep = min(self.min_per_level, pts_all.shape[1])
            reserved_pts.append(pts_all[:, :keep])
            reserved_score.append(sc[:, :keep])
            reserved_level.append(torch.full_like(idx[:, :keep], lv))
            if pts_all.shape[1] > keep:
                pool_pts.append(pts_all[:, keep:])
                pool_score.append(sc[:, keep:])
                pool_level.append(torch.full_like(idx[:, keep:], lv))

        need = self.k - self.min_per_level * len(self.levels)
        if need > 0:
            if not pool_pts:
                raise RuntimeError("no spare peaks to fill the candidate budget")
            pts = torch.cat(pool_pts, 1)
            score = torch.cat(pool_score, 1)
            level = torch.cat(pool_level, 1)
            if pts.shape[1] < need:
                raise RuntimeError(f"only {pts.shape[1]} spare peaks for a budget of {need}")
            order = score.topk(need, dim=1).indices
            reserved_pts.append(pts.gather(1, order[..., None].expand(-1, -1, 2)))
            reserved_score.append(score.gather(1, order))
            reserved_level.append(level.gather(1, order))

        pts = torch.cat(reserved_pts, 1)                                    # (B,K,2)
        scores = torch.cat(reserved_score, 1)
        levels = torch.cat(reserved_level, 1)
        eps = 0.5 / float(max(maps[self.levels[0]].shape[-1], maps[self.levels[0]].shape[-2]))
        pts = pts.clamp(eps, 1.0 - eps)

        injected = torch.zeros_like(scores)
        if training and self.inject_gt_ratio > 0 and gt_centers is not None:
            pts, levels, injected = self._inject(pts, levels, scores, gt_centers, gt_mask)

        raw = self._gather_raw(dense, pts, levels)
        step = self._gather_step(dense, pts, levels)
        boxes = decode_coarse_v4(raw, pts, step, self.num_classes, self.max_wh)
        return {
            "pts": pts, "boxes": boxes, "raw": raw,
            "scores": objectness(raw, self.num_classes) * ((scores > 0) | (injected > .5)),
            "objectness": objectness(raw, self.num_classes),
            "quality": raw[..., self.num_classes + 4].sigmoid(),
            "quality_logits": raw[..., self.num_classes + 4],
            "cls_logits": raw[..., :self.num_classes],
            "levels": levels, "injected": injected, "dense": dense,
            "level_names": self.levels,
        }

    # ------------------------------------------------------------------ #
    def _gather_step(self, dense, pts, levels):
        feats = []
        for lv, name in enumerate(self.levels):
            hh, ww = dense[name].shape[-2:]
            step = torch.full_like(pts, 0.)
            step[..., 0] = 1.0 / max(ww - 1, 1)
            step[..., 1] = 1.0 / max(hh - 1, 1)
            feats.append(step)
        steps = torch.stack(feats, 0)                                       # (L,B,K,2)
        idx = levels[None, ..., None].expand(1, -1, -1, 2)
        return steps.gather(0, idx).squeeze(0)

    def _gather_raw(self, dense, pts, levels):
        """Bilinear-sample each candidate from *its own* level."""
        gathered = [sample_points(dense[name], pts) for name in self.levels]
        stacked = torch.stack(gathered, 0)                                  # (L,B,K,C+5)
        idx = levels[None, ..., None].expand(1, -1, -1, stacked.shape[-1])
        return stacked.gather(0, idx).squeeze(0)

    def _inject(self, pts, levels, scores, gt, mask):
        pts = pts.clone()
        levels = levels.clone()
        flags = torch.zeros_like(scores)
        if mask is None:
            mask = gt.abs().sum(-1) > 0
        n = min(self.k - 1, max(1, round(self.k * self.inject_gt_ratio)))
        for bi in range(pts.shape[0]):
            valid = gt[bi][mask[bi] > 0.5]
            if valid.numel() == 0:
                continue
            target = valid[torch.arange(n, device=gt.device) % len(valid)].clone()
            exact = min(n, len(valid))
            target[exact:] += torch.randn_like(target[exact:]) * 0.005
            replace = scores[bi].argsort()[:n]
            pts[bi, replace] = target.clamp(0, 1).to(pts.dtype)
            levels[bi, replace] = 0        # injected points are read from the finest level
            flags[bi, replace] = 1
        return pts, levels, flags


# --------------------------------------------------------------------------- #
# multi-scale RoIAlign
# --------------------------------------------------------------------------- #
def assign_levels(boxes: torch.Tensor, level_shapes, k0: float = 4.0) -> torch.Tensor:
    """FPN-style level assignment by box area, in normalised coordinates.

    ``level_shapes`` is a list of ``(H, W)`` per level.  A box of normalised area
    ``a`` is assigned to the level whose cell area is closest to ``a / k0`` in log
    space.  Everything is expressed as a fraction of the image, so no camera
    parameter and no cross-modal coordinate is involved.
    """
    area = (boxes[..., 2].clamp_min(1e-8) * boxes[..., 3].clamp_min(1e-8))
    cells = torch.tensor([1.0 / (h * w) for h, w in level_shapes],
                         dtype=boxes.dtype, device=boxes.device)
    ratio = torch.log(area[..., None] / (k0 * cells)[None, None, :].clamp_min(1e-12))
    return ratio.abs().argmin(-1)


class MultiScaleRoIAlign(nn.Module):
    """``boxes [B,K,4]`` (normalised cx,cy,w,h) -> ``[B,K,C,r,r]`` multi-scale RoI features."""

    def __init__(self, channels: int = 128, levels=("P2", "P3", "P4", "P5"),
                 out_size: int = 3, sampling_ratio: int = 2, k0: float = 4.0):
        super().__init__()
        self.channels = int(channels)
        self.levels = tuple(levels)
        self.out_size = int(out_size)
        self.sampling_ratio = int(sampling_ratio)
        self.k0 = k0

    def forward(self, maps, boxes):
        out_shape = boxes.shape[:2] + (self.channels, self.out_size, self.out_size)
        out = boxes.new_zeros(out_shape)
        level_shapes = [tuple(maps[n].shape[-2:]) for n in self.levels]
        assign = assign_levels(boxes, level_shapes, self.k0)                 # (B,K)
        b, k, _ = boxes.shape
        for lv, name in enumerate(self.levels):
            feat = maps[name]
            hh, ww = feat.shape[-2:]
            sel = (assign == lv)                                            # (B,K)
            if not bool(sel.any()):
                continue
            rows = sel.nonzero(as_tuple=False)                              # (N,2)
            bidx = rows[:, 0]
            box = boxes[bidx, rows[:, 1]]
            x0 = (box[:, 0] - box[:, 2] / 2) * ww
            y0 = (box[:, 1] - box[:, 3] / 2) * hh
            x1 = (box[:, 0] + box[:, 2] / 2) * ww
            y1 = (box[:, 1] + box[:, 3] / 2) * hh
            rois = torch.stack((bidx.to(box.dtype), x0, y0, x1, y1), -1)
            sampled = roi_align(feat, rois, self.out_size, self.sampling_ratio)   # (N,C,r,r)
            # out-of-place scatter: an in-place ``out[...] = sampled`` would not
            # record autograd for ``sampled`` and would silently detach the whole
            # RoI path (its parameters would then be dead).
            out = out.index_put((bidx, rows[:, 1]), sampled, accumulate=False)
        return out


def roi_align(feat: torch.Tensor, rois: torch.Tensor, out_size: int,
              sampling_ratio: int = 2) -> torch.Tensor:
    """RoIAlign with the boxes already expressed in feature-map pixels.

    ``rois`` is ``(N,5)`` as ``[batch_index, x0, y0, x1, y1]``.  Uses
    ``torchvision.ops.roi_align`` when available (the reference implementation) and
    falls back to a vectorised bilinear ``grid_sample`` otherwise.
    """
    try:
        from torchvision.ops import roi_align as tv_roi_align
        return tv_roi_align(feat, rois, output_size=(out_size, out_size),
                            spatial_scale=1.0, sampling_ratio=sampling_ratio,
                            aligned=True)
    except (ImportError, RuntimeError):
        return _roi_align_grid_sample(feat, rois, out_size)


def _roi_align_grid_sample(feat: torch.Tensor, rois: torch.Tensor,
                           out_size: int) -> torch.Tensor:
    """Reference-equivalent RoIAlign via ``grid_sample``.

    Convention: a box coordinate ``x`` lives in the continuous space where feature
    pixel ``i`` covers ``[i, i+1)`` -- the same convention ``torchvision.ops.roi_align``
    uses for ``aligned=True``.  That maps to ``grid_sample`` with
    ``align_corners=False`` and ``gx = 2 * x / W - 1``; using the
    ``align_corners=True`` mapping instead would introduce a systematic half-pixel
    shift, which is exactly the kind of silent misalignment ROIAlign is supposed
    to make impossible.
    """
    n = rois.shape[0]
    if n == 0:
        return feat.new_zeros(0, feat.shape[1], out_size, out_size)
    idx = rois[:, 0].long()
    boxes = rois[:, 1:5]
    hh, ww = feat.shape[-2:]
    # bin centres of an ``out_size x out_size`` grid inside each box
    step = (torch.arange(out_size, dtype=feat.dtype, device=feat.device) + 0.5) / out_size
    xs = boxes[:, 0:1] + (boxes[:, 2:3] - boxes[:, 0:1]) * step[None, :]    # (N,out)
    ys = boxes[:, 1:2] + (boxes[:, 3:4] - boxes[:, 1:2]) * step[None, :]    # (N,out)
    px = xs[:, None, :].expand(n, out_size, out_size)                       # (N,out,out)
    py = ys[:, :, None].expand(n, out_size, out_size)
    grid = torch.stack((px * 2 / ww - 1, py * 2 / hh - 1), -1)              # (N,out,out,2)
    feats = feat.index_select(0, idx)                                       # (N,C,H,W)
    return F.grid_sample(feats, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


# --------------------------------------------------------------------------- #
# proposal loss
# --------------------------------------------------------------------------- #
def candidate_losses_v4(cand, boxes, mask, labels=None, num_classes=1, max_wh=0.5,
                        lambda_pbox: float = 1.5, lambda_quality: float = 0.5,
                        centers=None, sizes=None, w_heat: float = 1.0):
    """``L_prop = L_heatmap + lambda_pbox * L_CIoU_coarse + lambda_quality * L_quality``."""
    dense = cand["dense"]
    levels = cand["level_names"]

    # ---- heatmap: one class-quality channel per class, per level ---------- #
    heat = boxes.new_zeros(())
    if w_heat > 0 and centers is not None and sizes is not None:
        n_terms = 0
        for name in levels:
            raw = dense[name]
            for c in range(num_classes):
                if num_classes == 1:
                    sel = mask
                elif labels is None:
                    sel = mask
                else:
                    sel = mask * (labels == c).to(mask.dtype)
                target = draw_gaussian(torch.zeros_like(raw[:, c:c + 1]), centers, sizes, sel)
                heat = heat + gaussian_focal_loss(raw[:, c:c + 1], target)
                n_terms += 1
        heat = heat / max(n_terms, 1)

    # ---- coarse box: CIoU at quantised GT centres on every level ---------- #
    losses = []
    for name in levels:
        raw = dense[name]
        hh, ww = raw.shape[-2:]
        step = raw.new_tensor([1.0 / max(ww - 1, 1), 1.0 / max(hh - 1, 1)])
        center = (boxes[..., :2] / step).round() * step
        sampled = sample_points(raw, center)
        predicted = decode_coarse_v4(sampled, center, step, num_classes, max_wh)
        valid = mask > 0.5
        losses.append(ciou_loss(predicted[valid], boxes[valid]).mean() if valid.any()
                      else predicted.sum() * 0.0)
    coarse = torch.stack(losses).mean()

    # ---- proposal quality: supervised by the actual coarse-box IoU -------- #
    pos, tgt = assign_candidates(cand["pts"], boxes, mask, topk=0)
    target = aligned_iou(cand["boxes"].detach(), tgt).detach() * pos
    quality = quality_focal_loss(cand["quality_logits"].float(), target)

    total = w_heat * heat + lambda_pbox * coarse + lambda_quality * quality
    return {"loss": total, "heat": heat.detach(), "coarse": coarse.detach(),
            "quality": quality.detach()}


__all__ = ["DenseProposalHeadV4", "MultiScaleCandidatesV4",
           "MultiScaleRoIAlign", "roi_align", "assign_levels", "decode_coarse_v4",
           "objectness", "candidate_losses_v4"]
