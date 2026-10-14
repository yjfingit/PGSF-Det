"""PGSF-Det v4 detection heads, decoding helper and losses.

Two heads, both shared between the fused path and the auxiliary un-fused path --
the spec's ``L_self`` is an auxiliary *loss* on the same heads fed with the raw
``z_m`` / ``S_m``, so no extra parameters (and no extra inference branch) exist:

* :class:`ClsHeadV4`   ``128 -> 128 -> C`` on a token.
* :class:`RegHeadV4`   a 1x1 conv over the RoI descriptor ``S``, global average
  pooling, then the coarse box as the reference for a 4-d delta.

Box convention (identical to v1/v2/v3 so training and both decoders cannot
disagree): ``center = base_center + tanh(d) * base_size / 2`` and
``size = base_size * exp(clamp(d, -2, 2))``.  The regression projection is
zero-initialised, so at step 0 the head outputs exactly the coarse box.

Loss follows the v4 spec exactly::

    L_final = L_QFL + lambda_iou * L_CIoU + lambda_l1 * L_L1
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.losses import assign_candidates, ciou_loss, quality_focal_loss
from pgsf.models.proposal import aligned_iou


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class ClsHeadV4(nn.Module):
    """``128 -> 128 -> C`` classification head on a fused (or raw) token."""

    def __init__(self, dim: int = 128, width: int = 128, num_classes: int = 1,
                 init_score: float = -4.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.body = nn.Sequential(nn.Linear(dim, width), nn.SiLU(inplace=True))
        self.out = nn.Linear(width, self.num_classes)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias.fill_(init_score)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(token))


class RegHeadV4(nn.Module):
    """RoI descriptor ``[B,K,C,w,h]`` + coarse box -> 4-d box delta."""

    def __init__(self, channels: int = 128, width: int = 128):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(inplace=True))
        self.box_embed = nn.Sequential(nn.Linear(4, channels), nn.SiLU(inplace=True),
                                       nn.Linear(channels, channels))
        nn.init.normal_(self.box_embed[-1].weight, std=0.01)
        nn.init.zeros_(self.box_embed[-1].bias)
        self.body = nn.Sequential(nn.Linear(2 * channels, width), nn.SiLU(inplace=True))
        self.out = nn.Linear(width, 4)
        # zero -> the head predicts exactly the coarse box at step 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, roi: torch.Tensor, coarse_box: torch.Tensor) -> torch.Tensor:
        b, k, c, hh, ww = roi.shape
        # the 1x1 conv is spatial: it must see (N,C,H,W), then the RoI token is pooled back
        pooled = self.reduce(roi.reshape(b * k, c, hh, ww)).mean((-2, -1)).reshape(b, k, c)
        size = coarse_box[..., 2:].clamp_min(1e-4).log()
        geom = torch.cat((coarse_box[..., :2], size), -1)
        h = torch.cat((pooled, self.box_embed(geom)), -1)
        return self.out(self.body(h))                                # (B,K,4)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_pred_v4(raw: torch.Tensor, coarse_box: torch.Tensor, max_wh: float = 0.5):
    """``raw (B,K,4+C)`` -> ``(boxes (B,K,4), class_logits (B,K,C))``."""
    base = coarse_box
    center = base[..., :2] + raw[..., :2].tanh() * base[..., 2:] * 0.5
    size = (base[..., 2:] * raw[..., 2:4].clamp(-2, 2).exp()).clamp(1e-5, max_wh)
    return torch.cat((center, size), -1), raw[..., 4:]


def decode_boxes_scores_v4(raw: torch.Tensor, coarse_box: torch.Tensor,
                           img_hw: torch.Tensor, proposal_scores: torch.Tensor | None = None,
                           score_power: float = 0.5, max_wh: float = 0.5):
    """Fixed-shape decode to pixel ``xyxy`` + scores + class ids (own pixel frame)."""
    boxes_n, logits = decode_pred_v4(raw, coarse_box, max_wh)
    center, half = boxes_n[..., :2], boxes_n[..., 2:] * 0.5
    scale = img_hw.flip(-1).repeat(1, 2)
    boxes = torch.cat((center - half, center + half), -1) * scale[:, None]
    probs = logits.sigmoid()
    scores, labels = probs.max(-1)
    if proposal_scores is not None:
        scores = scores * proposal_scores.clamp_min(1e-8).pow(score_power)
    return boxes, scores, labels


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
class DetectionLossV4(nn.Module):
    """``L_QFL + lambda_iou * L_CIoU + lambda_l1 * L_L1``, class aware."""

    def __init__(self, num_classes: int = 1, lambda_iou: float = 2.5, lambda_l1: float = 1.0,
                 max_wh: float = 0.5, center_thr: float = 0.05, pos_thr: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lambda_iou, self.lambda_l1 = lambda_iou, lambda_l1
        self.max_wh, self.center_thr, self.pos_thr = max_wh, center_thr, pos_thr

    def forward(self, raw, cand_pts, coarse_box, gt_boxes, gt_mask, gt_labels=None,
                quality_aware: bool = True):
        boxes, logits = decode_pred_v4(raw, coarse_box, self.max_wh)
        pos, tgt = assign_candidates(cand_pts, gt_boxes, gt_mask, self.pos_thr,
                                     self.center_thr, topk=0)
        c = logits.shape[-1]
        if c == 1 or gt_labels is None:
            core = pos[..., None].expand(-1, -1, c)
        else:
            idx = label_slot_index(cand_pts, gt_boxes, tgt, gt_mask)
            labels = torch.gather(gt_labels, 1, idx).clamp(0, c - 1)
            core = F.one_hot(labels, c).to(logits.dtype) * pos[..., None]
        if quality_aware:
            target = core * (aligned_iou(boxes.detach(), tgt).detach() * pos)[..., None]
        else:
            target = core
        l_qfl = quality_focal_loss(logits.float(), target)

        pb, tb = boxes[pos > 0.5], tgt[pos > 0.5]
        if pb.numel() == 0:
            zero = boxes.sum() * 0.0
            l_iou = l_l1 = zero
        else:
            l_iou = ciou_loss(pb, tb).mean()
            l_l1 = (pb - tb).abs().mean()
        total = l_qfl + self.lambda_iou * l_iou + self.lambda_l1 * l_l1
        return {"loss": total, "qfl": l_qfl.detach(), "iou": l_iou.detach(),
                "l1": l_l1.detach(), "pos_ratio": pos.mean().detach()}


def label_slot_index(cand_pts, gt_boxes, target_of, gt_mask):
    """GT slot of every candidate (B,K), recovered from the assigned target box."""
    b, k, _ = cand_pts.shape
    m = gt_boxes.shape[1]
    if m == 0:
        return torch.zeros(b, k, dtype=torch.long, device=cand_pts.device)
    distance = (target_of.unsqueeze(2) - gt_boxes.unsqueeze(1)).abs().sum(-1)
    distance = distance.masked_fill((gt_mask <= 0.5).unsqueeze(1), float("inf"))
    return distance.argmin(-1)


def class_probabilities(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``[...,C]`` logits -> a proper distribution.

    ``C == 1`` keeps the historical single-logit sigmoid convention and returns a
    two-way ``[p, 1-p]``; ``C > 1`` uses softmax over the classes.
    """
    if int(num_classes) == 1:
        p = logits.sigmoid()
        return torch.cat((p, 1.0 - p), -1)
    return logits.softmax(-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two probability rows (nats, in [0, log2])."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


__all__ = ["ClsHeadV4", "RegHeadV4", "decode_pred_v4", "decode_boxes_scores_v4",
           "DetectionLossV4", "class_probabilities", "js_divergence"]
