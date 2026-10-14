"""PGSF-Det v3 detection head.

The head consumes, per candidate:

* ``token``     -- the **enhanced** object token produced by the correspondence
                   module (its classification input);
* ``roi_feat``  -- this modality's own RoI spatial descriptor (P2/P3/P4/P5
                   sampling around the coarse box);
* ``coarse_box``-- this modality's own coarse box.

Two properties are structural, not conventions to be remembered:

* **Box regression never sees the other modality.**  Its inputs are the enhanced
  token, the *own* RoI descriptor and the *own* coarse box, so an RGB box can
  only ever be produced in RGB normalised coordinates and a TIR box in TIR
  coordinates.  Copying the counterpart box is not reachable by construction.
* **The head starts as the coarse box.**  The box projection is zero-initialised,
  and the decoder interprets it as ``center = base + tanh(d)*base_size/2``,
  ``size = base * exp(clamp(d,-2,2))``.  At initialisation the output is exactly
  the coarse box, so step-0 gradients never fight a random box prior.

Multi-class: the head emits ``4 + num_classes`` channels.  ``num_classes == 1``
reproduces the historical single-logit sigmoid head bit-for-bit, so v1/v2
checkpoints and configs keep working; ``num_classes > 1`` switches to one sigmoid
logit per class and the decoder reports ``argmax`` as the class.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.losses import assign_candidates, ciou_loss, nwd_loss, quality_focal_loss, sigmoid_focal_loss
from pgsf.models.proposal import aligned_iou


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _base_box(coarse_box, cand_pts, default_size: float = 0.05):
    if coarse_box is not None:
        return coarse_box
    if cand_pts is None:
        raise ValueError("need either coarse_box or cand_pts")
    return torch.cat((cand_pts, cand_pts.new_full(cand_pts.shape[:2] + (2,), default_size)), -1)


def decode_pred_v3(raw: torch.Tensor, coarse_box: torch.Tensor | None = None,
                   max_wh: float = 0.5, cand_pts: torch.Tensor | None = None):
    """Raw v3 head output -> ``(boxes (B,K,4), class_logits (B,K,C))``.

    The box convention is exactly :func:`pgsf.models.head.decode_pred` with
    ``base_boxes``: a bounded centre offset inside half the coarse size and a
    bounded log-size residual.  Keeping one convention means the loss and both
    decoders cannot disagree.
    """
    base = _base_box(coarse_box, cand_pts)
    center = base[..., :2] + raw[..., :2].tanh() * base[..., 2:] * 0.5
    size = (base[..., 2:] * raw[..., 2:4].clamp(-2, 2).exp()).clamp(1e-5, max_wh)
    return torch.cat((center, size), -1), raw[..., 4:]


def decode_boxes_scores_v3(raw: torch.Tensor, coarse_box: torch.Tensor,
                           img_hw: torch.Tensor, proposal_scores: torch.Tensor | None = None,
                           score_power: float = 0.5, max_wh: float = 0.5):
    """Fixed-shape (CUDA-graph friendly) decode to pixel ``xyxy`` + scores + labels."""
    boxes_n, logits = decode_pred_v3(raw, coarse_box, max_wh)
    center, half = boxes_n[..., :2], boxes_n[..., 2:] * 0.5
    scale = img_hw.flip(-1).repeat(1, 2)
    boxes = torch.cat((center - half, center + half), -1) * scale[:, None]
    probs = logits.sigmoid()
    scores, labels = probs.max(-1)
    if proposal_scores is not None:
        scores = scores * proposal_scores.clamp_min(1e-8).pow(score_power)
    return boxes, scores, labels


# --------------------------------------------------------------------------- #
# head
# --------------------------------------------------------------------------- #
class DetHeadV3(nn.Module):
    """``[enhanced token, own RoI, own coarse box] -> (box, class logits)``."""

    def __init__(self, token_dim: int = 128, roi_dim: int = 64, width: int = 128,
                 n_layers: int = 2, num_classes: int = 1, max_wh: float = 0.5,
                 init_score: float = -4.0):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = int(num_classes)
        self.max_wh = max_wh

        # --- classification tower: enhanced token only ---------------------- #
        cls_layers: list[nn.Module] = []
        c = token_dim
        for _ in range(max(0, n_layers - 1)):
            cls_layers += [nn.LayerNorm(c), nn.Linear(c, width), nn.SiLU(inplace=True)]
            c = width
        self.cls_body = nn.Sequential(*cls_layers) if cls_layers else nn.Identity()
        self.cls_out = nn.Linear(c, self.num_classes)
        nn.init.normal_(self.cls_out.weight, std=0.01)
        nn.init.zeros_(self.cls_out.bias)
        with torch.no_grad():
            self.cls_out.bias.fill_(init_score)

        # --- box tower: enhanced token + own RoI + own coarse box ----------- #
        self.roi_project = nn.Sequential(nn.LayerNorm(roi_dim), nn.Linear(roi_dim, roi_dim))
        nn.init.normal_(self.roi_project[-1].weight, std=0.01)
        nn.init.zeros_(self.roi_project[-1].bias)
        self.box_embed = nn.Sequential(nn.Linear(4, roi_dim), nn.SiLU(inplace=True),
                                       nn.Linear(roi_dim, roi_dim))
        nn.init.normal_(self.box_embed[-1].weight, std=0.01)
        nn.init.zeros_(self.box_embed[-1].bias)

        box_in = token_dim + 2 * roi_dim
        box_layers: list[nn.Module] = []
        c2 = box_in
        for _ in range(max(0, n_layers - 1)):
            box_layers += [nn.LayerNorm(c2), nn.Linear(c2, width), nn.SiLU(inplace=True)]
            c2 = width
        self.box_body = nn.Sequential(*box_layers) if box_layers else nn.Identity()
        self.box_out = nn.Linear(c2, 4)
        # zero -> the head predicts exactly the coarse box at step 0
        nn.init.zeros_(self.box_out.weight)
        nn.init.zeros_(self.box_out.bias)

    def forward(self, token: torch.Tensor, roi_feat: torch.Tensor,
                coarse_box: torch.Tensor) -> torch.Tensor:
        cls = self.cls_out(self.cls_body(token))                       # (B,K,C)
        pos = coarse_box[..., :2]
        size = coarse_box[..., 2:].clamp_min(1e-4).log()
        geom = torch.cat((pos, size), -1)
        h = torch.cat((token, self.roi_project(roi_feat), self.box_embed(geom)), -1)
        box = self.box_out(self.box_body(h))                           # (B,K,4)
        return torch.cat((box, cls), -1)                               # (B,K,4+C)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
class DetectionLossV3(nn.Module):
    """Focal classification (multi-class capable) + CIoU (+ optional NWD) per modality."""

    def __init__(self, w_cls: float = 1.0, w_box: float = 2.5, w_nwd: float = 0.5,
                 use_nwd: bool = True, pos_thr: float = 0.5, center_thr: float = 0.05,
                 max_wh: float = 0.5, num_classes: int = 1):
        super().__init__()
        self.w_cls, self.w_box, self.w_nwd = w_cls, w_box, w_nwd
        self.use_nwd = use_nwd
        self.pos_thr, self.center_thr = pos_thr, center_thr
        self.max_wh = max_wh
        self.num_classes = int(num_classes)

    def forward(self, raw: torch.Tensor, cand_pts: torch.Tensor,
                gt_boxes: torch.Tensor, gt_mask: torch.Tensor,
                gt_labels: torch.Tensor | None = None,
                base_boxes: torch.Tensor | None = None,
                quality_aware: bool = False):
        pred, logits = decode_pred_v3(raw, base_boxes, self.max_wh, cand_pts)
        pos, tgt = assign_candidates(cand_pts, gt_boxes, gt_mask, self.pos_thr,
                                     self.center_thr, topk=0 if quality_aware else 8)
        c = logits.shape[-1]

        # ---- class target ------------------------------------------------ #
        if gt_labels is None or c == 1:
            # Single-class / legacy convention: positives are 1, background 0.
            core = pos[..., None].expand(-1, -1, c)
        else:
            _, target_of = assign_candidates(cand_pts, gt_boxes, gt_mask,
                                            self.pos_thr, self.center_thr, topk=0)
            idx = matched_label_index(cand_pts, gt_boxes, target_of, gt_mask)
            labels = torch.gather(gt_labels, 1, idx).clamp(0, c - 1)
            core = F.one_hot(labels, c).to(logits.dtype) * pos[..., None]

        if quality_aware:
            iou = aligned_iou(pred.detach(), tgt).detach() * pos
            target = core * iou[..., None]
            l_cls = quality_focal_loss(logits.float(), target)
        else:
            target = core
            l_cls = sigmoid_focal_loss(logits.float(), target)

        pb, tb = pred[pos > 0.5], tgt[pos > 0.5]
        if pb.numel() == 0:
            zero = pred.sum() * 0.0
            l_box, l_nwd = zero, zero
        else:
            l_box = ciou_loss(pb, tb).mean()
            l_nwd = nwd_loss(pb, tb).mean() if self.use_nwd else l_box.new_zeros(())
        total = self.w_cls * l_cls + self.w_box * l_box
        if self.use_nwd:
            total = total + self.w_nwd * l_nwd
        return {"loss_cls": l_cls.detach(), "loss_box": l_box.detach(),
                "loss_nwd": l_nwd.detach(), "loss": total,
                "pos_ratio": pos.mean().detach()}


def matched_label_index(cand_pts: torch.Tensor, gt_boxes: torch.Tensor,
                        target_of: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Index of the GT slot each candidate is assigned to (B,K).

    ``target_of`` (B,K,4) is the assigned GT box; we recover the slot by exact
    comparison against the GT table, falling back to slot 0 (whose loss weight is
    zero for unmatched candidates).
    """
    b, k, _ = cand_pts.shape
    m = gt_boxes.shape[1]
    if m == 0:
        return torch.zeros(b, k, dtype=torch.long, device=cand_pts.device)
    distance = (target_of.unsqueeze(2) - gt_boxes.unsqueeze(1)).abs().sum(-1)   # (B,K,M)
    distance = distance.masked_fill((gt_mask <= 0.5).unsqueeze(1), float("inf"))
    return distance.argmin(-1)


__all__ = ["DetHeadV3", "DetectionLossV3", "decode_pred_v3", "decode_boxes_scores_v3"]
