"""Loss functions.

Total objective (per the task spec)::

    L = L_det^RGB + L_det^TIR + lambda_h L_heat + lambda_g L_geo
        + lambda_a L_assoc

Every term has an unambiguous supervision source:

=================  ===========================================================
term               supervised by
=================  ===========================================================
``L_cls``          candidate <-> GT assignment (IoU / centre-distance based)
``L_box``          assigned candidates -> GT box (CIoU, plus optional NWD)
``L_heat``         CenterNet Gaussian target on the P3 grid, both modalities
``L_geo``          Gaussian NLL of the predicted correspondence, target = the
                   *other* modality's GT centre (train split only)
``L_assoc``        is the best cross-modal key better than the null token?
                   (BCE on the null-vs-real attention logit gap)
=================  ===========================================================

Every term is differentiable in at least one parameter that is not otherwise
reachable, so a gradient check can prove that no module is silently dead.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# basic
# --------------------------------------------------------------------------- #
def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "mean") -> torch.Tensor:
    """Binary focal loss on logits (targets in [0,1])."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = w * (1 - pt).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def gaussian_focal_loss(pred: torch.Tensor, gt: torch.Tensor,
                        alpha: float = 2.0, gamma: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on a heatmap *logit* map.

    Positive locations are the exact GT centres; everything with ``gt < 1`` is
    treated as negative and re-weighted by ``(1 - gt)^gamma`` so that the ring
    around a true centre is not punished as hard as a far-away background cell.
    """
    pred = torch.clamp(pred.float().sigmoid(), 1e-4, 1 - 1e-4)
    gt = gt.float()
    pos = gt.ge(0.999).float()
    neg = 1.0 - pos
    neg_w = torch.pow(1.0 - gt, gamma)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_w * neg

    n_pos = pos.sum()
    if n_pos == 0:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def quality_focal_loss(logits,targets,beta=2.0):
    """Quality focal: BCE modulated by |quality-sigmoid(logit)|^beta."""
    logits=logits.float();targets=targets.float()
    probability=logits.sigmoid()
    return (F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
            *(targets-probability).abs().pow(beta)).mean()


# --------------------------------------------------------------------------- #
# box losses
# --------------------------------------------------------------------------- #
def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU loss on normalised (cx,cy,w,h) boxes. Returns per-box loss."""
    pcx, pcy, pw, ph = pred.float().unbind(-1)
    tcx, tcy, tw, th = target.float().unbind(-1)
    pa = (pw * ph).clamp_min(eps)
    ta = (tw * th).clamp_min(eps)

    dx, dy = pcx - tcx, pcy - tcy
    dw, dh = pw - tw, ph - th
    # enclosing box
    cw = torch.maximum(pcx + pw / 2, tcx + tw / 2) - torch.minimum(pcx - pw / 2, tcx - tw / 2)
    ch = torch.maximum(pcy + ph / 2, tcy + th / 2) - torch.minimum(pcy - ph / 2, tcy - th / 2)
    c2 = (cw.clamp_min(eps) ** 2 + ch.clamp_min(eps) ** 2)

    # IoU part
    lt = torch.maximum(pcx - pw / 2, tcx - tw / 2)
    rb = torch.minimum(pcx + pw / 2, tcx + tw / 2)
    tp = torch.maximum(pcy - ph / 2, tcy - th / 2)
    bt = torch.minimum(pcy + ph / 2, tcy + th / 2)
    iw = (rb - lt).clamp_min(0)
    ih = (bt - tp).clamp_min(0)
    inter = iw * ih
    union = pa + ta - inter
    iou = inter / union.clamp_min(eps)

    # centre / aspect terms
    enclose_c2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    rho2 = enclose_c2 / c2.clamp_min(eps)
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(tw / th.clamp_min(eps)) - torch.atan(pw / ph.clamp_min(eps)), 2)
    with torch.no_grad():
        alpha_v = v / (1 - iou + v).clamp_min(eps)
    ciou = iou - (rho2 + alpha_v * v)
    return 1.0 - ciou


def nwd_loss(pred: torch.Tensor, target: torch.Tensor,
             alpha: float = 0.5, eps: float = 1e-7, normalizer: float = .05) -> torch.Tensor:
    """Gaussian box Wasserstein distance, scaled to normalized image coordinates.

    sqrt(||dc||^2 + ||dsize||^2/4), rather than a near-zero squared-distance
    penalty. normalizer=.05 corresponds to a 5% normalized-frame distance.
    """
    delta=(pred.float()-target.float())
    squared=delta[...,:2].square().sum(-1)+delta[...,2:].square().sum(-1)/4
    distance=(squared+eps).sqrt()-math.sqrt(eps)
    return -torch.expm1(-2*alpha*distance.clamp_min(0)/normalizer)


# --------------------------------------------------------------------------- #
# candidate <-> GT assignment
# --------------------------------------------------------------------------- #
def assign_candidates(cand_pts: torch.Tensor, gt_boxes: torch.Tensor, gt_mask: torch.Tensor,
                      pos_thr: float = 0.5, center_thr: float = 0.05,
                      topk: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized expanded-box/center assignment, keeping top-k per GT.

    Candidates are points, so no fictitious point-box IoU is used. pos_thr is
    retained for the public API; actual box IoU supervises quality/regression.
    """
    b,k,_=cand_pts.shape;m=gt_boxes.shape[1]
    if m==0:
        return cand_pts.new_zeros(b,k),cand_pts.new_zeros(b,k,4)
    delta=cand_pts.float().unsqueeze(2)-gt_boxes.float()[:,None,:,:2]
    distance=delta.norm(dim=-1)
    inside=(delta.abs()<=gt_boxes.float()[:,None,:,2:].clamp_min(1e-4)).all(-1)
    match=inside&(distance<=center_thr)&(gt_mask[:,None,:]>.5)
    if topk and topk<k:
        d=distance.masked_fill(~match,float('inf')).transpose(1,2)
        index=d.topk(topk,dim=-1,largest=False).indices
        keep=torch.zeros_like(d,dtype=torch.bool).scatter_(-1,index,True).transpose(1,2)
        match=match&keep
    assigned=distance.masked_fill(~match,float('inf')).argmin(-1)
    positive=match.any(-1)
    target=gt_boxes.gather(1,assigned[...,None].expand(-1,-1,4))
    target=torch.where(positive[...,None],target,torch.zeros_like(target))
    return positive.float(),target


# --------------------------------------------------------------------------- #
# heatmap target
# --------------------------------------------------------------------------- #
def gaussian_radius(h: float, w: float, min_iou: float = 0.5) -> float:
    """CenterNet radius: smallest r such that a Gaussian with that sigma keeps IoU >= min_iou."""
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_iou) / (1 + min_iou)
    r1 = (b1 - torch.sqrt(torch.clamp(b1 ** 2 - 4 * a1 * c1, min=0.0))) / (2 * a1)
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_iou) * w * h
    r2 = (b2 - torch.sqrt(torch.clamp(b2 ** 2 - 4 * a2 * c2, min=0.0))) / (2 * a2)
    a3, b3, c3 = 4 * min_iou, -2 * min_iou * (h + w), (min_iou - 1) * w * h
    r3 = (b3 + torch.sqrt(torch.clamp(b3 ** 2 - 4 * a3 * c3, min=0.0))) / (2 * a3)
    return float(torch.clamp(torch.min(torch.min(r1, r2), r3), min=0.0))


def draw_gaussian(heat: torch.Tensor, centers: torch.Tensor, sizes: torch.Tensor,
                  mask: torch.Tensor, sigma_scale: float = 2.0) -> torch.Tensor:
    """Render CenterNet Gaussian targets.

    heat: (B,1,H,W) to fill; centers (B,M,2) normalised (x,y); sizes (B,M,2)
    normalised (w,h); mask (B,M).

    The Gaussian radius is derived from the *smaller* target side, so a 20x20 px
    drone gets a tight peak and a 200x200 blob gets a broad one -- a single global
    radius would either drown small targets in positives or leave large ones
    unsupervised.
    """
    b, _, H, W = heat.shape
    out = heat.clone()
    m = centers.shape[1]
    if m == 0:
        return out
    xs = torch.arange(W, device=heat.device, dtype=heat.dtype).view(1, W)   # (1,W)
    ys = torch.arange(H, device=heat.device, dtype=heat.dtype).view(1, H)   # (1,H)
    active=(mask>.5).any(0).nonzero(as_tuple=False).flatten().tolist()
    for i in active:
        c = centers[:, i] * torch.tensor([W - 1, H - 1], device=heat.device,
                                         dtype=heat.dtype)                   # (B,2)
        c = c.round().clamp_min(0)  # guarantee one exact positive pixel per GT
        s = sizes[:, i] * torch.tensor([W, H], device=heat.device, dtype=heat.dtype)
        valid = mask[:, i] > 0.5                                            # (B,)
        r = (sigma_scale * s.min(dim=1).values / 6.0).clamp(0.7, 8.0)        # (B,)
        # NOTE: subtract the coordinate vector from the per-batch centre, not the
        # other way round -- ``xs - c`` would broadcast xs into a leading axis and
        # silently produce a 4-D tensor.
        dx = (c[:, 0:1] - xs) ** 2                                          # (B,W)
        dy = (c[:, 1:2] - ys) ** 2                                          # (B,H)
        d2 = dy.unsqueeze(2) + dx.unsqueeze(1)                               # (B,H,W)
        g = torch.exp(-d2 / (2 * (r ** 2).view(b, 1, 1)))
        g = torch.where(valid.view(b, 1, 1), g, torch.zeros_like(g))
        out[:, 0] = torch.maximum(out[:, 0], g)
    return out


# --------------------------------------------------------------------------- #
# detection loss for one modality
# --------------------------------------------------------------------------- #
class DetectionLoss(nn.Module):
    """Focal classification + CIoU (+ optional NWD) for one modality."""

    def __init__(self, w_cls: float = 1.0, w_box: float = 2.5, w_nwd: float = 0.5,
                 use_nwd: bool = True, pos_thr: float = 0.5, center_thr: float = 0.05,
                 max_wh: float = 0.5):
        super().__init__()
        self.w_cls, self.w_box, self.w_nwd = w_cls, w_box, w_nwd
        self.use_nwd = use_nwd
        self.pos_thr, self.center_thr = pos_thr, center_thr
        self.max_wh = max_wh

    def forward(self, raw: torch.Tensor, cand_pts: torch.Tensor,
                gt_boxes: torch.Tensor, gt_mask: torch.Tensor, base_boxes: torch.Tensor | None = None,
                quality_aware: bool = False) -> Dict[str, torch.Tensor]:
        """raw (B,K,5) from DetHead; cand_pts (B,K,2) normalised."""
        from pgsf.models.head import decode_pred
        pred, score_logit = decode_pred(raw, cand_pts, self.max_wh, base_boxes)
        pos, tgt = assign_candidates(cand_pts, gt_boxes, gt_mask,
                                     self.pos_thr, self.center_thr,topk=0 if quality_aware else 8)
        cls_target=pos
        if quality_aware:
            from pgsf.models.proposal import aligned_iou
            cls_target=aligned_iou(pred.detach(),tgt).detach()*pos
        l_cls = quality_focal_loss(score_logit,cls_target) if quality_aware else sigmoid_focal_loss(score_logit.float(),cls_target)
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


# --------------------------------------------------------------------------- #
# geometry + association losses
# --------------------------------------------------------------------------- #
def geometry_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
                 valid: torch.Tensor, sigma_ref: float = 0.05,
                 minimum_variance: float | None = None) -> torch.Tensor:
    """Masked Gaussian NLL relative to N(0, sigma_ref^2 I).

    Subtracting a constant changes the displayed value, not the gradients or
    the relative optimization strength of other losses. sigma is covariance.
    """
    valid = valid.bool()
    if valid.sum() == 0:
        return mu.sum() * 0.0
    # Promote to float32 before the linear solve: under AMP the activations arrive
    # as fp16 while the identity regulariser is fp32, and linalg.solve refuses
    # mixed dtypes. fp32 is also numerically required here -- the covariance is
    # tiny for well-localised targets, so fp16 would underflow.
    mu_v = mu[valid].float()
    s_v = sigma[valid].float()
    t_v = target[valid].float()
    d = (t_v - mu_v).unsqueeze(-1)                             # (N,2,1)
    s = s_v + 1e-4 * torch.eye(2, device=mu.device, dtype=torch.float32)
    sol = torch.linalg.solve(s, d)
    quad = (d.transpose(1, 2) @ sol).squeeze(-1).squeeze(-1)     # (N,)
    logdet = torch.logdet(s)
    # Optional fixed shift to the theoretical lower bound. The 1e-4 I
    # regularizer is included: covariance >= v I gives logdet >= 2 log(v+1e-4).
    # No clamp/abs: this changes neither gradients nor weighting.
    logdet_ref = (2*math.log(minimum_variance+1e-4) if minimum_variance is not None
                  else 4.0 * math.log(sigma_ref))
    return (0.5 * quad + 0.5 * (logdet - logdet_ref)).mean()


def association_loss(null_logit: torch.Tensor, best_real_logit: torch.Tensor,
                     positive: torch.Tensor) -> torch.Tensor:
    """Binary CE: is the best cross-modal key a better explanation than "no match"?

    null_logit / best_real_logit are the (unnormalised) scores of the null token
    and of the best real key; ``positive`` is 1 when a true correspondence
    exists, 0 when the modality has no evidence at that candidate.
    """
    diff = best_real_logit - null_logit
    tgt = positive.to(diff.dtype)
    return F.binary_cross_entropy_with_logits(diff, tgt)


def correspondence_mask(query_pts: torch.Tensor, key_pts: torch.Tensor,
                        gt_query: torch.Tensor, gt_key: torch.Tensor,
                        gt_mask: torch.Tensor, frame_valid: torch.Tensor,
                        center_thr: float = 0.05) -> torch.Tensor:
    """GT identity mask (B,Q,K), independent of candidate order and misalignment.

    key_pts may be shared (B,K,2) or per-query (B,Q,K,2). Boxes at the same GT
    index are paired objects. Background queries and absent keys select null.
    """
    if key_pts.dim() == 3:
        key_pts = key_pts.unsqueeze(1).expand(-1, query_pts.shape[1], -1, -1)
    b, q, k, _ = key_pts.shape
    if gt_query.shape[1] == 0:
        return torch.zeros(b, q, k, device=query_pts.device, dtype=torch.bool)

    def identity(points, boxes):
        center = boxes[:, None, None, :, :2]
        size = boxes[:, None, None, :, 2:].clamp_min(1e-6)
        delta = points.unsqueeze(-2) - center
        # Same expanded-box/center condition as the detector assignment, but
        # keep all matching keys instead of capping the positives per object.
        near = (delta.abs() <= size).all(-1) & (delta.norm(dim=-1) <= center_thr)
        near &= (gt_mask[:, None, None] > 0.5)
        near &= ((points>=0)&(points<=1)).all(-1).unsqueeze(-1)
        distance = delta.norm(dim=-1).masked_fill(~near, float("inf"))
        idx = distance.argmin(-1)
        return idx, near.any(-1)

    qi, qvalid = identity(query_pts.unsqueeze(2), gt_query)
    ki, kvalid = identity(key_pts, gt_key)
    return (qi == ki) & qvalid & kvalid & (frame_valid[:, None, None] > 0.5)


def supervised_association_loss(attn: torch.Tensor, matches: torch.Tensor,
                                has_null: bool = True, negative_weights=None) -> torch.Tensor:
    """NLL of probability on same-object keys, or on null if no match exists.

    The final attention entry is null. Several keys may represent the same GT;
    supervision sums their mass and does not impose one-to-one matching.
    """
    if attn.shape[-1] != matches.shape[-1] + int(has_null):
        raise ValueError("association key mask must exclude exactly the null key")
    real = (attn[..., :matches.shape[-1]].float() * matches).sum(-1)
    positive=matches.any(-1)
    if not has_null:
        return -real[positive].clamp_min(1e-8).log().mean() if positive.any() else attn.sum()*0.
    probability=torch.where(positive,real,attn[...,-1].float())
    loss=-probability.clamp_min(1e-8).log()
    if negative_weights is None:return loss.mean()
    # Balance positive matches and hard background queries instead of letting
    # the many background candidates make always-null a cheap solution.
    values=[]
    if positive.any():values.append(loss[positive].mean())
    if (~positive).any():
        weight=negative_weights[~positive].detach().float()
        values.append((loss[~positive]*weight).sum()/weight.sum().clamp_min(1e-6))
    return torch.stack(values).mean()
