"""COCO-style AP evaluation, computed per modality.

UAV-CB uses a single-target protocol, but the metric code is the full COCO AP so
that a future multi-object version needs no change.  ``AP50 / AP75 / AP50:95`` use
the standard 101-point interpolation and 0.5:0.05:0.95 IoU thresholds.

The matcher is greedy on descending score and takes **at most one GT per
prediction** (standard 1-to-1 matching); unmatched predictions count as FP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4), b (M,4) in xyxy -> (N,M) IoU."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


# --------------------------------------------------------------------------- #
def average_precision(recall: np.ndarray, precision: np.ndarray,
                      mode: str = "101point") -> float:
    """COCO 101-point precision envelope, with an optional VOC area variant."""
    if not recall.size:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    if mode != "area":
        indices = np.searchsorted(recall, np.linspace(0., 1., 101), side="left")
        sampled = np.zeros(101)
        valid = indices < envelope.size
        sampled[valid] = envelope[indices[valid]]
        return float(sampled.mean())
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], envelope, [0.0]))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def filtered_ground_truth(batch, modality: str):
    """Return one (N,4) pixel box tensor per image, excluding padded rows."""
    return [boxes[mask > 0.5] for boxes, mask in
            zip(batch[f"boxes_{modality}_px"], batch.get("mask_"+modality,batch["mask"]))]


# --------------------------------------------------------------------------- #
def eval_detection(preds: Sequence[Dict[str, torch.Tensor]],
                   gts: Sequence[torch.Tensor],
                   iou_thrs: Optional[Sequence[float]] = None,
                   max_det: int = 50) -> Dict[str, float]:
    """
    preds: list (per image) of {"boxes": (N,4) xyxy, "scores": (N,)}
    gts:   list (per image) of (M,4) xyxy (possibly empty)

    **Both must be in the same coordinate space.** Passing pixel predictions against
    normalised GT (or vice versa) yields IoU = 0 everywhere and a silent AP = 0 --
    a failure that looks exactly like "the model does not learn". The sanity check
    below turns that into a loud error.
    """
    iou_thrs = list(iou_thrs) if iou_thrs is not None else \
        [round(0.5 + 0.05 * i, 2) for i in range(10)]

    if len(preds) != len(gts):
        raise ValueError("predictions and ground truth must have the same image count")
    n_img = len(preds)
    n_gt = int(sum(g.shape[0] for g in gts))
    if n_img == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP50:95": 0.0, "n_gt": 0}

    # coordinate-space guard
    pred_max = max((float(p["boxes"].max()) for p in preds if p["boxes"].numel()),
                   default=0.0)
    gt_max = max((float(g.max()) for g in gts if g.numel()), default=0.0)
    if pred_max > 1.5 and 0 < gt_max <= 1.5:
        raise AssertionError(
            "eval_detection got pixel-space predictions (max "
            f"{pred_max:.1f}) against normalised GT (max {gt_max:.3f}); "
            "every IoU would be 0. Pass boxes_<modality>_px as the ground truth.")

    aps = []
    for thr in iou_thrs:
        scores_per_img, matched_per_img = [], []
        npos = 0
        for i in range(n_img):
            g = gts[i].cpu().numpy() if gts[i].numel() else np.zeros((0, 4))
            p = preds[i]
            b = p["boxes"].cpu().numpy() if p["boxes"].numel() else np.zeros((0, 4))
            s = p["scores"].cpu().numpy() if p["scores"].numel() else np.zeros((0,))
            npos += g.shape[0]
            order = np.argsort(-s, kind="stable")[:max_det]
            b, s = b[order], s[order]
            scores_per_img.append(s)
            hit = np.zeros(b.shape[0], dtype=bool)
            if b.shape[0] and g.shape[0]:
                iou = box_iou_matrix(b, g)
                used = np.zeros(g.shape[0], dtype=bool)
                # Score-ordered predictions claim at most one unused GT.
                for pi in range(b.shape[0]):
                    available = np.where(used, -1.0, iou[pi])
                    gi = int(available.argmax())
                    if available[gi] >= thr:
                        used[gi] = True
                        hit[pi] = True
            matched_per_img.append(hit)

        if npos == 0:
            aps.append(float("nan"))
            continue
        all_s = np.concatenate([s for s in scores_per_img if s.size]) \
            if any(s.size for s in scores_per_img) else np.zeros(0)
        all_m = np.concatenate([m for m in matched_per_img if m.size]) \
            if any(m.size for m in matched_per_img) else np.zeros(0, bool)
        if all_s.size == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-all_s, kind="stable")
        tp = all_m[order].astype(np.float64)
        fp = 1.0 - tp
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        recall = ctp / npos
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        aps.append(average_precision(recall, precision))

    aps_arr = np.array([a for a in aps if not np.isnan(a)], dtype=np.float64)
    def at(threshold):
        values = [a for t, a in zip(iou_thrs, aps) if abs(t - threshold) < 1e-6 and not np.isnan(a)]
        return float(values[0]) if values else 0.0
    ap50, ap75 = at(0.5), at(0.75)
    ap5095 = float(aps_arr.mean()) if aps_arr.size else 0.0
    return {
        "AP50": float(ap50),
        "AP75": float(ap75),
        "AP50:95": ap5095,
        "AP": ap5095,
        "n_gt": npos,
        "per_iou": [float(a) for a in aps],
    }


# --------------------------------------------------------------------------- #
def small_object_recall(preds, gts, area_thr: float = 32 * 32,
                        iou_thr: float = 0.5) -> float:
    """Recall restricted to small GT boxes (COCO: sqrt(area) < 32 px)."""
    n_hit = n_tot = 0
    for p, g in zip(preds, gts):
        g = g.cpu().numpy() if g.numel() else np.zeros((0, 4))
        if g.shape[0] == 0:
            continue
        areas = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        small = areas < area_thr
        if not small.any():
            continue
        b = p["boxes"].cpu().numpy() if p["boxes"].numel() else np.zeros((0, 4))
        s = p["scores"].cpu().numpy() if p["scores"].numel() else np.zeros((0,))
        if b.shape[0] == 0:
            n_tot += int(small.sum())
            continue
        if s.size:
            keep = np.argsort(-s)[:50]
            b = b[keep]
        iou = box_iou_matrix(b, g[small])                    # (N, n_small)
        for j in range(iou.shape[1]):
            n_tot += 1
            if iou[:, j].max(initial=0.0) >= iou_thr:
                n_hit += 1
    return float(n_hit / max(n_tot, 1))


def precision_recall_at(preds, gts, conf_thr: float = 0.05,
                        iou_thr: float = 0.5) -> Dict[str, float]:
    """Single-score-threshold precision / recall / F1 (the UAV-CB style protocol)."""
    tp = fp = 0
    npos = 0
    for p, g in zip(preds, gts):
        g = g.cpu().numpy() if g.numel() else np.zeros((0, 4))
        b = p["boxes"].cpu().numpy() if p["boxes"].numel() else np.zeros((0, 4))
        s = p["scores"].cpu().numpy() if p["scores"].numel() else np.zeros((0,))
        npos += g.shape[0]
        keep = np.argsort(-s, kind="stable")
        keep = keep[s[keep] >= conf_thr]
        used = np.zeros(g.shape[0], dtype=bool)
        for i in keep:
            if g.shape[0] == 0:
                fp += 1
                continue
            iou = box_iou_matrix(b[i][None], g)[0]
            iou = np.where(used, -1.0, iou)
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not used[j]:
                tp += 1
                used[j] = True
            else:
                fp += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(npos, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "npos": npos}


# --------------------------------------------------------------------------- #
def match_accuracy(pred_rgb, pred_tir, gt_rgb, gt_tir,
                   iou_thr: float = 0.5, tol: float = 0.05) -> Dict[str, float]:
    """Fraction of images whose top detections hit the same paired GT identity.

    Compare each prediction to GT in its own modality's pixel space. This is a
    detection-pair success rate; it is not an attention association accuracy.
    """
    correct = total = 0
    for pr, pt, gr, gt in zip(pred_rgb, pred_tir, gt_rgb, gt_tir):
        total += 1
        if not pr["boxes"].numel() or not pt["boxes"].numel() or not gr.numel() or not gt.numel():
            continue
        ir, it = int(pr["scores"].argmax()), int(pt["scores"].argmax())
        riou = box_iou_matrix(pr["boxes"][ir:ir+1].cpu().numpy(), gr.cpu().numpy())[0]
        tiou = box_iou_matrix(pt["boxes"][it:it+1].cpu().numpy(), gt.cpu().numpy())[0]
        rgt, tgt = int(riou.argmax()), int(tiou.argmax())
        if rgt == tgt and riou[rgt] >= iou_thr and tiou[tgt] >= iou_thr:
            correct += 1
    return {"match_acc": float(correct / max(total, 1)),
            "n": int(total), "n_correct": int(correct)}
