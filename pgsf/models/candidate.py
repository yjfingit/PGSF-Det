"""High-recall candidate generation.

Given the neck pyramid we predict a **center heatmap** on the P3 grid
(stride 8).  The heatmap is supervised with CenterNet-style Gaussian targets and
CenterNet's penalty-reduced focal loss, which gives an extremely high-recall
proposal source with almost no background false positives.

From the heatmap we keep ``n_cand`` points per modality:

* ``n_global`` *global* top-k peaks (after local NMS)
* the remaining budget as *partition* candidates -- the best response inside
  each cell of a ``side x side`` grid.  This guarantees spatial coverage, so a
  drone in a corner cannot be missed because a distant blob scored slightly
  higher.

Local NMS (3x3 max-pool peak picking) prevents candidates piling up on one blob.

During training we additionally inject GT-perturbed candidates (random jitter
around each GT center), replacing the *lowest-scoring* predicted ones so the
candidate count stays constant.  This keeps the positive/negative balance sane
while the heatmap is still poorly calibrated.  At inference only predicted
candidates are used -- GT is never touched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import ConvBNAct, DWConvBNAct


# --------------------------------------------------------------------------- #
# sampling helpers -- all coordinates are normalised to [0,1], order (x, y)
# --------------------------------------------------------------------------- #
def sample_points(feat: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``feat`` (B,C,H,W) at normalised ``pts`` (B,K,2 as x,y).

    Returns (B, K, C).
    """
    grid = (pts * 2.0 - 1.0).unsqueeze(1)  # (B,1,K,2)
    out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out.squeeze(2).transpose(1, 2).contiguous()  # (B,K,C)


def bilinear_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Pixel-centre coordinates of an (h,w) grid, **normalised to [0,1]**.

    Every consumer of candidate points (grid_sample, scatter_points, the geometry
    prior) works in normalised units, so the normalisation has to happen here.
    Using pixel indices instead is a silent, far-reaching bug: grid_sample would
    clamp everything to the border and the scatter would index out of bounds.
    """
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    x = xs / max(w - 1, 1)
    y = ys / max(h - 1, 1)
    return torch.stack([x, y], dim=-1).flatten(0, 1)  # (H*W, 2) as x,y


# --------------------------------------------------------------------------- #
# heatmap head
# --------------------------------------------------------------------------- #
class HeatmapHead(nn.Module):
    """P2 (fine texture) + P3 (semantics) -> 1-channel center logits at P3 size."""

    def __init__(self, ch: int, hidden: int = 64):
        super().__init__()
        self.p2_branch = nn.Sequential(DWConvBNAct(ch, hidden, 3, 1),
                                       ConvBNAct(hidden, hidden, 1, 1))
        self.p3_branch = nn.Sequential(DWConvBNAct(ch, hidden, 3, 1),
                                       ConvBNAct(hidden, hidden, 1, 1))
        self.fuse = nn.Sequential(
            ConvBNAct(2 * hidden, hidden, 3, 1),
            DWConvBNAct(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        # low initial response -> early candidates stay spread out instead of
        # collapsing onto the image centre
        nn.init.constant_(self.fuse[-1].bias, -4.0)
        nn.init.normal_(self.fuse[-1].weight, std=0.01)

    def forward(self, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        a = F.interpolate(self.p2_branch(f2), size=f3.shape[-2:], mode="nearest")
        b = self.p3_branch(f3)
        return self.fuse(torch.cat([a, b], dim=1))  # (B,1,H,W) logits


# --------------------------------------------------------------------------- #
# peak picking
# --------------------------------------------------------------------------- #
def local_nms_topk(heat: torch.Tensor, k: int, kernel: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy local NMS via max-pool peak picking (fully vectorised).

    ``heat`` (B,1,H,W) -> flat indices (B,k) sorted by descending score and the
    corresponding scores.  If the map has fewer true peaks than ``k`` we fall
    back to raw top-k so the budget is always filled.
    """
    b, _, h, w = heat.shape
    pooled = F.max_pool2d(heat, kernel, 1, kernel // 2)
    keep = (heat >= pooled).flatten(1).float()          # (B, HW)
    # Always rank peaks before non-peaks; fill only the remaining slots. Raw
    # top-k of the entire map would undo suppression whenever peak count < k.
    masked = heat.flatten(1) + keep * 2.
    k = min(k, masked.shape[1])
    _, idx = torch.topk(masked, k, dim=1)
    scores=(heat.flatten(1)*keep).gather(1,idx)
    return idx, scores


# --------------------------------------------------------------------------- #
# candidate generator
# --------------------------------------------------------------------------- #
class CandidateGenerator(nn.Module):
    """Candidate points / scores / sampled features for **one** modality."""

    def __init__(self, ch: int, n_cand: int = 64, n_global: int = 48,
                 nms_kernel: int = 5, n_partitions: int = 16,
                 inject_gt_ratio: float = 0.25):
        super().__init__()
        if n_cand > n_global + n_partitions:
            raise ValueError(f"n_cand={n_cand} exceeds n_global+n_partitions budget")
        self.n_cand = n_cand
        self.n_global = min(n_global, n_cand)
        self.n_part = n_cand - self.n_global
        self.nms_kernel = nms_kernel
        self.inject_gt_ratio = inject_gt_ratio
        self.side = int(round(n_partitions ** 0.5))
        if self.side * self.side != n_partitions:
            raise ValueError("n_partitions must be a perfect square")
        self.n_cells = n_partitions
        self.per_cell = max(1, self.n_part // self.n_cells) if self.n_part > 0 else 0
        self.n_part = self.per_cell * self.n_cells  # actual partition budget

    # ------------------------------------------------------------------ #
    def forward(self, heat: torch.Tensor, f3: torch.Tensor,
                gt_centers: Optional[torch.Tensor] = None,
                training: bool = False, gt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        heat: (B,1,H,W) center logits at P3 resolution
        f3:   (B,C,H,W) neck features at the *same* resolution as ``heat``
        gt_centers: (B,M,2) normalised GT centers, training only
        """
        b, _, h, w = heat.shape
        dev, dt = heat.device, heat.dtype
        heat_s = torch.sigmoid(heat)
        grid = bilinear_grid(h, w, dev, dt)  # (H*W, 2)

        idx_g, sc_g = local_nms_topk(heat_s, self.n_global, self.nms_kernel)
        pts_g = grid[idx_g]                     # (B, n_global, 2)

        if self.per_cell > 0:
            idx_p, sc_p = self._partition_topk(heat_s, self.per_cell)
            pts_p = grid[idx_p]                 # (B, n_part, 2)
        else:
            pts_p, sc_p = pts_g[:, :0], sc_g[:, :0]

        pts = torch.cat([pts_g, pts_p], dim=1)
        scores = torch.cat([sc_g, sc_p], dim=1)

        # keep candidates inside the valid sampling area (grid_sample border pad)
        eps = 0.5 / float(max(h, w))
        pts = pts.clamp(min=eps, max=1.0 - eps)

        injected = torch.zeros(b, pts.shape[1], device=dev, dtype=dt)
        if training and self.inject_gt_ratio > 0 and gt_centers is not None \
                and gt_centers.shape[1] > 0:
            pts, scores, injected = self._inject_gt(pts, scores, gt_centers, gt_mask)

        feats = sample_points(f3, pts)  # (B,K,C)
        return {
            "pts": pts,               # (B,K,2) normalised x,y
            "scores": scores,         # (B,K)   heatmap probability
            "feats": feats,           # (B,K,C)
            "injected": injected,     # (B,K)   1 = GT-perturbed candidate
            "heat": heat,             # (B,1,H,W) logits
            "heat_size": (h, w),
        }

    # ------------------------------------------------------------------ #
    def _partition_topk(self, heat_s: torch.Tensor, per: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Best ``per`` peaks inside each cell of a side x side grid.

        Implemented by zero-padding the heatmap to a multiple of ``side`` and
        folding (cell_y, cell_x) into the batch dimension -- no python loop.

        Two things must be handled carefully here:
          1. ``block index`` encodes (batch, cell_y, cell_x) as
             ``b*side^2 + cell_y*side + cell_x``, so it must be decoded with
             ``// side`` and ``% side`` before being added to the local coords;
             treating it as a plain row offset overruns the map.
          2. the padded cells may point past ``H*W``; those candidates are
             dropped (score set to -inf and index clamped) rather than wrapping.
        """
        b, _, h, w = heat_s.shape
        s = self.side
        hp, wp = (h + s - 1) // s, (w + s - 1) // s
        padded = F.pad(heat_s, (0, s * wp - w, 0, s * hp - h))     # (B,1,s*hp,s*wp)
        blocks = padded.view(b, 1, s, hp, s, wp) \
                     .permute(0, 1, 2, 4, 3, 5).reshape(b * s * s, 1, hp, wp)
        idx, sc = local_nms_topk(blocks, per, self.nms_kernel)      # (B*s*s, per)
        idx = idx.view(b, s * s, per)
        sc = sc.view(b, s * s, per)

        cell = torch.arange(s * s, device=heat_s.device).view(1, s * s, 1)
        cell_y = torch.div(cell, s, rounding_mode="floor")          # (1,s*s,1)
        cell_x = cell % s
        gy = torch.div(idx, wp, rounding_mode="floor") + cell_y * hp
        gx = idx % wp + cell_x * wp
        valid = (gy < h) & (gx < w)
        g = (gy * w + gx).reshape(b, -1).clamp(0, h * w - 1)
        sc = torch.where(valid.reshape(b, -1), sc.reshape(b, -1),
                         torch.full_like(sc.reshape(b, -1), -1.0))
        return g, sc

    # ------------------------------------------------------------------ #
    def _inject_gt(self, pts: torch.Tensor, scores: torch.Tensor,
                   gt_centers: torch.Tensor, gt_mask: Optional[torch.Tensor] = None):
        """Replace low-score candidates with valid GTs only; keep empty images intact."""
        b, k, _ = pts.shape
        n_inj = min(k - 1, max(1, int(round(k * self.inject_gt_ratio))))
        if gt_mask is None:
            gt_mask = (gt_centers.abs().sum(-1) > 0)
        rows, score_rows, flags = [], [], []
        for bi in range(b):
            valid = gt_centers[bi][gt_mask[bi] > 0.5]
            if not valid.numel():
                rows.append(pts[bi]); score_rows.append(scores[bi])
                flags.append(torch.zeros_like(scores[bi]))
                continue
            idx = torch.arange(n_inj, device=pts.device) % valid.shape[0]
            inj = valid[idx].clone()
            # Reserve exact centers as well as jittered positives for tiny boxes.
            exact = min(n_inj, valid.shape[0])
            inj[exact:] += torch.randn_like(inj[exact:]) * 0.01
            inj = inj.clamp(1e-3, 1.0 - 1e-3)
            keep_idx = scores[bi].argsort(descending=True)[:k - n_inj]
            rows.append(torch.cat([pts[bi, keep_idx], inj]))
            score_rows.append(torch.cat([scores[bi, keep_idx], scores.new_ones(n_inj)]))
            flags.append(torch.cat([scores.new_zeros(k - n_inj), scores.new_ones(n_inj)]))
        return torch.stack(rows), torch.stack(score_rows), torch.stack(flags)
