"""PGSF-Det v3 proposals: P2/P3 dense prediction, 128 sparse candidates and 8x8 grid tokens.

The v3 candidate budget is **128 per modality** (the v2 budget was 64).  The
construction itself is unchanged -- P2 peaks, P3 peaks and a 4x4 partition
supplement, then cross-level de-duplication down to a constant output width --
so the dense/coarse/quality supervision described in :mod:`pgsf.models.proposal`
carries over verbatim.

New in v3 is :class:`GridTokens`: a small fixed set of pooled feature tokens
(8x8 = 64) with their normalised cell centres.  They enlarge the *retrieval pool*
of the cross-modal correspondence module without adding proposals, so a query
whose correspondent was not proposed as a peak can still be anchored nearby.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import DWConvBNAct
from pgsf.models.candidate import bilinear_grid, sample_points
from pgsf.models.proposal import (
    MultiScaleCandidates,
    aligned_iou,
    decode_coarse,
    proposal_losses,
)


class DenseProposalHeadV3(nn.Module):
    """P2 + P3 dense head: center logit plus a coarse ``(dx, dy, w, h, obj, quality)``.

    Identical in semantics to :class:`pgsf.models.proposal.DenseProposalHead`; the
    class exists so the v3 graph is self-describing and the proposal branch of v3
    can evolve without touching the v2 code path.
    """

    def __init__(self, channels: int = 128, hidden: int = 32, max_wh: float = 0.5):
        super().__init__()
        self.max_wh = max_wh

        def head():
            h = nn.Sequential(DWConvBNAct(channels, hidden),
                              DWConvBNAct(hidden, hidden), nn.Conv2d(hidden, 6, 1))
            nn.init.normal_(h[-1].weight, std=0.01)
            nn.init.zeros_(h[-1].bias)
            with torch.no_grad():
                h[-1].bias[0] = -4.0
                h[-1].bias[3:5] = math.log(0.03 / max_wh / (1 - 0.03 / max_wh))
            return h

        self.p2 = head()
        self.p3 = head()

    def forward(self, p2: torch.Tensor, p3: torch.Tensor):
        return {"P2": self.p2(p2), "P3": self.p3(p3)}


class CandidateV3(MultiScaleCandidates):
    """v2 candidate construction with the **128** budget as the v3 default.

    Output keys are exactly those consumed by the shared proposal loss, the
    detector assignment and the v2/v3 decoding paths
    (``pts``/``feats``/``scores``/``quality``/``objectness``/``quality_logits``/
    ``boxes``/``injected``/``levels``/``dense``), so no downstream code needs a
    v3-specific branch.
    """

    def __init__(self, channels: int = 128, n_cand: int = 128,
                 inject_gt_ratio: float = 0.25, max_wh: float = 0.5,
                 nms_kernel: int = 5):
        super().__init__(channels, n_cand=n_cand, inject_gt_ratio=inject_gt_ratio,
                         max_wh=max_wh, nms_kernel=nms_kernel)


class GridTokens(nn.Module):
    """``grid x grid`` pooled tokens from several neck levels, with cell centres.

    Adaptive average pooling keeps the token count independent of the input
    resolution, which is what makes the retrieval pool comparable between the
    1280x800 RGB frame and the 640x512 TIR frame.

    Returns
        tokens (B, grid*grid, C)  normalised-coordinate cell centres (grid*grid, 2)
    """

    def __init__(self, channels: int = 128, grid: int = 8,
                 levels=("P3", "P4")):
        super().__init__()
        self.grid = int(grid)
        self.levels = tuple(levels)
        in_ch = channels * len(self.levels)
        self.proj = nn.Sequential(nn.LayerNorm(in_ch), nn.Linear(in_ch, channels),
                                  nn.SiLU())
        ys, xs = torch.meshgrid(torch.arange(self.grid), torch.arange(self.grid),
                                indexing="ij")
        pts = torch.stack(((xs + 0.5) / self.grid, (ys + 0.5) / self.grid), -1)
        self.register_buffer("grid_pts", pts.reshape(-1, 2).float())

    def forward(self, maps):
        pooled = []
        for name in self.levels:
            fm = maps[name]
            p = F.adaptive_avg_pool2d(fm, (self.grid, self.grid))     # (B,C,G,G)
            pooled.append(p.flatten(2).transpose(1, 2))               # (B,G*G,C)
        tokens = self.proj(torch.cat(pooled, -1))
        return tokens, self.grid_pts.to(tokens)


__all__ = [
    "DenseProposalHeadV3",
    "CandidateV3",
    "GridTokens",
    "decode_coarse",
    "aligned_iou",
    "sample_points",
    "bilinear_grid",
    "proposal_losses",
]
