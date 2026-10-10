"""Detection heads.

Architecture required by the task: ``128 -> 128 -> 5``, predicting
``(x_c, y_c, w, h, s)`` where ``s`` is the object confidence.

``s`` is implemented as a **single logit** (not a 2-class softmax) because UAV-CB
has a single foreground class; a 1-way sigmoid focal loss is strictly better
conditioned for a single-class problem and keeps the "score" directly comparable
across modalities.  Multi-object expansion is unaffected: each candidate carries
its own score, so up to ``max_det`` boxes can be emitted per frame.

An auxiliary 1x1 *preliminary* head predicts a coarse box for every candidate
before the late fusion; those boxes are what the late-stage box prior
``P(b_j^T | b_i^R)`` operates on.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import DWConvBNAct


class DetHead(nn.Module):
    """Candidate head: 128 -> 128 -> 5.

    Output semantics (raw network outputs -> predicted box), applied in
    :func:`decode_pred` so that **training and inference cannot disagree**:

    =============  ==========================  =========================
    channel        activation                  meaning
    =============  ==========================  =========================
    ``0`` (cx)     ``tanh(x) * 0.5``           x offset from the candidate point
    ``1`` (cy)     ``tanh(y) * 0.5``           y offset from the candidate point
    ``2`` (w)      ``sigmoid(x) * max_w``      box width  (normalised)
    ``3`` (h)      ``sigmoid(y) * max_h``      box height (normalised)
    ``4`` (s)      raw logit                   object confidence
    =============  ==========================  =========================

    Two decisions matter here:

    * **Size must be positive.** An unconstrained linear output can predict a
      negative width; CIoU then clamps the area to ``eps`` and the gradient
      vanishes, so the box regression silently stops learning while the loss still
      looks finite. ``sigmoid`` removes that failure mode by construction.
    * **Position is a residual around the candidate point**, not an absolute
      coordinate. The heatmap already localises the target to within a cell, so
      predicting an offset halves the dynamic range the head has to cover and keeps
      the prediction inside the frame without clipping.
    """

    def __init__(self, in_ch: int = 128, width: int = 128, n_layers: int = 2,
                 init_score: float = -4.0, max_wh: float = 0.5):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.LayerNorm(c), nn.Linear(c, width), nn.SiLU(inplace=True)]
            c = width
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(c, 5)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[4] = init_score
            # start from a small-drone size (~3% of the frame), matching the UAV-CB
            # median box of 68x52 px on a 1920x1080 image
            bias_w = math.log(0.03 / max_wh / (1 - 0.03 / max_wh))
            self.out.bias[2] = bias_w
            self.out.bias[3] = bias_w
        self.in_ch = in_ch
        self.width = width
        self.feat_dim = c
        self.max_wh = max_wh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,K,C) -> raw (B,K,5). Use :func:`decode_pred` for a box."""
        return self.out(self.body(x))


def decode_pred(raw: torch.Tensor, cand_pts: torch.Tensor,
                max_wh: float = 0.5, base_boxes: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw head output -> (boxes (B,K,4) normalised cx,cy,w,h, score logits (B,K)).

    ``cand_pts`` is (B,K,2) normalised. This is the single place where the head's
    output convention is defined -- losses and inference both call it.
    """
    if base_boxes is not None:
        # Candidate-region refinement: box scale sets the center search radius.
        center = base_boxes[..., :2] + raw[..., :2].tanh() * base_boxes[..., 2:] * .5
        size = (base_boxes[..., 2:] * raw[..., 2:4].clamp(-2,2).exp()).clamp(1e-5,max_wh)
        return torch.cat((center,size),-1),raw[...,4]
    off = torch.tanh(raw[..., :2]) * 0.5                    # (B,K,2) in [-0.5, 0.5]
    ctr = cand_pts + off
    wh = torch.sigmoid(raw[..., 2:4]) * max_wh              # (B,K,2) in (0, max_wh]
    boxes = torch.cat([ctr, wh], dim=-1)
    return boxes, raw[..., 4]


class BoxHead(nn.Module):
    """Auxiliary box head used *before* late fusion (coarse size estimate only).

    Input is a candidate sequence (B,K,C), so this uses Linear layers rather
    than convolutions -- a 1x1 conv over candidates would mix unrelated targets.
    """

    def __init__(self, in_ch: int = 128, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, width), nn.SiLU(inplace=True),
        )
        self.out = nn.Linear(width, 4)
        # Small-gain init, NOT zero: a zero final layer would block all gradient
        # from the loss back to ``body`` and to the refine module feeding it.
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,K,C) -> (B,K,4) normalised (cx,cy,w,h)."""
        return self.out(self.body(x))


class MultiScaleBoxRefine(nn.Module):
    """Refine a candidate by sampling P2/P3/P4/P5 around it.

    The candidate centre is the position prior; each pyramid level is queried with
    ``grid_sample`` at a fixed 9-point pattern scaled by the candidate's box size,
    and max+mean pooling over the 9 samples gives a per-level descriptor.  Every
    level is first compressed to ``hidden`` channels so the fused descriptor has a
    fixed width regardless of how many levels or channels the neck produces.

    This is what lets the final head localise a 20x20 px drone without paying for
    RoIAlign: P2 contributes the fine texture, P5 the context that separates a
    drone from a similarly-sized background blob.
    """

    _OFFSETS = ((0, 0), (-1, -1), (0, -1), (1, -1),
                (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

    def __init__(self, in_ch: int, n_levels: int = 4, hidden: int = 64):
        super().__init__()
        self.n_levels = n_levels
        self.hidden = hidden
        self.n_pts = len(self._OFFSETS)
        self.register_buffer('_sampling_offsets',torch.tensor(self._OFFSETS,dtype=torch.float32),persistent=False)
        # per-level compression: (2 * in_ch) [max,mean] -> hidden.
        # LayerNorm (not BatchNorm) because the tensor here is (B, hidden, 1, K):
        # BatchNorm over that layout would compute statistics across candidates of
        # the *same* image, which couples unrelated targets.
        self.reduce = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * in_ch), nn.Linear(2 * in_ch, hidden))
            for _ in range(n_levels)
        ])
        self.merge = nn.Sequential(
            nn.LayerNorm(hidden * n_levels),
            nn.Linear(hidden * n_levels, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        # small-gain init keeps the descriptor informative from step 0
        nn.init.normal_(self.merge[-1].weight, std=0.01)
        nn.init.zeros_(self.merge[-1].bias)

    def forward(self, maps, pts: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        maps: dict of (B,C,H,W) keyed P2..P5 (all with C = neck_out)
        pts:  (B,K,2) normalised candidate centres
        boxes:(B,K,4) normalised (cx,cy,w,h) -- gives the sampling scale
        Returns (B,K,hidden)
        """
        b, k, _ = pts.shape
        base = self._sampling_offsets.to(pts).view(1, 1, self.n_pts, 2)
        # (B,K,1,2) * (1,1,9,2) -> (B,K,9,2)
        scale = boxes[..., 2:].unsqueeze(2) * 0.4
        off = base * scale

        desc = []
        for i, name in enumerate(sorted(maps.keys())):      # P2, P3, P4, P5
            fm = maps[name]
            q = (pts.unsqueeze(2) + off).reshape(b, k * self.n_pts, 2).clamp(1e-3, 1 - 1e-3)
            grid = (q * 2 - 1).unsqueeze(1)
            sampled = nn.functional.grid_sample(fm, grid, mode="bilinear",
                                                padding_mode="border",
                                                align_corners=True)   # (B,C,1,K*9)
            sampled = sampled.squeeze(2).transpose(1, 2).reshape(b, k, self.n_pts, fm.shape[1])
            pooled = torch.cat([sampled.max(dim=2).values,
                                sampled.mean(dim=2)], dim=-1)           # (B,K,2C)
            desc.append(self.reduce[i](pooled))                          # (B,K,hidden)

        return self.merge(torch.cat(desc, dim=-1))                       # (B,K,hidden)
