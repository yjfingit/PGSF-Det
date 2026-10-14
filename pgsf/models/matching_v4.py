"""PGSF-Det v4 CFPM: coarse-to-fine cross-modal matching at object level.

The module replaces v2's fitted affine prior and v3's Gaussian anchors with a
two-stage, **coordinate-prior-free** procedure.  For the RGB -> TIR direction:

1. **Coarse semantic retrieval.**  ``Q_R = W_Q z_R`` and ``K_T = W_K z_T`` with
   ``W_Q, W_K`` of shape ``128 x 64``; the similarity matrix is
   ``A_RT = Q_R K_T^T / sqrt(64)`` of shape ``[B,K,K]``, and each RGB candidate
   keeps the **Top-L** (default 2) TIR candidates.  Only appearance is used: the
   centre coordinates are never compared, so nothing presumes the two modalities
   share a coordinate frame.

2. **Fine local probing.**  For every retained anchor the pair
   ``[query token, anchor token]`` (256-d) is mapped by ``OffsetMLP: 256 -> 128 -> 8``
   to four 2-D offsets.  Sampling coordinates are

       p_ils = centre_il + tanh(dp_ils) * size_il

   i.e. the offsets are scaled by the **anchor's own box width/height**, which is
   why no camera parameter (focal length, baseline, pre-fitted warp) appears
   anywhere.  Each probe point also predicts a 4-way softmax over P2-P5, and the
   four level features are combined with those weights (a soft, learned equivalent
   of FPN level selection).

   Two anchors x four probes = **8 local features**, aggregated by attention into
   ``z_T->R``.  The aggregation is ``attention_output - query`` and is
   zero-initialised by construction, so ``z_T->R = 0`` at step 0.

The TIR -> RGB direction is the exact mirror with independent parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.candidate import sample_points
from pgsf.models.fusion import SparseCrossAttention


class CoarseToFineMatching(nn.Module):
    """One direction of the v4 cross-modal matcher (instantiated twice)."""

    def __init__(self, dim: int = 128, retrieval_dim: int = 64, top_l: int = 2,
                 samples: int = 4, levels=("P2", "P3", "P4", "P5"), heads: int = 4,
                 ff_mult: int = 4, offset_scale: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.retrieval_dim = int(retrieval_dim)
        self.top_l = int(top_l)
        self.samples = int(samples)
        self.levels = tuple(levels)
        self.n_levels = len(self.levels)
        self.offset_scale = float(offset_scale)

        # ---- coarse retrieval: 128 x 64 projections ------------------------- #
        self.q_proj = nn.Linear(dim, retrieval_dim, bias=False)
        self.k_proj = nn.Linear(dim, retrieval_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)

        # ---- fine probing --------------------------------------------------- #
        self.offset = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(inplace=True),
                                    nn.Linear(dim, samples * 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.level_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(inplace=True),
                                        nn.Linear(dim, samples * self.n_levels))
        nn.init.normal_(self.level_head[-1].weight, std=0.01)
        nn.init.zeros_(self.level_head[-1].bias)
        self.point_embed = nn.Parameter(torch.zeros(samples, dim))
        self.level_embed = nn.Parameter(torch.zeros(self.n_levels, dim))

        # ---- attention aggregation over the L*S local features -------------- #
        self.agg = SparseCrossAttention(dim, heads, ff_mult)
        # ``SparseCrossAttention`` ships with a zero-initialised output projection so
        # that, elsewhere, the module starts as an exact identity.  Here that would
        # stack a *second* multiplicative zero on top of TD-Fusion's gates
        # (``f_cls = z + r * W_c(c)``, ``W_c`` zero-init) and starve the whole
        # local-probing branch of gradient.  The aggregation is a feature
        # extractor, not a gate, so it gets the ordinary small-gain init.
        nn.init.normal_(self.agg.w_o.weight, std=0.02)
        nn.init.zeros_(self.agg.w_o.bias)
        nn.init.normal_(self.agg.ff[-1].weight, std=0.02)
        nn.init.zeros_(self.agg.ff[-1].bias)

    # ------------------------------------------------------------------ #
    def similarity(self, z_q: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
        """``A = Q K^T / sqrt(retrieval_dim)``, shape ``[B,K,K]``."""
        q = self.q_proj(z_q)
        k = self.k_proj(z_other)
        return q @ k.transpose(1, 2) / math.sqrt(self.retrieval_dim)

    # ------------------------------------------------------------------ #
    def forward(self, z_q: torch.Tensor, z_other: torch.Tensor,
                box_other: torch.Tensor, other_maps):
        """``z_q [B,K,C]``, ``z_other [B,K,C]``, ``box_other [B,K,4]`` (the other
        modality's own normalised boxes), ``other_maps`` the other modality's
        P2-P5 features.  Returns ``(z_other_to_self [B,K,C], stats)``."""
        b, k, c = z_q.shape
        l = min(self.top_l, k)
        s = self.samples

        sim = self.similarity(z_q, z_other)                                  # (B,K,K)
        top = sim.topk(l, dim=-1)
        anchor = top.indices                                                 # (B,K,L)
        # gather needs matching ranks: flatten (B,K,L) -> (B, K*L) first
        flat_anchor = anchor.reshape(b, k * l)
        z_anchor = torch.gather(z_other, 1, flat_anchor[..., None].expand(-1, -1, c)
                                ).reshape(b, k, l, c)
        box_anchor = torch.gather(box_other, 1, flat_anchor[..., None].expand(-1, -1, 4)
                                  ).reshape(b, k, l, 4)

        pair = torch.cat((z_q[:, :, None].expand_as(z_anchor), z_anchor), -1)  # (B,K,L,2C)
        delta = self.offset(pair).view(b, k, l, s, 2).tanh() * self.offset_scale
        # offsets are scaled by the ANCHOR box size -> no camera parameter
        local_pts = box_anchor[..., None, :2] + delta * box_anchor[..., None, 2:]
        level_weights = self.level_head(pair).view(b, k, l, s, self.n_levels).softmax(-1)

        flat = local_pts.reshape(b, k * l * s, 2)
        stacked = torch.stack([sample_points(other_maps[name], flat)
                               for name in self.levels], -2)                 # (B,K*L*S,Lv,C)
        local = (stacked * level_weights.reshape(b, k * l * s, self.n_levels, 1)).sum(-2)
        local = local.reshape(b, k, l * s, c)                                # (B,K,8,C)

        point_id = torch.arange(l * s, device=z_q.device) % s
        soft_level = (level_weights.reshape(b, k, l * s, self.n_levels, 1)
                      * self.level_embed[None, None, None]).sum(-2)
        keys = local + self.point_embed[point_id][None, None] + soft_level

        out, attn = self.agg(z_q, keys, keys, return_attn=True)
        # ``out = query + W_O(ctx) + FF(...)`` with both projections zero-initialised,
        # so the pure cross-modal message is ``out - query`` and it starts at exactly 0.
        message = out - z_q

        stats = {
            "sim": sim, "anchor_idx": anchor, "anchor_scores": top.values,
            "margin": top.values[..., 0] - (top.values[..., 1] if l > 1 else torch.zeros_like(top.values[..., 0])),
            "local_pts": local_pts, "level_weights": level_weights,
            "attn": attn, "top_l": l, "samples": s,
        }
        return message, stats


__all__ = ["CoarseToFineMatching", "gt_slot_index", "identity_matches"]


# --------------------------------------------------------------------------- #
# instance identity -- the ONLY place correspondence labels enter
# --------------------------------------------------------------------------- #
def gt_slot_index(points, boxes, mask, center_thr: float = 0.05):
    """Nearest GT slot of every point, using the detector's own assignment rule.

    ``points (B,N,2)``, ``boxes (B,M,4)``, ``mask (B,M)`` ->
    ``(index (B,N) long, valid (B,N) bool)``.  A point is valid only when it lies
    inside the expanded GT box *and* within ``center_thr``, which is exactly the
    condition ``assign_candidates`` uses, so supervision and detection agree.
    """
    b, n, _ = points.shape
    m = boxes.shape[1]
    if m == 0:
        return (torch.zeros(b, n, dtype=torch.long, device=points.device),
                torch.zeros(b, n, dtype=torch.bool, device=points.device))
    delta = points.unsqueeze(2) - boxes[:, None, :, :2]
    inside = (delta.abs() <= boxes[:, None, :, 2:].clamp_min(1e-6)).all(-1)
    inside = inside & (delta.norm(dim=-1) <= center_thr) & (mask[:, None, :] > 0.5)
    distance = delta.norm(dim=-1).masked_fill(~inside, float("inf"))
    return distance.argmin(-1), inside.any(-1)


def identity_matches(q_pts, q_boxes, q_mask, k_pts, k_boxes, k_mask, corr_pairs,
                     center_thr: float = 0.05, frame_valid=None):
    """``(B,Nq,Nk)`` boolean: query candidate ``n`` and key candidate ``m`` are the same object.

    ``corr_pairs`` (B, Mq, Mk) is the only correspondence input, and it must come
    from the dataset.  Nothing here assumes the ``i``-th target on one side matches
    the ``i``-th on the other, that equal class ids mean the same object, or that
    the nearest centres are the same object; both sides may hold different numbers
    of objects.  A query with no counterpart simply gets an all-False row and the
    matching loss ignores it.
    """
    qi, qv = gt_slot_index(q_pts, q_boxes, q_mask, center_thr)
    ki, kv = gt_slot_index(k_pts, k_boxes, k_mask, center_thr)
    n = q_pts.shape[1]
    if corr_pairs.shape[1] == 0 or corr_pairs.shape[2] == 0:
        return torch.zeros(q_pts.shape[0], n, k_pts.shape[1], dtype=torch.bool,
                           device=q_pts.device)
    # two hops: candidate -> its own GT slot, then that GT slot -> the counterpart's GT slot
    row = corr_pairs.gather(1, qi[..., None].expand(-1, -1, corr_pairs.shape[-1]))
    pair = row.gather(2, ki[:, None, :].expand(-1, n, -1))
    match = (pair > 0.5) & kv[:, None, :] & qv[..., None]
    if frame_valid is not None:
        match = match & (frame_valid[:, None, None] > 0.5)
    return match
