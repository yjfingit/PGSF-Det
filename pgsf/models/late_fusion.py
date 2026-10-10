"""Late-stage object-level bidirectional Transformer.

After the neck we run a first (preliminary) detection on the pyramid, take the
best ``n_cand_late`` (32) candidates per modality as **object tokens**, and let
them attend across modalities::

    M_ij = q_i^R . k_j^T / sqrt(d) + lambda_p * log P(b_j^T | b_i^R)

plus one learnable **null token** per modality, so a candidate with no counterpart
is not forced onto an unrelated object.  A reliability gate (A4) then blends the
cross-modal update with the original feature, and each modality keeps its **own**
enhanced features and its own detection head -- the two output streams stay
independent at the very end.

This module deliberately does **not** enforce a one-to-one matching.  Several
queries may attend to the same key; the association supervision only asks "is the
best available correspondence better than the null token?".
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.fusion import SparseCrossAttention
from pgsf.models.geometry import invert_affine


class BoxProbHead(nn.Module):
    """Log P(b_j | b_i) under a Gaussian on the normalised (cx,cy,w,h) box."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(4))
        nn.init.zeros_(self.log_std)

    def forward(self, box_q: torch.Tensor, box_k: torch.Tensor,
                A: torch.Tensor, b: torch.Tensor,
                lam: float = 1.0) -> torch.Tensor:
        """
        box_q (B,Nq,4), box_k (B,Nk,4) in normalised (cx,cy,w,h).
        ``A`` is the 2x3 homogeneous affine (its 3rd column is the translation),
        so ``b`` is accepted for interface symmetry but not used.
        Returns (B,Nq,Nk) additive log-prior (already scaled by ``lam``).
        """
        cq, sq = box_q[..., :2], box_q[..., 2:].clamp_min(1e-4)
        ck = box_k
        hom = torch.cat([cq, torch.ones_like(cq[..., :1])], dim=-1)
        pred_c = hom @ A.to(device=cq.device, dtype=cq.dtype).transpose(-1, -2)                                 # (B,Nq,2)
        # residual in normalised units; size residual weighted by 4 to balance units
        dc = ck.unsqueeze(1)[..., :2] - pred_c.unsqueeze(2)               # (B,Nq,Nk,2)
        ds = (ck.unsqueeze(1)[..., 2:] - sq.unsqueeze(2)) * 4.0
        r = torch.cat([dc, ds], dim=-1)
        var = torch.exp(self.log_std).to(cq.dtype).clamp_min(1e-4)
        return lam * (-0.5 * (r ** 2 / var).sum(-1))


class LateFusion(nn.Module):
    """Bidirectional object-level cross attention with box prior, null token and gate."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, ff_mult: int = 4,
                 lambda_box: float = 0.35, use_gate: bool = True, use_box_prior: bool = True):
        super().__init__()
        self.attn_r2t = SparseCrossAttention(d_model, n_heads, ff_mult)
        self.attn_t2r = SparseCrossAttention(d_model, n_heads, ff_mult)
        self.box_prior = BoxProbHead()
        self.lambda_box = lambda_box if use_box_prior else 0.0
        self.use_box_prior = use_box_prior
        self.use_gate = use_gate
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_token, std=0.02)
        self.null_bias = nn.Parameter(torch.zeros(1))
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model + 1, d_model // 2), nn.SiLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, q_feat: torch.Tensor, q_box: torch.Tensor,
                k_feat: torch.Tensor, k_box: torch.Tensor,
                A: torch.Tensor, b: torch.Tensor, direction: str = "r2t"):
        """
        q_feat (B,Nq,C), q_box (B,Nq,4); k_feat (B,Nk,C), k_box (B,Nk,4)
        """
        if direction not in ("r2t", "t2r"):
            raise ValueError(direction)
        if direction == "t2r":
            A = invert_affine(A)
        bsz, nq, c = q_feat.shape
        null = self.null_token.expand(bsz, 1, c)
        keys = torch.cat([k_feat, null], dim=1)
        values = keys

        if self.use_box_prior:
            prior = self.box_prior(q_box, k_box, A, b, self.lambda_box)   # (B,Nq,Nk)
            prior = F.pad(prior, (0, 1), value=0.0)
        else:
            prior = None

        bias = torch.cat([torch.zeros(k_feat.shape[1], device=q_feat.device,
                                      dtype=q_feat.dtype), self.null_bias.view(1)])
        attn_mod = self.attn_r2t if direction == "r2t" else self.attn_t2r
        z, attn = attn_mod(q_feat, keys, values, log_prior=prior, type_bias=bias,
                            return_attn=True)

        if self.use_gate:
            gamma = torch.sigmoid(self.gate(torch.cat([q_feat, z, attn[..., -1:]], dim=-1)))
        else:
            gamma = torch.ones(bsz, nq, 1, device=q_feat.device, dtype=q_feat.dtype)
        out = q_feat + gamma * (z - q_feat)

        return {
            "feats": out,
            "attn": attn,                      # (B,Nq,Nk+1)
            "gamma": gamma.squeeze(-1),
            "null_attn": attn[..., -1],        # attention mass on the null token
            "real_attn": attn[..., :-1].max(dim=-1).values,
        }


def select_topk_candidates(cand: Dict[str, torch.Tensor], boxes: torch.Tensor,
                           k: int) -> Dict[str, torch.Tensor]:
    """Keep the top-``k`` candidates by preliminary score, aligned across tensors.

    Injected (GT-perturbed) candidates are **always kept** regardless of their
    score: right after training starts the heatmap is uninformative, so a pure
    score ranking would throw away every positive and the box/class losses would
    see zero positives.  Keeping them guarantees a non-empty positive set from
    step 0; at convergence their scores are learned to be high anyway.
    """
    scores = cand["scores"]                                  # (B,N)
    inj = cand.get("injected")
    if inj is not None:
        # rank = +inf for injected candidates so they occupy the top slots
        rank = scores + inj * 10.0
    else:
        rank = scores
    k = min(k, rank.shape[1])
    idx = rank.topk(k, dim=1).indices                        # (B,k)
    c = cand["feats"].shape[-1]
    selected = {
        "idx": idx,
        "pts": torch.gather(cand["pts"], 1, idx[..., None].expand(-1, -1, 2)),
        "feats": torch.gather(cand["feats"], 1, idx[..., None].expand(-1, -1, c)),
        "scores": torch.gather(scores, 1, idx),
        "boxes": torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 4)),
        "injected": (torch.gather(inj, 1, idx) if inj is not None
                     else torch.zeros_like(idx, dtype=boxes.dtype)),
    }
    for name in ("quality","quality_logits","levels","objectness"):
        if name in cand:selected[name]=cand[name].gather(1,idx)
    return selected
