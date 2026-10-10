"""Mid-stage probabilistic-geometry-guided sparse cross-modal Transformer.

This is the first (mid-stage) fusion.  For every candidate of modality A we build
a **sparse key set** drawn from modality B:

  * ``n_local`` (8)  **local deformable samples** around the geometry-predicted
    position mu.  Offsets are predicted from the query feature, so the model can
    pull the sample location towards where the target actually is.
  * ``n_global`` (16) **global high-response candidates** of B -- the fallback path
    that keeps working when the geometry prior is wrong.
  * 1 **learnable null token** -- lets the query declare "B carries no evidence
    here".  Essential for UAV-CB frames where the drone is invisible in TIR: those
    must not be forced to import garbage from the other modality.

Attention logits::

    a_ij = q_i . k_j / sqrt(d_h) + lambda_g * log P_G(p_j | p_i)   (local keys)
    a_ij = q_i . k_j / sqrt(d_h) + type_bias                      (null key)

The geometric log-prior is applied **only** to real spatial keys.  Global and null
keys get their own learned type bias, so "spatially plausible" is never conflated
with "semantically relevant".

Reliability gate::

    gamma_i = sigmoid( MLP([f_i, z_i, c_g, u_i]) )
    f_i'    = f_i + gamma_i * (z_i - f_i)

``c_g`` is geometry confidence relative to the initial prior and ``u_i`` the matching
uncertainty.  ``W_O`` inside :class:`SparseCrossAttention` is **zero-initialised**,
so at step 0 the module is exactly the identity and cannot damage the already
trained single-modality features -- this is what makes stage-2 initialisation safe.

Enhanced candidates are scattered back onto the feature maps with a differentiable
splat (weight-normalised), so the second half of the network sees spatially
enriched features rather than only per-candidate vectors.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.candidate import sample_points
from pgsf.models.geometry import ProbGeoPrior, invert_affine


# --------------------------------------------------------------------------- #
# differentiable scatter (write candidates back onto a feature map)
# --------------------------------------------------------------------------- #
def scatter_points(pts: torch.Tensor, vals: torch.Tensor, height: int, width: int
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear splat of ``vals`` (B,K,C) at normalised ``pts`` (B,K,2) onto (B,C,H,W).

    Overlapping candidates are handled by **weight normalisation**: each target
    cell receives ``sum_k v_k w_k / (sum_k w_k + eps)``, i.e. the weighted mean,
    rather than a 3x-inflated value from naive scatter-add.  The coverage mask is
    returned so the caller can rescale the added residual by how many cells
    actually received data.

    Returns (map (B,C,H,W), coverage (B,1,H,W) in [0,1]).
    """
    b, k, c = vals.shape
    dev, dt = vals.device, vals.dtype
    acc_dt = torch.float32 if dt in (torch.float16,torch.bfloat16) else dt
    vals = vals.to(acc_dt)
    gx = (pts[..., 0].clamp(0, 1) * max(width - 1, 0)).clamp(0, max(width - 1, 0))
    gy = (pts[..., 1].clamp(0, 1) * max(height - 1, 0)).clamp(0, max(height - 1, 0))
    x0 = gx.floor().long().clamp(0, max(width - 1, 0))
    y0 = gy.floor().long().clamp(0, max(height - 1, 0))
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    fx = (gx - x0.to(gx.dtype)).clamp(0, 1)
    fy = (gy - y0.to(gy.dtype)).clamp(0, 1)

    # scatter_add_ indexes dim 1 of ``self``, so the flattened spatial axis must
    # BE dim 1: num/den are (B, H*W, C).
    num = torch.zeros(b, height * width, c, device=dev, dtype=acc_dt)
    den = torch.zeros(b, height * width, 1, device=dev, dtype=acc_dt)
    for xx, yy, wgt in ((x0, y0, (1 - fx) * (1 - fy)), (x1, y0, fx * (1 - fy)),
                        (x0, y1, (1 - fx) * fy), (x1, y1, fx * fy)):
        idx = (yy * width + xx).unsqueeze(-1)                       # (B,K,1)
        wgt3 = wgt.unsqueeze(-1).to(acc_dt)                                    # (B,K,1)
        num.scatter_add_(1, idx.expand(b, k, c), vals * wgt3)
        den.scatter_add_(1, idx, wgt3)
    out = num / den.clamp_min(1e-8)
    # cells with no coverage must stay exactly zero rather than become noise;
    # ``torch.where`` also avoids the systematic 1/(1+eps) shrink that adding eps
    # to the denominator would impose on *every* covered cell.
    out = torch.where(den > 0, out, torch.zeros_like(out))
    coverage = (den.squeeze(-1) > 0).to(dt)                       # (B, H*W)
    return out.view(b, height, width, c).permute(0, 3, 1, 2).contiguous().to(dt), \
        coverage.view(b, 1, height, width)


# --------------------------------------------------------------------------- #
# sparse cross-modal attention
# --------------------------------------------------------------------------- #
class SparseCrossAttention(nn.Module):
    """Multi-head cross attention over a heterogeneous (local/global/null) key set."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.w_o = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.w_o.weight)   # zero-init -> residual starts as identity
        nn.init.zeros_(self.w_o.bias)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                log_prior: Optional[torch.Tensor] = None,
                prior_mask: Optional[torch.Tensor] = None,
                type_bias: Optional[torch.Tensor] = None,
                return_attn: bool = False, key_mask: Optional[torch.Tensor] = None):
        """
        query : (B, Nq, C)
        key, value : (B, Nk, C)  -- one shared key set for every query (late fusion),
                    or (B, Nq, K, C) -- a *per-query* key set (mid fusion)
        log_prior  (B, Nq, K), added only where prior_mask is True
        prior_mask (B, Nq, K) bool
        type_bias  (K,) or (B, K)
        Returns (B, Nq, C) and optionally the head-averaged attention (B, Nq, K).
        """
        if query.dim() != 3:
            raise ValueError(f"query must be (B,Nq,C), got {tuple(query.shape)}")
        b, nq, c = query.shape
        if key.dim() == 3:                       # shared key set -> broadcast
            nk = key.shape[1]
            key = key.unsqueeze(1).expand(b, nq, nk, c)
            value = value.unsqueeze(1).expand(b, nq, nk, c)
        elif key.dim() == 4:
            nk = key.shape[2]
        else:
            raise ValueError(f"key must be (B,Nk,C) or (B,Nq,K,C), got {tuple(key.shape)}")

        # kk is (B, Nq, Nk, h, dh)
        # NB: the query *must* go through q_proj -- skipping it would leave q_proj
        # unused (dead parameters) and deny the model any query-side transformation.
        q = self.q_proj(self.q_norm(query))
        q = q.view(b, nq, self.h, self.dh).transpose(1, 2)                # (b,h,nq,d)
        kk = self.k_norm(key).view(b, nq, nk, self.h, self.dh)               # (b,nq,nk,h,d)
        attn = torch.einsum("bhqd,bqkhd->bhqk", q, kk) * self.scale

        if log_prior is not None:
            lp = log_prior.unsqueeze(1)                          # (B,1,Nq,Nk)
            if prior_mask is not None:
                lp = torch.where(prior_mask.unsqueeze(1), lp, torch.zeros_like(lp))
            attn = attn + lp
        if type_bias is not None:
            tb = type_bias
            if tb.dim() == 1:
                tb = tb.view(1, 1, 1, -1)
            elif tb.dim() == 2:
                tb = tb.unsqueeze(1).unsqueeze(1)               # (b,1,1,K)
            attn = attn + tb
        if key_mask is not None:
            attn = attn.masked_fill(~key_mask.unsqueeze(1), -1e4)
        attn = attn.float().softmax(dim=-1)
        attn_d = self.drop(attn)
        vv = value.view(b, nq, nk, self.h, self.dh)             # (b,nq,nk,h,d)
        ctx = torch.einsum("bhqk,bqkhd->bhqd", attn_d.to(vv.dtype), vv)     # (b,h,nq,d)
        out = query + self.w_o(ctx.transpose(1, 2).reshape(b, nq, c))
        out = out + self.ff(self.ff_norm(out))
        if return_attn:
            return out, attn.mean(dim=1)                        # (B,Nq,Nk)
        return out


# --------------------------------------------------------------------------- #
# mid-stage fusion
# --------------------------------------------------------------------------- #
class MidFusion(nn.Module):
    """Bidirectional probabilistic-geometry-guided sparse fusion (RGB <-> TIR)."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_local: int = 8,
                 n_global: int = 16, lambda_geo: float = 0.35,
                 use_geo: bool = True, use_gate: bool = True,
                 ff_mult: int = 4):
        super().__init__()
        self.n_local = n_local
        self.n_global = n_global
        self.use_geo = use_geo
        self.use_gate = use_gate
        self.lambda_geo = lambda_geo

        # one geometry model, shared by both directions (same scene, same prior)
        self.geo = ProbGeoPrior(d_model, hidden=d_model, use_residual=use_geo)

        self.attn_r2t = SparseCrossAttention(d_model, n_heads, ff_mult)
        self.attn_t2r = SparseCrossAttention(d_model, n_heads, ff_mult)

        # deformable local offsets, predicted from the query feature
        self.off_r2t = self._make_offset(d_model, n_local)
        self.off_t2r = self._make_offset(d_model, n_local)

        self.k_local = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_token, std=0.02)
        # [global, null]
        self.type_bias = nn.Parameter(torch.zeros(2))

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model + 2, d_model // 2), nn.SiLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _make_offset(d_model: int, n_local: int) -> nn.Sequential:
        m = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(inplace=True),
                          nn.Linear(d_model, n_local * 2))
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
        return m

    # ------------------------------------------------------------------ #
    def forward(self, q_feat: torch.Tensor, q_pts: torch.Tensor,
                other_map: torch.Tensor, other_global: torch.Tensor,
                direction: str = "r2t",
                maps: Optional[Dict[str, torch.Tensor]] = None,
                affine: Optional[torch.Tensor] = None,
                other_global_pts: Optional[torch.Tensor] = None):
        """
        q_feat      (B,Nq,C)  query candidate features
        q_pts       (B,Nq,2)  normalised query centers
        other_map   (B,C,H,W) other-modality feature map (P3 grid)
        other_global(B,Ng,C)  other-modality global candidates
        maps        optional dict of maps to write the enhanced candidates back into
        """
        b, nq, c = q_feat.shape
        assert direction in ("r2t", "t2r")
        ng = other_global.shape[1]
        if affine is None:
            affine = self.geo.A if direction == "r2t" else invert_affine(self.geo.A)
        if not self.use_geo:
            affine = q_pts.new_tensor([[1., 0., 0.], [0., 1., 0.]])
        mu, L, extra = self.geo(q_pts, q_feat, affine=affine)
        conf = self.geo.confidence(extra).unsqueeze(-1)                    # (B,Nq,1)

        # ---- local deformable samples around the geometry prediction
        off_mlp = self.off_r2t if direction == "r2t" else self.off_t2r
        off = off_mlp(q_feat).view(b, nq, self.n_local, 2).tanh() * 0.05
        local_pts = (mu.unsqueeze(2) + off).clamp(1e-3, 1 - 1e-3)
        local_feat = sample_points(other_map, local_pts.reshape(b, nq * self.n_local, 2))
        local_k = self.k_local(local_feat.view(b, nq, self.n_local, c))

        L_rep = L.unsqueeze(2).expand(-1, -1, self.n_local, -1, -1)
        lp_local = ProbGeoPrior.log_prob(local_pts, mu.unsqueeze(2), L_rep)   # (B,Nq,L)

        # ---- global candidates (fallback path) and the null token
        # Per the spec the geometric log-prior is applied to the **local spatial
        # keys only**. Global candidates are spatially unconstrained by design
        # (that is precisely why they are the fallback when the geometry prior is
        # wrong), so scoring them with log P_G would defeat their purpose; they get
        # the learned type bias instead.
        glob_k = self.k_global(other_global).unsqueeze(1).expand(b, nq, ng, c)
        null = self.null_token.expand(b, nq, 1, c)
        keys = torch.cat([local_k, glob_k, null], dim=2)          # (B,Nq,K,C)
        values = keys
        n_keys = self.n_local + ng + 1

        if self.use_geo:
            # prior lives on the first ``n_local`` keys; zeros elsewhere
            prior_mask = torch.zeros(b, nq, n_keys, dtype=torch.bool,
                                     device=q_feat.device)
            prior_mask[..., :self.n_local] = True
            log_prior = torch.cat([lp_local * self.lambda_geo,
                                   lp_local.new_zeros(b, nq, ng + 1)], dim=2)
        else:
            prior_mask, log_prior = None, None

        bias = torch.cat([
            torch.zeros(self.n_local, device=q_feat.device, dtype=q_feat.dtype),
            self.type_bias[0].expand(ng),
            self.type_bias[1].expand(1),
        ], dim=0)

        # Each query has its own key set (its local samples differ), so the
        # attention runs on the (B, Nq, K, C) per-query key layout.
        attn_mod = self.attn_r2t if direction == "r2t" else self.attn_t2r
        z, attn = attn_mod(q_feat, keys, values,
                           log_prior=log_prior, prior_mask=prior_mask,
                           type_bias=bias, return_attn=True)

        # Matching uncertainty must include attention ambiguity and null mass,
        # rather than duplicating the geometry entropy already represented by conf.
        entropy = -(attn.float() * attn.float().clamp_min(1e-8).log()).sum(-1)
        unc = ((entropy / math.log(n_keys) + attn[..., -1]) * 0.5).unsqueeze(-1).to(q_feat.dtype)
        if self.use_gate:
            gamma = torch.sigmoid(self.gate(torch.cat([q_feat, z, conf, unc], dim=-1)))
        else:
            gamma = torch.ones(b, nq, 1, device=q_feat.device, dtype=q_feat.dtype)
        out = q_feat + gamma * (z - q_feat)

        # association evidence for L_assoc: null-key score vs. best real-key score
        n_real = self.n_local + ng
        null_logit = attn[..., -1].clamp_min(1e-8).log()          # (B,Nq)
        best_real_logit = attn[..., :n_real].max(dim=-1).values.clamp_min(1e-8).log()             # (B,Nq)
        stats = {
            "conf": conf.squeeze(-1),
            "unc": unc.squeeze(-1),
            "gamma": gamma.squeeze(-1),
            "mu": mu,
            "sigma": L,
            "null_logit": null_logit,
            "best_real_logit": best_real_logit,
            "attn": attn,
            "key_pts": (torch.cat([local_pts, other_global_pts.unsqueeze(1).expand(-1, nq, -1, -1)], dim=2)
                        if other_global_pts is not None else local_pts),
        }

        new_maps = None
        if maps is not None:
            new_maps = {}
            for name, fm in maps.items():
                h, w = fm.shape[-2:]
                splat, cov = scatter_points(q_pts, out - q_feat, h, w)
                new_maps[name] = fm + splat * cov
        return {"feats": out, "stats": stats, "maps": new_maps}
