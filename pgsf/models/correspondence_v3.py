"""PGSF-Det v3 cross-modal correspondence and rejection-gated fusion.

The module implements **one** object-level fusion, in three explicit steps, and
runs it in both directions (RGB->TIR and TIR->RGB):

1. **Coarse retrieval.**  Each of the ``N`` object tokens of the query modality
   scores every key of the other modality -- its ``N`` candidate tokens, its
   ``8x8 = 64`` grid tokens and one null token -- with a **64-d Q/K dot
   product**.  The ``top_k`` (= 2) highest-scoring *real* keys become the
   anchors.  There is deliberately **no** train-fitted affine and **no** raw
   coordinate distance in this step: the initial correspondence is decided by
   learned appearance alone, so it transfers to a camera pair whose geometry was
   never measured.

2. **Local probabilistic matching.**  Each anchor predicts a 2-D Gaussian
   ``N(mu, Sigma)``: ``mu`` is where the correspondent is believed to be and
   ``Sigma`` how unsure we are.  ``Sigma`` drives the sampling displacement, so an
   uncertain anchor probes a wider neighbourhood instead of always sending the
   same fixed ring.  Each Gaussian contributes ``samples`` (= 8) points, spread
   over P2-P5 (2 per level) -- 16 points per object -- and the local features at
   those points are what actually verify the correspondence.

3. **Rejection fusion.**  A null probability ``p_null`` and a gate ``g`` blend the
   cross-modal message ``m`` into the query token::

       z' = z + g * (1 - p_null) * W_m(m)

   ``W_m`` is zero-initialised, so the module starts as the exact identity and the
   network starts as pure single-modality detection.  When the match is
   unreliable (``p_null -> 1``) or the counterpart is missing, the token is left
   untouched -- a wrong fusion is strictly worse than no fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.candidate import sample_points
from pgsf.models.fusion import SparseCrossAttention
from pgsf.models.fusion_v2 import gaussian_log_prob


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reference_points(per_level: int) -> torch.Tensor:
    """Unit-scale sampling pattern for one level: the centre, or an even-angle ring."""
    if per_level <= 1:
        return torch.zeros(1, 2)
    angles = torch.arange(per_level, dtype=torch.float32) * (2 * math.pi / per_level)
    return torch.stack((angles.cos(), angles.sin()), -1)


class GaussianAnchorHead(nn.Module):
    """Per-anchor ``N(mu, Sigma)`` with an analytic, always-PD Cholesky factor.

    ``Sigma = B B^T + floor^2 I`` with ``B`` lower triangular.  The Cholesky of
    ``B B^T + floor^2 I`` is built in closed form (no ``linalg`` call), which keeps
    the covariance positive definite for *any* network output -- including the
    large off-diagonal regime where merely flooring a diagonal Cholesky would
    already be indefinite -- and keeps the forward free of cuSOLVER workspace
    allocation, so the module stays CUDA-graph capturable.
    """

    def __init__(self, dim: int = 128, hidden: int = 128,
                 sigma_floor: float = 0.02, init_sigma: float = 0.05):
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.init_sigma = float(init_sigma)
        self.register_buffer("log_sigma_init", torch.tensor(math.log(float(init_sigma))))
        # inputs are [query token, anchor token, anchor point (x, y, 0)] -> 2C + 3
        in_dim = 2 * dim + 3
        self.mu_res = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
                                    nn.Linear(hidden, 2))
        nn.init.normal_(self.mu_res[-1].weight, std=0.01)
        nn.init.zeros_(self.mu_res[-1].bias)
        self.unc = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
                                 nn.Linear(hidden, 3))
        nn.init.zeros_(self.unc[-1].weight)
        nn.init.zeros_(self.unc[-1].bias)

    def forward(self, q_feat: torch.Tensor, a_pts: torch.Tensor, a_feat: torch.Tensor):
        """q_feat (B,N,C); a_pts (B,N,A,2); a_feat (B,N,A,C) -> mu, L."""
        h = torch.cat((q_feat[:, :, None].expand_as(a_feat), a_feat,
                       torch.cat((a_pts, torch.zeros_like(a_pts[..., :1])), -1)), -1)
        mu = a_pts + self.mu_res(h) * 0.5          # bounded residual around the anchor
        raw = self.unc(h).float()
        s1 = F.softplus(raw[..., 0] + self.log_sigma_init) + 1e-4
        s2 = raw[..., 1]
        s3 = F.softplus(raw[..., 2] + self.log_sigma_init) + 1e-4
        floor2 = self.sigma_floor ** 2
        c11 = s1.square() + floor2
        c12 = s1 * s2
        det = (s1.square() * s3.square() + floor2 * (s1.square() + s2.square() + s3.square())
               + floor2 ** 2)
        l11 = c11.sqrt()
        l21 = c12 / l11
        l22 = (det / c11).sqrt()
        L = torch.stack((l11, torch.zeros_like(l11), l21, l22), -1).reshape(*mu.shape[:-1], 2, 2)
        return mu, L

    def confidence(self, L: torch.Tensor) -> torch.Tensor:
        """Bounded trust in a Gaussian, relative to the initial prior's entropy."""
        logdet = (torch.log(L[..., 0, 0].clamp_min(1e-6))
                  + torch.log(L[..., 1, 1].clamp_min(1e-6)))
        entropy = logdet + math.log(2 * math.pi * math.e)
        h_ref = math.log(2 * math.pi * math.e * self.init_sigma ** 2)
        return torch.sigmoid(h_ref - entropy)

    @torch.no_grad()
    def set_sigma_init(self, sigma: float) -> None:
        self.log_sigma_init.fill_(math.log(max(float(sigma), 1e-4)))


# --------------------------------------------------------------------------- #
# instance identity (the only place correspondence labels enter)
# --------------------------------------------------------------------------- #
def gt_slot_index(points: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor,
                  center_thr: float = 0.05):
    """Nearest GT slot of every point, using the detector's own assignment rule.

    ``points (B,N,2)``, ``boxes (B,M,4)``, ``mask (B,M)`` ->
    ``(index (B,N) long, valid (B,N) bool)``.  A point is valid only when it lies
    inside the expanded GT box *and* within ``center_thr`` -- the same condition
    ``assign_candidates`` uses, so supervision and detection agree on identity.
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


def identity_matches(q_pts: torch.Tensor, q_boxes: torch.Tensor, q_mask: torch.Tensor,
                     k_pts: torch.Tensor, k_boxes: torch.Tensor, k_mask: torch.Tensor,
                     corr_pairs: torch.Tensor, center_thr: float = 0.05,
                     frame_valid: torch.Tensor | None = None) -> torch.Tensor:
    """(B,N,M) boolean: query candidate ``n`` and key candidate ``m`` are the same object.

    ``corr_pairs`` (B, Mq, Mk) is the *only* place instance correspondence enters,
    and it is supplied by the dataset.  Nothing here assumes the ``i``-th target on
    one side matches the ``i``-th on the other, and the two sides may hold
    different numbers of objects.  A query with no counterpart simply gets an
    all-False row, which the null supervision turns into "do not fuse".
    """
    qi, qv = gt_slot_index(q_pts, q_boxes, q_mask, center_thr)
    ki, kv = gt_slot_index(k_pts, k_boxes, k_mask, center_thr)
    n, m = q_pts.shape[1], k_pts.shape[1]
    if corr_pairs.shape[1] == 0 or corr_pairs.shape[2] == 0:
        return torch.zeros(q_pts.shape[0], n, m, dtype=torch.bool, device=q_pts.device)
    # Two hops are required: candidate -> GT slot of the query, then GT slot of the
    # key.  Indexing ``corr_pairs`` with candidate indices directly would assume the
    # candidate order and the annotation order coincide -- exactly the assumption
    # this function exists to avoid.
    row = corr_pairs.gather(1, qi[..., None].expand(-1, -1, corr_pairs.shape[-1]))
    pair = row.gather(2, ki[:, None, :].expand(-1, n, -1))
    match = (pair > 0.5) & kv[:, None, :] & qv[..., None]
    if frame_valid is not None:
        match = match & (frame_valid[:, None, None] > 0.5)
    return match


def match_target_center(matches: torch.Tensor, key_slot: torch.Tensor,
                        other_centers: torch.Tensor) -> torch.Tensor:
    """Centre (in the other modality's GT table) of the matched object, per query.

    ``matches (B,N,M)``, ``key_slot (B,M)`` (key candidate -> GT slot),
    ``other_centers (B,Mk,2)`` -> ``(B,N,2)``.  Queries without a match get the GT
    centre of slot 0; their loss weight is zero.
    """
    b, n, m = matches.shape
    first = matches.float().argmax(-1)                                  # (B,N)
    slot = torch.gather(key_slot, 1, first) if m else torch.zeros(
        b, n, dtype=torch.long, device=matches.device)
    return torch.gather(other_centers, 1, slot[..., None].expand(-1, -1, 2))


# --------------------------------------------------------------------------- #
# the correspondence module
# --------------------------------------------------------------------------- #
class BidirectionalCorrespondence(nn.Module):
    """Retrieval -> Gaussian anchors -> local verification -> rejection fusion."""

    _DIRECTIONS = ("r2t", "t2r")

    def __init__(self, dim: int = 128, retrieval_dim: int = 64, top_k: int = 2,
                 samples: int = 8, levels=("P2", "P3", "P4", "P5"), heads: int = 4,
                 ff_mult: int = 4, sigma_floor: float = 0.02, init_sigma: float = 0.05,
                 lambda_local: float = 0.35, use_null: bool = True,
                 use_gate: bool = True):
        super().__init__()
        self.dim = dim
        self.retrieval_dim = retrieval_dim
        self.top_k = int(top_k)
        self.samples = int(samples)
        self.levels = tuple(levels)
        self.n_levels = len(self.levels)
        if self.samples % self.n_levels != 0:
            raise ValueError("samples must divide evenly across the P2-P5 levels")
        self.per_level = self.samples // self.n_levels
        self.use_null = use_null
        self.use_gate = use_gate
        self.lambda_local = lambda_local
        self.sigma_floor = float(sigma_floor)
        self.init_sigma = float(init_sigma)
        self.evidence_dim = 3

        # --- 1. retrieval: one SHARED embedding space for Q and K ----------- #
        self.q_proj = nn.Linear(dim, retrieval_dim, bias=False)
        self.k_cand = nn.Linear(dim, retrieval_dim, bias=False)
        self.k_grid = nn.Linear(dim, retrieval_dim, bias=False)
        self.null_retrieval = nn.Parameter(torch.randn(1, 1, retrieval_dim) * 0.02)

        # --- 2. anchors + local verification (direction specific) ---------- #
        self.anchor = nn.ModuleDict({
            d: GaussianAnchorHead(dim, dim, sigma_floor, init_sigma) for d in self._DIRECTIONS})
        self.attn = nn.ModuleDict({
            d: SparseCrossAttention(dim, heads, ff_mult) for d in self._DIRECTIONS})
        self.k_anchor = nn.ModuleDict({d: nn.Linear(dim, dim) for d in self._DIRECTIONS})
        self.k_local = nn.ModuleDict({d: nn.Linear(dim, dim) for d in self._DIRECTIONS})
        self.level_embed = nn.Parameter(torch.zeros(self.n_levels, dim))
        # ``samples`` points on the unit ring, later split across P2-P5
        # (``per_level`` each).  P must be ``samples``, not ``per_level``: the
        # level loop slices *this* axis.
        self.register_buffer("reference", reference_points(self.samples).float())

        if use_null:
            self.null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.null_bias = nn.Parameter(torch.zeros(1))

        # --- 3. rejection fusion ------------------------------------------- #
        self.mix = nn.ModuleDict({d: nn.Linear(dim, dim) for d in self._DIRECTIONS})
        for d in self._DIRECTIONS:
            nn.init.zeros_(self.mix[d].weight)
            nn.init.zeros_(self.mix[d].bias)
        if use_gate:
            self.gate = nn.ModuleDict({
                d: nn.Sequential(nn.Linear(2 * dim + self.evidence_dim, dim // 2),
                                 nn.SiLU(inplace=True), nn.Linear(dim // 2, 1))
                for d in self._DIRECTIONS})
            for d in self._DIRECTIONS:
                nn.init.zeros_(self.gate[d][-1].weight)
                nn.init.zeros_(self.gate[d][-1].bias)
        self.null_head = nn.ModuleDict({
            d: nn.Sequential(nn.Linear(2 * dim + self.evidence_dim, dim // 2),
                             nn.SiLU(inplace=True), nn.Linear(dim // 2, 1))
            for d in self._DIRECTIONS})
        for d in self._DIRECTIONS:
            nn.init.zeros_(self.null_head[d][-1].weight)
            nn.init.zeros_(self.null_head[d][-1].bias)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def set_sigma_init(self, sigma: float) -> None:
        for head in self.anchor.values():
            head.set_sigma_init(sigma)

    # ------------------------------------------------------------------ #
    def forward(self, q_feat: torch.Tensor, q_pts: torch.Tensor,
                other_feat: torch.Tensor, other_pts: torch.Tensor,
                other_grid_feat: torch.Tensor, other_grid_pts: torch.Tensor,
                other_maps, direction: str = "r2t", other_dense=None,
                other_boxes=None, other_mask=None, q_boxes=None, q_mask=None,
                corr_pairs=None, frame_valid=None, center_thr: float = 0.05):
        """One direction of the object-level fusion.

        Only ``q_feat/q_pts/other_*`` are required; the identity arguments
        (``corr_pairs`` and the boxes/masks) are optional and, when absent, the
        module simply returns no supervision targets.
        """
        if direction not in self._DIRECTIONS:
            raise ValueError(direction)
        b, n, c = q_feat.shape
        m = other_feat.shape[1]
        g = other_grid_feat.shape[1]
        a = min(self.top_k, m + g)
        if other_grid_pts.dim() == 2:
            # the grid centres are a constant of the module, not of the batch
            other_grid_pts = other_grid_pts[None].expand(b, -1, -1)

        # ---------------- 1. coarse retrieval ------------------------------ #
        q = self.q_proj(q_feat)                                          # (B,N,R)
        keys_ret = torch.cat((self.k_cand(other_feat), self.k_grid(other_grid_feat)), 1)
        null_ret = self.null_retrieval.expand(b, 1, -1)
        logits = (torch.cat((keys_ret, null_ret), 1) @ q.transpose(1, 2)
                  / math.sqrt(self.retrieval_dim)).transpose(1, 2)       # (B,N,M+G+1)
        real_logits = logits[..., :m + g]
        anchor_idx = real_logits.topk(a, dim=-1).indices                  # (B,N,A)

        all_pts = torch.cat((other_pts, other_grid_pts), 1)               # (B,M+G,2)
        all_feat = torch.cat((other_feat, other_grid_feat), 1)            # (B,M+G,C)
        flat_idx = anchor_idx.reshape(b, n * a)
        a_pts = torch.gather(all_pts, 1, flat_idx[..., None].expand(-1, -1, 2)
                             ).reshape(b, n, a, 2)
        a_feat = torch.gather(all_feat, 1, flat_idx[..., None].expand(-1, -1, c)
                              ).reshape(b, n, a, c)

        # ---------------- 2. local probabilistic matching ------------------- #
        head = self.anchor[direction]
        mu, L = head(q_feat, a_pts, a_feat)                               # (B,N,A,2),(B,N,A,2,2)
        conf = head.confidence(L)                                         # (B,N,A)
        at_mean = gaussian_log_prob(mu, mu, L)                            # (B,N,A)

        ref = self.reference.to(mu)[None, None, None]                     # (1,1,1,P,2)
        local_pts = mu.unsqueeze(3) + (L.unsqueeze(3) @ ref.unsqueeze(-1)).squeeze(-1)

        collect = lambda name: dict(feat=[], prior=[], valid=[], conf=[], level=[])
        acc = collect(None)
        for lv, name in enumerate(self.levels):
            sl = slice(lv * self.per_level, (lv + 1) * self.per_level)
            pts_l = local_pts[:, :, :, sl]                                # (B,N,A,p,2)
            flat = pts_l.reshape(b, n * a * self.per_level, 2)
            acc["feat"].append(sample_points(other_maps[name], flat)
                               .reshape(b, n, a * self.per_level, c))
            lp = gaussian_log_prob(pts_l, mu.unsqueeze(3), L.unsqueeze(3))
            prior = (lp - at_mean.unsqueeze(-1)).reshape(b, n, a * self.per_level)
            if other_dense is not None and name in other_dense:
                cov = sample_points(other_dense[name][:, 5:6], flat).reshape(
                    b, n, a * self.per_level).sigmoid()
                prior = prior + 0.25 * cov.clamp_min(1e-4).log()
                acc["valid"].append(((pts_l >= 0) & (pts_l <= 1)).all(-1).reshape(
                    b, n, a * self.per_level) & (cov > 0.0))
            else:
                acc["valid"].append(((pts_l >= 0) & (pts_l <= 1)).all(-1).reshape(
                    b, n, a * self.per_level))
            acc["prior"].append(prior.clamp(-16.0, 0.0))
            acc["conf"].append(conf.unsqueeze(-1).expand(-1, -1, -1, self.per_level)
                               .reshape(b, n, a * self.per_level))
            acc["level"].append(torch.full((a * self.per_level,), lv, dtype=torch.long,
                                           device=q_feat.device))
        local_feat = torch.cat(acc["feat"], 2)                            # (B,N,S,C)
        local_prior = torch.cat(acc["prior"], 2) * self.lambda_local \
            * torch.cat(acc["conf"], 2)
        local_valid = torch.cat(acc["valid"], 2)
        n_local = local_feat.shape[2]
        level_index = torch.cat(acc["level"], 0)

        anchor_keys = self.k_anchor[direction](a_feat)                    # (B,N,A,C)
        local_keys = (self.k_local[direction](local_feat)
                      + self.level_embed[level_index][None, None])

        keys = [anchor_keys, local_keys]
        priors = [q_feat.new_zeros(b, n, a), local_prior]
        masks = [torch.ones(b, n, a, dtype=torch.bool, device=q_feat.device), local_valid]
        bias_parts = [q_feat.new_zeros(a), q_feat.new_zeros(n_local)]
        if self.use_null:
            keys.append(self.null_token.expand(b, n, 1, c))
            priors.append(q_feat.new_zeros(b, n, 1))
            masks.append(torch.ones(b, n, 1, dtype=torch.bool, device=q_feat.device))
            bias_parts.append(self.null_bias)
        keys = torch.cat(keys, 2)
        log_prior = torch.cat(priors, 2)
        key_mask = torch.cat(masks, 2)
        type_bias = torch.cat(bias_parts, 0)

        z, attn = self.attn[direction](q_feat, keys, keys, log_prior=log_prior,
                                       type_bias=type_bias, key_mask=key_mask,
                                       return_attn=True)
        null_mass = attn[..., -1] if self.use_null else attn.new_zeros(b, n)
        entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1) / math.log(max(keys.shape[2], 2))
        best_real = real_logits.max(-1).values
        runner_up = torch.topk(real_logits, min(2, real_logits.shape[-1]),
                               dim=-1).values[..., -1]
        evidence = torch.stack((conf.mean(-1), entropy, best_real - runner_up), -1)

        # ---------------- 3. rejection fusion ------------------------------ #
        h_gate = torch.cat((q_feat, z, evidence.to(q_feat.dtype)), -1)
        p_null = self.null_head[direction](h_gate).sigmoid().squeeze(-1)   # (B,N)
        if self.use_gate:
            gamma = self.gate[direction](h_gate).sigmoid().squeeze(-1)
        else:
            gamma = q_feat.new_ones(b, n)
        message = self.mix[direction](z)
        enhanced = q_feat + (gamma * (1.0 - p_null)).unsqueeze(-1) * message

        stats = {
            "enhanced": enhanced, "p_null": p_null, "gamma": gamma,
            "attn": attn, "n_anchor": a, "n_local_key": n_local, "n_cand_other": m,
            "n_grid_other": g, "anchor_idx": anchor_idx, "retrieval_logits": logits,
            "local_pts": local_pts, "mu": mu, "sigma": L, "conf": conf,
            "null_mass": null_mass, "local_valid": local_valid,
            "match_logit": (attn[..., :a + n_local].max(-1).values.clamp_min(1e-8).log()
                            - null_mass.clamp_min(1e-8).log()) if self.use_null else
                           attn[..., :a + n_local].max(-1).values,
        }

        # ---- optional supervision targets (only with instance labels) ------ #
        if corr_pairs is not None:
            k_slot, k_valid = gt_slot_index(other_pts, other_boxes, other_mask, center_thr)
            matches_key = identity_matches(q_pts, q_boxes, q_mask, other_pts, other_boxes,
                                           other_mask, corr_pairs, center_thr, frame_valid)
            # A grid key has no GT identity, so it can never be a labelled positive.
            # Widening the tables to (M+G) keeps the gather below in range.
            long_zeros = torch.zeros(b, g, dtype=torch.long, device=q_pts.device)
            bool_zeros = torch.zeros(b, g, dtype=torch.bool, device=q_pts.device)
            matches_all = torch.cat((matches_key, bool_zeros[:, None].expand(b, n, g)), 2)
            slot_all = torch.cat((k_slot, long_zeros), 1)
            valid_all = torch.cat((k_valid, bool_zeros), 1)
            # gather needs matching rank: flatten (B,N,M+G) to (B, N*(M+G))
            match_rows = matches_all.reshape(b, n * (m + g))
            matches_anchor = match_rows.gather(1, flat_idx).reshape(b, n, a)
            anchor_slot = slot_all.gather(1, flat_idx).reshape(b, n, a)
            anchor_has_identity = valid_all.gather(1, flat_idx).reshape(b, n, a)
            stats.update({
                "matches_key": matches_key, "matches_anchor": matches_anchor,
                "key_slot": k_slot, "key_valid": k_valid,
                "anchor_slot": anchor_slot,
                "anchor_has_identity": anchor_has_identity,
            })
        return {"feats": enhanced, "stats": stats}


__all__ = ["BidirectionalCorrespondence", "GaussianAnchorHead", "identity_matches",
           "gt_slot_index", "match_target_center", "reference_points"]
