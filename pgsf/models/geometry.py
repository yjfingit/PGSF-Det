"""Probabilistic geometric prior  P(p_T | p_R) = N(mu, Sigma).

Why a *probability* and not a warp
----------------------------------
UAV-CB is explicitly **non-registered**: the visible and thermal cameras have a
different optical centre, so the target position obeys a global affine relation
``p_T ~ A p_R + b`` **plus** a residual that varies per frame (platform
vibration, parallax).  Hard-registering with a single affine would therefore
mis-align a noticeable fraction of frames, and forcing a match there would
inject wrong evidence into the fusion.

So we model the correspondence as a 2-D Gaussian:

    mu  = A p_R + b + delta(p_R)        (delta = small learned residual MLP)
    Sigma = L L^T + eps I               (L lower-triangular, predicted per point)

The covariance is the important part: a **large** Sigma means "the geometry
prior is untrustworthy here", and the fusion gate (see :mod:`pgsf.models.fusion`)
down-weights the cross-modal evidence instead of corrupting the feature.

Everything here works in **normalised [0,1] coordinates**, so the affine is
resolution independent and the same weights transfer between the 1280x768 RGB
frame and the 640x512 TIR frame.

``fit_affine`` is called **only on the training split** -- test GT never
participates in fitting.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


def fit_affine(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares affine  dst ~= A [src,1]  in homogeneous coordinates.

    src: (N,2) normalised source points, dst: (N,2) normalised targets.
    Returns ``(A (2,3), b (2,))`` where ``b`` is just the third column of ``A``,
    kept separate so callers can use either the plain 2x2 part or the full
    homogeneous form.

    Uses ``torch.linalg.lstsq`` on float64 for numerical stability, with a
    pseudo-inverse fallback when the system is rank deficient (e.g. a training
    split where the target never moves).
    """
    n = src.shape[0]
    x = torch.cat([src, torch.ones(n, 1, dtype=src.dtype, device=src.device)], dim=1)  # (N,3)
    xt = x.to(torch.float64)
    yt = dst.to(torch.float64)
    try:
        sol = torch.linalg.lstsq(xt, yt).solution            # (3,2)
    except Exception:                                        # pragma: no cover
        sol = torch.linalg.pinv(xt) @ yt
    A = sol[:2].T.to(src.dtype).contiguous()                 # (2,2)
    b = sol[2].to(src.dtype).contiguous()                    # (2,)
    A3 = torch.cat([A, b.view(2, 1)], dim=1).contiguous()    # (2,3)
    return A3, b


def invert_affine(A: torch.Tensor) -> torch.Tensor:
    """Reverse a normalized homogeneous 2x3 affine (supports batched matrices).

    Pseudoinverse also gives a finite fallback for rank-deficient training fits.
    """
    linear = torch.linalg.pinv(A[..., :2].float())
    offset = -(linear @ A[..., 2:].float())
    return torch.cat([linear, offset], dim=-1).to(A.dtype)


def augment_affine(A: torch.Tensor, transform: Optional[torch.Tensor]) -> torch.Tensor:
    """A_aug = T A T^-1 for paired coordinate augmentation, e.g. a flip."""
    if transform is None:
        return A
    T = transform.to(device=A.device, dtype=torch.float32)
    bottom = A.new_tensor([0., 0., 1.]).expand(*A.shape[:-2], 1, 3)
    homogeneous = torch.cat([A, bottom], dim=-2).float()
    return (T @ homogeneous @ torch.linalg.inv(T))[..., :2, :].to(A.dtype)


class ProbGeoPrior(nn.Module):
    """Predicts mu / Sigma for the cross-modal correspondence of each point."""

    def __init__(self, in_ch: int = 128, hidden: int = 128,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        # global affine, stored as buffers so checkpoints carry the fitted geometry
        # global affine, stored as buffers so checkpoints carry the fitted geometry.
        # ``A`` is the full homogeneous 2x3 matrix (its 3rd column *is* the
        # translation), so the forward pass uses ``A`` alone and never adds ``b``.
        self.register_buffer("A", torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        self.register_buffer("b", torch.zeros(2))

        if use_residual:
            self.residual = nn.Sequential(
                nn.Linear(in_ch, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, 2),
            )
            # start at exactly zero so the initial behaviour is pure global affine
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        # per-point uncertainty: 3 raw values -> lower-triangular L (2x2)
        self.unc = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.unc[-1].weight)
        nn.init.zeros_(self.unc[-1].bias)
        self.eps = 1e-4
        # A prior sigma of ~0.05 in normalised units (~5% of the frame), matching
        # the measured cross-modal centre offset spread on UAV-CB.
        self.log_sigma_init = float(torch.log(torch.tensor(0.05)))
        # **Floor on sigma.** The Gaussian NLL is minimised at sigma -> 0 whenever mu
        # is on target (the density grows without bound), so an unbounded sigma makes
        # the objective unbounded from below: during training the term drove itself to
        # -11 and dominated every other loss. Flooring sigma keeps L_geo a bounded
        # *correction*. The floor is half the prior, so the model may legitimately
        # become twice as confident as the global affine but no more.
        self.sigma_floor = 0.02

    # ------------------------------------------------------------------ #
    def set_affine(self, A: torch.Tensor, b: torch.Tensor) -> None:
        self.A.copy_(A.to(self.A.device, self.A.dtype))
        self.b.copy_(b.to(self.b.device, self.b.dtype))

    @torch.no_grad()
    def fit_from_pairs(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        A, b = fit_affine(src.detach().float().cpu(), dst.detach().float().cpu())
        self.set_affine(A, b)

    # ------------------------------------------------------------------ #
    def forward(self, pts: torch.Tensor, feat: torch.Tensor,
                affine: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pts:  (B,K,2) normalised source points
        feat: (B,K,C) source candidate features

        Returns
            mu    (B,K,2)   expected target position
            L     (B,K,2,2) Cholesky factor of Sigma
            extra (B,K,3)   [log det Sigma, sigma_max, sigma_min] statistics
        """
        # global affine part (A already carries the translation column)
        hom = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)  # (B,K,3)
        A = self.A if affine is None else affine
        mu = hom @ A.to(device=pts.device, dtype=pts.dtype).transpose(-1, -2)                            # (B,K,2)

        if self.use_residual:
            mu = mu + self.residual(feat)

        # lower-triangular from 3 raw values: L = [[softplus(s1),0],[s2,softplus(s3)]]
        # ``sigma_floor`` keeps Sigma bounded away from singular (see __init__).
        raw = self.unc(feat).float()
        s1 = F.softplus(raw[..., 0] + self.log_sigma_init) + self.eps
        s2 = raw[..., 1]
        s3 = F.softplus(raw[..., 2] + self.log_sigma_init) + self.eps
        # A diagonal floor on a Cholesky factor does not bound covariance
        # eigenvalues when the off-diagonal is large. Use BB^T + floor^2 I,
        # with an analytic 2x2 Cholesky in FP32 (no inference linalg call).
        floor2=self.sigma_floor**2
        c11=s1.square()+floor2;c12=s1*s2
        determinant=s1.square()*s3.square()+floor2*(s1.square()+s2.square()+s3.square())+floor2**2
        l11=c11.sqrt();l21=c12/l11;l22=(determinant/c11).sqrt()
        L=torch.stack((l11,torch.zeros_like(l11),l21,l22),-1).reshape(*pts.shape[:2],2,2)

        sigma_max = l11 + l21.abs()
        sigma_min = l22
        logdet = torch.log(l11*l22)
        extra = torch.stack([logdet, sigma_max, sigma_min], dim=-1)
        return mu, L, extra

    # ------------------------------------------------------------------ #
    @staticmethod
    def log_prob(p: torch.Tensor, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """log N(p; mu, L L^T) for p (B,K,2), mu (B,K,2), L (B,K,2,2).

        The triangular solve is done in float32: under AMP ``p`` arrives as fp16
        while ``L`` may be fp32, and ``linalg.solve_triangular`` requires matching
        dtypes. fp16 would also underflow for well-localised Gaussians.
        """
        d = (p.float() - mu.float()).unsqueeze(-1)                      # (B,K,2,1)
        Lf = L.float()
        sol = torch.linalg.solve_triangular(Lf, d, upper=False)          # (B,K,2,1)
        m2 = (sol ** 2).sum(dim=(-1, -2))                                # (B,K)
        logdet = torch.log(Lf[..., 0, 0] * Lf[..., 1, 1])                # (B,K)
        return (-0.5 * m2 - logdet - _LOG_2PI).to(p.dtype)

    # ------------------------------------------------------------------ #
    @staticmethod
    def entropy_from_extra(extra: torch.Tensor) -> torch.Tensor:
        """Differential entropy (nats) of the predicted 2-D Gaussian."""
        logdet = extra[..., 0]
        return logdet + math.log(2 * math.pi * math.e)

    def confidence(self, extra: torch.Tensor) -> torch.Tensor:
        """Bounded trust relative to the initial prior's entropy; wider is lower."""
        h_ref = math.log(2 * math.pi * math.e * 0.05 ** 2)
        return torch.sigmoid(h_ref - self.entropy_from_extra(extra))


class BoxProb(nn.Module):
    """Gaussian on the candidate box, used as the late-stage prior  P(b_j^T | b_i^R).

    We model the *relative* box correspondence (log-size ratio + centre offset)
    because absolute sizes live in different pixel scales for the two modalities.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, 2))  # log sigma for (dx, dy)
        nn.init.zeros_(self.log_std)

    def forward(self, boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Pairwise log P(b_j | b_i).

        boxes_a: (B,Na,4) boxes of modality A, boxes_b: (B,Nb,4) of modality B,
        in normalised (cx,cy,w,h).
        Returns (B,Na,Nb) log-probability.
        """
        ca, size_a = boxes_a[..., :2], boxes_a[..., 2:].clamp_min(1e-4)
        cb, size_b = boxes_b[..., :2], boxes_b[..., 2:].clamp_min(1e-4)
        # predict the correspondence in B from A using the global affine
        hom = torch.cat([ca, torch.ones_like(ca[..., :1])], dim=-1)
        pred_c = hom @ self.A.to(ca.dtype).T + self.b.to(ca.dtype)      # (B,Na,2)
        pred_sz = size_a                                              # sizes are resolution-normalised
        dc = cb.unsqueeze(1) - pred_c.unsqueeze(2)                     # (B,Na,Nb,2)
        ds = (size_b.unsqueeze(1) - pred_sz.unsqueeze(2)) * 4.0         # scaled log-size residual
        r = torch.cat([dc, ds], dim=-1)                                # (B,Na,Nb,4)
        var = torch.exp(self.log_std).to(ca.dtype).clamp_min(1e-4)
        # centre and size residuals have different units -> separate variance
        var = torch.cat([var, var * 4.0], dim=-1)                       # (4,)
        return -0.5 * (r ** 2 / var).sum(-1)

    def set_affine(self, A: torch.Tensor, b: torch.Tensor) -> None:
        if not hasattr(self, "A"):
            self.register_buffer("A", A.clone())
            self.register_buffer("b", b.clone())
        else:
            self.A.copy_(A.to(self.A.device, self.A.dtype))
            self.b.copy_(b.to(self.b.device, self.b.dtype))

    def fit_from_pairs(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        A, b = fit_affine(src.detach().float().cpu(), dst.detach().float().cpu())
        self.set_affine(A, b)


def add_independent(logits: torch.Tensor, prior_logp: torch.Tensor,
                    lam: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``Softmax``-compatible additive prior:  logits + lam * log P.

    ``mask`` (same shape as prior_logp, bool) restricts the prior to a subset of
    keys -- e.g. only the *local* spatial candidates, leaving global and
    null candidates to their own type bias.  Masked-out entries get a large
    negative prior so they cannot accidentally win through the prior term.
    """
    if mask is None:
        return logits + lam * prior_logp
    neg = torch.finfo(logits.dtype).min / 4
    return logits + lam * torch.where(mask, prior_logp, torch.full_like(prior_logp, neg))
