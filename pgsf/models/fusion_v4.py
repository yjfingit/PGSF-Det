"""PGSF-Det v4 TD-Fusion: asymmetric fusion for classification and localization.

The core idea of v4, in one equation pair.  With ``z`` the object token of the
current modality and ``z'`` the message retrieved from the other modality:

* **Consistency token** (512 -> 256 -> 128)::

      c = MLP([ z, z', z * z', |z - z'| ])

* **Reliability gate** (sigmoid of a small MLP over
  ``[z, z', |z - z'|, q_self, q_other, top1-top2 margin]``)::

      r = sigmoid(MLP(...))          in (0, 1)

* **Classification** consumes the cross-modal semantics directly::

      f_cls = z + r * W_c(c)

* **Localization** never receives the other modality's spatial features.  The
  cross-modal consistency only produces a *channel modulation* of the modality's
  own RoI descriptor::

      gamma  = tanh(W_r c)
      F_reg  = S * (1 + alpha * r * gamma)          alpha: learnable scalar, init 0.1

  so the spatial support of ``F_reg`` is ``S`` (this modality's own RoI), and the
  final box boundary is decided by this modality alone.  ``alpha`` is learned, so
  the network can decide how much (or whether at all) to let consistency touch
  localization.

``W_c`` and ``W_r`` are zero-initialised, which makes the whole module the exact
identity at step 0: the model starts as two independent detectors and can only
gain from the cross-modal path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricFusion(nn.Module):
    """One direction of TD-Fusion (RGB->TIR or TIR->RGB)."""

    def __init__(self, dim: int = 128, alpha_init: float = 0.1,
                 gate_hidden: int = 128, cons_hidden: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.consistency = nn.Sequential(
            nn.Linear(4 * dim, cons_hidden), nn.SiLU(inplace=True),
            nn.Linear(cons_hidden, dim))
        nn.init.normal_(self.consistency[-1].weight, std=0.01)
        nn.init.zeros_(self.consistency[-1].bias)

        # [z, z', |z - z'|] (3C) + [q_self, q_other, margin] (3)
        self.gate = nn.Sequential(nn.Linear(3 * dim + 3, gate_hidden), nn.SiLU(inplace=True),
                                  nn.Linear(gate_hidden, 1))
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        nn.init.zeros_(self.gate[-1].bias)

        self.w_cls = nn.Linear(dim, dim)
        self.w_reg = nn.Linear(dim, dim)
        nn.init.zeros_(self.w_cls.weight)
        nn.init.zeros_(self.w_cls.bias)
        nn.init.zeros_(self.w_reg.weight)
        nn.init.zeros_(self.w_reg.bias)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor, z_other: torch.Tensor,
                quality: torch.Tensor | None = None,
                quality_other: torch.Tensor | None = None,
                margin: torch.Tensor | None = None,
                other_available: torch.Tensor | None = None):
        """``z, z_other [B,K,C]`` -> fused classification token, reliability and modulation."""
        diff = (z - z_other).abs()
        consistency = self.consistency(torch.cat((z, z_other, z * z_other, diff), -1))
        if quality is None:
            quality = z.new_ones(*z.shape[:2])
        if quality_other is None:
            quality_other = z.new_ones(*z.shape[:2])
        if margin is None:
            margin = z.new_zeros(*z.shape[:2])
        evidence = torch.stack((quality, quality_other, margin), -1)
        r = self.gate(torch.cat((z, z_other, diff, evidence), -1)).sigmoid()   # (B,K,1)
        if other_available is not None:
            # a dropped sensor must behave exactly like an independent detector
            r = r * other_available.reshape(-1, 1, 1)
        f_cls = z + r * self.w_cls(consistency)
        gamma = torch.tanh(self.w_reg(consistency))                            # (B,K,C)
        return {"f_cls": f_cls, "consistency": consistency, "gamma": gamma,
                "r": r.squeeze(-1), "alpha": self.alpha}

    # ------------------------------------------------------------------ #
    def modulate(self, roi: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor):
        """``F_reg = S * (1 + alpha * r * gamma)`` for ``roi [B,K,C,w,h]``.

        ``r`` may arrive as ``[B,K]`` or ``[B,K,1]``; it is broadcast over channels
        only, never over the spatial axes -- that is what keeps the localization
        support identical to this modality's own RoI.
        """
        if r.dim() == 2:
            r = r.view(r.shape[0], r.shape[1], 1)
        gain = (1.0 + self.alpha * r * gamma).unsqueeze(-1).unsqueeze(-1)
        return roi * gain


__all__ = ["AsymmetricFusion"]
