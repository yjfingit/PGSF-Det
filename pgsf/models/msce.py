"""MSCE -- Multi-Scale Small-target Contrast Enhancement.

Motivation
----------
A drone in UAV-CB occupies ~30x30 px at most, often fewer. On the P2 feature map
(stride 4) it is a 3-7 cell blob sitting on top of high-energy background
(building facades, power lines, cloud edges). A plain conv cannot easily separate
"small high-contrast blob" from "small high-contrast background detail" because
it lacks an explicit notion of *contrast against the local surround*.

MSCE therefore computes four complementary cues on P2:

1. 3x3 depthwise conv  -> local texture / edges
2. 5x5 depthwise conv  -> mid-scale structure
3. 7x7 depthwise conv  -> wider context
4. multi-scale Gaussian difference + local contrast
   (a small object is a *positive* deviation from a blurred background, so
   ``|x - blur(x)|`` and ``x - blur(x)`` both carry signal; we keep both signs
   by concatenating signed difference and absolute difference)

The four cues are concatenated, compressed by 1x1 and squashed by a sigmoid into
a saliency map ``S in [0,1]``. The map is used **only as a bounded residual
gate**::

    alpha = alpha_max * tanh(a)          # a is learnable, |alpha| <= alpha_max
    F'   = F * (1 + alpha * S)

Never ``F' = S * F``: that would destroy the original features whenever ``S`` is
small, which is exactly what happens on frames where the drone is dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import ConvBNAct


class GaussianDiff(nn.Module):
    """Fixed-kernel Gaussian blurs (3/5/7) implemented as depthwise convs."""

    def __init__(self, channels: int):
        super().__init__()
        for k in (3, 5, 7):
            sigma = k / 6.0
            coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = (g / g.sum()).view(1, 1, 1, k).repeat(channels, 1, 1, 1)
            self.register_buffer(f"gk{k}", g)

    def blur(self, x: torch.Tensor, k: int) -> torch.Tensor:
        c = x.shape[1]
        w = getattr(self, f"gk{k}").to(x.dtype).expand(c, 1, 1, k)
        p = k // 2
        x = F.conv2d(x, w, padding=(0, p), groups=c)
        wt = w.transpose(2, 3).contiguous()
        x = F.conv2d(x, wt, padding=(p, 0), groups=c)
        return x


class MSCE(nn.Module):
    """Multi-scale small-object contrast enhancement (residual gated)."""

    def __init__(self, channels: int, width: int = 64, alpha_max: float = 0.5):
        super().__init__()
        mid = max(16, width)
        self.dw3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.dw5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.dw7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.blur = GaussianDiff(channels)
        # signed + absolute gaussian differences, each at 3 scales, are 2x3 -> 6
        self.contrast = ConvBNAct(6 * channels, mid, 1, 1)
        self.texture = ConvBNAct(3 * channels, mid, 1, 1)
        self.fuse = nn.Sequential(
            ConvBNAct(2 * mid, mid, 3, 1),
            nn.Conv2d(mid, 1, 1, 1, bias=True),
        )
        # start close to identity: a = atanh(0.1/alpha_max)
        import math
        a0 = math.atanh(min(0.5, 0.1 / max(alpha_max, 1e-6)))
        self.alpha_raw = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        self.alpha_max = float(alpha_max)
        # bias so that S starts around 0.5, not 0.09
        nn.init.constant_(self.fuse[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, H, W) -> (enhanced x, saliency map (B,1,H,W))."""
        t = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        t = self.texture(t)

        b3, b5, b7 = self.blur.blur(x, 3), self.blur.blur(x, 5), self.blur.blur(x, 7)
        c = torch.cat([x - b3, (x - b3).abs(), x - b5, (x - b5).abs(),
                       x - b7, (x - b7).abs()], dim=1)
        c = self.contrast(c)

        s = torch.sigmoid(self.fuse(torch.cat([t, c], dim=1)))
        alpha = self.alpha_max * torch.tanh(self.alpha_raw)
        out = x * (1.0 + alpha * s)
        return out, s


class LiteMSCE(nn.Module):
    """Compute four saliency cues in a narrow branch, gate original features."""
    def __init__(self, channels, width=24, alpha_max=.5):
        super().__init__()
        self.reduce = ConvBNAct(channels, width, 1)
        self.texture = nn.ModuleList(nn.Conv2d(width, width, k, padding=k//2,
                                               groups=width) for k in (3,5,7))
        self.blur = GaussianDiff(width)
        self.fuse = nn.Sequential(ConvBNAct(9*width, width, 1),
                                  nn.Conv2d(width, 1, 1))
        self.alpha_raw = nn.Parameter(torch.tensor(.2))
        self.alpha_max = alpha_max

    def forward(self, x):
        h = self.reduce(x)
        cues = [layer(h) for layer in self.texture]
        for k in (3,5,7):
            d = h - self.blur.blur(h,k)
            cues.extend((d,d.abs()))
        saliency = self.fuse(torch.cat(cues,1)).sigmoid()
        return x * (1 + self.alpha_max*self.alpha_raw.tanh()*saliency), saliency
