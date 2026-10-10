"""Lightweight PAN-FPN neck.

Every output level is projected to ``neck_out`` channels (default 128) so that
downstream candidate embedding / transformer dimension is uniform.  P2 is never
dropped or heavily compressed -- it carries the small-drone detail.

Two independent necks (RGB / TIR), weight independent.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgsf.models.backbone import ConvBNAct, DWConvBNAct


class PANFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 384), neck_dim: int = 384,
                 out_ch: int = 128, repeats: int = 3):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        self.out_ch = out_ch

        # top-down (FPN) lateral + refine
        self.lat3 = ConvBNAct(c3, neck_dim, 1, 1)
        self.lat4 = ConvBNAct(c4, neck_dim, 1, 1)
        self.lat5 = ConvBNAct(c5, neck_dim, 1, 1)
        self.td3 = nn.Sequential(*[DWConvBNAct(neck_dim, neck_dim, 3, 1)
                                   for _ in range(repeats)])
        self.td4 = nn.Sequential(*[DWConvBNAct(neck_dim, neck_dim, 3, 1)
                                   for _ in range(repeats)])
        self.td5 = ConvBNAct(neck_dim, neck_dim, 3, 1)

        # bottom-up (PAN)
        self.bu4 = ConvBNAct(neck_dim, neck_dim, 3, 1)
        self.bu3 = nn.Sequential(*[DWConvBNAct(neck_dim, neck_dim, 3, 1)
                                   for _ in range(repeats)])
        self.bu2 = ConvBNAct(neck_dim, neck_dim, 3, 1)
        # P2 lateral: lifts the 64-channel stride-4 feature into neck_dim so it can
        # take part in the top-down fusion. P2 itself is never discarded, only widened.
        self.lat2 = ConvBNAct(c2, neck_dim, 1, 1)

        # keep P2: only a light 1x1 (no stride change, no spatial pooling)
        self.out2 = nn.Conv2d(neck_dim, out_ch, 1, bias=False)
        self.out3 = nn.Conv2d(neck_dim, out_ch, 1, bias=False)
        self.out4 = nn.Conv2d(neck_dim, out_ch, 1, bias=False)
        self.out5 = nn.Conv2d(neck_dim, out_ch, 1, bias=False)
        for m in (self.out2, self.out3, self.out4, self.out5):
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p2, p3, p4, p5 = feats["P2"], feats["P3"], feats["P4"], feats["P5"]

        l5 = self.td5(self.lat5(p5))
        l4 = self.lat4(p4) + F.interpolate(l5, size=p4.shape[-2:], mode="nearest")
        l4 = self.td4(l4)
        l3 = self.lat3(p3) + F.interpolate(l4, size=p3.shape[-2:], mode="nearest")
        l3 = self.td3(l3)
        l2 = self.lat2(p2) + F.interpolate(l3, size=p2.shape[-2:], mode="nearest")
        l2 = self.bu2(l2)

        u4 = self.bu4(l4 + F.interpolate(l5, size=l4.shape[-2:], mode="nearest"))
        u3 = self.bu3(l3 + F.interpolate(u4, size=l3.shape[-2:], mode="nearest"))

        return {
            "P2": self.out2(l2),
            "P3": self.out3(u3),
            "P4": self.out4(u4),
            "P5": self.out5(l5),
        }


class DualNeck(nn.Module):
    def __init__(self, in_channels, neck_dim: int, out_ch: int, repeats: int):
        super().__init__()
        self.rgb = PANFPN(in_channels, neck_dim, out_ch, repeats)
        self.tir = PANFPN(in_channels, neck_dim, out_ch, repeats)

    def forward(self, rgb_feats, tir_feats):
        return self.rgb(rgb_feats), self.tir(tir_feats)
