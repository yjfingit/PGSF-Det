"""Lightweight CSP-style dual-branch CNN backbone (YOLOv8 C2f inspired).

Design notes
------------
* The two branches (RGB / TIR) are **structurally identical but weight
  independent** -- each modality has a very different appearance statistics
  (visible texture vs. thermal contrast), so sharing weights hurts.
* The stem is a plain 3x3 conv (not depthwise) to preserve low-level detail.
* After the stem, blocks are ``C2f`` with depthwise-separable bottlenecks to keep
  the parameter count low, plus a multi-scale depthwise (3/5/7 kernel) context
  block before P5 so the deepest stage sees multiple receptive fields.
* P2 keeps stride 4 with 64 channels: for UAV-CB the drone occupies only a few
  dozen pixels, and this is the resolution where the target is still separable
  from background clutter.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# primitive blocks
# --------------------------------------------------------------------------- #
def autopad_k(k: int) -> int:
    return k // 2


class ConvBNAct(nn.Sequential):
    """Conv -> BN -> SiLU with 'same' padding."""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, autopad_k(k), groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class DWConvBNAct(nn.Sequential):
    """Depthwise separable conv block."""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__(
            nn.Conv2d(cin, cin, k, s, autopad_k(k), groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class Bottleneck(nn.Module):
    """1x1 -> kxk(dw) -> 1x1 residual block, expansion 0.5."""

    def __init__(self, cin: int, cout: int, k: int = 3, e: float = 0.5, use_dw: bool = True):
        super().__init__()
        mid = max(8, int(cin * e))
        if use_dw:
            self.cv1 = ConvBNAct(cin, mid, 1, 1)
            self.cv2 = DWConvBNAct(mid, cout, k, 1)
        else:
            self.cv1 = ConvBNAct(cin, mid, k, 1)
            self.cv2 = ConvBNAct(mid, cout, k, 1)
        self.add = (cin == cout) and (k == 3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        if self.add:
            y = y + x
        return self.act(y)


class C2f(nn.Module):
    """CSP bottleneck with 2 conv splits and ``n`` bottlenecks (YOLOv8 C2f)."""

    def __init__(self, cin: int, cout: int, n: int = 1, k: int = 3, e: float = 0.5,
                 use_dw: bool = True):
        super().__init__()
        self.c = max(8, int(cout * e))
        self.cv1 = ConvBNAct(cin, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, cout, 1, 1)
        self.m = nn.ModuleList(
            [Bottleneck(self.c, self.c, k, 1.0, use_dw) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: List[torch.Tensor] = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class MultiScaleDW(nn.Module):
    """Parallel 3/5/7 depthwise convs -> concat -> 1x1.

    Cheap way to give a single stage several receptive fields, which matters for
    distinguishing a 10px drone from a 30px background blob.
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        # depthwise branches must keep in==out channels, so each branch emits
        # ``cin`` channels and the 1x1 fuse compresses the 3*cin concatenation.
        def branch(k: int):
            return nn.Sequential(
                nn.Conv2d(cin, cin, k, 1, k // 2, groups=cin, bias=False),
                nn.BatchNorm2d(cin), nn.SiLU(inplace=True))

        self.b3 = branch(3)
        self.b5 = branch(5)
        self.b7 = branch(7)
        self.fuse = ConvBNAct(3 * cin, cout, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1))


class SPPFLite(nn.Module):
    """Space-to-depth free, cheap spatial pyramid (max 5 + avg 5)."""

    def __init__(self, cin: int, cout: int, k: int = 5):
        super().__init__()
        mid = max(8, cin // 2)
        self.cv1 = ConvBNAct(cin, mid, 1, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)
        self.avg = nn.AvgPool2d(k, 1, k // 2)
        self.cv2 = ConvBNAct(4 * mid, cout, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([y, y1, y2, self.avg(y)], dim=1))


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
class LightCSPBackbone(nn.Module):
    """Returns P2/P3/P4/P5 at strides 4/8/16/32.

    Channel plan for width_mult=1.0: (64, 128, 256, 384).
    """

    def __init__(self, in_ch: int = 3, width_mult: float = 1.0,
                 depths=(3, 6, 9, 3), use_dw: bool = True):
        super().__init__()
        w = width_mult

        def ch(c: int) -> int:
            return max(8, int(round(c * w / 8)) * 8)

        c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(384)
        self.out_channels = (c2, c3, c4, c5)

        # stride 2 -> 4:  plain (non-depthwise) stem, two convs
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, max(16, c2 // 2), 3, 2),
            ConvBNAct(max(16, c2 // 2), c2, 3, 2),
        )
        # stride 8
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), C2f(c3, c3, n=depths[0],
                                                                 use_dw=use_dw))
        # stride 16
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), C2f(c4, c4, n=depths[1],
                                                                 use_dw=use_dw))
        # stride 32 + multi-scale context + SPP
        self.stage5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), C2f(c5, c5, n=depths[2],
                                                                 use_dw=use_dw))
        self.msdw = MultiScaleDW(c5, c5)
        self.spp = SPPFLite(c5, c5)
        # extra depth applied only at P5 (cheapest place to add capacity)
        self.stage5b = C2f(c5, c5, n=depths[3], use_dw=use_dw)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        p2 = self.stem(x)          # /4
        p3 = self.stage3(p2)       # /8
        p4 = self.stage4(p3)       # /16
        p5 = self.stage5(p4)       # /32
        p5 = self.spp(self.msdw(p5))
        p5 = self.stage5b(p5)
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}


class DualBackbone(nn.Module):
    """Two weight-independent :class:`LightCSPBackbone` instances."""

    def __init__(self, rgb_in_ch: int = 3, tir_in_ch: int = 1, width_mult: float = 1.0,
                 depths=(3, 6, 9, 3), use_dw: bool = True):
        super().__init__()
        self.rgb = LightCSPBackbone(rgb_in_ch, width_mult, depths, use_dw)
        self.tir = LightCSPBackbone(tir_in_ch, width_mult, depths, use_dw)

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor):
        return self.rgb(rgb), self.tir(tir)


class YOLO11Backbone(nn.Module):
    """Unmodified YOLO11s backbone, with fully loaded COCO weights per modality.

    The head/neck of the source detector are discarded. TIR stem uses the sum of
    RGB input-channel weights, equivalent to the source stem on replicated gray.
    Requires ultralytics only when this backbone is selected.
    """
    def __init__(self, in_ch=3, weights=""):
        super().__init__()
        from ultralytics import YOLO
        from pathlib import Path
        if weights and not Path(weights).is_file():
            raise FileNotFoundError(f"pretrained backbone not found: {weights}")
        source = YOLO(weights or "yolo11s.yaml").model
        self.layers = nn.ModuleList(list(source.model.children())[:11])
        self.out_channels = (128, 256, 256, 512)
        self.pretrained = bool(weights)
        if in_ch == 1:
            old = self.layers[0].conv
            stem = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride,
                             old.padding, bias=old.bias is not None)
            with torch.no_grad():
                stem.weight.copy_(old.weight.sum(1, keepdim=True))
                if old.bias is not None:
                    stem.bias.copy_(old.bias)
            self.layers[0].conv = stem
        elif in_ch != 3:
            raise ValueError("YOLO11 backbone accepts RGB or single-channel TIR")

    def forward(self, x):
        out = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in (2, 4, 6, 10):
                out[{2: "P2", 4: "P3", 6: "P4", 10: "P5"}[i]] = x
        return out
