#!/usr/bin/env python
"""Isolate the heatmap loss: which variant actually forms a peak at the GT?

Prints, for each variant, the analytic gradient of L_heat w.r.t. the logit at the
GT centre after normalising by the number of background cells -- that ratio is what
determines whether the peak can grow at all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, math
from pgsf.losses import gaussian_focal_loss, draw_gaussian

H, W = 96, 160          # RGB P3 grid
gt = torch.zeros(1, 1, H, W)
gt[0, 0, 48, 80] = 1.0
sizes = torch.tensor([[[0.067, 0.05]]])
centers = torch.tensor([[[80 / (W - 1), 48 / (H - 1)]]])
mask = torch.ones(1, 1)
target = draw_gaussian(torch.zeros(1, 1, H, W), centers, sizes, mask)
n_pos_cells = int((target >= 0.999).sum())
print(f"grid {H}x{W} = {H*W} cells; cells with target>=0.999: {n_pos_cells}")
print(f"target>0.5: {int((target>0.5).sum())}   target>0.1: {int((target>0.1).sum())}")

for p in (0.018, 0.05, 0.2, 0.5, 0.8):
    z = torch.full((1, 1, H, W), math.log(p / (1 - p)), requires_grad=True)
    loss = gaussian_focal_loss(z, target)
    loss.backward()
    g_c = float(z.grad[0, 0, 48, 80])
    g_bg_mean = float(z.grad[0, 0].mean())
    print(f"  p={p:5.3f}  loss={float(loss):8.4f}  "
          f"grad@GT={g_c:+.3e}  grad@bg(mean)={g_bg_mean:+.3e}  "
          f"-> {'peak grows' if g_c < 0 else 'peak pushed DOWN'}")

print("\n--- variant: normalise neg by a fraction ---")
for neg_scale in (1.0, 0.5, 0.1, 0.01):
    p = 0.018
    pp = torch.tensor(p)
    g_pos = (-(1 - pp) ** 2 / pp + 2 * math.log(p) * (1 - pp)) * pp * (1 - pp)
    g_neg = (-(pp ** 2 + (1 - pp) * 2 * pp)) * pp * (1 - pp)
    n_neg = float((target < 0.999).sum())
    gt_total = float(g_pos) + neg_scale * float(g_neg) * n_neg
    print(f"  neg_scale={neg_scale:<5}  GT gradient = {gt_total:+.5f} "
          f"({'GROWS' if gt_total < 0 else 'SHRINKS'})")
