#!/usr/bin/env python
"""Diagnose why AP is 0: heatmap peaks, candidate recall, assignment, box range."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from pgsf.config import load_config
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.engine.trainer import fit_affine_np
from pgsf.models.detector import PGSFDet
from pgsf.losses import assign_candidates, draw_gaussian

cfg = load_config("configs/pgsf_stage2.yaml")
dev = torch.device("cuda")
frames = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes)["train"]
by_scene = {}
for f in frames:
    by_scene.setdefault(f.scene, []).append(f)
scenes = sorted(by_scene)
picked = [by_scene[scenes[i % len(scenes)]][i // len(scenes)] for i in range(4)]
ds = UAVCBDataset(picked, cfg.data.rgb_size, cfg.data.tir_size, train=False)
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)

m = PGSFDet(cfg).to(dev).eval()
src = torch.tensor([f.rgb_box[:2] for f in picked if f.rgb_box is not None])
dst = torch.tensor([f.tir_box[:2] for f in picked if f.tir_box is not None])
m.set_affine(*fit_affine_np(src, dst))

batch = next(iter(loader))
batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
with torch.no_grad():
    out = m(batch)

for name in ("rgb", "tir"):
    cand = out[f"cand_{name}"]
    heat = out[f"heat_{name}"]
    hs = heat.sigmoid()
    gt_c = batch[f"centers_{name}"]
    gt_s = batch[f"sizes_{name}"]
    H, W = hs.shape[-2:]
    # GT centre on the heat grid
    gx = (gt_c[:, 0, 0] * (W - 1)).round().long()
    gy = (gt_c[:, 0, 1] * (H - 1)).round().long()
    at_gt = hs[:, 0, gy, gx]
    gmax = hs.flatten(1).max(1).values
    print(f"\n=== {name} ===")
    print(f"  heat grid {H}x{W}  GT centre pixel ({gy.tolist()},{gx.tolist()})")
    print(f"  heat at GT   : {at_gt.tolist()}  (want ~1.0)")
    print(f"  heat max     : {gmax.tolist()}")
    # is the GT a local peak?
    ys = torch.arange(H, device=dev).view(1, H, 1)
    xs = torch.arange(W, device=dev).view(1, 1, W)
    r = max(2.0, float(gt_s[:, 0, 1].min() * H))
    window = ((ys - gy.view(-1, 1, 1)) ** 2 + (xs - gx.view(-1, 1, 1)) ** 2) <= r * r
    masked = torch.where(window.unsqueeze(1), hs, torch.zeros_like(hs))
    print(f"  max in +-{r:.0f}px window: {masked.flatten(1).max(1).values.tolist()}")
    print(f"  -> GT is local peak: {bool((masked.flatten(1).max(1).values >= at_gt - 1e-6).all())}")

    # candidate recall: is any candidate within 0.05 of the GT centre?
    d = torch.norm(cand["pts"] - gt_c[:, 0:1], dim=-1)     # (B,K)
    hit = (d < 0.05).any(1)
    print(f"  candidates within 0.05 of GT: {hit.tolist()}  "
          f"min dist per sample: {d.min(1).values.tolist()}")
    print(f"  top-1 cand dist: {d.min(1).values.tolist()}")
    print(f"  top-5 cand dists: {d.topk(5, largest=False).values.tolist()}")

    # assignment on the SELECTED 32 candidates
    sel = out[f"sel_{name}"]
    pos, tgt = assign_candidates(sel["pts"], batch[f"boxes_{name}"], batch["mask"])
    print(f"  positive ratio among the 32 selected: {pos.mean().item():.4f} "
          f"(per sample: {pos.sum(1).tolist()})")

    pred = out[f"pred_{name}"]
    print(f"  pred xywh range: cx {pred[...,0].min():.3f}..{pred[...,0].max():.3f}  "
          f"w {pred[...,2].min():.3f}..{pred[...,2].max():.3f}")
    print(f"  pred score logit range: {pred[...,4].min():.3f}..{pred[...,4].max():.3f}")

# what the GT looks like on the grid
h_r = draw_gaussian(torch.zeros_like(out["cand_rgb"]["heat"]),
                    batch["centers_rgb"], batch["sizes_rgb"], batch["mask"])
print(f"\nGT gaussian target: max={h_r.max():.3f}, "
      f"cells>0.5={(h_r > 0.5).sum().item()}, total cells={h_r.shape[-2]*h_r.shape[-1]}")
print(f"GT sizes (normalised): {batch['sizes_rgb'][:, 0].tolist()}")
