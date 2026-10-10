#!/usr/bin/env python
"""Does the neck actually carry drone information at P3 resolution?

If a linear probe cannot recover the target centre from the P2/P3 features, no
amount of heatmap training will help -- the information is already lost.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import DataLoader
from pgsf.config import load_config
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.models.detector import PGSFDet

cfg = load_config("configs/pgsf_stage2.yaml")
cfg.model.width_mult = 0.5
cfg.model.backbone_depths = (1, 1, 1, 1)
cfg.model.neck_dim = 128
cfg.model.neck_repeats = 1
dev = torch.device("cuda")

frames = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes)["train"]
rng = np.random.RandomState(0)
idx = rng.permutation(len(frames))[:16]
picked = [frames[i] for i in idx]
ds = UAVCBDataset(picked, cfg.data.rgb_size, cfg.data.tir_size, train=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)

m = PGSFDet(cfg).to(dev).eval()
feats, labels = [], []
with torch.no_grad():
    for batch in loader:
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        f = m.rgb_branch.features(batch["rgb"])
        for lvl in ("P2", "P3"):
            x = f[lvl]                                  # (B,C,H,W)
            B, C, H, W = x.shape
            gt = batch["centers_rgb"]                    # (B,M,2)
            gx = (gt[:, 0, 0] * (W - 1)).round().long().clamp(0, W - 1)
            gy = (gt[:, 0, 1] * (H - 1)).round().long().clamp(0, H - 1)
            # feature at the GT centre vs. random background
            b = torch.arange(B, device=dev)
            at = x[b, :, gy, gx]                        # (B,C)
            rb = torch.randint(0, H, (B,), device=dev)
            rc = torch.randint(0, W, (B,), device=dev)
            bg = x[b, :, rb, rc]
            feats.append(at.cpu().numpy()); labels.append(bg.cpu().numpy())
            print(f"  {lvl}: GT-centre ||f||={at.norm(dim=1).mean():.3f}  "
                  f"random-bg ||f||={bg.norm(dim=1).mean():.3f}  "
                  f"ratio={float(at.norm(dim=1).mean() / bg.norm(dim=1).mean()):.3f}")

A = np.concatenate(feats)     # (N, C)
B_ = np.concatenate(labels)
print(f"\ncollected {A.shape[0]} GT-centre vectors and {B_.shape[0]} background ones, dim={A.shape[1]}")
# simple separability: can a linear classifier separate them (train/test split)?
mu_a, mu_b = A.mean(0), B_.mean(0)
d = mu_a - mu_b
proj_a, proj_b = A @ d, B_ @ d
print(f"mean projection on (mu_gt - mu_bg): GT {proj_a.mean():.4f}+-{proj_a.std():.4f}  "
      f"BG {proj_b.mean():.4f}+-{proj_b.std():.4f}")
sep = (proj_a.mean() - proj_b.mean()) / np.sqrt(0.5 * (proj_a.var() + proj_b.var()))
print(f"effect size (Cohen's d) = {sep:.3f}")
print("  (>0.5 = clearly separable, ~0 = drone signal absent)")
print(f"raw cosine(GT, BG) = {float((A @ B_.T).mean() / (np.linalg.norm(A,axis=1)*np.linalg.norm(B_,axis=1)).mean()):.3f}")
