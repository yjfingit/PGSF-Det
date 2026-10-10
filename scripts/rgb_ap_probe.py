#!/usr/bin/env python
"""Why is RGB AP50 pinned at 0 while TIR climbs?

The pixel-space GT is computed with the *original* image size (1920x1080), while
the prediction is produced in the *network input* frame (1280x768). Those differ
by a constant factor, so a plain IoU is still a valid IoU -- the aspect ratio is
what matters. This script checks the two hypotheses that could pin RGB at 0:

  H1  aspect-ratio mismatch  -> RGB IoU is systematically lower than TIR's
  H2  RGB has a real problem -> RGB IoU is fine but the score is below threshold

It reports, on real frames, the GT aspect ratio, the achieved IoU, and the score
of the best-overlapping prediction for both modalities.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from pgsf.config import load_config
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.engine.trainer import fit_affine_np
from pgsf.models.detector import PGSFDet, box_iou
from pgsf.models.head import decode_pred

cfg = load_config("configs/pgsf_stage2.yaml")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes)["train"]
ds = UAVCBDataset(frames[:16], cfg.data.rgb_size, cfg.data.tir_size, train=False)

print(f"network input : RGB {tuple(cfg.data.rgb_size)}  TIR {tuple(cfg.data.tir_size)}")
print(f"original size : RGB {frames[0].rgb_size}  TIR {frames[0].tir_size}")
print()
ar_in = cfg.data.rgb_size[1] / cfg.data.rgb_size[0]
ar_or = frames[0].rgb_size[0] / frames[0].rgb_size[1]
print(f"RGB aspect ratio: input {ar_in:.4f} vs original {ar_or:.4f} "
      f"-> distortion {abs(ar_in - ar_or) / ar_or * 100:.2f}%")
print()

m = PGSFDet(cfg).to(dev).eval()
src = torch.tensor(np.array([f.rgb_box[:2] for f in frames[:16] if f.rgb_box is not None]))
dst = torch.tensor(np.array([f.tir_box[:2] for f in frames[:16] if f.tir_box is not None]))
m.set_affine(*fit_affine_np(src, dst))

for name, hw in (("rgb", "rgb_hw"), ("tir", "tir_hw")):
    rows = []
    for i in range(0, len(ds), 4):
        items = [ds[j] for j in range(i, min(i + 4, len(ds)))]
        batch = collate(items)
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = m(batch)
        sel = out[f"sel_{name}"]
        boxes, sc = decode_pred(out[f"pred_{name}"], sel["pts"], cfg.model.max_wh)
        H, W = batch[hw][0].tolist()
        px = torch.stack([
            (boxes[..., 0] - boxes[..., 2] / 2) * W,
            (boxes[..., 1] - boxes[..., 3] / 2) * H,
            (boxes[..., 0] + boxes[..., 2] / 2) * W,
            (boxes[..., 1] + boxes[..., 3] / 2) * H,
        ], dim=-1)
        gt = batch[f"boxes_{name}_px"].to(dev)
        for b in range(px.shape[0]):
            g = gt[b][gt[b].any()]
            if g.numel() == 0:
                continue
            iou = box_iou(px[b], g)                       # (N,1)
            best = int(iou.max())
            rows.append({
                "iou": float(iou[best]),
                "score": float(sc[b, best].sigmoid()),
                "top_score": float(sc[b].sigmoid().max()),
                "gt_wh": g[0, 2:].tolist(),
            })
    ious = np.array([r["iou"] for r in rows])
    scs = np.array([r["score"] for r in rows])
    tops = np.array([r["top_score"] for r in rows])
    wh = np.array([r["gt_wh"] for r in rows])
    print(f"=== {name.upper()} ({len(rows)} frames, untrained model) ===")
    print(f"  GT size (px): median {np.median(wh, 0)[0]:.1f} x {np.median(wh, 0)[1]:.1f}"
          f"   aspect(w/h) median {np.median(wh[:, 0] / wh[:, 1]):.3f}")
    print(f"  best IoU     : median {np.median(ious):.4f}  max {ious.max():.4f}"
          f"  (>=0.5 count: {(ious >= 0.5).sum()}/{len(ious)})")
    print(f"  score at that box: median {np.median(scs):.4f}  max {scs.max():.4f}")
    print(f"  top-1 score  : median {np.median(tops):.4f}  max {tops.max():.4f}")
    print(f"  aspect distorts IoU? {'YES -> H1' if np.median(ious) < 0.1 else 'no -> H2'}")
