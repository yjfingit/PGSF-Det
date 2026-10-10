#!/usr/bin/env python
"""Overfit + diagnose in one pass: after training, report heatmap peak location,
candidate recall, assignment positives and box error -- the four stages that
determine AP."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from pgsf.config import load_config
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.engine.trainer import fit_affine_np
from pgsf.models.detector import PGSFDet
from pgsf.models.head import decode_pred
from pgsf.losses import assign_candidates, draw_gaussian
from pgsf.utils.metrics import eval_detection, precision_recall_at

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4
n_img = int(sys.argv[3]) if len(sys.argv) > 3 else 4

cfg = load_config("configs/pgsf_stage2.yaml")
dev = torch.device("cuda")
frames = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes)["train"]
by = {}
for f in frames:
    by.setdefault(f.scene, []).append(f)
scenes = sorted(by)
picked = [by[scenes[i % len(scenes)]][i // len(scenes)] for i in range(n_img)]
ds = UAVCBDataset(picked, cfg.data.rgb_size, cfg.data.tir_size, train=True)
loader = DataLoader(ds, batch_size=min(2, n_img), shuffle=True, collate_fn=collate,
                    drop_last=False)

m = PGSFDet(cfg).to(dev)
import numpy as np
src = torch.tensor(np.array([f.rgb_box[:2] for f in picked if f.rgb_box is not None]))
dst = torch.tensor(np.array([f.tir_box[:2] for f in picked if f.tir_box is not None]))
m.set_affine(*fit_affine_np(src, dst))

opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
# NO_AMP=1 runs in fp32. The CenterNet focal loss contains -log(p) on the positive
# cell; when p is small that term is ~9, and multiplied by w_heat=8 the total loss
# easily exceeds the fp16 range, so GradScaler keeps inf-lerping and silently skips
# optimiser steps. Running fp32 isolates that possibility.
use_amp = os.environ.get("NO_AMP", "0") != "1"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"AMP={'on' if use_amp else 'off'}  w_heat={cfg.model.w_heat}  w_geo={cfg.model.w_geo}")
m.train()
it = 0
n_skip = 0
while it < steps:
    for batch in loader:
        if it >= steps: break
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = m(batch); loss, terms = m.compute_losses(out)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            old = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() < old:
                n_skip += 1
        else:
            loss.backward(); opt.step()
        it += 1
print(f"trained {it} steps, final loss {float(loss):.4f}, "
      f"GradScaler reductions={n_skip} (scale now {scaler.get_scale():.0f})")

# ---- diagnose in EVAL mode (no GT injection) ----
batch = next(iter(loader))
batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
m.eval()
with torch.no_grad():
    out = m(batch)
    res, _ = m.inference(batch, max_det=50, score_thr=cfg.eval.conf_thr,
                          nms_thr=cfg.eval.nms_thr)

for name in ("rgb", "tir"):
    heat = out[f"heat_{name}"].sigmoid()
    H, W = heat.shape[-2:]
    gt_c = batch[f"centers_{name}"]; gt_s = batch[f"sizes_{name}"]
    gx = (gt_c[:, 0, 0] * (W - 1)).round().long()
    gy = (gt_c[:, 0, 1] * (H - 1)).round().long()
    at_gt = heat[:, 0, gy, gx]
    # global argmax
    flat = heat.flatten(1).argmax(1)
    amx, amy = flat % W, flat // W
    err_px = torch.sqrt((amx.float() - gx.float())**2 + (amy.float() - gy.float())**2)
    print(f"\n=== {name} (grid {H}x{W}) ===")
    print(f"  heat@GT={at_gt.tolist()}  heat_max={heat.flatten(1).max(1).values.tolist()}")
    print(f"  global argmax at ({amy.tolist()},{amx.tolist()}) vs GT ({gy.tolist()},{gx.tolist()})"
          f"  err={err_px.tolist()} px")
    cand = out[f"cand_{name}"]; sel = out[f"sel_{name}"]
    d = torch.norm(cand["pts"] - gt_c[:, 0:1], dim=-1)
    ds_ = torch.norm(sel["pts"] - gt_c[:, 0:1], dim=-1)
    print(f"  cand  min-dist to GT: {d.min(1).values.tolist()}")
    print(f"  sel32 min-dist to GT: {ds_.min(1).values.tolist()}")
    pos, _ = assign_candidates(sel["pts"], batch[f"boxes_{name}"], batch["mask"],
                               cfg.model.assign_pos_thr, cfg.model.assign_center_thr)
    print(f"  assignment positives (eval, no injection): {pos.sum(1).tolist()} / {sel['pts'].shape[1]}")
    boxes, sc = decode_pred(out[f"pred_{name}"], sel["pts"], cfg.model.max_wh)
    print(f"  pred box w,h: mean {boxes[...,2:].mean().item():.4f} "
          f"(GT {gt_s[:, 0].mean().item():.4f})")
    print(f"  top score: {sc.sigmoid().flatten().topk(3).values.tolist()}")
    # --- the decisive numbers: centre error and IoU of the *best* candidate ---
    g = batch[f"boxes_{name}"][:, 0]                    # (B,4) normalised GT
    cx, cy, w, h = boxes.unbind(-1)                       # (B,K)
    gcx, gcy, gw, gh = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    cdist = torch.sqrt((cx - gcx) ** 2 + (cy - gcy) ** 2)          # (B,K)
    ix0 = torch.maximum(cx - w / 2, gcx - gw / 2)
    iy0 = torch.maximum(cy - h / 2, gcy - gh / 2)
    ix1 = torch.minimum(cx + w / 2, gcx + gw / 2)
    iy1 = torch.minimum(cy + h / 2, gcy + gh / 2)
    inter = (ix1 - ix0).clamp_min(0) * (iy1 - iy0).clamp_min(0)
    union = w * h + gw * gh - inter
    iou_all = inter / union.clamp_min(1e-9)                         # (B,K)
    order = sc.argsort(dim=1, descending=True)
    ar = torch.arange(iou_all.shape[0], device=iou_all.device)
    print(f"  centre dist of the TOP-1 candidate: {cdist[ar, order[:, 0]].tolist()}")
    print(f"  IoU of the TOP-1 candidate      : {iou_all[ar, order[:, 0]].tolist()}")
    print(f"  best IoU among all candidates   : {iou_all.max(1).values.tolist()}  "
          f"(need >= 0.5 for AP50)")
    gts = [b for b in batch[f"boxes_{name}"]]
    ap = eval_detection(res[name], gts, cfg.eval.iou_thrs, 50)
    pr = precision_recall_at(res[name], gts, cfg.eval.score_thr)
    print(f"  AP50={ap['AP50']:.3f} AP={ap['AP50:95']:.3f} P={pr['precision']:.3f} R={pr['recall']:.3f}")

# also check TRAIN-mode assignment (with injection)
m.train()
with torch.no_grad():
    out = m(batch)
    for name in ("rgb", "tir"):
        sel = out[f"sel_{name}"]
        pos, _ = assign_candidates(sel["pts"], batch[f"boxes_{name}"], batch["mask"],
                                   cfg.model.assign_pos_thr, cfg.model.assign_center_thr)
        print(f"  [train mode] {name} positives: {pos.sum(1).tolist()} / {sel['pts'].shape[1]}"
              f"  injected kept: {int(sel['injected'].sum().item())}")
