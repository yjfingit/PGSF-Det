#!/usr/bin/env python
"""Fast A/B test on the heatmap hypothesis.

Diagnosis so far: the heatmap never forms a peak (heat@GT stays ~0.012 vs the
0.018 initialisation), so eval-mode candidates miss the target and AP stays 0.

Rather than train for 1500 steps to find out, this script trains three short
variants and reports heat@GT / argmax error / candidate recall for each:

  V1  baseline                       (as configured)
  V2  w_heat x 8                     (is the heatmap simply under-weighted?)
  V3  only heatmap + box head train  (can the features support it at all?)

Whichever variant lifts heat@GT tells us where the real bottleneck is.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader
from pgsf.config import load_config
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.engine.trainer import fit_affine_np
from pgsf.models.detector import PGSFDet
from pgsf.utils.metrics import eval_detection

STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "5e-4"))
N_IMG = int(os.environ.get("N_IMG", "4"))

base_cfg = load_config("configs/pgsf_stage2.yaml")
frames = load_split_manifest(base_cfg.data.split_dir, base_cfg.data.root,
                             base_cfg.data.scenes)["train"]
by = {}
for f in frames:
    by.setdefault(f.scene, []).append(f)
scenes = sorted(by)
picked = [by[scenes[i % len(scenes)]][i // len(scenes)] for i in range(N_IMG)]
ds = UAVCBDataset(picked, base_cfg.data.rgb_size, base_cfg.data.tir_size, train=True)
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, drop_last=False)
src = torch.tensor(np.array([f.rgb_box[:2] for f in picked if f.rgb_box is not None]))
dst = torch.tensor(np.array([f.tir_box[:2] for f in picked if f.tir_box is not None]))
A, b = fit_affine_np(src, dst)
dev = torch.device("cuda")


def run(tag, w_heat=1.0, heat_only=False):
    torch.manual_seed(0)
    cfg = load_config("configs/pgsf_stage2.yaml")
    cfg.model.w_heat = w_heat
    m = PGSFDet(cfg).to(dev)
    m.set_affine(A, b)
    if heat_only:
        # freeze everything except the heatmap head -> isolates feature quality
        for n_, p_ in m.named_parameters():
            p_.requires_grad_ = ("heat" in n_)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    m.train()
    t0 = time.time(); it = 0
    while it < STEPS:
        for batch in loader:
            if it >= STEPS: break
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                out = m(batch)
                loss, terms = m.compute_losses(out)
                if heat_only:
                    # ``terms`` holds *detached* values for logging, so recompute the
                    # heatmap term from the live logits instead of reusing it.
                    from pgsf.losses import draw_gaussian, gaussian_focal_loss
                    h_r = draw_gaussian(torch.zeros_like(out["cand_rgb"]["heat"]),
                                        batch["centers_rgb"], batch["sizes_rgb"],
                                        batch["mask"])
                    h_t = draw_gaussian(torch.zeros_like(out["cand_tir"]["heat"]),
                                        batch["centers_tir"], batch["sizes_tir"],
                                        batch["mask"])
                    loss = w_heat * (gaussian_focal_loss(out["heat_rgb"], h_r) +
                                     gaussian_focal_loss(out["heat_tir"], h_t))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); it += 1
    # evaluate in EVAL mode (no GT injection)
    m.eval()
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = m(batch)
        res, _ = m.inference(batch, max_det=50, score_thr=cfg.eval.conf_thr,
                             nms_thr=cfg.eval.nms_thr)
    heat = out["heat_rgb"].sigmoid()
    H, W = heat.shape[-2:]
    gt_c = batch["centers_rgb"]
    gx = (gt_c[:, 0, 0] * (W - 1)).round().long()
    gy = (gt_c[:, 0, 1] * (H - 1)).round().long()
    at_gt = heat[:, 0, gy, gx]
    flat = heat.flatten(1).argmax(1)
    err = torch.sqrt((flat % W - gx.float())**2 + (flat // W - gy.float())**2)
    cand = out["cand_rgb"]
    d = torch.norm(cand["pts"] - gt_c[:, 0:1], dim=-1)
    hit5 = (d < 0.05).any(1)
    gts = list(batch["boxes_rgb"])
    ap = eval_detection(res["rgb"], gts, cfg.eval.iou_thrs, 50)
    print(f"{tag:28s} loss={float(loss):7.3f} heat@GT={at_gt.mean():.4f} "
          f"heatmax={heat.max():.4f} argmax_err={err.mean():6.1f}px "
          f"cand_hit@0.05={int(hit5.sum())}/{hit5.numel()} "
          f"cand_min={d.min(1).values.mean():.4f} AP50={ap['AP50']:.3f} "
          f"({time.time()-t0:.0f}s)")
    return float(at_gt.mean()), float(ap["AP50"])


print(f"STEPS={STEPS} LR={LR} N_IMG={N_IMG}\n")
run("V1 baseline (w_heat=1)")
run("V2 w_heat=8", w_heat=8.0)
run("V3 heatmap head only", heat_only=True)
