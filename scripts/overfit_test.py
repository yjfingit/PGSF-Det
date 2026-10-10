#!/usr/bin/env python
"""Small-sample overfit test on **real** UAV-CB data.

This is the decisive check that the supervision is wired correctly: if the loss
cannot drive AP50 to a high value on 4-8 real images, no amount of training time
will help, and the bug is in data / assignment / gradient rather than capacity.

Run::

    python scripts/overfit_test.py --config configs/pgsf_stage2.yaml \
        --n 4 --steps 300 --batch 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pgsf.config import load_config  # noqa: E402
from pgsf.data import UAVCBDataset, collate, load_split_manifest
from pgsf.data.uavcb import make_augment  # noqa: E402
from pgsf.engine.trainer import fit_affine_np  # noqa: E402
from pgsf.models.detector import PGSFDet  # noqa: E402
from pgsf.utils.metrics import eval_detection, precision_recall_at, filtered_ground_truth  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--n", type=int, default=4, help="number of real images")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--augment", action="store_true",
                   help="keep random augmentation ON. Off by default: overfitting "
                        "means memorising a FIXED input->output map, so scale jitter / "
                        "flip / photometric noise work against the very thing being "
                        "measured, and they also make evaluation non-deterministic.")
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt",default=None,help="initialize a warmup/joint small-sample checkpoint")
    p.add_argument("--save-ckpt",default=None,help="save best small-sample weights; not a SOTA checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    frames = load_split_manifest(cfg.data.split_dir, cfg.data.root,
                                cfg.data.scenes)["train"]
    if not frames:
        raise RuntimeError(f"no training frames under {cfg.data.split_dir}")
    # deterministic pick, prefer a spread across scenes
    by_scene: dict = {}
    for f in frames:
        by_scene.setdefault(f.scene, []).append(f)
    picked, scenes = [], sorted(by_scene)
    i = 0
    while len(picked) < args.n:
        picked.append(by_scene[scenes[i % len(scenes)]][i // len(scenes)])
        i += 1
    print(f"[data] overfitting on {len(picked)} real training frames: "
          f"{[(f.scene, f.frame_id) for f in picked]}")

    # train=args.augment (default False): the evaluation and the training input must
    # be the SAME fixed image, otherwise "overfit" measures nothing.
    ds = UAVCBDataset(picked, cfg.data.rgb_size, cfg.data.tir_size,
                      train=args.augment,augment=make_augment(cfg.data,args.augment))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=collate, drop_last=len(picked) > args.batch)

    model = PGSFDet(cfg).to(device)
    if args.ckpt:
        ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
        transfer=ck.get('config',{}).get('stage')==1 and cfg.stage==2
        model.load_checkpoint_weights(ck.get('ema') or ck['model'],transfer)
    # fit the affine prior on exactly these frames (training data, as in production)
    src = torch.tensor([f.rgb_box[:2] for f in picked if f.rgb_box is not None])
    dst = torch.tensor([f.tir_box[:2] for f in picked if f.tir_box is not None])
    A, b = fit_affine_np(src, dst)
    model.set_affine(A, b)
    print(f"[geo] affine fitted on {len(src)} frames: A={A.tolist()} b={b.tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=128., enabled=(device.type == "cuda"))

    history = []
    best_score=-1.
    t0 = time.time()
    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            model.set_training_progress(step/max(args.steps,1))
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(batch)
                loss, terms = model.compute_losses(out)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, loader, device, cfg)
                row = {"step": step, "loss": float(loss),
                       **{k: float(v) for k, v in terms.items()}, **m}
                history.append(row)
                score=(m['rgb_AP50']+m['tir_AP50'])/2
                if args.save_ckpt and score>best_score:
                    best_score=score
                    os.makedirs(os.path.dirname(args.save_ckpt) or '.',exist_ok=True)
                    torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'ema':None,
                                'kind':'real training-frame overfit, not validation/test performance',
                                'step':step,'metrics':m},args.save_ckpt)
                print(f"[step {step:4d}] loss={float(loss):.4f} "
                      f"AP50 rgb={m['rgb_AP50']:.3f} tir={m['tir_AP50']:.3f} "
                      f"AP rgb={m['rgb_AP']:.3f} tir={m['tir_AP']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    final = history[-1]
    ok = final["rgb_AP50"] > 0.5 and final["tir_AP50"] > 0.5
    print("\n=== overfit verdict ===")
    print(f"  final RGB AP50={final['rgb_AP50']:.3f} AP={final['rgb_AP']:.3f} "
          f"P={final['rgb_P']:.3f} R={final['rgb_R']:.3f}")
    print(f"  final TIR AP50={final['tir_AP50']:.3f} AP={final['tir_AP']:.3f} "
          f"P={final['tir_P']:.3f} R={final['tir_R']:.3f}")
    print("  VERDICT: " + ("PASS (supervision is wired correctly)"
                          if ok else "FAIL (investigate data / assignment / loss)"))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "history": history}, f, indent=2)
        print(f"  wrote {args.out}")
    return 0 if ok else 1


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    preds = {"rgb": [], "tir": []}
    gts = {"rgb": [], "tir": []}
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        res, _ = model.inference(batch, max_det=cfg.eval.max_det,
                                 score_thr=cfg.eval.conf_thr, nms_thr=cfg.eval.nms_thr)
        for name in ("rgb", "tir"):
            preds[name].extend(res[name])
            # pixel-space GT: ``inference`` returns pixel coordinates, so the GT must
            # match. Using the normalised boxes here makes every IoU 0 and AP 0.
            gts[name].extend(filtered_ground_truth(batch, name))
    model.train()
    out = {}
    for name in ("rgb", "tir"):
        ap = eval_detection(preds[name], gts[name], cfg.eval.iou_thrs, cfg.eval.max_det)
        pr = precision_recall_at(preds[name], gts[name], cfg.eval.score_thr)
        out[f"{name}_AP50"] = ap["AP50"]
        out[f"{name}_AP"] = ap["AP50:95"]
        out[f"{name}_P"] = pr["precision"]
        out[f"{name}_R"] = pr["recall"]
    return out


if __name__ == "__main__":
    sys.exit(main())
