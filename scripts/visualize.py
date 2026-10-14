#!/usr/bin/env python
"""Visualise PGSF-Det detections on UAV-CB frames.

Draws RGB and TIR side by side in a common figure with both modalities' boxes, so
the cross-modal consistency (or its absence) is directly visible.

Run::

    python scripts/visualize.py --ckpt outputs/pgsf_stage2/ckpt/best.pt \
        --config configs/pgsf_stage2.yaml --n 8 --split val --out outputs/vis
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from pgsf.config import load_config  # noqa: E402
from pgsf.data import UAVCBDataset, load_split_manifest  # noqa: E402
from pgsf.models.detector import PGSFDet
from pgsf.data.uavcb import make_augment  # noqa: E402

BOX_COLORS = {"rgb": (0, 220, 60), "tir": (255, 120, 0), "gt": (240, 40, 40)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="outputs/vis")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def draw(img: np.ndarray, boxes, scores, color, label: str) -> Image.Image:
    im = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    for b, s in zip(boxes, scores):
        if s < 0.01:
            continue
        d.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=3)
        d.text((b[0] + 2, max(0, b[1] - 12)), f"{s:.2f}", fill=color)
    d.text((6, 6), label, fill=(255, 255, 255))
    return im


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGSFDet(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("ema") or ck["model"]
    missing = model.load_checkpoint_weights(state)
    if missing.missing_keys:
        print(f"[warn] missing keys: {len(missing.missing_keys)}")
    print(f"[load] {args.ckpt}")

    frames = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes)[args.split]
    frames = frames[::args.stride][:args.n]
    if not frames:
        print("no frames"); return
    # augmentation off: we want the original image with boxes on it
    ds = UAVCBDataset(frames, cfg.data.rgb_size, cfg.data.tir_size, train=False,augment=make_augment(cfg.data,False))
    os.makedirs(args.out, exist_ok=True)

    for i in range(len(ds)):
        item = ds[i]
        batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v])
                 for k, v in item.items()}
        res, _ = model.inference(batch, max_det=cfg.eval.max_det,
                                 score_thr=args.score_thr, nms_thr=cfg.eval.nms_thr)
        # Each modality's GT is filtered by *its own* mask, so a TIR-only object
        # can no longer show up as a zero box on the RGB panel.
        rgb_gt = item["boxes_rgb_px"][item.get("mask_rgb", item["mask"]) > 0.5].numpy()
        tir_gt = item["boxes_tir_px"][item.get("mask_tir", item["mask"]) > 0.5].numpy()

        # draw on the *original* resolution images, so boxes map 1:1
        rgb_raw = np.array(Image.open(item_frame_rgb(frames, i)).convert("RGB"))
        tir_raw = np.array(Image.open(item_frame_tir(frames, i)).convert("L").convert("RGB"))
        rb = res["rgb"][0]["boxes"].cpu().numpy()
        rs = res["rgb"][0]["scores"].cpu().numpy()
        tb = res["tir"][0]["boxes"].cpu().numpy()
        ts = res["tir"][0]["scores"].cpu().numpy()

        im_r = draw(rgb_raw, rb, rs, BOX_COLORS["rgb"], "RGB pred")
        d = ImageDraw.Draw(im_r)
        for b in rgb_gt:
            if b.any():
                d.rectangle(list(b), outline=BOX_COLORS["gt"], width=2)
        im_t = draw(tir_raw, tb, ts, BOX_COLORS["tir"], "TIR pred")
        d2 = ImageDraw.Draw(im_t)
        for b in tir_gt:
            if b.any():
                d2.rectangle(list(b), outline=BOX_COLORS["gt"], width=2)

        # scale both to a common height for the side-by-side figure
        h = 540
        w_r = int(im_r.width * h / im_r.height)
        w_t = int(im_t.width * h / im_t.height)
        canvas = Image.new("RGB", (w_r + w_t + 8, h + 24), (20, 20, 20))
        canvas.paste(im_r.resize((w_r, h)), (0, 24))
        canvas.paste(im_t.resize((w_t, h)), (w_r + 8, 24))
        out_path = os.path.join(args.out, f"{frames[i].scene}_{frames[i].frame_id}.jpg")
        canvas.save(out_path, quality=92)
        print(f"[{i+1}/{len(ds)}] {frames[i].scene}/{frames[i].frame_id}  "
              f"rgb_top={float(rs.max()) if rs.size else 0:.3f} "
              f"tir_top={float(ts.max()) if ts.size else 0:.3f}  -> {out_path}")


def item_frame_rgb(frames, i):
    return frames[i].rgb_path


def item_frame_tir(frames, i):
    return frames[i].tir_path


if __name__ == "__main__":
    main()
