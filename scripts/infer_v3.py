#!/usr/bin/env python
"""PGSF-Det **v3** inference entry point.

Runs the detector on image pairs and writes per-image detections as JSON.  The
model never touches annotations: the batch it receives contains only
``rgb``/``tir`` and their native ``*_hw`` sizes, which is exactly the contract
``PGSFDet.inference`` guarantees (``tests/test_v3.py::test_inference_needs_no_gt``
pins it down).

Examples
--------
A held-out split (ground truth is read only to compute the metric)::

    python scripts/infer_v3.py --ckpt outputs/v3_stage2/ckpt/best.pt \
        --split val --n 32 --out outputs/v3_val_dets.json

Arbitrary image pairs, no annotations anywhere::

    python scripts/infer_v3.py --ckpt outputs/v3_stage2/ckpt/best.pt \
        --pair /data/a_visible.jpg /data/a_thermal.jpg \
        --pair /data/b_visible.jpg /data/b_thermal.jpg \
        --out outputs/v3_pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pgsf.config import load_config  # noqa: E402
from pgsf.data.uavcb import PairAugment  # noqa: E402
from pgsf.models.detector import PGSFDet  # noqa: E402


def load_model(cfg, ckpt: str, device):
    model = PGSFDet(cfg).to(device).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = state.get("ema") or state.get("model") or state
        model.load_checkpoint_weights(weights, stage1_transfer=False)
    return model


def images_batch(rgb_path: str, tir_path: str, cfg, aug: PairAugment, device):
    """GT-free batch: images + native sizes only."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8, copy=True)
    tir = np.array(Image.open(tir_path).convert("L"), dtype=np.uint8, copy=True)
    rgb_t, tir_t, *_ = aug.augment_boxes(
        rgb, tir, None, None, (rgb.shape[1], rgb.shape[0]), (tir.shape[1], tir.shape[0]),
        cfg.data.rgb_size, cfg.data.tir_size)
    return {
        "rgb": torch.from_numpy(rgb_t)[None].to(device),
        "tir": torch.from_numpy(tir_t)[None].to(device),
        "rgb_hw": torch.tensor(_hw(rgb.shape[1], rgb.shape[0]), dtype=torch.float32,
                               device=device),
        "tir_hw": torch.tensor(_hw(tir.shape[1], tir.shape[0]), dtype=torch.float32,
                               device=device),
    }


def _hw(width: int, height: int):
    """Native (W,H) -> the ``[H, W]`` row the model's decoder expects."""
    return [[height, width]]


def main():
    ap = argparse.ArgumentParser(description="PGSF-Det v3 inference")
    ap.add_argument("--config", default="configs/pgsf_v3_stage2.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--pair", nargs=2, action="append", default=[],
                    metavar=("RGB", "TIR"), help="explicit image pair (repeatable)")
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(cfg, args.ckpt, device)
    aug = PairAugment(train=False)
    score_thr = cfg.eval.score_thr if args.score_thr is None else args.score_thr

    sources = []
    if args.pair:
        sources = [(r, t) for r, t in args.pair]
    elif args.split:
        from pgsf.data.uavcb import load_split_manifest, list_frames
        fixed = load_split_manifest(cfg.data.split_dir, cfg.data.root, cfg.data.scenes,
                                    cfg.data.class_names)
        frames = (fixed.get(args.split) or []) if fixed else []
        if not frames:
            frames = list_frames(cfg.data.root, cfg.data.scenes, cfg.data.class_names)
        sources = [(f.rgb_path, f.tir_path) for f in frames[:args.n]]
    else:
        raise SystemExit("provide either --pair RGB TIR ... or --split {train,val,test}")

    results = []
    for rgb_path, tir_path in sources:
        batch = images_batch(rgb_path, tir_path, cfg, aug, device)
        with torch.no_grad():
            detections, _ = model.inference(batch, max_det=cfg.eval.max_det,
                                            score_thr=score_thr, nms_thr=cfg.eval.nms_thr)
        entry = {"rgb_path": rgb_path, "tir_path": tir_path}
        for name in ("rgb", "tir"):
            det = detections[name][0]
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det.get("labels")
            labels = None if labels is None else labels.cpu().numpy().tolist()
            # Pixel boxes are emitted in each modality's own frame; they are never
            # rescaled into the other modality's coordinates.
            entry[name] = {"boxes": boxes.tolist(), "scores": scores.tolist(),
                           "labels": labels}
        results.append(entry)
        print(f"{os.path.basename(rgb_path)}: "
              f"rgb={len(entry['rgb']['scores'])} tir={len(entry['tir']['scores'])}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "ckpt": args.ckpt,
                       "score_thr": score_thr, "images": results}, f, indent=2)
        print(f"[out] {args.out}")
    return results


if __name__ == "__main__":
    main()
