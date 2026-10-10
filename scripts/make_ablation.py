#!/usr/bin/env python
"""Generate the ablation configs A0 - A6 from the A6 (full) baseline.

A0  dual CNN + independent detection              (no MSCE, no fusion)
A1  + MSCE                                        (small-target enhancement)
A2  + mid-stage Transformer **without** geometry    (plain cross-modal attention)
A3  + probabilistic geometric prior                (log P_G inside attention)
A4  + reliability gate and null matching
A5  + late-stage object-level Transformer
A6  + NWD auxiliary localisation                   (= the full model)

Every variant keeps the *same* backbone, neck, candidate budget, optimiser and
data split, so a difference in AP is attributable to the added module alone.

Run::

    python scripts/make_ablation.py            # writes configs/ablation_A*.yaml
    python scripts/make_ablation.py --train    # and prints the run commands
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgsf.config import load_config  # noqa: E402

# name -> (model flag overrides)
ABLATIONS = {
    "A0": dict(use_msce=False, use_mid_fusion=False, use_geo_prior=False,
               use_gate=False, use_late_fusion=False, use_nwd=False),
    "A1": dict(use_msce=True, use_mid_fusion=False, use_geo_prior=False,
               use_gate=False, use_late_fusion=False, use_nwd=False),
    "A2": dict(use_msce=True, use_mid_fusion=True, use_geo_prior=False,
               use_gate=False, use_late_fusion=False, use_nwd=False),
    "A3": dict(use_msce=True, use_mid_fusion=True, use_geo_prior=True,
               use_gate=False, use_late_fusion=False, use_nwd=False),
    "A4": dict(use_msce=True, use_mid_fusion=True, use_geo_prior=True,
               use_gate=True, use_late_fusion=False, use_nwd=False),
    "A5": dict(use_msce=True, use_mid_fusion=True, use_geo_prior=True,
               use_gate=True, use_late_fusion=True, use_nwd=False),
    "A6": dict(use_msce=True, use_mid_fusion=True, use_geo_prior=True,
               use_gate=True, use_late_fusion=True, use_nwd=True),
}

DESCRIPTIONS = {
    "A0": "dual CNN + independent detection",
    "A1": "A0 + MSCE small-target enhancement",
    "A2": "A1 + mid-stage Transformer (no geometry prior)",
    "A3": "A2 + probabilistic geometric prior",
    "A4": "A3 + reliability gate and null matching",
    "A5": "A4 + late object-level Transformer",
    "A6": "A5 + NWD auxiliary localisation (full model)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/pgsf_stage2.yaml")
    ap.add_argument("--out-dir", default="configs/ablation")
    ap.add_argument("--train", action="store_true",
                    help="also print the training command for each variant")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = load_config(args.base)
    # keep the runs comparable: same budget for every variant
    base.optim.epochs = 30

    for tag, flags in ABLATIONS.items():
        cfg = load_config(args.base)
        for k, v in flags.items():
            setattr(cfg.model, k, v)
        # A0/A1 need stage 2 enabled so the code path is identical minus the modules
        cfg.model.use_null=tag in ("A4","A5","A6")
        cfg.stage = 2
        cfg.optim.epochs = base.optim.epochs
        cfg.name = f"pgsf_{tag}"
        cfg.out_dir = f"outputs/ablation_{tag}"
        path = os.path.join(args.out_dir, f"pgsf_{tag}.yaml")
        cfg.save(path)
        print(f"{tag}: {DESCRIPTIONS[tag]}")
        print(f"     {path}")
        if args.train:
            print(f"     CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config {path}")

    # a runnable summary table
    print("\n| variant | modules on | AP(RGB) | AP(TIR) | small-R | match acc |")
    print("|---|---|---|---|---|---|")
    for tag, flags in ABLATIONS.items():
        on = []
        if flags["use_msce"]:
            on.append("MSCE")
        if flags["use_mid_fusion"]:
            on.append("Mid" + ("+geo" if flags["use_geo_prior"] else ""))
        if flags["use_gate"]:
            on.append("gate+null")
        if flags["use_late_fusion"]:
            on.append("Late")
        if flags["use_nwd"]:
            on.append("NWD")
        print(f"| {tag} | {', '.join(on) or 'none'} | | | | |")


if __name__ == "__main__":
    main()
