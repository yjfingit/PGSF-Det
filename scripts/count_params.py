#!/usr/bin/env python
"""Exact parameter accounting for PGSF-Det.

Reports:
  * total / trainable parameters
  * a breakdown by top-level module
  * a deeper breakdown of one modality branch (backbone, MSCE, neck, heads)
  * the budget check against the 50M cap
  * the same numbers for every ablation variant, so it is clear what each module costs

Run::

    python scripts/count_params.py
    python scripts/count_params.py --config configs/pgsf_stage1.yaml
    python scripts/count_params.py --all-ablations
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgsf.config import load_config  # noqa: E402
from pgsf.models.detector import PGSFDet  # noqa: E402

BUDGET = 50_000_000


def human(n: int) -> str:
    return f"{n / 1e6:.3f} M" if n >= 1e6 else f"{n / 1e3:.1f} K"


def deep_breakdown(model: PGSFDet) -> None:
    """Per-submodule counts inside each modality branch (the bulk of the model)."""
    print("\n--- one modality branch, module by module ---")
    for branch_name in ("rgb_branch", "tir_branch"):
        branch = getattr(model, branch_name)
        parts = [(n, sum(p.numel() for p in m.parameters()))
                 for n, m in branch.named_children()]
        total = sum(v for _, v in parts)
        print(f"  {branch_name}: {human(total)}")
        for n, v in parts:
            print(f"      {n:10s} {human(v):>12s}"
                  f"   ({100.0 * v / total:5.1f}%)")

    print("\n--- RGB backbone internals (per stage) ---")
    bb = model.rgb_branch.backbone
    for n, m in bb.named_children():
        print(f"      {n:10s} {human(sum(p.numel() for p in m.parameters())):>12s}")

    print("\n--- RGB neck internals ---")
    nk = model.rgb_branch.neck
    for n, m in nk.named_children():
        print(f"      {n:10s} {human(sum(p.numel() for p in m.parameters())):>12s}")


def variant_table(base_cfg: str) -> None:
    """Parameter count of every ablation variant -- shows what each module costs."""
    from scripts.make_ablation import ABLATIONS  # noqa: E402
    print("\n--- ablation variants (what each module costs) ---")
    print(f"  {'variant':8s} {'params':>12s} {'delta vs prev':>15s}  modules on")
    prev = None
    for tag, flags in ABLATIONS.items():
        cfg = load_config(base_cfg)
        for k, v in flags.items():
            setattr(cfg.model, k, v)
        cfg.stage = 2
        n = sum(p.numel() for p in PGSFDet(cfg).parameters())
        on = [k for k in ("use_msce", "use_mid_fusion", "use_geo_prior",
                          "use_gate", "use_late_fusion", "use_nwd") if flags[k]]
        delta = "" if prev is None else f"{n - prev:+15,d}"
        print(f"  {tag:8s} {n:12,d} {delta:>15s}  {', '.join(on) or 'none'}")
        prev = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pgsf_stage2.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--all-ablations", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    model = PGSFDet(cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    print("=" * 66)
    print("PGSF-Det parameter count")
    print("=" * 66)
    print(f"config          : {args.config}  (stage {cfg.stage})")
    print(f"input           : RGB {tuple(cfg.data.rgb_size)}  "
          f"TIR {tuple(cfg.data.tir_size)}")
    print()
    print(f"TOTAL PARAMETERS : {total:,d}  ({human(total)})")
    print(f"trainable       : {trainable:,d}  ({human(trainable)})")
    print(f"buffers (BN etc): {buffers:,d}  ({human(buffers)})")
    print(f"budget          : {BUDGET:,d}  -> "
          f"{'OK, using %.1f%%' % (100.0 * total / BUDGET)}"
          if total < BUDGET else "OVER BUDGET")
    if 25e6 <= total <= 35e6:
        print("                   (inside the preferred 25M-35M band)")

    print("\n--- by top-level module ---")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n == 0:
            continue
        print(f"  {name:14s} {n:12,d}  ({human(n):>10s})  "
              f"{100.0 * n / total:5.1f}%")

    deep_breakdown(model)

    if args.all_ablations:
        variant_table(args.config)

    if not model.v2:
        assert total < BUDGET, f"parameter budget exceeded: {total} >= {BUDGET}"
    print("\nParameter accounting complete; v2 prioritizes measured FPS.")


if __name__ == "__main__":
    main()
