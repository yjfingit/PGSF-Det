#!/usr/bin/env python
"""PGSF-Det **v3** train / eval entry point.

A thin, opinionated front end for ``scripts/train.py``: it defaults the config to
the v3 recipe, refuses to run against a non-v3 architecture (so a typo cannot
silently train the wrong graph), and prints the v3-specific guard rails that
matter for a 2 x 10GB RTX 3080 host.

Examples
--------
Stage 1 (independent branches, no cross-modal coupling)::

    python scripts/train_v3.py --config configs/pgsf_v3_stage1.yaml

Stage 2 (adds the single object-level correspondence fusion)::

    python scripts/train_v3.py --config configs/pgsf_v3_stage2.yaml \
        --ckpt outputs/v3_stage1/ckpt/best.pt

Evaluate a checkpoint (no GT is used by the model, only by the metric)::

    python scripts/train_v3.py --config configs/pgsf_v3_stage2.yaml \
        --eval-only --ckpt outputs/v3_stage2/ckpt/best.pt --split val

Two GPUs, 2 samples each, AMP on::

    python scripts/train_v3.py --gpus 0,1 --set optim.batch_size=2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "configs/pgsf_v3_stage2.yaml"


def _ensure_config_arg() -> None:
    """Inject the default config when the caller did not pass one."""
    if "--config" not in sys.argv[1:] and not any(
            a.startswith("--config=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--config", DEFAULT_CONFIG]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[])
    known, _ = parser.parse_known_args()

    from pgsf.config import load_config
    cfg = load_config(known.config, known.set)
    arch = int(cfg.model.architecture_version)
    if arch < 3:
        raise SystemExit(f"{known.config} declares architecture_version={arch}; "
                         "scripts/train_v3.py only runs the v3 graph. Use "
                         "scripts/train.py for v1/v2.")

    batch = cfg.optim.batch_size
    print("[v3] candidates/modality =", cfg.model.n_cand,
          "| object tokens   =", cfg.model.n_cand_late)
    print("[v3] retrieval dim =", cfg.model.retrieval_dim,
          "| top-k anchors   =", cfg.model.corr_top_k,
          "| samples/anchor  =", cfg.model.corr_samples,
          "| grid tokens     =", cfg.model.v3_grid ** 2)
    print(f"[v3] AMP={cfg.optim.amp} per-GPU batch={batch} "
          f"img={tuple(cfg.data.rgb_size)}/{tuple(cfg.data.tir_size)}")
    if batch > 2:
        print("[v3][warn] per-GPU batch > 2 is above the 10GB RTX 3080 budget "
              "this recipe was calibrated for; watch the reported peak VRAM.")
    if cfg.stage >= 2 and not cfg.model.use_v3_fusion:
        print("[v3][warn] stage 2 with use_v3_fusion=false trains independent "
              "detection only -- that is the A0-style ablation, not v3.")

    _ensure_config_arg()
    from scripts.train import main as shared_main
    shared_main()


if __name__ == "__main__":
    main()
