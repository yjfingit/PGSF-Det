#!/usr/bin/env python
"""PGSF-Det **v4** train / eval entry point.

A thin, opinionated front end for ``scripts/train.py``: it defaults the config to
the v4 recipe, refuses to run a non-v4 architecture (so a typo cannot silently
train the wrong graph), and prints the v4 knobs that matter.

Examples
--------
Stage 1 (independent branches, no cross-modal interaction)::

    python scripts/train_v4.py --config configs/pgsf_v4_stage1.yaml

Stage 2 (adds the single object-level interaction: CFPM + TD-Fusion)::

    python scripts/train_v4.py --config configs/pgsf_v4_stage2.yaml \
        --ckpt outputs/v4_stage1/ckpt/best.pt

Evaluate / two GPUs with AMP::

    python scripts/train_v4.py --config configs/pgsf_v4_stage2.yaml \
        --eval-only --ckpt outputs/v4_stage2/ckpt/best.pt --split val
    python scripts/train_v4.py --gpus 0,1 --set optim.batch_size=2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "configs/pgsf_v4_stage2.yaml"


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
    if arch != 4:
        raise SystemExit(f"{known.config} declares architecture_version={arch}; "
                         "scripts/train_v4.py only runs the v4 graph. Use "
                         "scripts/train.py / train_v3.py for v1-v3.")

    m = cfg.model
    print(f"[v4] K = {m.n_cand} candidates/modality over {list(m.cand_levels)} "
          f"(reserved {m.cand_min_per_level}/level)")
    print(f"[v4] ROI S = [B,K,{m.neck_out},{m.roi_size},{m.roi_size}] | retrieval dim = "
          f"{m.retrieval_dim} | Top-L = {m.td_top_l} | samples/anchor = {m.td_samples} "
          f"-> {m.td_top_l * m.td_samples} local features")
    print(f"[v4] alpha_init = {m.alpha_init} | classes = {m.num_classes} | "
          f"match={m.use_match_loss} (w={m.lambda_match}) | cons={m.use_cons_loss} (w={m.w_cons})")
    print(f"[v4] losses: iou={m.lambda_iou} l1={m.lambda_l1} self={m.w_self} "
          f"prop={m.w_prop} pbox={m.lambda_pbox} quality={m.lambda_quality}")
    print(f"[v4] AMP={cfg.optim.amp} per-GPU batch={cfg.optim.batch_size} "
          f"img={tuple(cfg.data.rgb_size)}/{tuple(cfg.data.tir_size)}")
    if cfg.optim.batch_size > 2:
        print("[v4][warn] per-GPU batch > 2 is above the 10GB RTX 3080 budget this "
              "recipe was calibrated for; watch the reported peak VRAM.")
    if cfg.stage >= 2 and not m.use_td_fusion:
        print("[v4][warn] stage 2 with use_td_fusion=false trains independent "
              "detection only -- that is the A0-style ablation, not v4.")

    _ensure_config_arg()
    from scripts.train import main as shared_main
    shared_main()


if __name__ == "__main__":
    main()

