#!/usr/bin/env python
"""PGSF-Det train / eval entry point.

Examples
--------
Stage 1 (single-modality pretraining)::

    python scripts/train.py --config configs/pgsf_stage1.yaml \
        --set data.max_train_per_scene=200 optim.epochs=2

Stage 2 (cross-modal joint training, from a stage-1 checkpoint)::

    python scripts/train.py --config configs/pgsf_stage2.yaml \
        --ckpt outputs/pgsf_stage1/ckpt/last.pt

Evaluate a checkpoint::

    python scripts/train.py --config configs/pgsf_stage2.yaml \
        --eval-only --ckpt outputs/pgsf_stage2/ckpt/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist

from pgsf.config import load_config  # noqa: E402
from pgsf.engine.trainer import PGSFTrainer  # noqa: E402
from pgsf.models.detector import PGSFDet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PGSF-Det")
    p.add_argument("--config", default=None, help="yaml config")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.lr=1e-3")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--ckpt", default=None, help="checkpoint to load (eval / finetune)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default=None)
    p.add_argument('--gpus',default='auto',help='idle GPUs automatically, or physical indices')
    p.add_argument('--gpu-max-used-mib',type=int,default=512)
    p.add_argument('--gpu-min-free-mib',type=int,default=9000)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if 'LOCAL_RANK' not in os.environ and cfg.device!='cpu':
        from pgsf.utils.gpu_select import select_gpus
        selected,_=select_gpus(args.gpus,args.gpu_max_used_mib,args.gpu_min_free_mib)
        if args.eval_only:selected=selected[:1]
        gpu_ids=[g['index'] for g in selected]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpu_ids)))
        print(f'[GPU auto-select] GPUs={gpu_ids}; batch=8/GPU; global batch={8*len(gpu_ids)}',flush=True)
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpu_ids)}',
                 str(Path(__file__).resolve()),*sys.argv[1:]]
        raise SystemExit(subprocess.call(command,env=env))
    if 'LOCAL_RANK' in os.environ:cfg.optim.batch_size=8
    if args.out_dir:
        cfg.out_dir = args.out_dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    world_size=int(os.environ.get('WORLD_SIZE','1'))
    if world_size>1:
        local_rank=int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device=torch.device('cuda',local_rank)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                              else "cpu")
    trainer = PGSFTrainer(cfg, device=device)
    if trainer.is_main:print(f"[data] {trainer.counts}")

    if args.resume:
        trainer.resume(args.resume)
    elif args.ckpt:
        trainer.load_weights(args.ckpt)

    if args.eval_only:
        res = trainer.evaluate(split=args.split, use_ema=bool(trainer.ema))
        if trainer.is_main:
            print(json.dumps(res, indent=2, default=float))
            with open(os.path.join(cfg.out_dir, f"eval_{args.split}.json"), "w") as f:
                json.dump(res, f, indent=2, default=float)
        if world_size>1:dist.destroy_process_group()
        return

    trainer.train()
    final = trainer.evaluate(split="val")
    if trainer.is_main:
        print(json.dumps(final, indent=2, default=float))
        with open(os.path.join(cfg.out_dir,'final_val.json'),'w') as f:json.dump(final,f,indent=2)
    if world_size>1:dist.destroy_process_group()


if __name__ == "__main__":
    main()
