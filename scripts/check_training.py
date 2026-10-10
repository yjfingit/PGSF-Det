#!/usr/bin/env python
"""Two real-frame integration smoke: both stages, validation, save and resume.

Run from the project root: python scripts/check_training.py
Uses temporary checkpoints. This is not a performance or overfit experiment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import json
import tempfile
import torch
from torch.utils.data import Subset
from pgsf.config import load_config
from pgsf.engine.trainer import PGSFTrainer

def setup(stage, out):
    cfg=load_config(f'configs/pgsf_stage{stage}.yaml')
    cfg.stage=stage;cfg.out_dir=out
    cfg.data.max_train_per_scene=2;cfg.data.num_workers=0
    cfg.data.rgb_size=[768,1280];cfg.data.hflip=1.
    cfg.optim.batch_size=1;cfg.optim.epochs=1;cfg.optim.lr=1e-4
    trainer=PGSFTrainer(cfg,device=torch.device('cuda'))
    # Only two real training frames and two validation frames for integration.
    trainer.loader=trainer._loader(Subset(trainer.datasets['train'],[0,1]),True,1,0)
    trainer.val_loader=trainer._loader(Subset(trainer.datasets['val'],[0,1]),False,1,0)
    trainer.test_loader=[]
    return trainer

def main():
    results={}
    Path('outputs/v2_validation').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pgsf-smoke-') as d:
        stage1=setup(1,d+'/stage1')
        A=stage1.model.affine[0].clone()
        assert torch.equal(A,stage1.ema.module.affine[0])
        stage1.train()
        path=stage1.save('last.pt')
        results['stage1']={'epoch':stage1.current_epoch,'skipped':stage1.n_skipped,
                           'last_loss':stage1.history[-1]['total']}
        del stage1;gc.collect();torch.cuda.empty_cache()
        stage2=setup(2,d+'/stage2')
        stage2.load_weights(path)
        assert stage2.start_epoch==0
        assert torch.allclose(stage2.model.affine[0],A)
        stage2.train()
        stage2.save('last.pt')
        stage2.resume(d+'/stage2/ckpt/last.pt')
        assert stage2.start_epoch==1
        results['stage2']={'epoch':stage2.current_epoch,'skipped':stage2.n_skipped,
                           'last_loss':stage2.history[-1]['total']}
    print('SMOKE PASSED',results)
    with open('outputs/v2_validation/real_smoke.json','w') as f:
        json.dump({'kind':'integration smoke, not an overfit or performance experiment',
                   'train_frames':2,'val_frames':2,'stages':results},f,indent=2)


if __name__ == "__main__":
    main()
