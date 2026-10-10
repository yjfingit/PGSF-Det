#!/usr/bin/env python
"""Real full-resolution two-GPU integration: train, validate, transfer and resume.

CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.run --standalone \
    --nproc_per_node=2 scripts/check_distributed.py
"""
import argparse,gc,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from pgsf.config import load_config
from pgsf.engine.trainer import PGSFTrainer


def make_trainer(stage,device,args):
    cfg=load_config(f'configs/pgsf_stage{stage}.yaml')
    cfg.optim.epochs=1;cfg.optim.batch_size=args.batch;cfg.optim.early_stop_patience=30
    cfg.optim.log_interval=1;cfg.data.num_workers=0
    cfg.data.zoom_prob=.5;cfg.data.scale_range=(.9,1.12)
    cfg.out_dir=f'{args.out_dir}/ddp_stage{stage}'
    t=PGSFTrainer(cfg,device=device)
    count=args.batch*dist.get_world_size()*2
    t.loader=t._loader(Subset(t.datasets['train'],range(count)),True,args.batch,0)
    t.val_loader=t._loader(Subset(t.datasets['val'],range(3)),False,4,0)
    return t


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=6)
    p.add_argument('--out-dir',default='outputs/fulltrain_preparation')
    args=p.parse_args()
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    result={'world_size':dist.get_world_size(),'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'train_frames':args.batch*dist.get_world_size()*2,'val_frames':3,'kind':'integration only, not performance experiment'}
    for stage in (1,2):
        t=make_trainer(stage,device,args)
        if stage==2:t.load_weights(f'{args.out_dir}/ddp_stage1/ckpt/last.pt')
        t.train()
        assert t.n_skipped==0
        assert t.best_metrics['rgb']['n_gt']==3,t.best_metrics
        assert t.best_metrics['tir']['n_gt']==3,t.best_metrics
        checksum=torch.stack([p.detach().double().sum() for p in t.model.parameters()])
        copies=[torch.empty_like(checksum) for _ in range(dist.get_world_size())]
        dist.all_gather(copies,checksum)
        assert all(torch.equal(copies[0],v) for v in copies),'DDP parameters diverged'
        path=t.save('last.pt');dist.barrier();t.resume(path)
        assert t.start_epoch==1
        assert t.early_stopping.best==t.best_metrics['AP50:95']
        result[f'stage{stage}']={'skipped':t.n_skipped,'peak_gpu_mib':t.history[-1]['peak_gpu_mib'],
                                 'validation_gt_per_modality':3,'parameters_synchronized':True}
        del t;gc.collect();torch.cuda.empty_cache();dist.barrier()
    if dist.get_rank()==0:
        Path(f'{args.out_dir}/ddp_smoke.json').write_text(json.dumps(result,indent=2))
        print('DDP SMOKE PASSED',json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
