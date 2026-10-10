#!/usr/bin/env python
"""Choose per-GPU batch by measured full-model DDP samples/sec.

This intentionally measures real forward, AMP backward, gradient reduction and
the configured DataLoader on 48 full-resolution training frames. It skips
validation/checkpointing so candidates can be compared on equal work.
"""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from pgsf.config import load_config
from pgsf.engine.trainer import PGSFTrainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True)
    p.add_argument('--frames',type=int,default=240);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--warmup',type=int,default=3)
    a=p.parse_args();local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    c=load_config('configs/pgsf_full200_stage2.yaml');c.optim.batch_size=a.batch;c.data.num_workers=4
    c.optim.log_interval=0;c.data.close_aug_epochs=0
    c.out_dir=f'outputs/fulltrain_preparation/batch_bench_{a.batch}'
    t=PGSFTrainer(c,device=device)
    t.loader=t._loader(Subset(t.datasets['train'],range(a.frames)),True,a.batch,c.data.num_workers)
    t.train_model.train();t.model.train()
    opt=t.optimizer
    times=[];iterator=iter(t.loader);samples=0
    torch.cuda.reset_peak_memory_stats(device)
    for index in range(a.warmup+a.steps):
        torch.cuda.synchronize(device);start=time.perf_counter()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(t.loader);batch=next(iterator)
        batch={k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss,_=t.train_model(batch)
        t.scaler.scale(loss).backward();t.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(t.model.parameters(),c.optim.grad_clip)
        old_scale=t.scaler.get_scale();t.scaler.step(opt);t.scaler.update()
        if t.scaler.get_scale()<old_scale:raise RuntimeError('AMP skipped update in batch benchmark')
        t.ema.update(t.model,index)
        torch.cuda.synchronize(device)
        if index>=a.warmup:times.append(time.perf_counter()-start);samples+=batch['rgb'].shape[0]
    local_stats=torch.tensor([sum(times),len(times),torch.cuda.max_memory_allocated(device)/2**30,samples],device=device,dtype=torch.float64)
    gathered=[torch.zeros_like(local_stats) for _ in range(dist.get_world_size())];dist.all_gather(gathered,local_stats)
    if dist.get_rank()==0:
        # All ranks have the same number of steps; aggregate elapsed work by max rank time.
        elapsed=max(float(x[0]) for x in gathered);n=int(min(float(x[1]) for x in gathered))
        result={'batch_per_gpu':a.batch,'global_batch':a.batch*dist.get_world_size(),'frames':a.frames,
                'measured_steps':n,'warmup_steps':a.warmup,'includes':'data wait, augmentation, H2D, AMP forward/backward, DDP, clip, AdamW, EMA',
                'step_ms':elapsed/n*1000,'global_samples_per_sec':sum(float(x[3]) for x in gathered)/elapsed,
                'max_peak_gpu_gib':max(float(x[2]) for x in gathered),'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}
        Path('outputs/fulltrain_preparation').mkdir(exist_ok=True);p=Path(f'outputs/fulltrain_preparation/batch_{a.batch}.json');p.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
