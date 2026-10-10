#!/usr/bin/env python
"""Persistently orchestrate full stage1 -> stage2 DDP training, with status JSON.

Run inside tmux using the dedicated PGSF-Det virtual environment. Stage2 starts
from stage1's best validation-selected EMA. Existing last.pt checkpoints resume
their optimizer, AMP, epoch and early-stopping state on an explicit restart.
"""
import argparse,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default='outputs/fulltrain_repaired_20261010')
    p.add_argument('--gpus',default='auto')
    p.add_argument('--gpu-max-used-mib',type=int,default=512)
    p.add_argument('--gpu-min-free-mib',type=int,default=9000)
    p.add_argument('--config-prefix',default='configs/pgsf_repaired')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    run=(root/a.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    os.chdir(root)
    import fcntl
    run_lock=(run/'launcher.lock').open('w')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A launcher already owns this run directory')
    from pgsf.utils.gpu_select import select_gpus
    selected,inventory=select_gpus(a.gpus,a.gpu_max_used_mib,a.gpu_min_free_mib)
    gpus=[g['index'] for g in selected]
    from pgsf.config import load_config
    configs=[load_config(f'{a.config_prefix}_stage{s}.yaml') for s in (1,2)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(root))
    for c in configs:c.optim.batch_size=8
    print(f'[GPU auto-select] GPUs={gpus}; batch=8/GPU; global batch={8*len(gpus)}',flush=True)
    status={'pid':os.getpid(),'python':sys.executable,'physical_gpus':gpus,'gpu_inventory':inventory,
            'epoch_budget':sum(c.optim.epochs for c in configs),'stages':[c.optim.epochs for c in configs],
            'early_stop_patience':[c.optim.early_stop_patience for c in configs],
            'per_gpu_batch':configs[0].optim.batch_size,'global_batch':configs[0].optim.batch_size*len(gpus),
            'started':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    latest=root/'outputs/latest_training_run.json'
    latest.write_text(json.dumps({'run_dir':str(run)},indent=2))
    def write(**update):
        status.update(update);status['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        temp=run/'status.json.tmp';temp.write_text(json.dumps(status,indent=2))
        temp.replace(run/'status.json')
    for stage in (1,2):
        directory=run/f'stage{stage}'
        if (directory/'completed.json').exists():continue
        cfg=root/f'{a.config_prefix}_stage{stage}.yaml'
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',
                 str(root/'scripts/train.py'),'--config',str(cfg),'--out-dir',str(directory),
                 '--set','optim.batch_size=8']
        checkpoint=directory/'ckpt/last.pt'
        if checkpoint.exists():command+=['--resume',str(checkpoint)]
        elif stage==2:
            checkpoint=run/'stage1/ckpt/best.pt'
            if not checkpoint.exists():
                write(state='failed',stage=stage,error='stage1 best.pt is missing');raise SystemExit(1)
            command+=['--ckpt',str(checkpoint)]
        print(f'Launching stage{stage}: '+ ' '.join(command),flush=True)
        write(state='running',stage=stage,command=command)
        with (run/f'stage{stage}_console.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root)
            write(child_pid=child.pid)
            code=child.wait()
        if code:
            write(state='failed',stage=stage,exit_code=code);raise SystemExit(code)
        completion={'finished':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    'best':json.loads((directory/'best_metrics.json').read_text())}
        (directory/'completed.json').write_text(json.dumps(completion,indent=2))
    write(state='complete',stage=2,child_pid=None)
    print('Full training completed. Best checkpoints and validation metrics saved.',flush=True)


if __name__=='__main__':main()
