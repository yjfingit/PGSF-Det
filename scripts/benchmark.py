#!/usr/bin/env python
"""Batch-1 dual-modality FP16 latency, including decode/NMS and GPU preprocessing.

Disk/JPEG decoding is excluded from pipeline latency; decoded uint8 source
frames start in CPU memory. GPU resize uses bilinear antialiasing. This is a
speed measurement, not an AP experiment. Run with CUDA_VISIBLE_DEVICES=7.
"""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from pgsf.config import load_config
from pgsf.models.detector import PGSFDet
from pgsf.utils.deploy import fuse_for_inference, CudaGraphForward
from pgsf.utils.profiler import count_params,count_flops


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pgsf_stage2.yaml');p.add_argument('--ckpt')
    p.add_argument('--iters',type=int,default=50);p.add_argument('--warmup',type=int,default=15)
    acceleration=p.add_mutually_exclusive_group()
    acceleration.add_argument('--compile',action='store_true')
    acceleration.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--set',nargs='*',default=[])
    p.add_argument('--out',default='outputs/v2_validation/benchmark.json');a=p.parse_args()
    cfg=load_config(a.config,a.set)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the GPU latency benchmark')
    torch.manual_seed(0);torch.backends.cudnn.benchmark=True
    model=PGSFDet(cfg).cuda().eval()
    if a.ckpt:
        ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
        model.load_checkpoint_weights(ck.get('ema') or ck['model'])
    from pgsf.data import load_split_manifest
    frame=load_split_manifest(cfg.data.split_dir,cfg.data.root,cfg.data.scenes)['train'][0]
    sources={name:torch.from_numpy(np.array(Image.open(path).convert(mode),copy=True))
             for name,path,mode in [('rgb',frame.rgb_path,'RGB'),('tir',frame.tir_path,'L')]}
    def preprocess():
        batch={}
        for name in ('rgb','tir'):
            src=sources[name].to('cuda',non_blocking=True)
            image=src.permute(2,0,1)[None] if name=='rgb' else src[None,None]
            image=torch.nn.functional.interpolate(image.float()/255.,size=getattr(cfg.data,name+'_size'),
                                                  mode='bilinear',align_corners=False,antialias=True)
            if name=='rgb' and cfg.data.rgb_normalization=='imagenet':
                image=(image-image.new_tensor([.485,.456,.406])[None,:,None,None])/image.new_tensor([.229,.224,.225])[None,:,None,None]
            batch[name]=image
            batch[name+'_hw']=torch.tensor([[src.shape[0],src.shape[1]]],device='cuda',dtype=torch.float32)
        return batch
    batch=preprocess()
    macs=count_flops(model,batch);params=count_params(model)
    model=fuse_for_inference(model)
    if a.compile:model.forward=torch.compile(model.forward,mode='reduce-overhead')
    if a.cuda_graph:model.forward=CudaGraphForward(model,batch,decode=True)
    def run_forward():return model(batch)
    def run_decode():return model.inference(batch,score_thr=cfg.eval.conf_thr,nms_thr=cfg.eval.nms_thr)
    def run_pipeline():return model.inference(preprocess(),score_thr=cfg.eval.conf_thr,nms_thr=cfg.eval.nms_thr)
    def measure(fn):
        values=[]
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            for _ in range(a.warmup):fn()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for _ in range(a.iters):
                start=time.perf_counter();fn();torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000)
        mean=float(np.mean(values))
        return {'mean_ms':mean,'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
                'fps_pairs':1000/mean,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    result={'gpu':torch.cuda.get_device_name(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'input_hw':{'rgb':cfg.data.rgb_size,'tir':cfg.data.tir_size},'batch_size':1,'precision':'FP16 autocast',
            'params':params['total'],'conv_linear_macs':macs,'fused_conv_bn':True,
            'compiled_forward':a.compile,'cuda_graph':a.cuda_graph,'graph_includes_fixed_decode':a.cuda_graph,
            'forward_only':measure(run_forward),'forward_decode_nms':measure(run_decode),
            'gpu_preprocess_forward_decode_nms':measure(run_pipeline),
            'notes':'Decoded CPU source images; excludes disk/JPEG decoding. CUDA graph forward_only includes fixed-shape box/score decode, excludes NMS. No AP claim; GPU resize differs slightly from PIL training resize.'}
    print(json.dumps(result,indent=2));Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
