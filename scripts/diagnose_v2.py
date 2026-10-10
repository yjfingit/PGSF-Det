#!/usr/bin/env python
"""Inspect candidate coverage, coarse boxes and final localization on training frames."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from pgsf.config import load_config
from pgsf.models.detector import PGSFDet
from pgsf.models.proposal import aligned_iou
from pgsf.models.head import decode_pred
from pgsf.data import UAVCBDataset,load_split_manifest,collate
from pgsf.data.uavcb import make_augment


def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True)
    p.add_argument('--config',default='configs/pgsf_stage2.yaml');p.add_argument('--n',type=int,default=4)
    a=p.parse_args();cfg=load_config(a.config)
    m=PGSFDet(cfg).cuda().eval();ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    m.load_checkpoint_weights(ck.get('ema') or ck['model'])
    frames=load_split_manifest(cfg.data.split_dir,cfg.data.root,cfg.data.scenes)['train']
    by={}
    for f in frames:by.setdefault(f.scene,[]).append(f)
    selected=[by[sorted(by)[i%len(by)]][i//len(by)] for i in range(a.n)]
    ds=UAVCBDataset(selected,cfg.data.rgb_size,cfg.data.tir_size,train=False,augment=make_augment(cfg.data,False))
    report=[]
    for i in range(len(ds)):
        b=collate([ds[i]]);b={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
        with torch.no_grad():
            out=m(b)
            for name in ('rgb','tir'):
                cand=out['cand_'+name];sel=out['sel_'+name];gt=b['boxes_'+name][:,:1]
                decoded,logits=decode_pred(out['pred_'+name],sel['pts'],cfg.model.max_wh,sel['boxes'])
                coarse_iou=aligned_iou(cand['boxes'],gt)
                final_iou=aligned_iou(decoded,gt)
                score=logits.sigmoid()*sel['objectness'].clamp_min(1e-8).pow(cfg.model.score_fusion_power)
                top=score.argmax(1)
                near=(cand['pts']-gt[...,:2]).norm(dim=-1).min()
                report.append({'frame':selected[i].scene,'modality':name,'nearest_candidate':float(near),
                               'best_coarse_iou':float(coarse_iou.max()),'best_final_iou':float(final_iou.max()),
                               'top_score_iou':float(final_iou[0,top[0]]),'max_score':float(score.max()),
                               'best_iou_score':float(score[0,final_iou.argmax()]),
                               'gt_wh':gt[0,0,2:].tolist()})
    print(json.dumps({'checkpoint_step':ck.get('step'),'frames':report},indent=2))

if __name__=='__main__':main()
