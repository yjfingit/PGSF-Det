"""P2/P3 dense proposals, quality ranking and fixed-budget sparse candidates."""
from __future__ import annotations
import math
import torch
from torch import nn
from pgsf.models.backbone import DWConvBNAct
from pgsf.models.candidate import bilinear_grid, sample_points, local_nms_topk, CandidateGenerator


def decode_coarse(raw, pts, step, max_wh=.5):
    """One sub-cell center offset and positive size, in normalized coordinates."""
    center = pts + raw[...,1:3].tanh() * step * .5
    size = raw[...,3:5].sigmoid() * max_wh
    return torch.cat((center,size),-1)


class DenseProposalHead(nn.Module):
    def __init__(self, channels=128, hidden=32, max_wh=.5):
        super().__init__()
        self.max_wh = max_wh
        def head():
            h = nn.Sequential(DWConvBNAct(channels,hidden),
                              DWConvBNAct(hidden,hidden),nn.Conv2d(hidden,6,1))
            nn.init.normal_(h[-1].weight,std=.01)
            nn.init.zeros_(h[-1].bias)
            with torch.no_grad():
                h[-1].bias[0] = -4.
                h[-1].bias[3:5] = math.log(.03/max_wh/(1-.03/max_wh))
            return h
        self.p2 = head(); self.p3 = head()

    def forward(self, p2, p3):
        return {"P2":self.p2(p2),"P3":self.p3(p3)}


class MultiScaleCandidates(nn.Module):
    """P2/P3 peaks plus spatial supplements; no GT access on inference paths."""
    def __init__(self, channels=128, n_cand=64, inject_gt_ratio=.25, max_wh=.5,
                 nms_kernel=5):
        super().__init__()
        if n_cand < 16:
            raise ValueError("multiscale candidate budget must be at least 16")
        self.n_cand=n_cand;self.inject_gt_ratio=inject_gt_ratio
        self.max_wh=max_wh;self.nms_kernel=nms_kernel
        self.partitioner=CandidateGenerator(channels,n_cand=64,n_global=48,
                                             inject_gt_ratio=0.)
        self.feat_proj=nn.Sequential(nn.LayerNorm(2*channels),nn.Linear(2*channels,channels),nn.SiLU())

    def forward(self, dense, maps, gt_centers=None, gt_mask=None, training=False):
        p2,p3=dense['P2'],dense['P3']
        b,_,h,w=p2.shape
        points=[];scores=[];levels=[]
        for lv,name in enumerate(('P2','P3')):
            raw=dense[name];hh,ww=raw.shape[-2:]
            rank=raw[:,:1].sigmoid()
            idx,sc=local_nms_topk(rank,self.n_cand,self.nms_kernel)
            points.append(bilinear_grid(hh,ww,raw.device,torch.float32)[idx])
            scores.append(sc);levels.append(torch.full_like(idx,lv))
        rank2=p2[:,:1].sigmoid()
        ip,sp=self.partitioner._partition_topk(rank2,1)
        pp=bilinear_grid(h,w,p2.device,torch.float32)[ip]
        # Preserve global and partition quotas before filling removed duplicates.
        n3=(self.n_cand-16)//3;n2=self.n_cand-16-n3
        pts=torch.cat((points[0][:,:n2],points[1][:,:n3],pp,
                       points[0][:,n2:],points[1][:,n3:]),1)
        sc=torch.cat((scores[0][:,:n2],scores[1][:,:n3],sp,
                      scores[0][:,n2:],scores[1][:,n3:]),1)
        lv=torch.cat((levels[0][:,:n2],levels[1][:,:n3],torch.zeros_like(ip),
                      levels[0][:,n2:],levels[1][:,n3:]),1)
        # Vectorized cross-level deduplication in P2 cells, with constant output K.
        xy=torch.stack((pts[...,0].float()*(w-1),pts[...,1].float()*(h-1)),-1).round().long()
        ids=xy[...,1]*w+xy[...,0]
        prior=torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).tril(-1)
        duplicate=((ids.unsqueeze(2)==ids.unsqueeze(1))&prior).any(-1)
        priority=-torch.arange(ids.shape[1],device=ids.device,dtype=torch.float32)[None].expand(b,-1)
        priority=priority.masked_fill(duplicate,-1e6)
        idx=priority.topk(self.n_cand,dim=1).indices
        pts=pts.gather(1,idx[...,None].expand(-1,-1,2))
        lv=lv.gather(1,idx);sc=sc.gather(1,idx)
        injected=torch.zeros_like(sc)
        if training and gt_centers is not None and self.inject_gt_ratio>0:
            pts,lv,injected=self._inject(pts,lv,sc,gt_centers,gt_mask)
        raw2=sample_points(p2,pts);raw3=sample_points(p3,pts)
        raw=torch.where((lv==0)[...,None],raw2,raw3)
        step2=torch.stack((torch.full_like(pts[...,0],1/max(w-1,1)),
                           torch.full_like(pts[...,1],1/max(h-1,1))),-1)
        step3=torch.stack((torch.full_like(pts[...,0],1/max(p3.shape[-1]-1,1)),
                           torch.full_like(pts[...,1],1/max(p3.shape[-2]-1,1))),-1)
        steps=torch.where((lv==0)[...,None],step2,step3)
        box=decode_coarse(raw,pts,steps,self.max_wh)
        quality=raw[...,5].sigmoid()
        feats=self.feat_proj(torch.cat((sample_points(maps['P2'],pts),sample_points(maps['P3'],pts)),-1))
        peak_or_injected=(sc>0)|(injected>.5)
        # Candidate recall is governed by the dense center heatmap. The learned
        # box-quality head is intentionally excluded from proposal ranking: early
        # low-IoU quality estimates otherwise hide true centers, then receive no
        # positive supervision and create a self-reinforcing collapse.
        objectness=raw[...,0].sigmoid()*peak_or_injected
        return {'pts':pts,'feats':feats,'scores':objectness,
                'quality':quality,'objectness':raw[...,0].sigmoid(),'quality_logits':raw[...,5],'boxes':box,
                'injected':injected,'levels':lv,'heat':p2[:,:1],
                'heat_size':(h,w),'dense':dense}

    def _inject(self, pts, levels, scores, gt, mask):
        pts=pts.clone();levels=levels.clone();flags=torch.zeros_like(scores)
        if mask is None:mask=gt.abs().sum(-1)>0
        n=min(self.n_cand-1,max(1,round(self.n_cand*self.inject_gt_ratio)))
        for bi in range(pts.shape[0]):
            valid=gt[bi][mask[bi]>.5]
            if valid.numel()==0:continue
            target=valid[torch.arange(n,device=gt.device)%len(valid)].clone()
            exact=min(n,len(valid))
            target[exact:]+=torch.randn_like(target[exact:])*.005
            replace=scores[bi].argsort()[:n]
            pts[bi,replace]=target.clamp(0,1).to(pts.dtype)
            levels[bi,replace]=0;flags[bi,replace]=1
        return pts,levels,flags


def aligned_iou(a,b):
    a=a.float();b=b.float()
    lo=torch.maximum(a[...,:2]-a[...,2:]/2,b[...,:2]-b[...,2:]/2)
    hi=torch.minimum(a[...,:2]+a[...,2:]/2,b[...,:2]+b[...,2:]/2)
    inter=(hi-lo).clamp_min(0).prod(-1)
    return inter/(a[...,2:].prod(-1)+b[...,2:].prod(-1)-inter).clamp_min(1e-8)


def proposal_losses(cand, boxes, mask, use_nwd=True, max_wh=.5):
    """Direct coarse-box supervision at quantized GT centers on BOTH scales.

    Quality is supervised on actual candidate boxes, including hard background.
    Heat supervision is added separately by the detector.
    """
    from pgsf.losses import ciou_loss,nwd_loss,assign_candidates
    losses=[]
    for raw in cand['dense'].values():
        h,w=raw.shape[-2:]
        step=raw.new_tensor([1/max(w-1,1),1/max(h-1,1)])
        center=(boxes[...,:2]/step).round()*step
        sampled=sample_points(raw,center)
        predicted=decode_coarse(sampled,center,step,max_wh)
        valid=mask>.5
        if valid.any():
            l=ciou_loss(predicted[valid],boxes[valid]).mean()
            if use_nwd:l=l+.5*nwd_loss(predicted[valid],boxes[valid]).mean()
            losses.append(l)
        else:losses.append(predicted.sum()*0.)
    pos,tgt=assign_candidates(cand['pts'],boxes,mask,topk=0)
    target=aligned_iou(cand['boxes'].detach(),tgt).detach()*pos
    quality=torch.nn.functional.binary_cross_entropy_with_logits(cand['quality_logits'].float(),target)
    return {'box':torch.stack(losses).mean(),'quality':quality}
