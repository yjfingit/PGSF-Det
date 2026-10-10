"""PGSF-Det v2 assembly, forward and explicitly supervised objectives.

Keeps the public PGSFDet API and the v1 path for controlled comparisons.
"""
import math
import torch
from torch import nn
from pgsf.models.proposal import DenseProposalHead,MultiScaleCandidates,proposal_losses
from pgsf.models.head import MultiScaleBoxRefine,DetHead
from pgsf.models.msce import LiteMSCE
from pgsf.models.neck import PANFPN
from pgsf.models.fusion_v2 import CovarianceMidFusion,RegionLateFusion
from pgsf.models.late_fusion import select_topk_candidates
from pgsf.losses import (DetectionLoss,draw_gaussian,gaussian_focal_loss,geometry_nll,
                         assign_candidates,correspondence_mask,supervised_association_loss)


class RegionBranch(nn.Module):
    def __init__(self,in_ch,cfg):
        super().__init__()
        from pgsf.models.detector import LightBackboneCtor
        m=cfg.model;self.backbone=LightBackboneCtor(in_ch,m)
        self.msce_on=m.use_msce
        if self.msce_on:self.msce=LiteMSCE(self.backbone.out_channels[0],m.msce_width,m.msce_alpha_max)
        self.neck=PANFPN(self.backbone.out_channels,m.neck_dim,m.neck_out,m.neck_repeats)
        self.heat=DenseProposalHead(m.neck_out,m.proposal_width,m.max_wh)
        self.cand=MultiScaleCandidates(m.neck_out,m.n_cand,m.inject_gt_ratio,m.max_wh,m.nms_kernel)
        self.roi_on=m.use_roi_tokens
        if self.roi_on:
            self.refine=MultiScaleBoxRefine(m.neck_out,n_levels=4,hidden=64)
            self.roi_project=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,m.neck_out))
            self.box_position=nn.Sequential(nn.Linear(4,m.neck_out),nn.SiLU(),nn.Linear(m.neck_out,m.neck_out))
            nn.init.normal_(self.roi_project[-1].weight,std=.01)
            nn.init.zeros_(self.roi_project[-1].bias)
            nn.init.normal_(self.box_position[-1].weight,std=.01)
            nn.init.zeros_(self.box_position[-1].bias)
        self.det=DetHead(m.neck_out,m.head_width,m.n_layers_head,max_wh=m.max_wh)
        # Relative width/height residuals start at zero, not the absolute-size bias.
        with torch.no_grad():self.det.out.bias[:4].zero_()

    def features(self,x):
        f=self.backbone(x)
        if self.msce_on:f['P2'],_=self.msce(f['P2'])
        return self.neck(f)


def build_modules(model,cfg):
    m=cfg.model
    model.rgb_branch=RegionBranch(3,cfg);model.tir_branch=RegionBranch(1,cfg)
    model.mid_on=m.use_mid_fusion and cfg.stage>=2
    model.late_on=m.use_late_fusion and cfg.stage>=2
    if (model.mid_on or model.late_on) and m.use_geo_prior and m.use_image_context:
        model.geometry_context=nn.Sequential(nn.LayerNorm(2*m.neck_out),nn.Linear(2*m.neck_out,m.context_dim),
                                             nn.SiLU(),nn.Linear(m.context_dim,m.context_dim))
    if model.mid_on:
        model.mid=CovarianceMidFusion(m.neck_out,m.n_heads,m.n_local_samples,m.n_global,
                                      m.context_dim,m.use_geo_prior,m.use_gate,m.use_null,
                                      m.use_covariance_sampling,m.lambda_geo,m.ff_mult)
    if model.late_on:
        model.late=RegionLateFusion(m.neck_out,m.n_heads,m.context_dim,m.use_geo_prior,
                                    m.use_gate,m.use_null,m.lambda_box,m.ff_mult,
                                    standalone_geometry=not model.mid_on)
    model.det_loss_rgb=DetectionLoss(m.w_cls,m.w_box,m.w_nwd,m.use_nwd,m.assign_pos_thr,m.assign_center_thr,m.max_wh)
    model.det_loss_tir=DetectionLoss(m.w_cls,m.w_box,m.w_nwd,m.use_nwd,m.assign_pos_thr,m.assign_center_thr,m.max_wh)
    model.n_cand_late=m.n_cand_late


def mask_for(batch,name):
    mask=batch.get('mask_'+name,batch['mask'])
    available=batch.get('modality_available')
    if available is not None:mask=mask*available[:,0 if name=='rgb' else 1,None]
    return mask


def paired_geometry_mask(batch):
    mask=mask_for(batch,'rgb')*mask_for(batch,'tir')
    if 'geometry_valid' in batch:
        # Clipped boxes remain useful for detection, but their shifted centers
        # no longer follow the known image transform exactly.
        mask=mask*batch['geometry_valid'].prod(-1)[:,None]
    return mask


def pair_affines(model,batch):
    """Known augmentation only; inference without metadata uses cached transforms."""
    A,Ar=model._affine_A,model._affine_reverse
    if 'geo_transform_rgb' in batch:
        T_r=batch['geo_transform_rgb'];T_t=batch['geo_transform_tir']
        inv_r=batch['geo_inverse_rgb'];inv_t=batch['geo_inverse_tir']
    elif 'geo_transform' in batch:
        T_r=T_t=batch['geo_transform']
        # Legacy metadata used a self-inverse horizontal flip.
        inv_r=inv_t=T_r
    else:return A,Ar
    bottom=A.new_tensor([[0.,0.,1.]])
    with torch.autocast(device_type=A.device.type,enabled=False):
        H=torch.cat((A,bottom),0).float();Hr=torch.cat((Ar,bottom),0).float()
        return (T_t.float()@H@inv_r.float())[:,:2],(T_r.float()@Hr@inv_t.float())[:,:2]


def global_candidates(cand,n):
    rank=cand['scores'].detach().masked_fill(cand['injected']>.5,-1.)
    idx=rank.topk(min(n,rank.shape[1]),1).indices
    return (cand['feats'].gather(1,idx[...,None].expand(-1,-1,cand['feats'].shape[-1])),
            cand['pts'].gather(1,idx[...,None].expand(-1,-1,2)),cand['quality'].gather(1,idx))


def select_regions(branch,maps,cand,k):
    selected=select_topk_candidates(cand,cand['boxes'],k)
    if branch.roi_on:
        # Sample from the coarse region's center and scale, rather than fixed .05 boxes.
        ctx=branch.refine(maps,selected['boxes'][...,:2],selected['boxes'])
        position=torch.cat((selected['boxes'][...,:2],selected['boxes'][...,2:].clamp_min(1e-5).log()),-1)
        selected['feats']=selected['feats']+branch.roi_project(ctx)+branch.box_position(position)
        selected['roi']=ctx
    return selected


def forward(model,batch):
    cfg=model.cfg;m=cfg.model;features={};candidates={};dense={}
    for name in ('rgb','tir'):
        branch=getattr(model,name+'_branch')
        features[name]=branch.features(batch[name])
        dense[name]=branch.heat(features[name]['P2'],features[name]['P3'])
        mask=mask_for(batch,name) if 'mask' in batch else None
        candidates[name]=branch.cand(dense[name],features[name],batch.get('centers_'+name),mask,model.training)
    pooled=torch.cat((features['rgb']['P5'].mean((-1,-2)),features['tir']['P5'].mean((-1,-2))),-1)
    context=model.geometry_context(pooled) if hasattr(model,'geometry_context') else pooled.new_zeros(pooled.shape[0],m.context_dim)
    A,Ar=pair_affines(model,batch);aux={};distributions={}
    if model.mid_on:
        # Both directions read the original maps/candidates before either writeback.
        globals={name:global_candidates(candidates[name],m.n_global) for name in ('rgb','tir')}
        results={}
        for name,other,direction,affine in [('rgb','tir','r2t',A),('tir','rgb','t2r',Ar)]:
            c=candidates[name]
            jitter=None
            if model.training and m.use_geo_prior and m.geo_jitter_prob>0:
                active=(torch.rand(c['pts'].shape[0],1,1,device=c['pts'].device)<m.geo_jitter_prob)
                jitter=torch.randn(c['pts'].shape[0],1,2,device=c['pts'].device)*m.geo_jitter_std*active
            if 'geometry_jitter_'+name in batch:jitter=batch['geometry_jitter_'+name][:,None]
            results[name]=model.mid(c['feats'],c['pts'],features[other],globals[other],direction,
                                    {k:features[name][k] for k in ('P3','P4')},affine,context,
                                    c['boxes'],c['quality'],dense[other],jitter)
        for name in ('rgb','tir'):
            candidates[name]['feats']=results[name]['feats']
            features[name].update(results[name]['maps'])
            distributions[name]=results[name]['stats']
            aux['mid_stats_'+name]=results[name]['stats']
    selected={name:select_regions(getattr(model,name+'_branch'),features[name],candidates[name],m.n_cand_late)
              for name in ('rgb','tir')}
    if model.late_on:
        results={}
        for name,other,direction,affine in [('rgb','tir','r2t',A),('tir','rgb','t2r',Ar)]:
            sel=selected[name];key=selected[other];distribution=None
            if name in distributions and m.use_geo_prior:
                idx=sel['idx'];stats=distributions[name]
                mu=stats['mu'].gather(1,idx[...,None].expand(-1,-1,2))
                L=stats['sigma'].gather(1,idx[...,None,None].expand(-1,-1,2,2))
                conf=stats['conf'].gather(1,idx)
                distribution=(mu,L,conf)
            results[name]=model.late(sel['feats'],sel['boxes'],key['feats'],key['boxes'],affine,context,
                                     direction,distribution,sel['quality'])
        for name in ('rgb','tir'):
            selected[name]['feats']=results[name]['feats'];aux['late_'+name]=results[name]
    out={'batch':batch,'aux':aux,'context':context}
    for name in ('rgb','tir'):
        out['pred_'+name]=getattr(model,name+'_branch').det(selected[name]['feats'])
        out['cand_'+name]=candidates[name];out['sel_'+name]=selected[name]
        out['heat_'+name]=dense[name]['P2'][:,:1]
        out['dense_'+name]=dense[name];aux['pre_box_'+name]=candidates[name]['boxes']
    return out


def association(model,attn,qpts,kpts,batch,qname,kname,scores):
    mask=mask_for(batch,qname)*mask_for(batch,kname)
    valid=mask.any(-1).to(mask.dtype)
    matches=correspondence_mask(qpts,kpts,batch['boxes_'+qname],batch['boxes_'+kname],mask,valid,
                                 model.cfg.model.assign_center_thr)
    weights=1+model.cfg.model.hard_negative_weight*scores.detach()
    return supervised_association_loss(attn,matches,model.cfg.model.use_null,weights)


def compute_losses(model,out):
    from pgsf.models.detector import matched_gt_index,gather_by_index
    m=model.cfg.model;b=out['batch'];terms={};total=out['pred_rgb'].sum()*0.
    for name in ('rgb','tir'):
        mask=mask_for(b,name);sel=out['sel_'+name];cand=out['cand_'+name]
        raw=out['pred_'+name];boxes=b['boxes_'+name]
        centers=b['centers_'+name];sizes=b['sizes_'+name];dense=out['dense_'+name]
        available=b.get('modality_available')
        if available is not None:
            active=available[:,0 if name=='rgb' else 1]>.5
            if not active.any():
                # A dropped sensor is not a labeled background-only image.
                zero=raw.sum()*0.
                for key in ('loss_cls','loss_box','loss_nwd','loss','pos_ratio','pre_box','pre_quality','heat'):
                    terms[name+'_'+key]=zero.detach()
                continue
            raw=raw[active];mask=mask[active];boxes=boxes[active]
            centers=centers[active];sizes=sizes[active]
            sel={k:v[active] if isinstance(v,torch.Tensor) else v for k,v in sel.items()}
            dense={k:v[active] for k,v in dense.items()}
            cand={k:cand[k][active] for k in ('pts','boxes','quality_logits')}
            cand['dense']=dense
        det=getattr(model,'det_loss_'+name)(raw,sel['pts'],boxes,mask,
                                          base_boxes=sel['boxes'],quality_aware=True)
        total=total+m.w_det*det['loss']
        for k,v in det.items():terms[name+'_'+k]=v.detach()
        pre=proposal_losses(cand,boxes,mask,m.use_nwd,m.max_wh)
        total=total+m.w_pre_box*pre['box']+m.w_pre_quality*pre['quality']
        terms[name+'_pre_box']=pre['box'].detach();terms[name+'_pre_quality']=pre['quality'].detach()
        heat=total.new_zeros(())
        for raw in dense.values():
            target=draw_gaussian(torch.zeros_like(raw[:,:1]),centers,sizes,mask)
            heat=heat+gaussian_focal_loss(raw[:,:1],target)
        total=total+m.w_heat*heat;terms[name+'_heat']=heat.detach()
    if model.mid_on:
        geo=total.new_zeros(());geo_raw=total.new_zeros(());assoc=total.new_zeros(())
        paired_mask=paired_geometry_mask(b)
        for name,other in [('rgb','tir'),('tir','rgb')]:
            cand=out['cand_'+name];stats=out['aux']['mid_stats_'+name]
            if m.use_geo_prior:
                pos,_=assign_candidates(cand['pts'],b['boxes_'+name],paired_mask)
                idx=matched_gt_index(cand['pts'],b['boxes_'+name],paired_mask)
                target=gather_by_index(b['centers_'+other],idx)
                geo=geo+geometry_nll(stats['mu'],stats['sigma']@stats['sigma'].transpose(-1,-2),target,pos,
                                      minimum_variance=model.mid.geo.sigma_floor**2)
                geo_raw=geo_raw+geometry_nll(stats['mu'],stats['sigma']@stats['sigma'].transpose(-1,-2),target,pos)
            assoc=assoc+association(model,stats['attn'],cand['pts'],stats['key_pts'],b,name,other,cand['scores'])
        total=total+m.w_geo*geo+m.w_assoc_mid*assoc
        terms['geo']=geo.detach();terms['assoc_mid']=assoc.detach()
        terms['geo_raw']=geo_raw.detach()
    if model.late_on:
        assoc=total.new_zeros(());size_loss=total.new_zeros(());size_raw=total.new_zeros(());late_geo=total.new_zeros(())
        paired_mask=paired_geometry_mask(b)
        for name,other in [('rgb','tir'),('tir','rgb')]:
            sel=out['sel_'+name];other_sel=out['sel_'+other];res=out['aux']['late_'+name]
            assoc=assoc+association(model,res['attn'],sel['pts'],other_sel['pts'],b,name,other,sel['scores'])
            if m.use_geo_prior:
                pos,_=assign_candidates(sel['pts'],b['boxes_'+name],paired_mask)
                idx=matched_gt_index(sel['pts'],b['boxes_'+name],paired_mask)
                if not model.mid_on:
                    target_center=gather_by_index(b['centers_'+other],idx)
                    late_geo=late_geo+geometry_nll(res['mu'],res['sigma']@res['sigma'].transpose(-1,-2),
                                                  target_center,pos,minimum_variance=model.late.geo.sigma_floor**2)
                target=gather_by_index(b['sizes_'+other],idx).clamp_min(1e-4).log()
                if (pos>.5).any():
                    residual=(res['size_mu'].float()-target)/res['size_std']
                    # Gaussian NLL shifted by its variance-only minimum, so the
                    # displayed geometry fusion term has a zero lower bound.
                    nll=.5*residual.square()+res['size_std'].log()
                    size_raw=size_raw+nll[pos>.5].mean()
                    size_loss=size_loss+(nll[pos>.5]+2.).mean()
        total=total+m.w_assoc_late*assoc+m.w_size_geo*size_loss
        terms['assoc_late']=assoc.detach();terms['size_geo']=size_loss.detach()
        terms['size_geo_raw']=size_raw.detach()
        if not model.mid_on and m.use_geo_prior:
            total=total+m.w_geo*late_geo;terms['geo']=late_geo.detach()
    terms['total']=total.detach()
    return total,terms
