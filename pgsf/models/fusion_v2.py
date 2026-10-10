"""Image-conditioned, covariance-guided sparse cross-modal fusion."""
from __future__ import annotations
import math
import torch
from torch import nn
from pgsf.models.geometry import ProbGeoPrior
from pgsf.models.candidate import sample_points
from pgsf.models.fusion import SparseCrossAttention,scatter_points


def gaussian_log_prob(p,mu,L):
    """Analytic triangular solve in float32, including under autocast."""
    d=p.float()-mu.float();L=L.float()
    x=d[...,0]/L[...,0,0].clamp_min(1e-6)
    y=(d[...,1]-L[...,1,0]*x)/L[...,1,1].clamp_min(1e-6)
    return -.5*(x.square()+y.square())-L[...,0,0].clamp_min(1e-6).log()-L[...,1,1].clamp_min(1e-6).log()-math.log(2*math.pi)


class ConditionalGeoPrior(ProbGeoPrior):
    """Condition mean/covariance on the query, coarse box and both images."""
    def __init__(self,dim=128,context_dim=128):
        super().__init__(dim+context_dim+4,hidden=dim)
        self.context_dim=context_dim

    def forward(self,pts,feat,affine=None,context=None,boxes=None):
        b,n,_=feat.shape
        if context is None:context=feat.new_zeros(b,self.context_dim)
        if boxes is None:boxes=torch.cat((pts,pts.new_full((b,n,2),.03)),-1)
        geometric=torch.cat((pts,boxes[...,2:].clamp_min(1e-4).log()),-1)
        h=torch.cat((feat,context[:,None].expand(-1,n,-1),geometric),-1)
        return super().forward(pts,h,affine)


class CovarianceMidFusion(nn.Module):
    def __init__(self,dim=128,heads=4,n_local=8,n_global=16,context_dim=128,
                 use_geo=True,use_gate=True,use_null=True,covariance_sampling=True,
                 lambda_geo=.35,ff_mult=4):
        super().__init__()
        self.use_geo=use_geo;self.use_gate=use_gate;self.use_null=use_null
        self.n_local=n_local;self.n_global=n_global;self.lambda_geo=lambda_geo
        self.covariance_sampling=covariance_sampling
        if use_geo:
            self.geo=ConditionalGeoPrior(dim,context_dim)
            self.geo_t2r=ConditionalGeoPrior(dim,context_dim)
        self.attn_r2t=SparseCrossAttention(dim,heads,ff_mult)
        self.attn_t2r=SparseCrossAttention(dim,heads,ff_mult)
        self.off_r2t=self.offset(dim,n_local);self.off_t2r=self.offset(dim,n_local)
        self.k_local=nn.Linear(dim,dim);self.k_global=nn.Linear(dim,dim)
        self.position=nn.Linear(3,dim,bias=False)
        if use_null:
            self.null_token=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.type_bias=nn.Parameter(torch.zeros(2 if use_null else 1))
        if use_gate:
            self.gate=nn.Sequential(nn.Linear(2*dim+4,dim//2),nn.SiLU(),nn.Linear(dim//2,1))
            nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

        angles=torch.arange(max(n_local-1,1))*2*math.pi/max(n_local-1,1)
        refs=torch.cat((torch.zeros(1,2),torch.stack((angles.cos(),angles.sin()),-1)),0)[:n_local]
        self.register_buffer('reference_points',refs)

    @staticmethod
    def offset(dim,n):
        h=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,n*2))
        nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
        return h

    def set_affine(self,A,b,reverse):
        if self.use_geo:
            self.geo.set_affine(A,b)
            self.geo_t2r.set_affine(reverse,reverse[:,2])

    def forward(self,q_feat,q_pts,other_maps,globals,direction='r2t',maps=None,
                affine=None,context=None,boxes=None,quality=None,other_dense=None,
                prior_jitter=None):
        b,n,c=q_feat.shape;gf,gp,gquality=globals
        if self.use_geo:
            geo=self.geo if direction=='r2t' else self.geo_t2r
            mu,L,extra=geo(q_pts,q_feat,affine,context,boxes)
            conf=geo.confidence(extra)
        else:
            mu=q_pts
            L=torch.eye(2,device=q_pts.device).expand(b,n,2,2)*.05
            conf=q_pts.new_ones(b,n)
        # Perturb the search only. NLL supervises the unperturbed mean so
        # augmentation noise does not teach artificial geometric uncertainty.
        search_mu=mu if prior_jitter is None else mu+prior_jitter
        off=self.off_r2t if direction=='r2t' else self.off_t2r
        residual=off(q_feat).view(b,n,self.n_local,2).tanh()*.5
        normalized=self.reference_points.to(q_feat)[None,None]+residual
        if self.covariance_sampling:
            displacement=(L.float().unsqueeze(2)@normalized.float().unsqueeze(-1)).squeeze(-1)
        else:displacement=normalized.float()*.05
        local_pts=search_mu.float().unsqueeze(2)+displacement
        valid_local=((local_pts>=0)&(local_pts<=1)).all(-1)
        split=(self.n_local+1)//2
        local=[];qualities=[]
        for name,part in [('P2',local_pts[:,:,:split]),('P3',local_pts[:,:,split:])]:
            count=part.shape[2]
            if count==0:continue
            local.append(sample_points(other_maps[name],part.reshape(b,n*count,2)).reshape(b,n,count,c))
            if other_dense is None:qualities.append(q_feat.new_ones(b,n,count))
            else:qualities.append(sample_points(other_dense[name][:,5:6],part.reshape(b,n*count,2)).reshape(b,n,count).sigmoid())
        local=torch.cat(local,2)
        level_id=normalized.new_zeros(b,n,self.n_local,1)
        level_id[:,:,split:]=1
        position=torch.cat((normalized,level_id),-1)
        keys_local=self.k_local(local)+self.position(position)
        keys_global=self.k_global(gf).unsqueeze(1).expand(-1,n,-1,-1)
        keys=[keys_local,keys_global]
        if self.use_null:keys.append(self.null_token.expand(b,n,1,c))
        keys=torch.cat(keys,2)
        ng=gf.shape[1]
        # Relative log density is non-positive; a bounded prior leaves appearance
        # and global fallback able to override a wrong geometric estimate.
        if self.use_geo:
            lp=gaussian_log_prob(local_pts,search_mu.unsqueeze(2),L.unsqueeze(2))
            at_mean=gaussian_log_prob(search_mu,search_mu,L).unsqueeze(-1)
            lp=(lp-at_mean).clamp(-16,0)*self.lambda_geo*conf.unsqueeze(-1)
        else:lp=q_pts.new_zeros(b,n,self.n_local)
        logprior=torch.cat((lp,lp.new_zeros(b,n,ng+int(self.use_null))),-1)
        key_quality=torch.cat((torch.cat(qualities,2),gquality[:,None].expand(-1,n,-1)),2)
        if self.use_null:key_quality=torch.cat((key_quality,key_quality.new_ones(b,n,1)),2)
        logprior=logprior+.25*key_quality.float().clamp_min(1e-4).log()
        bias=torch.cat((q_feat.new_zeros(self.n_local),self.type_bias[0].expand(ng),
                        self.type_bias[1:].expand(int(self.use_null))))
        key_mask=torch.cat((valid_local,valid_local.new_ones(b,n,ng+int(self.use_null))),-1)
        attn_module=self.attn_r2t if direction=='r2t' else self.attn_t2r
        z,attn=attn_module(q_feat,keys,keys,log_prior=logprior,type_bias=bias,
                          key_mask=key_mask,return_attn=True)
        null=attn[...,-1] if self.use_null else attn.new_zeros(b,n)
        entropy=-(attn*attn.clamp_min(1e-8).log()).sum(-1)/math.log(max(keys.shape[2],2))
        if quality is None:quality=q_feat.new_ones(b,n)
        if self.use_gate:
            evidence=torch.stack((conf,entropy,null,quality),-1).to(q_feat.dtype)
            gamma=self.gate(torch.cat((q_feat,z,evidence),-1)).sigmoid()
        else:gamma=q_feat.new_ones(b,n,1)
        enhanced=q_feat+gamma*(z-q_feat)
        new_maps={}
        if maps is not None:
            for name,fm in maps.items():
                delta,cov=scatter_points(q_pts,enhanced-q_feat,*fm.shape[-2:])
                new_maps[name]=fm+delta*cov
        key_pts=torch.cat((local_pts,gp.unsqueeze(1).expand(-1,n,-1,-1)),2)
        return {'feats':enhanced,'maps':new_maps,'stats':{
            'mu':mu,'search_mu':search_mu,'sigma':L,'conf':conf,'attn':attn,'key_pts':key_pts,
            'gamma':gamma.squeeze(-1),'unc':entropy,'null_attn':null,
            'local_pts':local_pts,'query_quality':quality}}


class RegionLateFusion(nn.Module):
    """Object-region attention with center distribution and learned log-size ratio."""
    def __init__(self,dim=128,heads=4,context_dim=128,use_geo=True,use_gate=True,
                 use_null=True,lambda_box=.35,ff_mult=4,standalone_geometry=False):
        super().__init__()
        self.use_geo=use_geo;self.use_gate=use_gate;self.use_null=use_null
        self.lambda_box=lambda_box
        self.attn_r2t=SparseCrossAttention(dim,heads,ff_mult)
        self.attn_t2r=SparseCrossAttention(dim,heads,ff_mult)
        self.query_project=nn.Linear(dim,dim);self.key_project=nn.Linear(dim,dim)
        if use_geo:
            if standalone_geometry:
                self.geo=ConditionalGeoPrior(dim,context_dim)
                self.geo_t2r=ConditionalGeoPrior(dim,context_dim)
            def size_head():
                h=nn.Sequential(nn.Linear(dim+context_dim,dim//2),nn.SiLU(),nn.Linear(dim//2,2))
                nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
                return h
            self.size_r2t=size_head();self.size_t2r=size_head()
            self.log_size_std=nn.Parameter(torch.zeros(2))
        if use_null:
            self.null_token=nn.Parameter(torch.randn(1,1,dim)*.02)
            self.null_bias=nn.Parameter(torch.zeros(1))
        if use_gate:
            self.gate=nn.Sequential(nn.Linear(2*dim+4,dim//2),nn.SiLU(),nn.Linear(dim//2,1))
            nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

    def set_affine(self,A,b,reverse):
        if self.use_geo and hasattr(self,'geo'):
            self.geo.set_affine(A,b);self.geo_t2r.set_affine(reverse,reverse[:,2])

    def forward(self,q_feat,q_box,k_feat,k_box,affine,context,direction='r2t',
                distribution=None,quality=None):
        b,n,c=q_feat.shape
        prior=None;size_mu=None;size_std=None
        if self.use_geo:
            if distribution is None:
                if hasattr(self,'geo'):
                    geo=self.geo if direction=='r2t' else self.geo_t2r
                    mu,L,extra=geo(q_box[...,:2],q_feat,affine,context,q_box)
                    conf=geo.confidence(extra)
                else:
                    hom=torch.cat((q_box[...,:2],torch.ones_like(q_box[...,:1])),-1)
                    mu=hom@affine.transpose(-1,-2)
                    L=torch.eye(2,device=q_feat.device).expand(b,n,2,2)*.05
                    conf=q_feat.new_full((b,n),.5)
            else:mu,L,conf=distribution
            center_lp=gaussian_log_prob(k_box[:,None,:,:2],mu[:,:,None],L[:,:,None])
            center_lp=center_lp-gaussian_log_prob(mu,mu,L).unsqueeze(-1)
            h=torch.cat((q_feat,context[:,None].expand(-1,n,-1)),-1)
            delta=(self.size_r2t if direction=='r2t' else self.size_t2r)(h)
            size_mu=q_box[...,2:].clamp_min(1e-4).log()+delta
            size_std=self.log_size_std.clamp(-2,2).exp()
            size_lp=-.5*((k_box[:,None,:,2:].clamp_min(1e-4).log()-size_mu[:,:,None])/size_std).square().sum(-1)
            prior=(center_lp+size_lp).clamp(-16,0)*self.lambda_box*conf.unsqueeze(-1)
            if self.use_null:prior=torch.nn.functional.pad(prior,(0,1))
        else:conf=q_feat.new_ones(b,n)
        keys=self.key_project(k_feat)
        if self.use_null:keys=torch.cat((keys,self.null_token.expand(b,1,c)),1)
        bias=None
        if self.use_null:bias=torch.cat((q_feat.new_zeros(k_feat.shape[1]),self.null_bias))
        # Projection enters a zero-initialized residual without replacing q_feat.
        query=q_feat+self.query_project(q_feat)*.1
        mod=self.attn_r2t if direction=='r2t' else self.attn_t2r
        z,attn=mod(query,keys,keys,log_prior=prior,type_bias=bias,return_attn=True)
        delta=z-query
        null=attn[...,-1] if self.use_null else attn.new_zeros(b,n)
        entropy=-(attn*attn.clamp_min(1e-8).log()).sum(-1)/math.log(max(keys.shape[1],2))
        if quality is None:quality=q_feat.new_ones(b,n)
        if self.use_gate:
            evidence=torch.stack((conf,entropy,null,quality),-1).to(q_feat.dtype)
            gamma=self.gate(torch.cat((q_feat,q_feat+delta,evidence),-1)).sigmoid()
        else:gamma=q_feat.new_ones(b,n,1)
        return {'feats':q_feat+gamma*delta,'attn':attn,'gamma':gamma.squeeze(-1),
                'null_attn':null,'real_attn':attn[...,:k_feat.shape[1]].max(-1).values,
                'size_mu':size_mu,'size_std':size_std,
                'mu':mu if self.use_geo else None,'sigma':L if self.use_geo else None}
