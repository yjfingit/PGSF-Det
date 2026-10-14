"""Two-stage trainer.

Stage 1 (``cfg.stage == 1``): only the two independent branches (backbone, MSCE,
candidate generation, detection heads).  The fusion modules are simply not
instantiated, so ``L_geo`` / ``L_assoc`` do not exist.  Goal: two branches with
high recall before any cross-modal coupling.

Stage 2 (``cfg.stage == 2``): everything, initialised from the stage-1
checkpoint.  The fusion residual branches are zero-initialised, so the network
starts *exactly* at the stage-1 function and can only improve from there.

Parameter groups: backbone at ``lr * lr_backbone_mult``, freshly initialised
fusion modules at ``lr * lr_new_mult``, everything else at ``lr``.
"""

from __future__ import annotations

import json
import math
import os
import time
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

from pgsf.data import UAVCBDataset, build_split, collate, list_frames, load_split_manifest
from pgsf.models.detector import PGSFDet, load_pretrained_backbone, param_groups
from pgsf.utils.metrics import (
    eval_detection,
    filtered_ground_truth,
    match_accuracy,
    precision_recall_at,
    small_object_recall,
)


# --------------------------------------------------------------------------- #
def build_datasets(cfg) -> Tuple[Dict[str, UAVCBDataset], Dict[str, int]]:
    d = cfg.data
    fixed: Dict[str, List] = {}
    if d.use_fixed_split and os.path.isdir(d.split_dir):
        fixed = load_split_manifest(d.split_dir, d.root, d.scenes, d.class_names)
    if fixed and fixed.get("train"):
        split = fixed
    else:
        frames = list_frames(d.root, d.scenes, d.class_names)
        split = build_split(frames, d.train_ratio, d.val_ratio, d.test_ratio, seed=d.seed)

    sets = {}
    from pgsf.data.uavcb import make_augment
    train_aug = make_augment(d,True)
    for k in ("train", "val", "test"):
        frames = split.get(k, [])
        if k == "test" and not d.allow_test:
            frames_for_ds = frames          # kept, but never used for fitting/selection
        else:
            frames_for_ds = frames
        if k == "train" and d.max_train_per_scene > 0:
            frames_for_ds = _subsample(frames_for_ds, d.max_train_per_scene, d.seed)
        sets[k] = UAVCBDataset(frames_for_ds, d.rgb_size, d.tir_size,
                               train=(k == "train"),
                               augment=train_aug if k == "train" else make_augment(d,False),
                               class_names=d.class_names,
                               pair_by_index=d.pair_by_index,
                               multi_object=d.multi_object)
    counts = {k: len(v) for k, v in sets.items()}
    return sets, counts


def _val_augment():
    from pgsf.data.uavcb import PairAugment
    return PairAugment(train=False)


def _subsample(frames: List, per_scene: int, seed: int) -> List:
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for f in frames:
        by[f.scene].append(f)
    out = []
    for s in sorted(by):
        idx = rng.permutation(len(by[s]))[:per_scene]
        out.extend([by[s][i] for i in sorted(idx)])
    return out


# --------------------------------------------------------------------------- #
class ModelEMA:
    """Exponential moving average of the weights (kept on the same device)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for pe, pm in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(self.module.buffers(), model.buffers()):
            be.copy_(bm)


class TrainingStep(nn.Module):
    """Keep loss computation inside DDP so unused parameters are identified correctly."""
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self, batch):
        return self.model.compute_losses(self.model(batch))


class DistributedEvalSampler(Sampler):
    """Partition validation without the duplicate padding of DistributedSampler."""
    def __init__(self, dataset, rank, world_size):
        self.indices=range(rank,len(dataset),world_size)

    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)


class EarlyStopping:
    def __init__(self, patience=0, min_delta=1e-4):
        self.patience=patience;self.min_delta=min_delta
        self.best=-float('inf');self.bad_epochs=0

    def update(self, score):
        improved=math.isfinite(score) and score>self.best+self.min_delta
        if improved:self.best=score;self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.patience>0 and self.bad_epochs>=self.patience

    def state_dict(self):return {'best':self.best,'bad_epochs':self.bad_epochs}

    def load_state_dict(self, state):
        self.best=state['best'];self.bad_epochs=state['bad_epochs']


# --------------------------------------------------------------------------- #
class PGSFTrainer:
    def __init__(self, cfg, model: Optional[PGSFDet] = None,
                 device: Optional[torch.device] = None):
        self.cfg = cfg
        self.distributed=dist.is_available() and dist.is_initialized()
        self.rank=dist.get_rank() if self.distributed else 0
        self.world_size=dist.get_world_size() if self.distributed else 1
        self.is_main=self.rank==0
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = device or torch.device(cfg.device)
        self.model = (model or PGSFDet(cfg)).to(self.device)
        self.datasets, self.counts = build_datasets(cfg)
        self.loader = self._loader(self.datasets["train"], shuffle=True,
                                   bs=cfg.optim.batch_size, workers=cfg.data.num_workers)
        self.val_loader = self._loader(self.datasets["val"], shuffle=False,
                                       bs=cfg.optim.batch_size, workers=max(2, cfg.data.num_workers // 2))
        self.test_loader = self._loader(self.datasets["test"], shuffle=False,
                                        bs=cfg.optim.batch_size, workers=2)
        self.optimizer = self._make_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", init_scale=128., enabled=cfg.optim.amp and
                                           self.device.type == "cuda")
        self.ema = ModelEMA(self.model, cfg.optim.ema_decay) if cfg.optim.ema else None
        self.out_dir = cfg.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "ckpt"), exist_ok=True)
        if self.is_main:cfg.save(os.path.join(self.out_dir, "config.yaml"))
        self.history: List[Dict[str, float]] = []
        self.n_skipped = 0
        self._fit_geometry_prior()
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_metrics = {"AP50:95": -1.0}
        self.early_stopping=EarlyStopping(cfg.optim.early_stop_patience,cfg.optim.early_stop_min_delta)
        if cfg.optim.pretrained_backbone:
            n, tot = load_pretrained_backbone(self.model, cfg.optim.pretrained_backbone)
            if self.ema:
                self.ema.module.load_state_dict(self.model.state_dict())
            print(f"[pretrain] loaded {n}/{tot} backbone tensors from "
                  f"{cfg.optim.pretrained_backbone}")
        self.train_model=TrainingStep(self.model)
        if self.distributed:
            self.train_model=DistributedDataParallel(
                self.train_model,device_ids=[self.device.index],output_device=self.device.index,
                find_unused_parameters=True,broadcast_buffers=True)
        if self.is_main:
            print(f'[train] world_size={self.world_size}, per_gpu_batch={cfg.optim.batch_size}, '
                  f'global_batch={cfg.optim.batch_size*self.world_size}',flush=True)
        torch.manual_seed(cfg.seed+self.rank)

    # ------------------------------------------------------------------ #
    def _loader(self, ds, shuffle: bool, bs: int, workers: int) -> DataLoader:
        distributed=getattr(self,'distributed',False)
        sampler=None
        if distributed:
            sampler=(DistributedSampler(ds,num_replicas=self.world_size,rank=self.rank,
                                         shuffle=True,seed=self.cfg.seed)
                     if shuffle else DistributedEvalSampler(ds,self.rank,self.world_size))
        generator=torch.Generator().manual_seed(self.cfg.seed+1000*getattr(self,'rank',0))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle and sampler is None,
                          sampler=sampler,num_workers=workers,generator=generator,
                          collate_fn=collate, drop_last=False, pin_memory=True,
                          persistent_workers=workers > 0,prefetch_factor=2 if workers else None)

    def _make_optimizer(self):
        groups = param_groups(self.model, self.cfg)
        name = self.cfg.optim.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=self.cfg.optim.lr,
                                     weight_decay=self.cfg.optim.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(groups, lr=self.cfg.optim.lr, momentum=0.9,
                                   weight_decay=self.cfg.optim.weight_decay, nesterov=True)
        raise ValueError(name)

    def _fit_geometry_prior(self) -> None:
        """Least-squares affine on the **training split only**.

        Uses GT centres of both modalities, normalised to [0,1].  After fitting we
        also log the residual spread, which quantifies how much uncertainty the
        probabilistic prior has to cover.
        """
        ds = self.datasets["train"]
        if len(ds) == 0:
            return
        if int(self.cfg.model.architecture_version) >= 3:
            # v3 replaces the fitted affine with learned appearance retrieval and
            # per-anchor uncertainty, so there is no global warp to fit.
            if self.is_main:
                print("[geo] architecture_version>=3: no global affine is fitted")
            return
        n = min(self.cfg.data.geo_prior_samples, len(ds))
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        src, dst = [], []
        for i in idx:
            f = ds.frames[int(i)]
            if f.rgb_box is None or f.tir_box is None:
                continue
            src.append(f.rgb_box[:2])
            dst.append(f.tir_box[:2])
        if len(src) < 8:
            print("[geo] too few paired GT to fit the affine prior; keeping identity")
            return
        src_t = torch.from_numpy(np.asarray(src, dtype=np.float32))
        dst_t = torch.from_numpy(np.asarray(dst, dtype=np.float32))
        A, b = fit_affine_np(src_t, dst_t)
        self.model.set_affine(A, b)
        if self.ema:
            self.ema.module.set_affine(A, b)
        hom = torch.cat([src_t, torch.ones(src_t.shape[0], 1)], dim=1)
        pred = hom @ A.T
        res = (pred - dst_t)
        # Seed the prior's initial uncertainty from the residual spread actually
        # measured on this training split, instead of a constant tuned on one
        # dataset. Two standard deviations are used so the prior's initial 95%
        # interval covers the observed spread. The affine itself is a trainable
        # parameter and keeps moving during training.
        sigma0 = 2.0 * float(res.std(0).mean())
        priors = list(_iter_geo_priors(self.model))
        if self.ema:
            priors += list(_iter_geo_priors(self.ema.module))
        if priors:
            sigma0 = max(sigma0, 2.0 * max(p.sigma_floor for p in priors))
            for p in priors:
                p.set_sigma_init(sigma0)
        stats = {
            "n_pairs": int(len(src)),
            "A": A.tolist(), "b": b.tolist(),
            "res_mean": res.mean(0).tolist(),
            "res_std": res.std(0).tolist(),
            "res_p95_abs": np.percentile(res.abs().numpy(), 95, axis=0).tolist(),
            "sigma_init": sigma0,
        }
        if self.is_main:
            with open(os.path.join(self.out_dir, "geo_prior.json"), "w") as f:
                json.dump(stats, f, indent=2)
            print(f"[geo] fitted affine on {len(src)} training pairs; "
                  f"residual std {np.round(res.std(0).numpy(), 4).tolist()} "
                  f"(normalised units)")

    # ------------------------------------------------------------------ #
    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())

    def _lr_at(self, epoch: int, step: int, total_steps: int) -> float:
        cfg = self.cfg.optim
        warm = max(1, int(cfg.warmup_epochs * max(1, total_steps // max(1, cfg.epochs))))
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, object]:
        cfg = self.cfg
        total_epochs = cfg.optim.epochs
        total_steps = total_epochs * max(1, len(self.loader))
        step = self.start_epoch * max(1, len(self.loader))
        best = self.best_metrics
        log_path = os.path.join(self.out_dir, "train.log")

        for epoch in range(self.start_epoch, total_epochs):
            self.model.train()
            self.train_model.train()
            if isinstance(self.loader.sampler,DistributedSampler):self.loader.sampler.set_epoch(epoch)
            aug=getattr(self.datasets['train'],'aug',None)
            mild=cfg.data.close_aug_epochs>0 and epoch>=total_epochs-cfg.data.close_aug_epochs
            if cfg.data.close_aug_epochs>0 and cfg.optim.early_stop_patience>0:
                mild=mild or self.early_stopping.bad_epochs>=max(1,cfg.optim.early_stop_patience//2)
            if aug is not None:aug.set_mild_mode(mild)
            if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.time()
            agg: Dict[str, float] = {}
            successful_steps = 0
            if self.is_main:
                print(f'\nStage {cfg.stage} | '+('mild augmentation' if mild else 'training augmentation'),flush=True)
                print(f'{"Epoch":>9} {"GPU_mem":>9} {"box_loss":>10} {"cls_loss":>10} '
                      f'{"nwd_loss":>10} {"heat_loss":>10} {"fusion_loss":>12} {"Instances":>10} {"Size(R/T)":>23}',flush=True)
            progress=tqdm(self.loader,disable=not self.is_main,file=sys.stdout,
                          mininterval=.2,miniters=1,
                          bar_format='{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for batch_idx,batch in enumerate(progress):
                batch = {k: (v.to(self.device, non_blocking=True)
                             if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                self.model.set_training_progress(step/max(total_steps,1))
                scale = self._lr_at(epoch, step, total_steps)
                for g in self.optimizer.param_groups:
                    if "base_lr" not in g:
                        g["base_lr"] = g["lr"]
                    g["lr"] = g["base_lr"] * scale

                with self._autocast():
                    loss, terms = self.train_model(batch)

                self.optimizer.zero_grad(set_to_none=True)
                finite=torch.isfinite(loss).to(torch.int32)
                if self.distributed:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():
                    # Finish DDP's reduction before a skipped optimizer update.
                    if self.distributed:self.scaler.scale(torch.nan_to_num(loss)*0.).backward()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.n_skipped += 1
                    if self.n_skipped == 1:
                        print(f"[warn] non-finite loss at step {step}, skipping", flush=True)
                    step += 1
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    cfg.optim.grad_clip if cfg.optim.grad_clip>0 else float('inf'))
                grad_finite=torch.isfinite(norm).to(torch.int32)
                if self.distributed:dist.all_reduce(grad_finite,op=dist.ReduceOp.MIN)
                old_scale = self.scaler.get_scale()
                if not grad_finite.item():
                    # Every rank skips together even if only one had an invalid
                    # local loss graph. Keep scaler and optimizer states aligned.
                    self.scaler.update(new_scale=old_scale*.5 if self.scaler.is_enabled() else None)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.n_skipped+=1;step+=1
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < old_scale:
                    self.n_skipped += 1
                    step += 1
                    continue
                if self.ema is not None:
                    self.ema.update(self.model, step)
                step += 1
                successful_steps += 1
                for k, v in terms.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                if self.is_main:
                    mean={k:v/successful_steps for k,v in agg.items()}
                    combined=lambda suffix:sum(mean.get(name+suffix,0.) for name in ('rgb','tir'))/2
                    fusion=sum(mean.get(key,0.)*weight for key,weight in [
                        ('geo',cfg.model.w_geo),('assoc_mid',cfg.model.w_assoc_mid),
                        ('assoc_late',cfg.model.w_assoc_late),('size_geo',cfg.model.w_size_geo),
                        ('corr',getattr(cfg.model,'w_corr',0.)),
                        ('match',getattr(cfg.model,'lambda_match',0.)),
                        ('cons',getattr(cfg.model,'w_cons',0.))])
                    mem=torch.cuda.max_memory_allocated(self.device)/2**30 if self.device.type=='cuda' else 0.
                    instances=sum(int(batch.get('mask_'+name,batch['mask']).sum()) for name in ('rgb','tir'))
                    size=f'{cfg.data.rgb_size[0]}x{cfg.data.rgb_size[1]}/{cfg.data.tir_size[0]}x{cfg.data.tir_size[1]}'
                    progress.set_description_str(
                        f'{epoch+1:>4}/{total_epochs:<4} {mem:>8.2f}G {combined("_loss_box"):>10.4f} '
                        f'{combined("_loss_cls"):>10.4f} {combined("_loss_nwd"):>10.4f} '
                        f'{combined("_heat"):>10.4f} {fusion:>12.4f} {instances:>10} {size:>23}',refresh=False)

            if successful_steps == 0:
                raise RuntimeError("no optimizer updates succeeded this epoch; inspect AMP gradients/loss")
            n = successful_steps
            if self.distributed:
                keys=sorted(agg)
                values=torch.tensor([agg[k] for k in keys]+[n],device=self.device,dtype=torch.float64)
                dist.all_reduce(values)
                agg=dict(zip(keys,values[:-1].tolist()));n=float(values[-1])
            row = {k: v / n for k, v in agg.items()}
            row["epoch"] = epoch + 1
            row["lr"] = self.optimizer.param_groups[1]["lr"]
            row["time_s"] = time.time() - t0
            row["skipped"] = self.n_skipped
            row['mild_augmentation']=mild
            row['world_size']=self.world_size
            row['peak_gpu_mib']=torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type=='cuda' else 0.

            val = self.evaluate(split="val")
            for modality in ("rgb", "tir"):
                for metric in ("AP50", "AP75", "AP50:95", "small_recall"):
                    row[f"val_{modality}_{metric}"] = val[modality][metric]
            score = 0.5 * (val["rgb"]["AP50:95"] + val["tir"]["AP50:95"])
            improved,stop=self.early_stopping.update(score)
            row['val_mean_AP50:95']=score
            row['early_stop_bad_epochs']=self.early_stopping.bad_epochs
            row['total_time_s']=time.time()-t0
            self.history.append(row)
            self.current_epoch = epoch + 1
            if self.is_main:
                print(f'Epoch time {row["total_time_s"]:.1f}s | lr {row["lr"]:.2e} | '
                      f'best mean mAP50-95 {self.early_stopping.best:.4f} | '
                      f'patience {self.early_stopping.bad_epochs}/{cfg.optim.early_stop_patience} | '
                      f'skipped {self.n_skipped}',flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            if improved:
                best = {"AP50:95": score, "epoch": epoch + 1,
                        "rgb": val["rgb"], "tir": val["tir"]}
                self.best_metrics = best
                self.save("best.pt")
            self.save("last.pt")
            if stop and epoch+1>=cfg.optim.min_epochs:
                if self.is_main:print(f'[early-stop] no validation improvement for {cfg.optim.early_stop_patience} epochs',flush=True)
                break
        if self.is_main:
            with open(os.path.join(self.out_dir, "best_metrics.json"), "w") as f:
                json.dump(best, f, indent=2)
        return best

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, split: str = "val", model: Optional[nn.Module] = None,
                 use_ema: bool = True) -> Dict[str, object]:
        mdl = model or (self.ema.module if (self.ema and use_ema) else self.model)
        mdl.eval()
        distributed=getattr(self,'distributed',False)
        if distributed:
            # Rank-0 EMA/BN buffers define one identical validation model on both GPUs.
            for tensor in list(mdl.parameters())+list(mdl.buffers()):dist.broadcast(tensor,src=0)
        loader = {"val": self.val_loader, "test": self.test_loader,
                  "train": self.loader}[split]
        e = self.cfg.eval
        preds = {"rgb": [], "tir": []}
        gts = {"rgb": [], "tir": []}
        boxes_px = {"rgb": [], "tir": []}
        main=getattr(self,'is_main',True)
        title=f'{"Class":>12} {"Images":>8} {"Instances":>10} {"Box(P":>10} {"R":>10} {"mAP50":>10} {"mAP50-95)":>12}'
        progress=tqdm(loader,desc=title,disable=not main,file=sys.stdout,
                      mininterval=.2,miniters=1,
                      bar_format='{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for batch in progress:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast('cuda',enabled=self.device.type=='cuda' and self.cfg.optim.amp):
                res, _ = mdl.inference(batch, max_det=e.max_det,
                                       score_thr=e.conf_thr, nms_thr=e.nms_thr)
            for name in ("rgb", "tir"):
                preds[name].extend({k:v.cpu() for k,v in detection.items()} for detection in res[name])
                # **Pixel-space GT.** The predictions returned by ``inference`` are in
                # each modality's own pixel coordinates, so the ground truth must be
                # too. Passing the *normalised* boxes here silently yields IoU = 0 for
                # every pair (a 0.5-wide normalised box never overlaps a 640-px-wide
                # pixel box), which reported AP = 0 even for a detection whose true
                # IoU was 0.76.  The assertion below guards against that regression.
                px = batch[f"boxes_{name}_px"]
                if float(px.max()) <= 1.5 and float(px.sum()) > 0:
                    raise AssertionError(
                        f"{name} GT looks normalised (max={float(px.max()):.3f}) "
                        "but predictions are in pixels")
                valid_boxes = filtered_ground_truth(batch, name)
                boxes_px[name].extend(v.cpu() for v in valid_boxes)
                gts[name].extend(v.cpu() for v in valid_boxes)
        if distributed:
            gathered=[None]*self.world_size
            dist.all_gather_object(gathered,{'preds':preds,'gts':gts})
            preds={name:[p for part in gathered for p in part['preds'][name]] for name in ('rgb','tir')}
            gts={name:[g for part in gathered for g in part['gts'][name]] for name in ('rgb','tir')}
            boxes_px=gts
        out: Dict[str, object] = {}
        for name in ("rgb", "tir"):
            out[name] = eval_detection(preds[name], gts[name], e.iou_thrs, e.max_det)
            out[name]["small_recall"] = small_object_recall(preds[name], gts[name])
            pr = precision_recall_at(preds[name], gts[name], e.score_thr)
            out[name].update({f"P_{k}" if k in ("precision", "recall", "f1") else k: v
                              for k, v in pr.items()})
        out["match"] = match_accuracy(preds["rgb"], preds["tir"],
                                      boxes_px["rgb"], boxes_px["tir"])
        out["split"] = split
        if main:
            for name in ('rgb','tir'):
                m=out[name]
                print(f'{name.upper():>12} {len(gts[name]):>8} {m["n_gt"]:>10} '
                      f'{m["P_precision"]:>10.4f} {m["P_recall"]:>10.4f} '
                      f'{m["AP50"]:>10.4f} {m["AP50:95"]:>12.4f}',flush=True)
            print(f'{"mean":>12} {len(gts["rgb"]):>8} {sum(out[n]["n_gt"] for n in ("rgb","tir")):>10} '
                  + ' '.join(f'{sum(out[n][k] for n in ("rgb","tir"))/2:>10.4f}'
                             for k in ('P_precision','P_recall','AP50','AP50:95')),flush=True)
        return out

    # ------------------------------------------------------------------ #
    def save(self, name: str) -> str:
        path = os.path.join(self.out_dir, "ckpt", name)
        if not getattr(self,'is_main',True):return path
        torch.save({"model": self.model.state_dict(),
                    "ema": self.ema.module.state_dict() if self.ema else None,
                    "config": self.cfg.to_dict(), "epoch": self.current_epoch,
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(), "history": self.history,
                    "best_metrics": self.best_metrics,
                    "early_stopping":self.early_stopping.state_dict() if hasattr(self,'early_stopping') else None,
                    "skipped":self.n_skipped if hasattr(self,'n_skipped') else 0}, path+'.tmp')
        os.replace(path+'.tmp',path)
        return path

    def load_weights(self, path: str) -> None:
        """Initialize/fine-tune without inheriting epochs or optimizer state."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        source_stage = ck.get("config", {}).get("stage")
        transfer = self.cfg.stage >= 2 and (source_stage == 1 or
                   (source_stage is None and not any(k.startswith(("mid.", "late.", "corr.", "match.", "td."))
                                                    for k in ck["model"])))
        # Stage-1 selection uses EMA validation; start stage 2 from those same weights.
        state=(ck.get('ema') or ck['model']) if transfer else ck['model']
        self.model.load_checkpoint_weights(state, stage1_transfer=transfer)
        if self.ema:
            self.ema.module.load_checkpoint_weights(ck.get("ema") or ck["model"],
                                                     stage1_transfer=transfer)
        if getattr(self,'is_main',True):
            print(f"[load] {path}" + (" (stage 1 EMA -> stage 2)" if transfer else ""))

    def resume(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        source_stage = ck.get("config", {}).get("stage", self.cfg.stage)
        if source_stage != self.cfg.stage:
            raise ValueError("--resume requires the same stage; use --ckpt for stage transfer")
        self.load_weights(path)
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        else:
            print("[resume] legacy checkpoint has no optimizer state; optimizer starts fresh")
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        self.start_epoch = self.current_epoch = int(ck.get("epoch", 0))
        self.history = ck.get("history", [])
        self.best_metrics = ck.get("best_metrics", {"AP50:95": -1.0})
        self.n_skipped=ck.get('skipped',0)
        if hasattr(self,'early_stopping'):
            if ck.get('early_stopping'):self.early_stopping.load_state_dict(ck['early_stopping'])
            else:
                self.early_stopping.best=self.best_metrics['AP50:95']
                self.early_stopping.bad_epochs=max(0,self.current_epoch-self.best_metrics.get('epoch',self.current_epoch))
        print(f"[resume] {path} @ epoch {self.start_epoch}")


# --------------------------------------------------------------------------- #
def _iter_geo_priors(module):
    """Yield every conditional-geometry prior contained in ``module``."""
    from pgsf.models.geometry import ProbGeoPrior
    for m in module.modules():
        if isinstance(m, ProbGeoPrior):
            yield m


def fit_affine_np(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares homogeneous affine -> (A (2,3), b (2,))."""
    from pgsf.models.geometry import fit_affine
    return fit_affine(src, dst)
