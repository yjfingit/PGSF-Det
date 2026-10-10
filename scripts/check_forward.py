#!/usr/bin/env python
"""Forward / backward / shape verification with random inputs.

Checks
------
1. every intermediate feature map has the expected shape (table in the task spec)
2. a full forward runs and produces finite outputs
3. a full backward produces a **non-zero gradient for every** trainable parameter
   group, and specifically for each module the ablation depends on
4. the fusion residual is identity at initialisation (zero-init check)
5. peak VRAM and latency

Run::

    python scripts/check_forward.py --config configs/pgsf_stage2.yaml --batch 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from pgsf.config import load_config  # noqa: E402
from pgsf.models.detector import PGSFDet  # noqa: E402
from pgsf.utils.profiler import PeakMemory, measure_latency  # noqa: E402


def make_fake_batch(cfg, device, batch=1):
    """Random batch with **synthetic but self-consistent** boxes.

    Shapes mirror the real dataset exactly: boxes/centers/sizes are padded to
    ``eval.max_det`` with a validity mask, so the loss and assignment code paths
    exercised here are the production ones.
    """
    hr, wr = cfg.data.rgb_size
    ht, wt = cfg.data.tir_size
    md = cfg.eval.max_det
    g = torch.Generator(device="cpu").manual_seed(0)

    def boxes(n, dev):
        """-> (n, md, 4) boxes, (n, md) mask, (n, md, 2) centers, (n, md, 2) sizes."""
        cx = torch.rand(n, 1, generator=g) * 0.6 + 0.2
        cy = torch.rand(n, 1, generator=g) * 0.6 + 0.2
        bw = torch.rand(n, 1, generator=g) * 0.03 + 0.008
        bh = torch.rand(n, 1, generator=g) * 0.03 + 0.008
        b0 = torch.cat([cx, cy, bw, bh], dim=1)               # (n,4)
        b = b0.new_zeros(n, md, 4)
        b[:, 0] = b0
        m = torch.zeros(n, md)
        m[:, 0] = 1.0
        ctr = torch.zeros(n, md, 2)
        ctr[:, 0] = b0[:, :2]
        siz = torch.zeros(n, md, 2)
        siz[:, 0] = b0[:, 2:]
        return b.to(dev), m.to(dev), ctr.to(dev), siz.to(dev)

    br, mr, cr, zr = boxes(batch, device)
    bt, mt, ct, zt = boxes(batch, device)
    # Make the TIR target a *perturbed* copy of the RGB one. This mimics the real
    # cross-modal centre offset measured on UAV-CB (mean dy = -0.035, std ~0.03
    # in normalised units) so the geometry prior gets a non-trivial signal.
    bt[:, 0, 0] = br[:, 0, 0]
    bt[:, 0, 1] = (br[:, 0, 1] - 0.035).clamp(0.01, 0.99)
    ct[:, 0] = bt[:, 0, :2]
    return {
        "rgb": torch.randn(batch, 3, hr, wr, generator=g).to(device),
        "tir": torch.rand(batch, 1, ht, wt, generator=g).to(device),
        "boxes_rgb": br, "boxes_tir": bt,
        "centers_rgb": cr, "centers_tir": ct,
        "sizes_rgb": zr, "sizes_tir": zt,
        "mask": mr, "valid": mr[:, 0],
        "rgb_hw": torch.tensor([[hr, wr]] * batch, dtype=torch.float32, device=device),
        "tir_hw": torch.tensor([[ht, wt]] * batch, dtype=torch.float32, device=device),
    }


def check_shapes(cfg, model, device, batch):
    """Assert the backbone pyramid matches the table in the task description."""
    print("\n=== 1. backbone pyramid shapes ===")
    rgb, tir = batch["rgb"], batch["tir"]
    with torch.no_grad():
        br = model.rgb_branch.backbone(rgb)
        bt = model.tir_branch.backbone(tir)
    names=("P2","P3","P4","P5")
    channels=model.rgb_branch.backbone.out_channels
    expect_rgb={name:(ch,(cfg.data.rgb_size[0]+stride-1)//stride,
                         (cfg.data.rgb_size[1]+stride-1)//stride)
                for name,ch,stride in zip(names,channels,(4,8,16,32))}
    expect_tir={name:(ch,(cfg.data.tir_size[0]+stride-1)//stride,
                         (cfg.data.tir_size[1]+stride-1)//stride)
                for name,ch,stride in zip(names,channels,(4,8,16,32))}
    ok = True
    for name, feats, expect in (("RGB", br, expect_rgb), ("TIR", bt, expect_tir)):
        for k, ch in (("P2", 0), ("P3", 1), ("P4", 2), ("P5", 3)):
            t = feats[k]
            e = expect[k]
            flag = "OK " if (t.shape[1] == e[0] and t.shape[2] == e[1]
                             and t.shape[3] == e[2]) else "BAD"
            ok &= flag == "OK "
            print(f"  {flag} {name} {k}: {tuple(t.shape)}  expected B,{e[0]},{e[1]},{e[2]}")
    assert ok, "backbone pyramid does not match the spec"
    print("  all levels match the spec")


def check_forward(cfg, model, device, batch):
    print("\n=== 2. forward ===")
    model.train()
    out = model(batch)
    terms = {k: float(v) for k, v in out["aux"].get("_", {}).items()} if False else {}
    loss, term = model.compute_losses(out)
    for k, v in term.items():
        assert torch.isfinite(v), f"loss {k} is not finite: {v}"
    print(f"  pred_rgb {tuple(out['pred_rgb'].shape)}  pred_tir {tuple(out['pred_tir'].shape)}")
    print(f"  cand_rgb {tuple(out['cand_rgb']['pts'].shape)} "
          f"cand_tir {tuple(out['cand_tir']['pts'].shape)}")
    print(f"  heat_rgb {tuple(out['heat_rgb'].shape)}  heat_tir {tuple(out['heat_tir'].shape)}")
    print("  losses: " + "  ".join(f"{k}={float(v):.4f}" for k, v in term.items()))
    del terms
    return loss, term


def check_backward(model, loss, cfg, batch, device, steps: int = 3):
    """Two-stage gradient audit.

    **Stage 1 (this step).**  A single backward pass.  Parameters whose gradient is
    legitimately zero at initialisation are expected and *not* a bug:

    * ``w_o`` / ``ff[-1]`` of both fusion attentions are **zero-initialised** so the
      fusion module starts as the exact identity -- the only reason stage-2
      initialisation from a stage-1 checkpoint is safe.
    * Consequently everything *upstream* of them (``q_proj``, ``ff.0``,
      ``k_norm``, the deformable-offset MLP, the geometry ``residual``/``unc``
      hidden layers) receives exactly zero gradient on the very first step.
      This is the same mechanism as LoRA's ``lora_B = 0``: the output projection
      is zero, so nothing upstream can move on step 0.

    **Stage 2 (a few real optimiser steps).**  After ``steps`` SGD updates the
    zero-initialised projections have become non-zero, the path opens, and every
    parameter must receive a gradient.  A parameter that is *still* dead here is a
    genuine wiring bug (an unused module, a detached tensor, a wrong index).
    """
    print("\n=== 3a. backward (step 0) ===")
    model.zero_grad(set_to_none=True)
    loss.backward()
    dead0, alive0, _ = _scan_grads(model)
    print(f"  params with non-zero grad: {alive0}, zero at init: {len(dead0)}")
    if dead0:
        print("  (expected at init, see docstring -- re-checked after "
              f"{steps} optimiser steps below)")
        print("   " + ", ".join(dead0[:8]) + (" ..." if len(dead0) > 8 else ""))

    print(f"\n=== 3b. gradient coverage after {steps} optimiser steps ===")
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        l, _ = model.compute_losses(out)
        l.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    out = model(batch)
    l, _ = model.compute_losses(out)
    l.backward()
    dead, alive, by_module = _scan_grads(model, detailed=True)
    print(f"  params with non-zero grad: {alive}, still dead: {len(dead)}")
    if dead:
        print("  !! GENUINELY DEAD: " + ", ".join(dead[:30]))
    for k in sorted(by_module):
        print(f"     {k:14s} |grad| = {by_module[k]:.4e}")

    assert not dead, (f"{len(dead)} parameters are still without gradient after "
                      f"{steps} optimiser steps -- real wiring bug: {dead[:10]}")

    # Every ablation-critical module must actually be reachable.  Match on a
    # *substring* rather than a dotted prefix, because top-level modules appear as
    # ``mid.xxx`` / ``rgb_branch.msce.xxx`` with no leading dot.
    names = list(by_module_flat(model))

    def used(pred: str) -> bool:
        return any(pred in n for n in names)

    if model.v2:
        required=["heat.p2","heat.p3","cand.feat_proj","det.out","neck.lat2"]
        if model.rgb_branch.roi_on:required += ["refine.merge","roi_project","box_position"]
        if model.rgb_branch.msce_on:required += ["msce.reduce","msce.alpha_raw"]
        if model.cfg.model.backbone_name=="yolo11s":required += ["backbone.layers.0"]
        else:required += ["backbone.stem","backbone.msdw"]
        if model.mid_on:
            required += ["mid.off_r2t","mid.off_t2r","mid.position","mid.k_local","mid.k_global"]
            if model.cfg.model.use_geo_prior:required += ["mid.geo.residual","mid.geo_t2r.residual","mid.geo.unc","mid.geo_t2r.unc"]
        if model.late_on:
            required += ["late.query_project","late.key_project"]
            if model.cfg.model.use_geo_prior:required += ["late.size_r2t","late.size_t2r","late.log_size_std"]
        if hasattr(model,"geometry_context"):required += ["geometry_context"]
    else:
        required=["heat.","det.out","refine.merge","box_pre.body","neck.lat2","backbone.stem","backbone.msdw"]
        if model.rgb_branch.msce_on:required += ["msce.","msce.alpha_raw"]
        if model.mid_on:required += ["mid.attn_r2t.q_proj","mid.attn_t2r.q_proj","mid.off_r2t","mid.off_t2r"]
        if model.late_on:required += ["late.box_prior.log_std"]
    if model.cfg.model.use_gate:
        if model.mid_on:required += ["mid.gate"]
        if model.late_on:required += ["late.gate"]

    missing = [p for p in required if not used(p)]
    if missing:
        print("  !! MODULE NEVER REACHED: " + ", ".join(missing))
    assert not missing, f"modules never reached by the loss: {missing}"
    print(f"  all {len(required)} critical module paths are reached by the loss")
    del cfg
    return alive


def _scan_grads(model, detailed: bool = False):
    dead, alive, by_module = [], 0, {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is not None and not torch.isfinite(g).all():
            raise AssertionError(f"non-finite gradient: {name}")
        if g is None or float(g.abs().sum()) == 0.0:
            dead.append(name)
            continue
        alive += 1
        if detailed:
            top = name.split(".")[0]
            by_module[top] = by_module.get(top, 0.0) + float(g.abs().sum())
    return dead, alive, by_module


def by_module_flat(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            yield n


def check_zero_init(cfg, model, device, batch):
    print("\n=== 4. fusion residual is identity at initialisation ===")
    if not (model.mid_on or model.late_on):
        print("  fusion disabled (ablation A0/A1) -- skipped")
        return
    model.eval()
    with torch.no_grad():
        out = model(batch)
        mid_on, late_on = model.mid_on, model.late_on
        try:
            model.mid_on = model.late_on = False
            independent = model(batch)
        finally:
            model.mid_on, model.late_on = mid_on, late_on
        for name in ("rgb", "tir"):
            delta = (out["pred_" + name] - independent["pred_" + name]).abs().max()
            assert float(delta) < 1e-5, f"{name} fusion changes initial detections"
            print(f"  {name}: max prediction difference = {float(delta):.2e}")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGSFDet(cfg).to(device)

    batch = make_fake_batch(cfg, device, args.batch)
    check_shapes(cfg, model, device, batch)
    check_zero_init(cfg, model, device, batch)
    loss, _ = check_forward(cfg, model, device, batch)
    check_backward(model, loss, cfg, batch, device)

    if not args.skip_latency:
        print("\n=== 5. memory / latency / params / FLOPs ===")
        from pgsf.utils.profiler import profile_model
        rep = profile_model(model, batch, device, iters=8)
        p = rep["params"]
        print(f"  parameters   : {p['total'] / 1e6:.2f} M "
              f"({'OK <50M' if p['total'] < 50e6 else 'OVER 50M'})")
        for k, v in sorted(p["submodules"].items()):
            print(f"      {k:14s} {v / 1e6:7.3f} M")
        print(f"  MACs         : {rep['macs'] / 1e9:.2f} G  "
              f"(FLOPs = 2xMACs: {rep['flops_2x_macs'] / 1e9:.2f} G)")
        print(f"  peak VRAM    : {rep['peak_vram_mib']:.1f} MiB (inference, batch="
              f"{args.batch})")
        print(f"  train peak   : {rep['train_peak_vram_mib']:.1f} MiB")
        print(f"  latency      : {rep['latency']['latency_ms']:.2f} ms  "
              f"({rep['latency']['fps']:.1f} FPS, batch={args.batch})")
        if not model.v2:
            assert p["total"] < 50e6, "parameter budget exceeded (50M)"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
