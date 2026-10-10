#!/usr/bin/env python
"""Is L_geo bounded below? Train ONLY the uncertainty head and watch it.

This is the decisive test for the sigma-collapse failure: if the NLL is unbounded
below, minimising it drives sigma -> 0 and the term dominates every other loss.

Run:  python scripts/geo_bound_check.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pgsf.models.geometry import ProbGeoPrior
from pgsf.losses import geometry_nll

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = ProbGeoPrior(32, hidden=32).to(dev)
pts = torch.rand(4, 64, 2, device=dev)
tgt = pts.clone()                      # mu is exactly on target: worst case for NLL
valid = torch.ones(4, 64, device=dev)
feat = torch.randn(4, 64, 32, device=dev, requires_grad=True)

opt = torch.optim.SGD(list(g.parameters()) + [feat], lr=0.5)
print(f"sigma_floor = {g.sigma_floor}, sigma_ref = 0.05")
print("step   sigma_xx  sigma_yy     L_geo")
for it in range(1, 201):
    mu, L, _ = g(pts, feat)
    cov = L @ L.transpose(-1, -2)
    loss = geometry_nll(mu, cov, tgt, valid)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it in (1, 5, 20, 50, 100, 200):
        print(f"{it:4d}   {float(L[:,:,0,0].mean()):8.4f}  {float(L[:,:,1,1].mean()):8.4f}"
              f"   {float(loss):+9.4f}")

final_sigma = float(L[:, :, 0, 0].mean())
final_loss = float(loss)
print(f"\nfinal sigma {final_sigma:.4f} (floor {g.sigma_floor}), L_geo {final_loss:+.4f}")
ok = final_sigma >= g.sigma_floor - 1e-4 and final_loss > -5.0
print("VERDICT:", "PASS - L_geo is bounded and cannot dominate" if ok
      else "FAIL - still collapsing")
sys.exit(0 if ok else 1)
