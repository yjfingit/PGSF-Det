#!/usr/bin/env python
"""Sanity-check the geometry NLL as a *training objective*.

The NLL of a 2-D Gaussian can be negative (density > 1), so a naive check
"is the loss positive?" is meaningless. What matters is:

1. NLL is minimised exactly at the true (mu, Sigma);
2. it is convex in Sigma's log-scale, so gradient descent shrinks an
   over-wide covariance instead of diverging;
3. its gradient magnitude stays in a range comparable to the other terms,
   i.e. w_geo * |dL/dSigma| is not orders of magnitude larger than the box loss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, math
from pgsf.losses import geometry_nll

torch.manual_seed(0)
B, K = 8, 16
target = torch.rand(B, K, 2) * 0.6 + 0.2
eye = torch.eye(2)

def nll(mu, cov):
    valid = torch.ones(B, K)
    cov = cov.expand(B, K, 2, 2).contiguous() if cov.dim() == 2 else cov
    return geometry_nll(mu, cov, target, valid)

print("=== 1. mu: minimum at the target ===")
for off in (0.0, 0.01, 0.05, 0.1):
    mu = target + off
    cov = eye * 0.01
    print(f"  offset={off:.3f}  NLL={float(nll(mu, cov)):8.4f}")

print("\n=== 2. Sigma: minimum at the true covariance ===")
resid = target - target.mean(1, keepdim=True)
emp = torch.einsum("bki,bkj->bij", resid, resid) / K
print(f"  empirical cov (avg diag): {emp.diagonal(dim1=-2, dim2=-1).mean(0).tolist()}")
for s in (0.001, 0.01, 0.1, 0.5, 2.0):
    cov = eye * s
    print(f"  sigma={s:<6} NLL={float(nll(target, cov)):8.4f}")

print("\n=== 3. gradient w.r.t. log-sigma ===")
for init_log_s, label in ((0.0, "sigma=1"), (2.0, "sigma=7.4")):
    log_s = torch.tensor([init_log_s], requires_grad=True)
    cov = eye * torch.exp(log_s)
    n = nll(target, cov)
    n.backward()
    g = float(log_s.grad)
    # dNLL/dlog(sigma) > 0  =>  gradient descent DECREASES log(sigma), i.e. shrinks
    # an over-wide covariance. That is the desired behaviour (the sign convention
    # is easy to misread, hence the explicit label).
    print(f"  d NLL / d log sigma at {label:<11}: {g:+.4f} "
          f"-> GD {'SHRINKS' if g > 0 else 'GROWS'} sigma")

print("\n=== 4. gradient magnitude vs the box loss ===")
mu = target + 0.05
cov = eye * 0.05
mu_p = mu.clone().requires_grad_(True)
nll(mu_p, cov).backward()
g_geo = float(mu_p.grad.abs().mean())
from pgsf.losses import ciou_loss
pb = torch.rand(64, 4) * 0.1 + torch.tensor([0.4, 0.4, 0.02, 0.02])
pb = pb.requires_grad_(True)
ciou_loss(pb, torch.rand(64, 4) * 0.1 + torch.tensor([0.4, 0.4, 0.02, 0.02])).mean().backward()
g_box = float(pb.grad.abs().mean())
print(f"  |d NLL / d mu|  = {g_geo:.4f}   (w_geo=0.1 -> {0.1*g_geo:.5f})")
print(f"  |d CIoU / d box| = {g_box:.4f}   (w_box=2.5 -> {2.5*g_box:.5f})")
print(f"  ratio after weighting: {0.1*g_geo/(2.5*g_box):.4f}")
