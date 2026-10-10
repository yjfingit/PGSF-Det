"""FLOPs / params / VRAM / latency profiler.

FLOPs are computed with a **hook-based counter on Conv2d and Linear** rather than
an external dependency, because the useful number here is "multiply-accumulates
for one forward of the full two-modality network", and that is exactly what hooks
measure.  Counting is done on a single sample, in ``eval`` mode, under
``torch.no_grad`` so no activation memory is retained.

FLOPs convention: one multiply-accumulate counts as 1 FLOP, i.e. the number
reported is MACs.  We also report ``2 x MACs`` (the FLOPs convention of most
papers) so the two are never confused.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # per top-level submodule
    mods = {name: sum(p.numel() for p in m.parameters())
            for name, m in model.named_children()}
    return {"total": total, "trainable": trainable,
            "submodules": mods, "megaparam": total / 1e6}


def count_flops(model: nn.Module, batch: Dict[str, torch.Tensor]) -> int:
    """MACs for one forward pass (counting only; no gradients are built)."""
    macs = [0]
    handles = []

    def conv_hook(m, i, o):
        # out_elements x (in_channels/groups * k*k)
        macs[0] += o.numel() * (m.in_channels // m.groups) * \
            m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        macs[0] += o.numel() * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(batch)
    model.train(was_training)
    for h in handles:
        h.remove()
    return int(macs[0])


# --------------------------------------------------------------------------- #
class PeakMemory:
    """Context manager measuring CUDA peak memory in MiB."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_mib = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.base = torch.cuda.memory_allocated(self.device) / 2 ** 20
        else:
            self.base = 0.0
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_mib = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
            self.delta_mib = self.peak_mib - self.base
        return False


@torch.no_grad()
def measure_latency(model: nn.Module, batch: Dict[str, torch.Tensor],
                    warmup: int = 10, iters: int = 50,
                    device: Optional[torch.device] = None) -> Dict[str, float]:
    """End-to-end latency of the **whole** two-modality forward.

    ``batch`` must be the dict the detector consumes (rgb, tir, centres, boxes,
    mask, valid, ...).  Inference-mode placeholders are supplied for the fields
    the forward touches but the losses ignore, so the measured time is a genuine
    inference-time cost rather than a training step.
    """
    device = device or next(model.parameters()).device
    model.eval().to(device)
    b = batch["rgb"].shape[0]
    md = int(batch["boxes_rgb"].shape[1])
    payload = {
        "rgb": batch["rgb"].to(device), "tir": batch["tir"].to(device),
        "centers_rgb": batch["centers_rgb"].to(device),
        "centers_tir": batch["centers_tir"].to(device),
        "sizes_rgb": batch["sizes_rgb"].to(device),
        "sizes_tir": batch["sizes_tir"].to(device),
        "boxes_rgb": batch["boxes_rgb"].to(device),
        "boxes_tir": batch["boxes_tir"].to(device),
        "mask": batch["mask"].to(device),
        "valid": batch["valid"].to(device),
        "rgb_hw": batch["rgb_hw"].to(device),
        "tir_hw": batch["tir_hw"].to(device),
    }
    assert payload["rgb"].shape[0] == b and payload["boxes_rgb"].shape[1] == md
    for _ in range(warmup):
        model(payload)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(payload)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": dt * 1000, "fps": 1.0 / dt, "iters": iters}


# --------------------------------------------------------------------------- #
def profile_model(model: nn.Module, batch_like: Dict[str, torch.Tensor],
                  device: torch.device, iters: int = 30,
                  measure_mem: bool = True) -> Dict[str, object]:
    """Full report: params, MACs, peak VRAM, latency.

    ``batch_like`` is the dict produced by ``check_forward.make_fake_batch`` (or a
    real batch from the dataloader).  The reported latency / MACs are for the
    **inference** path; the training peak memory is measured on a real
    forward+backward with the full loss, which is what determines batch size.
    """
    model = model.to(device).eval()
    payload = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch_like.items()}

    with PeakMemory(device) as mem:
        macs = count_flops(model, payload)
        lat = measure_latency(model, payload, iters=iters, device=device)
        if measure_mem and device.type == "cuda":
            model.train()
            with torch.enable_grad():
                out = model(payload)
                loss, _ = model.compute_losses(out)
                loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            train_peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
            model.eval()
        else:
            train_peak = float("nan")

    return {
        "params": count_params(model),
        "macs": macs,
        "flops_2x_macs": macs * 2,
        "latency": lat,
        "peak_vram_mib": mem.peak_mib if measure_mem else float("nan"),
        "train_peak_vram_mib": train_peak,
        "input_shapes": {"rgb": list(payload["rgb"].shape),
                         "tir": list(payload["tir"].shape)},
    }
