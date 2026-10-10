"""PGSF-Det: Probabilistic-Geometry-guided Sparse Fusion detector for
un-registered RGB-T small drone detection (UAV-CB).

Sub-packages
------------
models   network building blocks + full detector
data     UAV-CB dataset / paired augmentation / sequence split
engine   trainer + COCO-style evaluator
losses   focal / CIoU / NWD / center heatmap / geometry NLL / association
utils    profiler (params, FLOPs, VRAM, latency) and misc helpers
"""

__version__ = "0.1.0"

from pgsf.config import PGSFConfig, load_config, save_config  # noqa: F401
