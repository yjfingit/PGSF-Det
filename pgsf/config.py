"""Configuration objects for PGSF-Det.

A single nested dataclass tree is used so that every model / data / loss switch is
reachable from one place, and so that the ablation grid (A0 - A6) is a pure
function of the config flags.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import yaml


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = "/home/yangjuanfeng/lab/shared/datasets/UAV-CB"
    scenes: Tuple[str, ...] = ("building", "cloud", "ground", "powerline", "vegetation")
    # When ``split_dir`` exists we reuse the repository's fixed paper-rule 6:2:2
    # manifest (seed 20261009).  This is what makes PGSF-Det numbers comparable
    # with the other baselines here.  Falls back to the internal sequence-level
    # split when the manifest is missing.
    split_dir: str = "/home/yangjuanfeng/lab/projects/SkyGuard-RGBT/dataset/UAV-CB/splits"
    use_fixed_split: bool = True
    # network input resolution (H, W) for each modality -- deliberately different,
    # because UAV-CB RGB (1920x1080) and TIR (640x512) have different native sizes.
    rgb_size: Tuple[int, int] = (800, 1280)
    tir_size: Tuple[int, int] = (512, 640)
    # internal sequence-level split ratios (only used when use_fixed_split=False)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # test split GT must never touch training / prior fitting / model selection
    allow_test: bool = False
    max_train_per_scene: int = 0  # 0 = use all (debug with e.g. 40)
    num_workers: int = 8
    seed: int = 0
    # augmentation
    scale_range: Tuple[float, float] = (0.75, 1.25)
    hflip: float = 0.5
    color_jitter: float = 0.3
    tir_gain_jitter: float = 0.15
    rgb_normalization: str = "imagenet"
    modality_dropout: float = 0.0
    degrade_prob: float = 0.0
    translation_prob: float = 0.0
    translation_range: float = 0.03
    zoom_prob: float = 0.0  # opt-in true canvas zoom; legacy resize path preserved
    blur_max: float = 1.5
    noise_std: float = 12.0
    close_aug_epochs: int = 0
    # geometric prior fitting: only drawn from the *training* split
    geo_prior_samples: int = 3000
    # --- multi-object / multi-class data (v3) ---------------------------------
    # ``class_names`` fixes the label -> class-id mapping.  A label outside this
    # list is dropped rather than silently remapped, so a typo cannot become class 0.
    class_names: Tuple[str, ...] = ("drone",)
    # Pad *every* annotated box (not only the first) and emit per-modality masks,
    # integer labels, and the instance-correspondence matrix ``corr_pairs``.
    # ``corr_pairs`` is what makes the correspondence losses computable; when a
    # dataset cannot provide it (no instance pairing), the objective falls back to
    # the two detection losses only.
    multi_object: bool = True
    # Pair the i-th object of RGB with the i-th of TIR.  This is an *assumption*
    # about the annotation order, so it is off by default; UAV-CB configs switch it
    # on because that dataset has exactly one object per modality, which makes the
    # pairing exact rather than assumed.
    pair_by_index: bool = False


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    architecture_version: int = 2
    backbone_name: str = "lightcsp"  # configs use locally pretrained yolo11s
    backbone_weights: str = ""
    # --- backbone -----------------------------------------------------------
    width_mult: float = 1.0
    backbone_depths: Tuple[int, ...] = (3, 6, 9, 3)
    use_dw_bottleneck: bool = True
    # --- neck ---------------------------------------------------------------
    neck_dim: int = 128
    neck_out: int = 128  # all pyramid levels unified to this width
    neck_repeats: int = 3
    # --- MSCE ---------------------------------------------------------------
    msce_alpha_max: float = 0.5
    msce_width: int = 24
    # --- candidate generation ------------------------------------------------
    n_cand: int = 64  # per modality at stage-1 candidate generation
    n_global: int = 16  # global high-response candidates fed to the fusion
    n_local_samples: int = 8  # local deformable sampling points
    heat_stride: int = 8  # legacy v1; v2 directly predicts on P2 and P3
    nms_kernel: int = 5  # local NMS window on the heatmap
    proposal_width: int = 32
    injection_end_fraction: float = 0.7
    inject_gt_ratio: float = 0.25  # training-time GT-perturbed candidates
    # --- transformers --------------------------------------------------------
    d_model: int = 128
    n_heads: int = 4  # 4 x 32 = 128
    n_cand_late: int = 32  # per modality entering the object-level fusion
    ff_mult: int = 4
    context_dim: int = 128
    use_image_context: bool = True
    use_covariance_sampling: bool = True
    use_roi_tokens: bool = True
    use_null: bool = True
    geo_jitter_prob: float = 0.15
    geo_jitter_std: float = 0.025
    hard_negative_weight: float = 2.0
    lambda_geo: float = 0.35  # weight of log P_G inside attention logits
    lambda_box: float = 0.35  # weight of log P(b_j | b_i) inside late attention
    # --- v3 object-level correspondence fusion ---------------------------------
    # One object-level fusion only: 128 candidates per modality, a 64-d Q/K coarse
    # retrieval over {candidates + 8x8 grid tokens + null}, Top-2 Gaussian anchors,
    # 8 local samples per anchor spread over P2-P5, and a rejection gate.
    use_v3_fusion: bool = True   # stage >= 2 only; False = independent detection
    retrieval_dim: int = 64      # Q/K width of the coarse retrieval
    corr_top_k: int = 2          # number of retrieved anchors
    corr_samples: int = 8        # samples per anchor Gaussian, over P2-P5
    corr_levels: Tuple[str, ...] = ("P2", "P3", "P4", "P5")
    corr_heads: int = 4
    corr_ff_mult: int = 4
    corr_sigma_floor: float = 0.02
    lambda_local: float = 0.35   # weight of log N(sample; mu, Sigma) inside attention
    corr_hard_negative_weight: float = 2.0
    v3_grid: int = 8             # grid tokens are v3_grid x v3_grid
    v3_grid_levels: Tuple[str, ...] = ("P3", "P4")
    roi_width: int = 64          # width of the RoI spatial descriptor
    num_classes: int = 1         # 1 keeps the legacy single-logit head
    w_corr: float = 0.3          # weight of (null + retrieval) correspondence terms
    w_corr_geo: float = 0.2      # weight of the anchor Gaussian NLL
    # --- v4: asymmetric semantic / localization fusion (TD-Fusion) -------------
    # One object-level cross-modal interaction.  ``n_cand`` is the K of the spec
    # (default 128 in the v4 configs); ``td_top_l`` the Top-L anchors, and
    # ``td_samples`` the sample points per anchor.
    use_td_fusion: bool = True   # stage >= 2 only; False = independent detection
    cand_levels: Tuple[str, ...] = ("P2", "P3", "P4", "P5")
    cand_min_per_level: int = 8  # reserved per-level quota of the K candidates
    roi_size: int = 3            # RoI descriptor S is [B,K,128,roi_size,roi_size]
    roi_sampling_ratio: int = 2
    td_top_l: int = 2
    td_samples: int = 4
    td_heads: int = 4
    td_ff_mult: int = 4
    alpha_init: float = 0.1      # learnable localization-modulation scale
    # --- v4 loss weights ------------------------------------------------------
    lambda_iou: float = 2.5      # CIoU weight in the final detection loss
    lambda_l1: float = 1.0       # L1 weight in the final detection loss
    w_self: float = 0.5          # auxiliary un-fused (self-modal) detection loss
    w_prop: float = 0.5          # proposal head loss
    lambda_pbox: float = 1.5     # coarse-box CIoU inside the proposal loss
    lambda_quality: float = 0.5  # proposal quality BCE inside the proposal loss
    lambda_match: float = 1.0    # bidirectional matching CE
    w_cons: float = 0.1          # class-consistency JS
    cons_start_fraction: float = 0.5   # from this training fraction on, mutual selection is trusted
    cons_margin: float = 1.0     # min top1-top2 similarity margin for the mutual-selection pair
    use_match_loss: bool = True
    use_cons_loss: bool = True
    # --- detection head -------------------------------------------------------
    head_width: int = 128
    n_layers_head: int = 2
    max_wh: float = 0.5  # upper bound of the predicted (normalised) box size
    # assignment thresholds for candidate <-> GT matching
    assign_pos_thr: float = 0.5      # retained legacy API field
    assign_center_thr: float = 0.05  # centre distance (normalised) threshold
    # --- loss weights ---------------------------------------------------------
    w_det: float = 1.0
    w_cls: float = 1.0
    w_box: float = 2.5
    w_nwd: float = 0.5
    # v2 supervises both dense scales. Historical v1 calibration does not
    # transfer to the pretrained backbone and quality-aware proposal heads.
    w_heat: float = 1.0
    w_pre_box: float = 1.5
    w_pre_quality: float = 0.5
    w_size_geo: float = 0.05
    score_fusion_power: float = 0.5
    # NLL may be negative. Its constant reference shift does not affect gradients;
    # tune this weight by supervision gradients and held-out validation.
    w_geo: float = 0.3
    w_assoc_mid: float = 0.5
    w_assoc_late: float = 0.5
    # --- ablation switches (A0 - A6) -----------------------------------------
    use_msce: bool = True  # A1
    use_mid_fusion: bool = True  # A2
    use_geo_prior: bool = True  # A3
    use_gate: bool = True  # A4
    use_late_fusion: bool = True  # A5
    use_nwd: bool = True  # A6


@dataclass
class OptimConfig:
    epochs: int = 60
    batch_size: int = 8
    lr: float = 2e-4
    lr_backbone_mult: float = 0.1  # backbone uses a smaller LR
    lr_new_mult: float = 2.0  # freshly initialised fusion modules use larger LR
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.02
    optimizer: str = "adamw"
    ema: bool = True
    ema_decay: float = 0.999
    amp: bool = True
    grad_clip: float = 5.0
    pretrained_backbone: str = ""  # path to a compatible CNN checkpoint
    freeze_backbone: bool = False
    early_stop_patience: int = 0  # 0 disables early stopping
    early_stop_min_delta: float = 1e-4
    log_interval: int = 50
    min_epochs: int = 0  # delay early stop until inference-like training is covered


@dataclass
class EvalConfig:
    # single-target protocol used by the UAV-CB literature: one GT per modality
    max_det: int = 50  # architecture keeps multi-target headroom up to 50
    score_thr: float = 0.05
    nms_thr: float = 0.45
    nms_topk: int = 300
    conf_thr: float = 0.001  # for AP computation keep everything
    # COCO AP uses 101 recall thresholds; single-target mode reports the same keys
    iou_thrs: Tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass
class PGSFConfig:
    name: str = "pgsf_a6"
    stage: int = 2  # 1 = single-modality pretrain, 2 = joint cross-modal
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "outputs/pgsf_a6"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


# --------------------------------------------------------------------------- #
# yaml loading with dotted override
# --------------------------------------------------------------------------- #
def _from_dict(cls, data: Dict[str, Any]):
    if data is None:
        return cls()
    kwargs = {}
    valid = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in data.items():
        if k not in valid:
            raise KeyError(f"unknown config key '{k}' for {cls.__name__}")
        kwargs[k] = v
    return cls(**kwargs)


def _apply_override(obj, dotted: str, value: str):
    """Apply 'model.wd=384'-style override, coercing to the field type."""
    parts = dotted.split(".")
    tgt = obj
    for p in parts[:-1]:
        if not hasattr(tgt, p):
            raise KeyError(f"unknown config section '{p}' in override '{dotted}'")
        tgt = getattr(tgt, p)
    leaf = parts[-1]
    if not hasattr(tgt, leaf):
        raise KeyError(f"unknown config key '{dotted}'")
    cur = getattr(tgt, leaf)
    if isinstance(value, str):
        if isinstance(cur, bool):
            value = value.lower() in ("1", "true", "yes", "y")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            value = int(value)
        elif isinstance(cur, float):
            value = float(value)
        elif isinstance(cur, (tuple, list)):
            parsed = yaml.safe_load(value)
            if not isinstance(parsed, (tuple, list)):
                parsed = [yaml.safe_load(x.strip()) for x in value.split(",")]
            value = tuple(parsed) if isinstance(cur, tuple) else list(parsed)
    setattr(tgt, leaf, value)


def load_config(path: str | None = None, overrides: List[str] | None = None) -> PGSFConfig:
    """Build a config from an optional yaml file plus dotted CLI overrides."""
    raw: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    name = raw.pop("name", None)
    stage = raw.pop("stage", None)
    seed = raw.pop("seed", None)
    device = raw.pop("device", None)
    out_dir = raw.pop("out_dir", None)

    cfg = PGSFConfig()
    cfg.data = _from_dict(DataConfig, raw.pop("data", None))
    cfg.model = _from_dict(ModelConfig, raw.pop("model", None))
    cfg.optim = _from_dict(OptimConfig, raw.pop("optim", None))
    cfg.eval = _from_dict(EvalConfig, raw.pop("eval", None))
    if name:
        cfg.name = name
    if stage is not None:
        cfg.stage = int(stage)
    if seed is not None:
        cfg.seed = int(seed)
        cfg.data.seed = int(seed)
    if device:
        cfg.device = device
    if out_dir:
        cfg.out_dir = out_dir
    if raw:
        raise KeyError(f"unknown top-level config keys: {list(raw)}")

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must look like key=value")
        k, v = ov.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
    return cfg


def save_config(cfg: PGSFConfig, path: str) -> None:
    cfg.save(path)
