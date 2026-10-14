"""UAV-CB dataset: LabelMe-style JSON, paired RGB/TIR, sequence-level split.

Facts established by ``probe_fast.py`` on the local copy of UAV-CB
(``/home/yangjuanfeng/lab/shared/datasets/UAV-CB``, 3393 paired frames):

* 5 scenes (building 1191, cloud 590, ground 168, powerline 555, vegetation 889).
* **Every frame has exactly one object**, label ``drone``, in both modalities.
* RGB is 1920x1080, TIR is 640x512 -> the two modalities are *not* registered.
* Box size (RGB pixels): median 68x52, p10 39x26, min 18x9 -> tiny targets.
* Normalised centre offset RGB->TIR: mean (0.002, -0.035), std (0.042, 0.028),
  p95 |offset| (0.080, 0.074).  There is a systematic vertical shift **and** a
  per-frame residual of the same order -- this is exactly why a single hard
  affine warp is not enough and why the prior is modelled as a Gaussian.
* Each scene contains exactly 2 sequence ids (the ``1`` / ``2`` token in
  ``<scene>_visible_<seq>_<frame>.jpg``).

Split protocol (sequence level, see :func:`build_split`)
---------------------------------------------------------
For every scene, sequence ``2`` goes **entirely** to *test*; sequence ``1`` is cut
*contiguously* into train / val.  No frame from a test sequence ever appears in
train or val, so val is also an honest estimate.  Test GT is never used for
prior fitting, model selection or any training signal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# label parsing
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    scene: str
    seq: str
    frame_id: str
    rgb_path: str
    tir_path: str
    rgb_size: Tuple[int, int]      # (W, H)
    tir_size: Tuple[int, int]
    # normalised (cx, cy, w, h) per modality -- the *first* annotated object,
    # kept for the geometric-prior fitting utilities that predate multi-object.
    rgb_box: Optional[np.ndarray]
    tir_box: Optional[np.ndarray]
    # Multi-object payload.  ``rgb_boxes``/``tir_boxes`` are (M,4) normalised and
    # ``rgb_labels``/``tir_labels`` are (M,) integer class ids; the two modalities
    # may hold different numbers of objects.  Declared *after* ``tir_box`` with
    # defaults so existing positional construction keeps working.
    rgb_boxes: Optional[np.ndarray] = None
    tir_boxes: Optional[np.ndarray] = None
    rgb_labels: Optional[np.ndarray] = None
    tir_labels: Optional[np.ndarray] = None


def parse_labelme(path: str) -> Tuple[List[np.ndarray], int, int]:
    """Return (normalised boxes, width, height) from a LabelMe JSON."""
    boxes, _, w, h = parse_labelme_ex(path)
    return boxes, w, h


def parse_labelme_ex(path: str) -> Tuple[List[np.ndarray], List[str], int, int]:
    """Return ``(normalised boxes, label names, width, height)`` from a LabelMe JSON.

    Every rectangle is kept -- the older ``parse_labelme`` kept only the first --
    so a multi-object frame is no longer silently truncated to one box.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    w, h = float(d["imageWidth"]), float(d["imageHeight"])
    boxes: List[np.ndarray] = []
    labels: List[str] = []
    for s in d.get("shapes", []):
        if s.get("shape_type") not in (None, "rectangle"):
            continue
        pts = np.asarray(s["points"], dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bw, bh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        boxes.append(np.array([(x0 + bw / 2) / w, (y0 + bh / 2) / h,
                               bw / w, bh / h], dtype=np.float32))
        labels.append(str(s.get("label", "")))
    return boxes, labels, int(w), int(h)


def class_ids(labels: Sequence[str], class_names: Sequence[str]) -> List[int]:
    """Map label names to contiguous ids; unknown names are **dropped**, not remapped.

    Silently mapping a typo to class 0 is how a mislabelled dataset becomes an
    unexplained accuracy floor, so a name outside ``class_names`` raises the
    per-object loss instead.
    """
    table = {name: i for i, name in enumerate(class_names)}
    return [table[name] for name in labels if name in table]


def list_frames(root: str, scenes: Sequence[str],
                class_names: Sequence[str] = ("drone",)) -> List[Frame]:
    frames: List[Frame] = []
    for scene in scenes:
        vdir = os.path.join(root, scene, "visible")
        tdir = os.path.join(root, scene, "thermal")
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".jpg"):
                continue
            stem = fn[:-4]
            tokens = stem.split("_")
            seq, frame_id = tokens[-2], tokens[-1]
            tir_fn = fn.replace("_visible_", "_thermal_")
            rgb_json = os.path.join(vdir, stem + ".json")
            tir_json = os.path.join(tdir, tir_fn[:-4] + ".json")
            if not (os.path.exists(os.path.join(tdir, tir_fn)) and
                    os.path.exists(rgb_json) and os.path.exists(tir_json)):
                continue
            rb, rlab_names, rw, rh = parse_labelme_ex(rgb_json)
            tb, tlab_names, tw, th = parse_labelme_ex(tir_json)
            # Keep only objects whose label is a known class; the box and its
            # label are filtered together so they can never drift apart.
            rb, rlab = _filter_known(rb, rlab_names, class_names, rw, rh)
            tb, tlab = _filter_known(tb, tlab_names, class_names, tw, th)
            frames.append(Frame(
                scene=scene, seq=seq, frame_id=frame_id,
                rgb_path=os.path.join(vdir, fn),
                tir_path=os.path.join(tdir, tir_fn),
                rgb_size=(rw, rh), tir_size=(tw, th),
                rgb_box=rb[0] if rb else None,
                tir_box=tb[0] if tb else None,
                rgb_boxes=np.asarray(rb, np.float32) if rb else None,
                tir_boxes=np.asarray(tb, np.float32) if tb else None,
                rgb_labels=np.asarray(rlab, np.int64) if rlab else None,
                tir_labels=np.asarray(tlab, np.int64) if tlab else None,
            ))
    return frames


def _filter_known(boxes, names, class_names, w, h):
    """Keep (box, class-id) pairs whose label is in ``class_names``."""
    table = {name: i for i, name in enumerate(class_names)}
    kept_boxes, kept_labels = [], []
    for box, name in zip(boxes, names):
        if name in table:
            kept_boxes.append(box)
            kept_labels.append(table[name])
    return kept_boxes, kept_labels


# --------------------------------------------------------------------------- #
# fixed split manifests (the project's own paper-rule 6:2:2 reconstruction)
# --------------------------------------------------------------------------- #
DEFAULT_SPLIT_DIR = "/home/yangjuanfeng/lab/projects/SkyGuard-RGBT/dataset/UAV-CB/splits"


def load_split_manifest(split_dir: str, root: str,
                        scenes: Sequence[str] = (),
                        class_names: Sequence[str] = ("drone",)) -> Dict[str, List[Frame]]:
    """Load a fixed ``{train,val,test}.txt`` manifest of visible annotation stems.

    ``dataset/UAV-CB/splits/README.md`` documents this as a **local reconstruction** of
    the paper's 6:2:2 stratified split (seed 20261009, 3383 pairs after excluding
    10 manually reviewed bad pairs).  Re-using it is what makes our numbers
    comparable with the other baselines in this repository.

    RGB and thermal always share split membership, so the pairing stays intact.
    """
    by_key: Dict[str, Frame] = {}
    for f in list_frames(root, scenes, class_names):
        rel = os.path.relpath(f.rgb_path, root).replace(os.sep, "/")
        by_key[rel[:-4]] = f                      # strip ".jpg"

    out: Dict[str, List[Frame]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(path):
            out[split] = []
            continue
        frames = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if not key:
                    continue
                # manifests store 'scene/visible/stem'; tolerate both forms
                for cand_key in (key, key + ".jpg"):
                    if cand_key in by_key:
                        frames.append(by_key[cand_key])
                        break
                else:
                    alt = key.replace("_visible_", "_thermal_")
                    if alt in by_key:
                        frames.append(by_key[alt])
        out[split] = frames
    return out


# --------------------------------------------------------------------------- #
# sequence-level split
# --------------------------------------------------------------------------- #
def build_split(frames: List[Frame], train_ratio: float = 0.8,
                val_ratio: float = 0.1, test_ratio: float = 0.1,
                test_seq: str = "2", seed: int = 0) -> Dict[str, List[Frame]]:
    """Sequence-level split.

    Sequences whose id equals ``test_seq`` go to *test* as a whole.  The remaining
    sequences are cut **contiguously** (train first, then val) so that temporally
    adjacent frames never straddle the train/val boundary.
    """
    rng = np.random.RandomState(seed)
    train: List[Frame] = []
    val: List[Frame] = []
    test: List[Frame] = []

    by_scene: Dict[str, Dict[str, List[Frame]]] = {}
    for f in frames:
        by_scene.setdefault(f.scene, {}).setdefault(f.seq, []).append(f)

    for scene in sorted(by_scene):
        seqs = by_scene[scene]
        test_seqs = sorted(s for s in seqs if s == test_seq)
        train_seqs = sorted(s for s in seqs if s != test_seq)
        for s in test_seqs:
            test.extend(sorted(seqs[s], key=lambda f: f.frame_id))
        # contiguous cut over the concatenated train sequences
        pool: List[Frame] = []
        for s in train_seqs:
            pool.extend(sorted(seqs[s], key=lambda f: f.frame_id))
        n = len(pool)
        if n == 0:
            continue
        # honour the requested ratios on the *non-test* pool, rescaled so that
        # train+val fill it exactly
        total = train_ratio + val_ratio
        n_train = int(round(n * train_ratio / total))
        n_val = n - n_train
        train.extend(pool[:n_train])
        val.extend(pool[n_train:n_train + n_val])
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# paired augmentation
# --------------------------------------------------------------------------- #
class PairAugment:
    """Geometry-preserving paired augmentation.

    Full frames are resized independently and flipped together. The returned
    transform lets the model conjugate the geometry prior for flipped inputs.
    Photometric jitter is applied independently per modality --
    that is exactly the kind of nuisance the fusion module must not depend on.
    """

    def __init__(self, scale_range=(0.75, 1.25), hflip: float = 0.5,
                 color_jitter: float = 0.3, tir_gain_jitter: float = 0.15,
                 train: bool = True, rgb_normalization="imagenet", modality_dropout=0.,
                 degrade_prob=0., translation_prob=0., translation_range=.03,
                 zoom_prob=0., blur_max=1.5, noise_std=12.):
        self.rgb_normalization=rgb_normalization
        self.modality_dropout=modality_dropout;self.degrade_prob=degrade_prob
        self.translation_prob=translation_prob;self.translation_range=translation_range
        self.scale_range = scale_range
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.tir_gain_jitter = tir_gain_jitter
        self.train = train
        self.zoom_prob=zoom_prob;self.blur_max=blur_max;self.noise_std=noise_std
        # Shared by persistent DataLoader workers, including DDP worker processes.
        self._mild=torch.zeros((),dtype=torch.uint8).share_memory_()

    def set_mild_mode(self, enabled):
        self._mild.fill_(int(enabled))

    def _resize(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR), copy=True)

    def __call__(self, rgb: np.ndarray, tir: np.ndarray,
                 rgb_box: Optional[np.ndarray], tir_box: Optional[np.ndarray],
                 rgb_size: Tuple[int, int], tir_size: Tuple[int, int],
                 out_rgb: Tuple[int, int], out_tir: Tuple[int, int], return_metadata=False):
        """Single-box entry point (unchanged contract, kept for compatibility).

        Accepts a ``(4,)`` box (or ``None``) per modality and returns it in the
        same shape.  Multi-object callers use :meth:`augment_boxes`, which is the
        single implementation both paths share.
        """
        r_in = _as_boxes(rgb_box)
        t_in = _as_boxes(tir_box)
        single_r = rgb_box is None or np.asarray(rgb_box).ndim == 1
        single_t = tir_box is None or np.asarray(tir_box).ndim == 1
        rgb, tir, rb, tb, _, _, meta = self.augment_boxes(
            rgb, tir, r_in, t_in, rgb_size, tir_size, out_rgb, out_tir)
        if single_r:
            rb = None if rb is None else rb[0]
        if single_t:
            tb = None if tb is None else tb[0]
        transform = meta if return_metadata else meta["rgb"]
        return rgb, tir, rb, tb, transform

    def augment_boxes(self, rgb: np.ndarray, tir: np.ndarray,
                      rgb_boxes: Optional[np.ndarray], tir_boxes: Optional[np.ndarray],
                      rgb_size: Tuple[int, int], tir_size: Tuple[int, int],
                      out_rgb: Tuple[int, int], out_tir: Tuple[int, int],
                      rgb_labels: Optional[np.ndarray] = None,
                      tir_labels: Optional[np.ndarray] = None):
        """Paired, geometry-preserving augmentation for **any** number of objects.

        ``rgb_boxes``/``tir_boxes`` are ``(M,4)`` normalised ``(cx,cy,w,h)`` (or
        ``None``); the two modalities may hold different ``M``.  Per-object
        clipping can drop an individual box (less than half visible), and the
        dropped row is removed from the box **and** label arrays together so a
        box can never inherit another object's class.

        Returns ``(rgb, tir, rgb_boxes, tir_boxes, rgb_labels, tir_labels, meta)``.
        """
        oh_r, ow_r = out_rgb
        oh_t, ow_t = out_tir
        flip = self.train and np.random.rand() < self.hflip
        mild = bool(self._mild.item())
        scale = float(np.random.uniform(*self.scale_range)) if self.train else 1.0
        if self.zoom_prob > 0:
            scale = (float(np.random.uniform(.96, 1.04) if mild else np.random.uniform(*self.scale_range))
                     if self.train and np.random.rand() < self.zoom_prob else 1.)

        def geom(img, ow, oh, boxes):
            transform = np.eye(3, dtype=np.float32)
            h, w = img.shape[:2]
            if scale != 1.0 and self.zoom_prob == 0:
                img = self._resize(img, max(1, round(w * scale)), max(1, round(h * scale)))
            img = self._resize(img, ow, oh)
            if self.zoom_prob > 0 and scale != 1.:
                sw, sh = max(1, round(ow * scale)), max(1, round(oh * scale))
                resized = self._resize(img, sw, sh)
                dx, dy = (ow - sw) // 2, (oh - sh) // 2
                fill = np.mean(img, axis=(0, 1)).astype(np.uint8)
                canvas = np.empty_like(img); canvas[...] = fill
                x0, x1 = max(0, -dx), min(sw, ow - dx); y0, y1 = max(0, -dy), min(sh, oh - dy)
                canvas[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = resized[y0:y1, x0:x1]
                img = canvas
                transform[0, 0] = sw / ow; transform[1, 1] = sh / oh
                transform[:2, 2] = [dx / ow, dy / oh]
                if boxes is not None:
                    boxes = boxes.copy()
                    boxes[:, :2] = boxes[:, :2] @ transform[:2, :2].T + transform[:2, 2]
                    boxes[:, 2:] *= np.array([sw / ow, sh / oh], np.float32)
            if flip:
                img = img[:, ::-1].copy()
            if boxes is not None:
                boxes = boxes.copy()
                if flip:
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            if flip:
                F = np.eye(3, dtype=np.float32); F[0] = [-1., 0., 1.]
                transform = F @ transform
            return img, boxes, transform

        rgb, rb, transform_r = geom(rgb, ow_r, oh_r, rgb_boxes)
        tir, tb, transform_t = geom(tir, ow_t, oh_t, tir_boxes)

        transforms = []
        geometry_valid = []

        def translate(img, boxes, base_transform):
            transform = base_transform.copy()
            if self.train and not mild and np.random.rand() < self.translation_prob:
                hh, ww = img.shape[:2]
                dx = int(round(np.random.uniform(-self.translation_range, self.translation_range) * ww))
                dy = int(round(np.random.uniform(-self.translation_range, self.translation_range) * hh))
                shifted = np.zeros_like(img)
                x0, x1 = max(0, -dx), min(ww, ww - dx)
                y0, y1 = max(0, -dy), min(hh, hh - dy)
                shifted[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = img[y0:y1, x0:x1]
                img = shifted
                transform[:2, 2] += np.array([dx / ww, dy / hh], np.float32)
                if boxes is not None:
                    boxes = boxes.copy()
                    boxes[:, :2] += np.array([dx / ww, dy / hh], np.float32)
            if boxes is not None and len(boxes):
                # A box is a usable geometric anchor only while it is fully inside.
                inside = (((boxes[:, :2] - boxes[:, 2:] / 2) >= 0).all(-1)
                          & ((boxes[:, :2] + boxes[:, 2:] / 2) <= 1).all(-1))
                geometry_valid.append(float(inside.all()))
                lo = np.maximum(boxes[:, :2] - boxes[:, 2:] / 2, 0)
                hi = np.minimum(boxes[:, :2] + boxes[:, 2:] / 2, 1)
                visible = np.maximum(hi - lo, 0).prod(-1) / np.maximum(boxes[:, 2:].prod(-1), 1e-8)
                keep = (visible >= .5) & (hi > lo).all(-1)
                # ``np.r_`` would stack the two (k,2) halves into (2k,2); the box
                # halves must be concatenated along the last axis to stay (k,4).
                boxes = (np.concatenate([((lo + hi) / 2)[keep], (hi - lo)[keep]], -1)
                         .astype(np.float32) if keep.any() else None)
            else:
                keep = np.ones(0 if boxes is None else len(boxes), np.bool_)
                geometry_valid.append(0.)
            transforms.append(transform)
            return img, boxes, keep

        rgb, rb, keep_r = translate(rgb, rb, transform_r)
        tir, tb, keep_t = translate(tir, tb, transform_t)
        # Boxes and labels are filtered by the *same* mask, so a surviving box can
        # never inherit a dropped object's class.
        rgb_labels = _subset_labels(rgb_labels, keep_r)
        tir_labels = _subset_labels(tir_labels, keep_t)

        available = np.ones(2, np.float32)
        if self.train and not mild and np.random.rand() < self.degrade_prob:
            from PIL import ImageFilter
            if np.random.randint(2) == 0:
                rgb = np.array(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(np.random.uniform(.2, self.blur_max))), copy=True)
            else:
                tir = np.clip(tir.astype(np.float32) + np.random.normal(0, self.noise_std, tir.shape), 0, 255).astype(np.uint8)
        if self.train and not mild and np.random.rand() < self.modality_dropout:
            dropped = int(np.random.randint(2)); available[dropped] = 0
            if dropped == 0:
                rgb = np.zeros_like(rgb)
            else:
                tir = np.zeros_like(tir)

        if self.train:
            if self.color_jitter > 0:
                j = self.color_jitter * (.5 if mild else 1.)
                for c in range(3):
                    rgb[..., c] = np.clip(rgb[..., c].astype(np.float32) *
                                          np.random.uniform(1 - j, 1 + j) +
                                          np.random.uniform(-18 * j, 18 * j), 0, 255)
            if self.tir_gain_jitter > 0:
                tir = np.clip(tir.astype(np.float32) *
                              np.random.uniform(1 - self.tir_gain_jitter,
                                                1 + self.tir_gain_jitter) +
                              np.random.uniform(-10, 10) * self.tir_gain_jitter, 0, 255)

        if available[0] == 0:
            rgb.fill(0)
        if available[1] == 0:
            tir.fill(0)
        rgb = rgb.astype(np.float32) / 255.0
        if self.rgb_normalization == "imagenet":
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        elif self.rgb_normalization != "unit":
            raise ValueError(self.rgb_normalization)
        tir = tir.astype(np.float32) / 255.0                      # (H,W) in [0,1]
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tir = np.ascontiguousarray(tir)[None]
        rb = None if rb is None else rb.astype(np.float32)
        tb = None if tb is None else tb.astype(np.float32)
        meta = {"rgb": transforms[0], "tir": transforms[1], "available": available,
                "geometry_valid": np.array(geometry_valid, np.float32)}
        return rgb, tir, rb, tb, rgb_labels, tir_labels, meta


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
class UAVCBDataset(Dataset):
    """Returns a dict consumed by :class:`pgsf.engine.trainer.PGSFTrainer`.

    Multi-object / multi-class / mixed-resolution safe: every annotated box of
    both modalities is kept (up to ``max_objs``), each modality carries its own
    padding mask and integer labels, and the two modalities may hold different
    numbers of objects.  Instances are **not** assumed to pair up by index --
    ``corr_pairs`` is emitted only when the pairing is unambiguous (one object on
    each side) or explicitly requested via ``pair_by_index``.
    """

    def __init__(self, frames: List[Frame], rgb_size=(768, 1280), tir_size=(512, 640),
                 train: bool = True, augment: Optional[PairAugment] = None,
                 max_objs: int = 50, class_names: Sequence[str] = ("drone",),
                 pair_by_index: bool = False, multi_object: bool = True):
        self.frames = frames
        self.rgb_size = tuple(rgb_size)
        self.tir_size = tuple(tir_size)
        self.train = train
        self.aug = augment or PairAugment(train=train)
        self.max_objs = int(max_objs)
        self.class_names = tuple(class_names)
        self.pair_by_index = bool(pair_by_index)
        self.multi_object = bool(multi_object)

    def __len__(self) -> int:
        return len(self.frames)

    def _load(self, path: str, thermal: bool) -> np.ndarray:
        img = Image.open(path)
        # ``copy()`` is essential: PIL may hand back a read-only buffer, which makes
        # the in-place colour jitter below raise.  uint8 keeps the memory small.
        if thermal:
            return np.array(img.convert("L"), dtype=np.uint8, copy=True)
        return np.array(img.convert("RGB"), dtype=np.uint8, copy=True)

    def _corr_pairs(self, mask_r: np.ndarray, mask_t: np.ndarray):
        """Instance correspondence matrix, or ``None`` when it cannot be justified.

        ``None`` is the honest answer for a multi-object frame with no explicit
        pairing: guessing "same index = same object" would inject wrong
        supervision, so the correspondence losses are simply not computed.
        """
        n_r, n_t = int((mask_r > 0.5).sum()), int((mask_t > 0.5).sum())
        if self.pair_by_index:
            pairs = np.zeros((self.max_objs, self.max_objs), np.float32)
            both = np.where((mask_r > 0.5) & (mask_t > 0.5))[0]
            pairs[both, both] = 1.0
            return pairs
        if n_r == 1 and n_t == 1:
            pairs = np.zeros((self.max_objs, self.max_objs), np.float32)
            pairs[int(np.argmax(mask_r)), int(np.argmax(mask_t))] = 1.0
            return pairs
        return None

    def __getitem__(self, i: int) -> Dict[str, object]:
        f = self.frames[i]
        rgb = self._load(f.rgb_path, thermal=False)
        tir = self._load(f.tir_path, thermal=True)

        r_boxes = f.rgb_boxes
        if r_boxes is None and f.rgb_box is not None:
            r_boxes = np.asarray(f.rgb_box, np.float32).reshape(1, 4)
        t_boxes = f.tir_boxes
        if t_boxes is None and f.tir_box is not None:
            t_boxes = np.asarray(f.tir_box, np.float32).reshape(1, 4)
        r_labels = f.rgb_labels if f.rgb_labels is not None else (
            None if r_boxes is None else np.zeros(len(r_boxes), np.int64))
        t_labels = f.tir_labels if f.tir_labels is not None else (
            None if t_boxes is None else np.zeros(len(t_boxes), np.int64))

        if not self.multi_object:
            # Legacy single-object behaviour: keep only the first annotated box.
            r_boxes = None if r_boxes is None else r_boxes[:1]
            t_boxes = None if t_boxes is None else t_boxes[:1]
            r_labels = None if r_labels is None else r_labels[:1]
            t_labels = None if t_labels is None else t_labels[:1]

        rgb, tir, rb, tb, rlab, tlab, meta = self.aug.augment_boxes(
            rgb, tir, r_boxes, t_boxes, f.rgb_size, f.tir_size,
            self.rgb_size, self.tir_size, r_labels, t_labels)

        boxes_r, mask_r, lab_r = _pad_objects(rb, rlab, self.max_objs)
        boxes_t, mask_t, lab_t = _pad_objects(tb, tlab, self.max_objs)
        centers_r, sizes_r = boxes_r[:, :2].copy(), boxes_r[:, 2:].copy()
        centers_t, sizes_t = boxes_t[:, :2].copy(), boxes_t[:, 2:].copy()
        n_r, n_t = float(mask_r.sum() > 0), float(mask_t.sum() > 0)

        out: Dict[str, object] = {
            "rgb": torch.from_numpy(rgb),
            "tir": torch.from_numpy(tir),
            # pixel boxes for evaluation / visualisation (each modality's own space)
            "boxes_rgb_px": torch.from_numpy(_to_px(boxes_r, mask_r, f.rgb_size)),
            "boxes_tir_px": torch.from_numpy(_to_px(boxes_t, mask_t, f.tir_size)),
            # normalised boxes + per-modality masks + integer class ids for losses
            "boxes_rgb": torch.from_numpy(boxes_r),
            "boxes_tir": torch.from_numpy(boxes_t),
            "centers_rgb": torch.from_numpy(centers_r),
            "centers_tir": torch.from_numpy(centers_t),
            "sizes_rgb": torch.from_numpy(sizes_r),
            "sizes_tir": torch.from_numpy(sizes_t),
            "mask": torch.from_numpy(np.maximum(mask_r, mask_t)),
            "mask_rgb": torch.from_numpy(mask_r),
            "mask_tir": torch.from_numpy(mask_t),
            "labels_rgb": torch.from_numpy(lab_r),
            "labels_tir": torch.from_numpy(lab_t),
            "modality_available": torch.from_numpy(meta["available"]),
            "geometry_valid": torch.from_numpy(meta["geometry_valid"]),
            "valid": torch.tensor(n_r * n_t, dtype=torch.float32),
            "scene": f.scene,
            "frame_id": f.frame_id,
            "rgb_hw": torch.tensor(f.rgb_size[::-1], dtype=torch.float32),
            "tir_hw": torch.tensor(f.tir_size[::-1], dtype=torch.float32),
            "geo_transform": torch.from_numpy(meta["rgb"]),
            "geo_transform_rgb": torch.from_numpy(meta["rgb"]),
            "geo_transform_tir": torch.from_numpy(meta["tir"]),
            "geo_inverse_rgb": torch.from_numpy(np.linalg.inv(meta["rgb"]).astype(np.float32)),
            "geo_inverse_tir": torch.from_numpy(np.linalg.inv(meta["tir"]).astype(np.float32)),
            "n_objects_rgb": torch.tensor(int(mask_r.sum()), dtype=torch.long),
            "n_objects_tir": torch.tensor(int(mask_t.sum()), dtype=torch.long),
            "index": i,
        }
        pairs = self._corr_pairs(mask_r, mask_t)
        if pairs is not None:
            out["corr_pairs"] = torch.from_numpy(pairs)
        return out


def _as_boxes(box) -> Optional[np.ndarray]:
    """``(4,)``/``None``/``(M,4)`` -> ``(M,4) float32`` or ``None``."""
    if box is None:
        return None
    arr = np.asarray(box, np.float32)
    if arr.size == 0:
        return None
    return arr.reshape(-1, 4)


def _subset_labels(labels, keep) -> Optional[np.ndarray]:
    """Apply a row-keep mask to a label array (``None`` in -> ``None`` out)."""
    if labels is None:
        return None
    labels = np.asarray(labels)
    keep = np.asarray(keep, np.bool_)
    if keep.size == 0:
        return labels[:0]
    if keep.size != labels.shape[0]:
        # No boxes survived (or none were given): the label array follows suit.
        return labels[:0]
    return labels[keep]


def _pad_objects(boxes, labels, max_objs: int):
    """``(K,4)`` boxes + ``(K,)`` labels -> padded ``(m,4)``, ``(m,)`` mask, ``(m,)`` ids.

    Raises on overflow instead of truncating: a silently dropped object is a
    training-set change that never shows up in any log.
    """
    m = int(max_objs)
    out_boxes = np.zeros((m, 4), np.float32)
    mask = np.zeros(m, np.float32)
    out_labels = np.full(m, -1, np.int64)
    if boxes is None or len(boxes) == 0:
        return out_boxes, mask, out_labels
    if len(boxes) > m:
        raise ValueError(f"{len(boxes)} objects exceed max_objs={m}; raise the budget "
                         "instead of truncating the annotation")
    k = len(boxes)
    out_boxes[:k] = np.asarray(boxes, np.float32)
    mask[:k] = 1.0
    if labels is None:
        out_labels[:k] = 0
    else:
        labels = np.asarray(labels, np.int64)
        if labels.shape[0] != k:
            raise ValueError(f"got {k} boxes but {labels.shape[0]} labels")
        out_labels[:k] = labels
    return out_boxes, mask, out_labels


def _to_px(boxes_norm: np.ndarray, mask: np.ndarray,
           size_wh: Tuple[int, int]) -> np.ndarray:
    """Normalised (cx,cy,w,h) -> pixel (x0,y0,x1,y1); masked entries become 0."""
    w, h = size_wh
    out = np.zeros_like(boxes_norm)
    if boxes_norm.size == 0:
        return out
    cx, cy, bw, bh = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
    out[:, 0] = (cx - bw / 2) * w
    out[:, 1] = (cy - bh / 2) * h
    out[:, 2] = (cx + bw / 2) * w
    out[:, 3] = (cy + bh / 2) * h
    out[mask == 0] = 0.0
    return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Collate that keeps string / metadata entries as python lists."""
    out: Dict[str, object] = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        elif isinstance(vals[0], (int, float, bool)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def make_augment(data_cfg, train=True):
    """One preprocessing contract for train, validation, overfit and visualization."""
    return PairAugment(data_cfg.scale_range,data_cfg.hflip,data_cfg.color_jitter,
                       data_cfg.tir_gain_jitter,train,data_cfg.rgb_normalization,
                       data_cfg.modality_dropout,data_cfg.degrade_prob,
                       data_cfg.translation_prob,data_cfg.translation_range,
                       data_cfg.zoom_prob,data_cfg.blur_max,data_cfg.noise_std)
