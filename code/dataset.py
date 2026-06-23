"""
dataset.py
----------
Gold Jewelry Authentication — Dataset construction, item-level splitting,
preprocessing pipeline, and augmentation.

Paper: "EfficientNet-Based Non-Destructive Visual Authentication of Gold Jewelry"
Target journal: MDPI Sensors
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# Label mapping
CLASS_TO_IDX = {"authentic": 1, "counterfeit": 0}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Subtype definitions (for per-subtype evaluation)
AUTHENTIC_SUBTYPES  = ["24K", "22K", "18K", "14K"]
COUNTERFEIT_SUBTYPES = ["gold_plated_brass", "gold_colored_alloy",
                         "tungsten_core", "gold_painted"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset metadata structure
# Expected directory layout:
#
#   data_root/
#     authentic/
#       item_001/   (one folder per physical item)
#         img_001.png
#         img_002.png
#         ...
#       item_002/
#         ...
#     counterfeit/
#       item_301/
#         ...
#
# Each item folder may contain an optional metadata.json:
#   {"subtype": "18K", "item_id": "item_001"}
# ─────────────────────────────────────────────────────────────────────────────

def load_item_metadata(data_root: str) -> List[Dict]:
    """
    Scan data_root and build a list of item-level records.
    Each record: {item_id, class_label, class_idx, subtype, images: [paths]}
    """
    records = []
    data_root = Path(data_root)

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue

        for item_dir in sorted(class_dir.iterdir()):
            if not item_dir.is_dir():
                continue

            # collect image paths
            images = sorted([
                str(p) for p in item_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}
            ])
            if not images:
                continue

            # optional metadata
            meta_path = item_dir / "metadata.json"
            subtype = "unknown"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                subtype = meta.get("subtype", "unknown")

            records.append({
                "item_id":    item_dir.name,
                "class_name": class_name,
                "class_idx":  class_idx,
                "subtype":    subtype,
                "images":     images,
            })

    return records


def item_level_split(
    records:    List[Dict],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split items (not images) into train / val / test partitions.
    Stratified by class_idx to preserve class balance.

    Critical: splitting is done at ITEM level to prevent data leakage
    where multiple images of the same physical item could appear across
    train and evaluation splits.
    """
    rng = random.Random(seed)

    # group items by class
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class_idx"]].append(rec)

    train_items, val_items, test_items = [], [], []

    for class_idx, items in by_class.items():
        items = items.copy()
        rng.shuffle(items)
        n      = len(items)
        n_tr   = int(n * train_frac)
        n_val  = int(n * val_frac)

        train_items.extend(items[:n_tr])
        val_items.extend(items[n_tr:n_tr + n_val])
        test_items.extend(items[n_tr + n_val:])

    return train_items, val_items, test_items


def make_kfold_splits(
    records: List[Dict],
    k:       int = 5,
    seed:    int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create k stratified item-level folds for cross-validation.
    Returns list of (train_items, val_items) tuples.
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class_idx"]].append(rec)

    # build k folds per class, then merge
    class_folds = {}
    for class_idx, items in by_class.items():
        items = items.copy()
        rng.shuffle(items)
        class_folds[class_idx] = [items[i::k] for i in range(k)]

    folds = []
    for fold_idx in range(k):
        val_items   = []
        train_items = []
        for class_idx in by_class:
            for f_i, fold in enumerate(class_folds[class_idx]):
                if f_i == fold_idx:
                    val_items.extend(fold)
                else:
                    train_items.extend(fold)
        folds.append((train_items, val_items))

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GoldJewelryDataset(Dataset):
    """
    PyTorch Dataset for gold jewelry authentication.
    Each sample is a single image; label is item-level class.
    """

    def __init__(
        self,
        item_records: List[Dict],
        transform:    Optional[object] = None,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int, str, str]] = []  # (path, label, subtype, item_id)

        for rec in item_records:
            for img_path in rec["images"]:
                self.samples.append((
                    img_path,
                    rec["class_idx"],
                    rec["subtype"],
                    rec["item_id"],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        path, label, subtype, item_id = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, subtype, item_id


# ─────────────────────────────────────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Training augmentation pipeline (Section III-C of paper):
    - Random horizontal flip (p=0.5)
    - Random rotation ±15°
    - Random perspective (distortion 0.2)
    - Color jitter: brightness/contrast/saturation ±30%, hue ±10%
    - Resize to 224×224, normalise with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Validation / test transform — resize and normalise only.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_cross_domain_transform(condition: str, severity: float = 1.0) -> transforms.Compose:
    """
    Cross-domain robustness evaluation transforms.

    Parameters
    ----------
    condition : str
        One of: 'lighting_bright', 'lighting_dark', 'jpeg_compression',
                'background_white', 'background_black', 'background_texture'
    severity : float
        Scaling factor for condition intensity (0.0 – 1.0).
    """
    base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if condition == "lighting_bright":
        # +40% brightness
        base.append(transforms.ColorJitter(brightness=(1.4 * severity, 1.4 * severity)))
    elif condition == "lighting_dark":
        # -40% brightness
        factor = max(0.1, 1.0 - 0.4 * severity)
        base.append(transforms.ColorJitter(brightness=(factor, factor)))
    elif condition == "jpeg_compression":
        # simulate JPEG at quality = int(50 + (1-severity)*45)
        quality = int(50 + (1.0 - severity) * 45)
        base.append(transforms.Lambda(
            lambda img: _jpeg_compress(img, quality)
        ))
    # background changes are handled at dataset level (image compositing)

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Compress PIL image to JPEG at given quality and reload."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    train_items: List[Dict],
    val_items:   List[Dict],
    test_items:  Optional[List[Dict]] = None,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    pin_memory:  bool = True,
) -> Dict[str, DataLoader]:

    loaders = {
        "train": DataLoader(
            GoldJewelryDataset(train_items, transform=get_train_transform()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            GoldJewelryDataset(val_items, transform=get_eval_transform()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    if test_items is not None:
        loaders["test"] = DataLoader(
            GoldJewelryDataset(test_items, transform=get_eval_transform()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def compute_class_weights(train_items: List[Dict]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy loss.
    w_c = N / (2 * n_c)
    """
    labels = [rec["class_idx"] for rec in train_items
              for _ in rec["images"]]
    n_total = len(labels)
    weights = torch.zeros(2)
    for c in range(2):
        n_c = labels.count(c)
        weights[c] = n_total / (2.0 * n_c) if n_c > 0 else 1.0
    return weights
