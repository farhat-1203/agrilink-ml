"""
modules/dataset.py  ─  PlantVillage Dataset + Augmentation Pipeline

PlantVillage folder layout:
    <root>/
        Apple___Apple_scab/       ← "crop___condition" naming
        Apple___Black_rot/
        Tomato___healthy/
        ...

Key design decisions:
  • Stratified split  → each class has the same val fraction
  • WeightedSampler   → rare classes are oversampled during training
  • Heavy augmentation (ColorJitter, blur, perspective, erasing)
    because PlantVillage images are lab-quality; field photos are noisy
  • Binary disease flag per sample (derived from "healthy" in folder name)
    used by the auxiliary loss head
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMG_MEAN, IMG_SIZE, IMG_STD,
    BATCH_SIZE, NUM_WORKERS, VAL_FRACTION,
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Augmentation transforms ───────────────────────────────────────────────────

def train_transforms() -> transforms.Compose:
    """
    Aggressive augmentation to bridge the gap between clean PlantVillage
    images and noisy real-world field photos.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),  # occlusion sim
    ])


def val_transforms() -> transforms.Compose:
    """Deterministic centre-crop — used for validation AND inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    Loads images from PlantVillage-style folder layout.

    Each __getitem__ returns:
        image       – (3, 224, 224) float tensor
        label       – int class index (multi-class disease)
        is_diseased – float 0.0 or 1.0 (binary auxiliary target)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose],
        split: str = "train",          # "train" | "val"
        val_fraction: float = VAL_FRACTION,
        seed: int = 42,
    ):
        self.transform = transform
        root = Path(root_dir)

        # Discover class folders
        class_dirs = sorted(d for d in root.iterdir()
                            if d.is_dir() and not d.name.startswith("."))
        if not class_dirs:
            raise ValueError(f"No class sub-folders found in {root_dir}")

        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}

        # Build flat sample list: (Path, class_idx, is_diseased)
        all_samples: List[Tuple[Path, int, bool]] = []
        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            diseased = "healthy" not in cls_dir.name.lower()
            for p in cls_dir.iterdir():
                if p.suffix.lower() in VALID_EXTS:
                    all_samples.append((p, idx, diseased))

        if not all_samples:
            raise ValueError(f"No images found under {root_dir}")

        # Stratified train/val split
        rng = random.Random(seed)
        by_class: dict = defaultdict(list)
        for s in all_samples:
            by_class[s[1]].append(s)

        train_s, val_s = [], []
        for cls_samples in by_class.values():
            rng.shuffle(cls_samples)
            n_val = max(1, int(len(cls_samples) * val_fraction))
            val_s.extend(cls_samples[:n_val])
            train_s.extend(cls_samples[n_val:])

        self.samples = train_s if split == "train" else val_s

        # Per-sample weights for balanced sampling
        counts = [0] * len(self.class_names)
        for _, idx, _ in self.samples:
            counts[idx] += 1
        inv = [1.0 / max(c, 1) for c in counts]
        self.sample_weights = [inv[idx] for _, idx, _ in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, class_idx, is_diseased = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))  # black placeholder

        if self.transform:
            img = self.transform(img)

        return {
            "image"      : img,
            "label"      : torch.tensor(class_idx,    dtype=torch.long),
            "is_diseased": torch.tensor(float(is_diseased), dtype=torch.float32),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Returns (train_loader, val_loader, class_names).

    WeightedRandomSampler is used on train_loader so rare disease classes
    are seen as often as common ones — critical for PlantVillage's class
    imbalance.
    """
    train_ds = PlantVillageDataset(root_dir, train_transforms(), split="train")
    val_ds   = PlantVillageDataset(root_dir, val_transforms(),   split="val")

    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, train_ds.class_names


# ── Synthetic data generator (no real dataset needed for CI / demo) ───────────

def generate_synthetic_dataset(
    out_dir: str,
    images_per_class: int = 80,
    img_size: int = 256,
    seed: int = 0,
) -> str:
    """
    Creates a tiny PlantVillage-style dataset using PIL + NumPy.
    Healthy classes → green ellipse.  Diseased → brown spots added.
    Returns the root directory path.
    """
    import numpy as np

    classes = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Potato___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Apple___healthy",
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Pepper___healthy",
        "Pepper___Bacterial_spot",
    ]

    rng = random.Random(seed)
    np.random.seed(seed)
    root = Path(out_dir)

    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        is_healthy = "healthy" in cls.lower()

        # Base colour: green for healthy, yellow-brown for diseased
        if is_healthy:
            base = (rng.randint(30, 80), rng.randint(120, 190), rng.randint(30, 80))
        else:
            base = rng.choice([(160,100,40),(140,115,55),(90,65,45),(175,155,75)])

        for i in range(images_per_class):
            arr = np.full((img_size, img_size, 3),
                          [rng.randint(20,50), rng.randint(70,130), rng.randint(20,50)],
                          dtype=np.uint8)
            # Draw ellipse (the crop surface)
            cx, cy = img_size//2, img_size//2
            rx, ry = img_size//3, img_size//4
            color = tuple(max(0, min(255, base[c]+rng.randint(-20,20))) for c in range(3))
            ys, xs = np.ogrid[:img_size, :img_size]
            mask = ((xs-cx)**2/rx**2 + (ys-cy)**2/ry**2) <= 1
            arr[mask] = color

            # Noise
            arr = np.clip(arr.astype(np.int16)+np.random.randint(-15,15,arr.shape),
                          0, 255).astype(np.uint8)

            # Disease spots
            if not is_healthy:
                for _ in range(rng.randint(4, 12)):
                    sx, sy = rng.randint(cx-rx, cx+rx), rng.randint(cy-ry, cy+ry)
                    sr = rng.randint(4, 13)
                    spot = (rng.randint(55,105), rng.randint(35,75), rng.randint(10,40))
                    ys2, xs2 = np.ogrid[:img_size, :img_size]
                    smask = (xs2-sx)**2 + (ys2-sy)**2 <= sr**2
                    arr[smask] = spot

            Image.fromarray(arr).save(cls_dir / f"{i:04d}.jpg", quality=85)

    total = len(classes) * images_per_class
    print(f"[dataset] Synthetic: {total} images | {len(classes)} classes → {out_dir}")
    return str(root)
