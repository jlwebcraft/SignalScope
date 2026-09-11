"""Dataset abstractions and loaders for SignalScope.

Supports standard folder structures (real/fake or real/synthetic),
metadata tracking of generator families, and generator-aware honest splits.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    torch = None
    transforms = None
    Dataset = object
    HAS_TORCH = False

from app.utils.image import load_image_safely
from app.utils.logger import logger


def get_default_transforms(
    image_size: int = 224,
    is_training: bool = False,
) -> transforms.Compose:
    """Returns standard torchvision transforms with normalization."""
    # ImageNet normalization standard for pretrained vision backbones
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


class SignalScopeDataset(Dataset):
    """PyTorch Dataset for authenticity classification.

    Tracks:
    - Image path
    - Label (0 for Real, 1 for Synthetic / AI)
    - Generator family name (e.g. 'Midjourney', 'StableDiffusion', 'DALL-E', 'FLUX', 'RealCamera')
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], int, str]],
        transform: Optional[Callable] = None,
    ) -> None:
        """Args:

        samples: List of tuples: (image_path, label_int, generator_name).
        transform: Optional torchvision transform callable.
        """
        self.samples = samples
        self.transform = transform or get_default_transforms(is_training=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        img_path, label, generator = self.samples[idx]

        try:
            pil_img = load_image_safely(img_path)
        except Exception as exc:
            logger.warning(f"Error loading {img_path}: {exc}. Using blank placeholder.")
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        tensor = self.transform(pil_img)
        target = torch.tensor([float(label)], dtype=torch.float32)
        metadata = {"path": str(img_path), "generator": generator}

        return tensor, target, metadata


def scan_dataset_directory(
    root_dir: Union[str, Path],
) -> List[Tuple[Path, int, str]]:
    """Scans a directory for real and synthetic images with fast scandir.

    Recognized folder conventions (case-insensitive):
    - root/REAL/*, root/FAKE/*
    - root/real/*, root/fake/*
    - root/authentic/*, root/synthetic/*
    - root/real/*, root/synthetic/<generator>/*
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        logger.warning(f"Dataset directory does not exist: {root_path}")
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: List[Tuple[Path, int, str]] = []

    real_folder_names = {"real", "0_real", "authentic"}
    fake_folder_names = {"fake", "synthetic", "1_fake", "ai"}

    # Inspect immediate child directories
    for child in root_path.iterdir():
        if not child.is_dir():
            continue

        cname_lower = child.name.lower()
        if cname_lower in real_folder_names:
            # Scan real images
            with os.scandir(child) as it:
                for entry in it:
                    if entry.is_file():
                        p = Path(entry.path)
                        if p.suffix.lower() in valid_extensions:
                            samples.append((p, 0, "real"))
        elif cname_lower in fake_folder_names:
            # Check if there are generator subdirectories or direct images
            has_subdirs = False
            for sub in child.iterdir():
                if sub.is_dir():
                    has_subdirs = True
                    gen_name = sub.name
                    with os.scandir(sub) as it:
                        for entry in it:
                            if entry.is_file():
                                p = Path(entry.path)
                                if p.suffix.lower() in valid_extensions:
                                    samples.append((p, 1, gen_name))
            if not has_subdirs:
                # Direct images in FAKE / synthetic folder
                with os.scandir(child) as it:
                    for entry in it:
                        if entry.is_file():
                            p = Path(entry.path)
                            if p.suffix.lower() in valid_extensions:
                                samples.append((p, 1, "synthetic"))

    # Fallback to rglob if standard folders were not found at root level
    if not samples:
        for f in root_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_extensions:
                parent_lower = f.parent.name.lower()
                if "real" in parent_lower or "authentic" in parent_lower:
                    samples.append((f, 0, "real"))
                elif "fake" in parent_lower or "synthetic" in parent_lower or "ai" in parent_lower:
                    samples.append((f, 1, f.parent.name))

    logger.info(
        f"Scanned {root_path}: found {len(samples):,} samples "
        f"({sum(1 for s in samples if s[1] == 0):,} real, {sum(1 for s in samples if s[1] == 1):,} synthetic)."
    )
    return samples


def create_honest_splits(
    samples: List[Tuple[Union[str, Path], int, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    holdout_generators: Optional[List[str]] = None,
) -> Tuple[List[Tuple[Path, int, str]], List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    """Partitions samples into train, val, and test splits without data leakage.

    Supports:
    - Generator-aware holdout: Specific generator families are reserved for unseen-generator evaluation.
    - Stratified splitting by binary class label for remaining data.
    - Deterministic random seeding.
    - Strict non-overlapping path verification.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    if not samples:
        return [], [], []

    rng = np.random.RandomState(seed)
    holdout_set = set(g.lower() for g in (holdout_generators or []))

    test_samples: List[Tuple[Path, int, str]] = []
    pool_samples: List[Tuple[Path, int, str]] = []

    for path, label, gen in samples:
        p = Path(path)
        if gen.lower() in holdout_set:
            test_samples.append((p, label, gen))
        else:
            pool_samples.append((p, label, gen))

    # Separate pool by class for stratification
    real_pool = [s for s in pool_samples if s[1] == 0]
    synth_pool = [s for s in pool_samples if s[1] == 1]

    def split_class_pool(pool: List[Tuple[Path, int, str]]):
        n = len(pool)
        indices = rng.permutation(n)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio)) if not holdout_generators else 0

        val_idx = set(indices[:n_val])
        test_idx = set(indices[n_val : n_val + n_test])

        val = [pool[i] for i in val_idx]
        test = [pool[i] for i in test_idx]
        train = [pool[i] for i in range(n) if i not in val_idx and i not in test_idx]
        return train, val, test

    real_train, real_val, real_test = split_class_pool(real_pool)
    synth_train, synth_val, synth_test = split_class_pool(synth_pool)

    train_samples = real_train + synth_train
    val_samples = real_val + synth_val
    test_samples = test_samples + real_test + synth_test

    # Shuffle final splits
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    # Verification of zero leakage / disjointness
    train_paths = set(str(s[0]) for s in train_samples)
    val_paths = set(str(s[0]) for s in val_samples)
    test_paths = set(str(s[0]) for s in test_samples)

    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected: Train and Val overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected: Train and Test overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected: Val and Test overlap!"

    logger.info(
        f"Honest Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} "
        f"(Holdout generators: {list(holdout_set) if holdout_set else 'None'})"
    )
    return train_samples, val_samples, test_samples
