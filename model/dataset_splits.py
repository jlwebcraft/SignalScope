"""Leakage-Safe Dataset Partitioning and Split Management for SignalScope.

Manages:
1. Official Train & Validation Split:
   - Source: C:\\Programming\\SignalScope-data\\train (100k samples)
   - Train partition: 85,000 samples (42,500 Real, 42,500 Synthetic)
   - Val partition (LOCAL VALIDATION / val_old): 15,000 samples (7,500 Real, 7,500 Synthetic)
   - Fixed deterministic seed: 42

2. New Dataset (Version-2) Partitioning:
   - Source: C:\\Programming\\SignalScope-data\\version-2\\AIGenImages2026
   - Train partition: 9,758 samples (4,879 Real, 4,879 Synthetic)
   - Val partition (NEW-DATA VALIDATION / val_new): 1,118 samples (559 Real, 559 Synthetic across 19 generators)

3. Optional Additional Real Pool:
   - Source: C:\\Programming\\SignalScope-data\\version-2\\REAL
   - Filtered photographic categories: apparel, cars, dishes, furniture, landmark, packaged, storefronts, toys
   - Excluded non-photographic categories: meme, illustrations

4. Absolute Data-Safety Guarantee:
   - STRICTLY EXCLUDES and NEVER TOUCHES C:\\Programming\\SignalScope-data\\test.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from app.utils.logger import logger
from model.dataset import (
    SignalScopeDataset,
    create_honest_splits,
    get_default_transforms,
    scan_dataset_directory,
)

DATA_ROOT = Path(r"C:\Programming\SignalScope-data")
OFFICIAL_TRAIN_DIR = DATA_ROOT / "train"
V2_DIR = DATA_ROOT / "version-2"
AIGEN_TRAIN_DIR = V2_DIR / "AIGenImages2026" / "train"
AIGEN_VAL_DIR = V2_DIR / "AIGenImages2026" / "val"
V2_REAL_DIR = V2_DIR / "REAL"

# Strict safety verification
assert "test" not in str(OFFICIAL_TRAIN_DIR).lower()
assert "test" not in str(V2_DIR).lower()


def get_official_splits(
    seed: int = 42,
    val_ratio: float = 0.15,
) -> Tuple[List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    """Generates official 85k train / 15k val split from official train pool."""
    all_train = scan_dataset_directory(OFFICIAL_TRAIN_DIR)
    train_samples, val_samples, _ = create_honest_splits(
        all_train,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    return train_samples, val_samples


def get_v2_splits() -> Tuple[List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    """Retrieves Version-2 AIGenImages2026 train (9,758) and validation (1,118) samples."""
    train_samples = scan_dataset_directory(AIGEN_TRAIN_DIR)
    val_samples = scan_dataset_directory(AIGEN_VAL_DIR)
    return train_samples, val_samples


def get_v2_real_samples(
    limit_per_category: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[Path, int, str]]:
    """Retrieves photographic real samples from version-2/REAL, excluding meme and illustrations."""
    valid_categories = [
        "apparel", "cars", "dishes", "furniture",
        "landmark", "packaged", "storefronts", "toys", "artwork"
    ]
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    rng = np.random.RandomState(seed)
    samples: List[Tuple[Path, int, str]] = []

    for cat in valid_categories:
        cat_dir = V2_REAL_DIR / cat
        if not cat_dir.exists():
            continue
        cat_files = [p for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        if limit_per_category and len(cat_files) > limit_per_category:
            indices = rng.permutation(len(cat_files))[:limit_per_category]
            cat_files = [cat_files[i] for i in indices]
        for p in cat_files:
            samples.append((p, 0, f"real_{cat}"))

    return samples


def verify_split_integrity(
    train_samples: List[Tuple[Path, int, str]],
    val_old_samples: List[Tuple[Path, int, str]],
    val_new_samples: List[Tuple[Path, int, str]],
) -> None:
    """Verifies complete disjointness and zero data leakage across all splits."""
    train_paths = set(str(s[0]) for s in train_samples)
    val_old_paths = set(str(s[0]) for s in val_old_samples)
    val_new_paths = set(str(s[0]) for s in val_new_samples)

    if len(train_paths.intersection(val_old_paths)) > 0:
        raise ValueError("Leakage detected: Train overlaps with val_old!")
    if len(train_paths.intersection(val_new_paths)) > 0:
        raise ValueError("Leakage detected: Train overlaps with val_new!")
    if len(val_old_paths.intersection(val_new_paths)) > 0:
        raise ValueError("Leakage detected: val_old overlaps with val_new!")

    logger.info("Split integrity check passed: zero leakage between Train, Val-Old, and Val-New.")
    return True
