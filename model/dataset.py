"""Dataset abstractions and loaders for SignalScope.

Supports standard folder structures (real/fake or real/synthetic),
metadata tracking of generator families, and generator-aware honest splits.
"""

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
    """Scans a directory for real and synthetic images.

    Recognized folder conventions:
    - root/real/*, root/fake/*
    - root/real/*, root/synthetic/*
    - root/real/*, root/synthetic/<generator>/*
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        logger.warning(f"Dataset directory does not exist: {root_path}")
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: List[Tuple[Path, int, str]] = []

    # Check for real images
    real_dirs = [root_path / "real", root_path / "0_real", root_path / "authentic"]
    for rdir in real_dirs:
        if rdir.exists() and rdir.is_dir():
            for f in rdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in valid_extensions:
                    samples.append((f, 0, "real"))

    # Check for synthetic images
    synth_dirs = [root_path / "fake", root_path / "synthetic", root_path / "1_fake", root_path / "ai"]
    for sdir in synth_dirs:
        if sdir.exists() and sdir.is_dir():
            for f in sdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in valid_extensions:
                    # Generator name from parent subfolder if present
                    generator = f.parent.name if f.parent != sdir else "unspecified_generator"
                    samples.append((f, 1, generator))

    logger.info(
        f"Scanned {root_path}: found {len(samples)} samples "
        f"({sum(1 for s in samples if s[1] == 0)} real, {sum(1 for s in samples if s[1] == 1)} synthetic)."
    )
    return samples
