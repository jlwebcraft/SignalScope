"""Robustness and Authenticity Stability benchmarking module for SignalScope.

Evaluates detector invariance across controlled real-world degradations:
1. Pristine Original
2. JPEG Recompression (Quality = 95, 85, 70)
3. Resolution Rescaling (Downsample to 70% and restore to native 32x32)
4. Screenshot Simulation (Subpixel blur + JPEG recompression)
5. Light Edit (90% center crop resized back to 32x32)

Computes:
- Prediction Flip Rate: fraction of hard decisions that invert under transformation
- Mean Absolute Probability Drift: average |p_trans - p_orig|
- Authenticity Stability Score: normalized, bounded [0, 1] consistency metric
"""

import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.utils.logger import logger
from model.dataset import get_default_transforms, get_native_transforms
from model.evaluate import calculate_metrics


# =====================================================================
# 1. Controlled Image Transformations (Operating on native 32x32)
# =====================================================================

def transform_identity(img: Image.Image) -> Image.Image:
    """Returns pristine original image copy."""
    return img.copy()


def transform_jpeg_recompression(img: Image.Image, quality: int = 70) -> Image.Image:
    """Simulates social media or web JPEG compression."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def transform_resize_down_up(img: Image.Image, scale_factor: float = 0.70) -> Image.Image:
    """Simulates resolution reduction followed by upsampling back to original 32x32."""
    w, h = img.size
    down_w = max(16, int(round(w * scale_factor)))
    down_h = max(16, int(round(h * scale_factor)))
    downsampled = img.resize((down_w, down_h), Image.Resampling.BILINEAR)
    return downsampled.resize((w, h), Image.Resampling.BICUBIC)


def transform_screenshot_simulation(img: Image.Image) -> Image.Image:
    """Simulates screenshot capture: subpixel blur + lossy re-encoding."""
    blurred = img.filter(ImageFilter.BoxBlur(radius=0.5))
    return transform_jpeg_recompression(blurred, quality=80)


def transform_light_crop_resize(img: Image.Image, crop_fraction: float = 0.90) -> Image.Image:
    """Simulates slight user crop followed by resizing back to 32x32."""
    w, h = img.size
    crop_w = int(round(w * crop_fraction))
    crop_h = int(round(h * crop_fraction))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


TRANSFORMATION_REGISTRY: Dict[str, Tuple[Callable[[Image.Image], Image.Image], Dict[str, Any]]] = {
    "original": (transform_identity, {"description": "Pristine native 32x32"}),
    "jpeg_95": (lambda img: transform_jpeg_recompression(img, quality=95), {"quality": 95, "description": "High-quality JPEG (Q=95)"}),
    "jpeg_85": (lambda img: transform_jpeg_recompression(img, quality=85), {"quality": 85, "description": "Standard web JPEG (Q=85)"}),
    "jpeg_70": (lambda img: transform_jpeg_recompression(img, quality=70), {"quality": 70, "description": "Aggressive social compression (Q=70)"}),
    "resize": (lambda img: transform_resize_down_up(img, scale_factor=0.70), {"scale_factor": 0.70, "description": "Bilinear downsample (0.7x) + Bicubic restore"}),
    "screenshot": (transform_screenshot_simulation, {"blur_radius": 0.5, "quality": 80, "description": "Subpixel box blur (r=0.5) + JPEG Q=80"}),
    "light_edit": (lambda img: transform_light_crop_resize(img, crop_fraction=0.90), {"crop_fraction": 0.90, "description": "90% center crop + Bilinear restore"}),
}


# =====================================================================
# 2. Metric Calculations
# =====================================================================

def compute_prediction_flip_rate(
    orig_probs: np.ndarray,
    trans_probs: np.ndarray,
    threshold: float = 0.50,
) -> float:
    """Calculates fraction of samples whose binary classification flipped.

    Flip = (orig_prob >= threshold) != (trans_prob >= threshold).
    """
    orig_decisions = (orig_probs >= threshold).astype(int)
    trans_decisions = (trans_probs >= threshold).astype(int)
    flips = np.sum(orig_decisions != trans_decisions)
    return float(flips / len(orig_probs)) if len(orig_probs) > 0 else 0.0


def compute_mean_probability_drift(
    orig_probs: np.ndarray,
    trans_probs: np.ndarray,
) -> float:
    """Calculates mean absolute difference: E[|p_trans - p_orig|]."""
    return float(np.mean(np.abs(trans_probs - orig_probs))) if len(orig_probs) > 0 else 0.0


def compute_authenticity_stability_score(
    orig_prob: float,
    trans_probs: List[float],
    threshold: float = 0.50,
) -> float:
    """Calculates Authenticity Stability Score S in [0, 1] for a single sample.

    Formula:
        S = C * (1.0 - 0.5 * (mean_drift + max_drift))
    where:
        C = fraction of transformations preserving hard decision (Consistency)
        mean_drift = (1/K) * sum(|p_k - p_0|)
        max_drift = max(|p_k - p_0|)

    Properties:
    - Invariant prediction (C=1, drift=0) -> S = 1.00
    - Complete volatility (C=0) -> S = 0.00
    - Multiplicatively penalizes prediction flipping
    - Bounded in [0, 1]
    """
    if not trans_probs:
        return 1.0

    orig_hard = int(orig_prob >= threshold)
    k = len(trans_probs)

    # Consistency C
    matching = sum(1 for p in trans_probs if int(p >= threshold) == orig_hard)
    consistency = matching / k

    # Drift terms
    drifts = [abs(p - orig_prob) for p in trans_probs]
    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))

    combined_penalty = 0.5 * (mean_drift + max_drift)
    score = consistency * max(0.0, 1.0 - combined_penalty)
    return float(max(0.0, min(1.0, score)))


# =====================================================================
# 3. Robustness Evaluation Dataset
# =====================================================================

class RobustnessDataset(Dataset):
    """Dataset applying a specific transformation to native 32x32 images.

    Returns:
    - spatial_tensor (for 224x224 ConvNeXt spatial backbone)
    - native_tensor (for 32x32 native FFT branch)
    - target (ground truth binary label)
    - path (source file path)
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], int, str]],
        transform_fn: Callable[[Image.Image], Image.Image],
        spatial_size: int = 224,
        native_size: int = 32,
    ) -> None:
        self.samples = samples
        self.transform_fn = transform_fn
        self.spatial_tf = get_default_transforms(image_size=spatial_size, is_training=False)
        self.native_tf = get_native_transforms(image_size=native_size, is_training=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_path, label, _ = self.samples[idx]
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")

        # Apply degradation to native image first
        perturbed_img = self.transform_fn(pil_img)

        spatial_tensor = self.spatial_tf(perturbed_img)
        native_tensor = self.native_tf(perturbed_img)
        target = torch.tensor([float(label)], dtype=torch.float32)

        return spatial_tensor, native_tensor, target, str(img_path)


# =====================================================================
# 4. Model Evaluation Runners
# =====================================================================

@torch.no_grad()
def evaluate_model_on_transformed_dataset(
    model: nn.Module,
    samples: List[Tuple[Union[str, Path], int, str]],
    transform_fn: Callable[[Image.Image], Image.Image],
    is_fusion: bool,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Evaluates a detector across samples under a specific transformation.

    Returns:
        (y_true, y_scores, image_paths)
    """
    model.eval()
    dataset = RobustnessDataset(samples, transform_fn=transform_fn)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_targets: List[int] = []
    all_scores: List[float] = []
    all_paths: List[str] = []

    for spatial_imgs, native_imgs, targets, paths in loader:
        spatial_imgs = spatial_imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if is_fusion:
                native_imgs = native_imgs.to(device, non_blocking=True)
                logits = model(spatial_imgs, x_native=native_imgs)
            else:
                logits = model(spatial_imgs)
            probs = torch.sigmoid(logits)

        all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
        all_scores.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
        all_paths.extend(paths)

    return np.array(all_targets, dtype=int), np.array(all_scores, dtype=float), all_paths
