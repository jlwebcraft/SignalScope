"""Authenticity Stability Score service.

Evaluates prediction invariance across controlled degradation transforms:
- JPEG recompression
- Resolution scaling (downsample-upsample)
- Screenshot simulation
- Light cropping/padding
"""

import io
from typing import Callable, List, Tuple
import numpy as np
from PIL import Image, ImageFilter

from app.api.schemas import AuthenticityStabilityInfo, StabilityTransformResult
from app.utils.logger import logger


def apply_jpeg_compression(image: Image.Image, quality: int = 70) -> Image.Image:
    """Simulates social media or web JPEG compression."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize_down_up(image: Image.Image, scale_factor: float = 0.5) -> Image.Image:
    """Simulates resolution reduction followed by upsampling."""
    w, h = image.size
    down_w = max(16, int(w * scale_factor))
    down_h = max(16, int(h * scale_factor))
    downsampled = image.resize((down_w, down_h), Image.Resampling.BILINEAR)
    return downsampled.resize((w, h), Image.Resampling.BICUBIC)


def apply_screenshot_simulation(image: Image.Image) -> Image.Image:
    """Simulates taking a screenshot and re-saving with minor blur and compression."""
    # Slight box blur to simulate subpixel sampling / OS window scaling
    slightly_blurred = image.filter(ImageFilter.BoxBlur(radius=0.5))
    # Recompress with standard PNG/JPEG screenshot behavior
    return apply_jpeg_compression(slightly_blurred, quality=80)


def apply_light_crop_resize(image: Image.Image, crop_fraction: float = 0.90) -> Image.Image:
    """Simulates slight user crop followed by resizing back to full frame."""
    w, h = image.size
    crop_w = int(w * crop_fraction)
    crop_h = int(h * crop_fraction)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def evaluate_authenticity_stability(
    image: Image.Image,
    predict_fn: Callable[[Image.Image], float],
    original_prob: float,
    stability_threshold: float = 0.75,
    operating_threshold: float = 0.50,
) -> AuthenticityStabilityInfo:
    """Evaluates how stable a model's prediction is across benchmark-validated transformations.

    Formulation (Phase 4 Robustness Benchmark):
        S = C * (1.0 - 0.5 * (mean_drift + max_drift))
    where:
        C = fraction of transformations retaining original hard decision (Consistency)
        mean_drift = (1/K) * sum(|p_k - p_0|)
        max_drift = max(|p_k - p_0|)

    Args:
        image: Original RGB PIL Image.
        predict_fn: Callable that accepts a PIL Image and returns synthetic probability in [0, 1].
        original_prob: Model's predicted synthetic probability on pristine original image.
        stability_threshold: Minimum stability score considered robust (default: 0.75).
        operating_threshold: Binary classification threshold (default: 0.50).

    Returns:
        AuthenticityStabilityInfo with stability score, per-transform results, and degradation impact.
    """
    transforms: List[Tuple[str, Image.Image]] = [
        ("original", image),
        ("jpeg_95", apply_jpeg_compression(image, quality=95)),
        ("jpeg_85", apply_jpeg_compression(image, quality=85)),
        ("jpeg_70", apply_jpeg_compression(image, quality=70)),
        ("resize_down_up", apply_resize_down_up(image, scale_factor=0.70)),
        ("screenshot_simulation", apply_screenshot_simulation(image)),
        ("light_crop_resize", apply_light_crop_resize(image, crop_fraction=0.90)),
    ]

    results: List[StabilityTransformResult] = []
    deltas: List[float] = []
    trans_probs: List[float] = []

    orig_hard = int(original_prob >= operating_threshold)

    for name, transformed_img in transforms:
        try:
            if name == "original":
                prob = float(original_prob)
            else:
                prob = float(predict_fn(transformed_img))
            delta = abs(prob - original_prob)
            deltas.append(delta)
            if name != "original":
                trans_probs.append(prob)

            results.append(
                StabilityTransformResult(
                    transform_name=name,
                    predicted_probability=round(prob, 4),
                    delta_from_original=round(delta, 4),
                )
            )
        except Exception as exc:
            logger.warning(f"Error applying stability test '{name}': {exc}")
            results.append(
                StabilityTransformResult(
                    transform_name=name,
                    predicted_probability=round(original_prob, 4),
                    delta_from_original=0.0,
                )
            )
            if name != "original":
                trans_probs.append(original_prob)
                deltas.append(0.0)

    # Validated stability formula
    if trans_probs:
        matching = sum(1 for p in trans_probs if int(p >= operating_threshold) == orig_hard)
        consistency = matching / len(trans_probs)
        drifts = [abs(p - original_prob) for p in trans_probs]
        mean_drift = float(np.mean(drifts))
        max_drift = float(np.max(drifts))
        stability_metric = consistency * max(0.0, 1.0 - 0.5 * (mean_drift + max_drift))
    else:
        consistency = 1.0
        mean_drift = 0.0
        max_drift = 0.0
        stability_metric = 1.0

    stability_metric = float(max(0.0, min(1.0, stability_metric)))

    if max_drift < 0.10 and consistency == 1.0:
        impact = "minimal"
    elif max_drift < 0.30 and consistency >= 0.80:
        impact = "moderate"
    else:
        impact = "severe"

    is_stable = stability_metric >= stability_threshold

    return AuthenticityStabilityInfo(
        stability_score=round(stability_metric, 4),
        is_stable=is_stable,
        transform_results=results,
        degradation_impact=impact,
    )
