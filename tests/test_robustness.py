"""Tests for Phase 4 Robustness and Authenticity Stability benchmarking."""

import numpy as np
from PIL import Image
import pytest

from app.services.stability import evaluate_authenticity_stability
from model.robustness import (
    compute_authenticity_stability_score,
    compute_mean_probability_drift,
    compute_prediction_flip_rate,
    transform_identity,
    transform_jpeg_recompression,
    transform_light_crop_resize,
    transform_resize_down_up,
    transform_screenshot_simulation,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Provides a synthetic 32x32 RGB test image."""
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_transformations_preserve_native_dimensions(sample_pil_image: Image.Image):
    """Verifies that all controlled transformations preserve native 32x32 dimensions."""
    orig_size = sample_pil_image.size  # (32, 32)

    t_ident = transform_identity(sample_pil_image)
    assert t_ident.size == orig_size

    t_jpeg95 = transform_jpeg_recompression(sample_pil_image, quality=95)
    assert t_jpeg95.size == orig_size

    t_jpeg70 = transform_jpeg_recompression(sample_pil_image, quality=70)
    assert t_jpeg70.size == orig_size

    t_resize = transform_resize_down_up(sample_pil_image, scale_factor=0.70)
    assert t_resize.size == orig_size

    t_screen = transform_screenshot_simulation(sample_pil_image)
    assert t_screen.size == orig_size

    t_crop = transform_light_crop_resize(sample_pil_image, crop_fraction=0.90)
    assert t_crop.size == orig_size


def test_prediction_flip_rate_calculation():
    """Tests flip rate metric calculation."""
    # Threshold 0.50
    orig = np.array([0.9, 0.8, 0.1, 0.2])
    # Case 1: no flips
    trans_same = np.array([0.95, 0.75, 0.05, 0.3])
    assert compute_prediction_flip_rate(orig, trans_same, threshold=0.50) == 0.0

    # Case 2: 2 flips out of 4 (sample 0 and sample 2)
    trans_flipped = np.array([0.4, 0.8, 0.6, 0.2])
    assert compute_prediction_flip_rate(orig, trans_flipped, threshold=0.50) == 0.50

    # Case 3: all flipped
    trans_all_flip = np.array([0.1, 0.2, 0.9, 0.8])
    assert compute_prediction_flip_rate(orig, trans_all_flip, threshold=0.50) == 1.0


def test_mean_probability_drift_calculation():
    """Tests mean probability drift metric calculation."""
    orig = np.array([0.80, 0.20])
    trans = np.array([0.85, 0.10])
    # |0.85 - 0.80| = 0.05, |0.10 - 0.20| = 0.10 -> mean = 0.075
    drift = compute_mean_probability_drift(orig, trans)
    assert pytest.approx(drift, rel=1e-4) == 0.075


def test_authenticity_stability_score_properties():
    """Verifies mathematical properties of Authenticity Stability Score S."""
    # Case A: Perfect stability (0 drift, no flips)
    score_perfect = compute_authenticity_stability_score(0.95, [0.95, 0.95, 0.95, 0.95])
    assert score_perfect == 1.0

    # Case B: Consistent prediction with moderate drift
    score_consistent = compute_authenticity_stability_score(0.90, [0.85, 0.80, 0.88, 0.82])
    assert 0.80 < score_consistent < 1.0

    # Case C: Flipped prediction penalizes score
    score_flipped = compute_authenticity_stability_score(0.90, [0.10, 0.20, 0.15, 0.05])
    assert score_flipped == 0.0  # C = 0 -> S = 0

    # Case D: Mixed predictions (some flips)
    score_mixed = compute_authenticity_stability_score(0.80, [0.85, 0.75, 0.40, 0.30])
    assert 0.0 < score_mixed < 0.50

    # Case E: Strictly bounded in [0, 1]
    for _ in range(50):
        p0 = np.random.uniform(0, 1)
        pk = list(np.random.uniform(0, 1, 6))
        s = compute_authenticity_stability_score(p0, pk)
        assert 0.0 <= s <= 1.0


def test_stability_service_deterministic():
    """Verifies that evaluate_authenticity_stability is deterministic and outputs schema-compliant results."""
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))

    def dummy_predict(image: Image.Image) -> float:
        return 0.95

    res1 = evaluate_authenticity_stability(img, dummy_predict, original_prob=0.95)
    res2 = evaluate_authenticity_stability(img, dummy_predict, original_prob=0.95)

    assert res1.stability_score == res2.stability_score
    assert res1.is_stable is True
    assert res1.degradation_impact == "minimal"
    assert len(res1.transform_results) == 7
    assert res1.stability_score == 1.0
