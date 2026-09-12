"""Tests for Authenticity Stability Score calculation and transformations."""

import pytest
from PIL import Image

from app.services.stability import (
    apply_jpeg_compression,
    apply_light_crop_resize,
    apply_resize_down_up,
    apply_screenshot_simulation,
    evaluate_authenticity_stability,
)


def test_stability_transforms_preserve_dimensions(sample_pil_image: Image.Image):
    w, h = sample_pil_image.size

    jpeg_img = apply_jpeg_compression(sample_pil_image, quality=60)
    assert jpeg_img.size == (w, h)
    assert jpeg_img.mode == "RGB"

    resized_img = apply_resize_down_up(sample_pil_image, scale_factor=0.5)
    assert resized_img.size == (w, h)
    assert resized_img.mode == "RGB"

    screenshot_img = apply_screenshot_simulation(sample_pil_image)
    assert screenshot_img.size == (w, h)
    assert screenshot_img.mode == "RGB"

    cropped_img = apply_light_crop_resize(sample_pil_image, crop_fraction=0.90)
    assert cropped_img.size == (w, h)
    assert cropped_img.mode == "RGB"


def test_evaluate_authenticity_stability_invariant_model(sample_pil_image: Image.Image):
    # Dummy predict_fn that is invariant
    dummy_invariant_fn = lambda img: 0.85

    stability = evaluate_authenticity_stability(
        image=sample_pil_image,
        predict_fn=dummy_invariant_fn,
        original_prob=0.85,
    )

    assert stability.stability_score >= 0.95
    assert stability.is_stable is True
    assert stability.degradation_impact == "minimal"
    assert len(stability.transform_results) == 7


def test_evaluate_authenticity_stability_volatile_model(sample_pil_image: Image.Image):
    # Dummy predict_fn that swings wildly
    call_count = [0]
    def volatile_predict(img):
        call_count[0] += 1
        return 0.10 if call_count[0] % 2 == 0 else 0.95

    stability = evaluate_authenticity_stability(
        image=sample_pil_image,
        predict_fn=volatile_predict,
        original_prob=0.90,
    )

    assert stability.stability_score < 0.70
    assert stability.is_stable is False
    assert stability.degradation_impact == "severe"
