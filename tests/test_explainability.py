"""Tests for Phase 5 Grad-CAM explainability, spectral evidence, and grounded explanations."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from model.architectures.convnext import build_convnext_tiny
from model.calibration import TemperatureScaler
from model.explainability import EvidenceExtractor, GradCAM, compute_spatial_attribution, extract_spectral_features


@pytest.fixture
def dummy_convnext_model() -> torch.nn.Module:
    """Instantiates a lightweight ConvNeXt model for unit testing."""
    model = build_convnext_tiny(pretrained=False)
    model.eval()
    return model


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Provides a synthetic 32x32 RGB PIL test image."""
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_gradcam_heatmap_generation_and_bounds(dummy_convnext_model: torch.nn.Module):
    """Verifies Grad-CAM heatmap generation, shape, bounds, and no NaN/Inf."""
    gradcam = GradCAM(dummy_convnext_model)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    heatmap = gradcam.generate_heatmap(x)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.shape == (7, 7)  # ConvNeXt stage 3 block feature resolution
    assert not np.isnan(heatmap).any()
    assert not np.isinf(heatmap).any()
    assert (heatmap >= 0.0).all()
    assert (heatmap <= 1.0).all()


def test_gradcam_overlay_rendering(dummy_convnext_model: torch.nn.Module, sample_pil_image: Image.Image):
    """Verifies Grad-CAM visual overlay generation on native image."""
    heatmap, overlay, conc = compute_spatial_attribution(
        dummy_convnext_model, sample_pil_image, device=torch.device("cpu")
    )

    assert isinstance(overlay, Image.Image)
    assert overlay.size == sample_pil_image.size
    assert overlay.mode == "RGB"
    assert 0.0 <= conc <= 1.0


def test_spectral_evidence_extraction(sample_pil_image: Image.Image):
    """Verifies 2D FFT spectrum, azimuthal profile, and energy ratio computation."""
    fft_2d, radial, hf_ratio, spec_img = extract_spectral_features(sample_pil_image)

    assert fft_2d.shape == (32, 32)
    assert len(radial) == 16
    assert 0.0 <= hf_ratio <= 1.0
    assert isinstance(spec_img, Image.Image)
    assert spec_img.size == (32, 32)
    assert not np.isnan(fft_2d).any()
    assert not np.isnan(radial).any()


def test_evidence_extractor_schema_and_uncertainty(
    dummy_convnext_model: torch.nn.Module,
    sample_pil_image: Image.Image,
    tmp_path: Path,
):
    """Verifies that EvidenceExtractor produces valid schema, evidence, and grounded explanations."""
    scaler = TemperatureScaler()
    extractor = EvidenceExtractor(
        classifier=dummy_convnext_model, scaler=scaler, device=torch.device("cpu")
    )

    evidence_obj = extractor.evaluate_image(
        sample_pil_image, artifact_save_dir=tmp_path, sample_id="unit_test_sample"
    )

    assert "verdict" in evidence_obj
    assert evidence_obj["verdict"] in {"likely_ai_generated", "likely_real", "uncertain"}
    assert "calibrated_probability" in evidence_obj
    assert 0.0 <= evidence_obj["calibrated_probability"] <= 1.0
    assert "confidence_level" in evidence_obj
    assert "evidence" in evidence_obj
    assert "spatial" in evidence_obj["evidence"]
    assert "frequency" in evidence_obj["evidence"]
    assert "robustness" in evidence_obj["evidence"]
    assert "explanation" in evidence_obj
    assert isinstance(evidence_obj["explanation"], str)
    assert len(evidence_obj["explanation"]) > 20

    # Check panel was saved
    expected_panel = tmp_path / "unit_test_sample_evidence_panel.png"
    assert expected_panel.exists()
