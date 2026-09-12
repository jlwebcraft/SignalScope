"""Tests for SignalScope baseline checkpoint save/load, inference runner, and threshold calibration.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from app.api.schemas import VerdictEnum
from app.inference.engine import SignalScopeInferenceEngine
from model.architectures.convnext import build_convnext_tiny
from model.evaluate import (
    calculate_metrics,
    find_optimal_threshold_youden,
    find_threshold_for_target_fpr,
)
from model.predict import run_prediction


def test_checkpoint_save_and_load(tmp_path: Path):
    """Verifies complete checkpoint creation, serialization, and reconstruction."""
    model = build_convnext_tiny(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    payload = {
        "epoch": 3,
        "model_name": "convnext_tiny.in12k_ft_in1k",
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": 0.952,
        "seed": 42,
    }
    torch.save(payload, ckpt_path)

    assert ckpt_path.exists()

    # Load back
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 3
    assert loaded["model_name"] == "convnext_tiny.in12k_ft_in1k"
    assert loaded["val_auc"] == 0.952
    assert "state_dict" in loaded

    # Verify model state loads cleanly
    new_model = build_convnext_tiny(pretrained=False)
    new_model.load_state_dict(loaded["state_dict"])
    new_model.eval()

    # Compare weights
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_inference_engine_with_checkpoint(tmp_path: Path, sample_pil_image: Image.Image):
    """Verifies SignalScopeInferenceEngine loads checkpoint and runs real inference."""
    model = build_convnext_tiny(pretrained=False)
    ckpt_path = tmp_path / "valid_model.pt"
    torch.save(
        {
            "model_name": "convnext_tiny.in12k_ft_in1k",
            "state_dict": model.state_dict(),
            "val_auc": 0.90,
        },
        ckpt_path,
    )

    engine = SignalScopeInferenceEngine(checkpoint_path=str(ckpt_path))
    assert engine.has_trained_weights is True
    assert engine.is_ready is True
    assert engine.model_name == "convnext_tiny.in12k_ft_in1k"

    result = engine.analyze(sample_pil_image)
    assert result.is_development_placeholder is False
    assert 0.0 <= result.probability <= 1.0
    assert result.verdict in [
        VerdictEnum.LIKELY_REAL,
        VerdictEnum.LIKELY_AI_GENERATED,
        VerdictEnum.UNCERTAIN,
    ]
    assert len(result.evidence) >= 2


def test_predict_runner_rejects_missing_checkpoint(temp_image_file: Path):
    """Verifies model/predict.py rejects missing or non-existent checkpoint gracefully."""
    ret = run_prediction(str(temp_image_file), checkpoint_path="nonexistent_checkpoint.pt")
    assert ret == 1


def test_predict_runner_succeeds_with_checkpoint(tmp_path: Path, temp_image_file: Path):
    """Verifies model/predict.py executes successfully with valid checkpoint."""
    model = build_convnext_tiny(pretrained=False)
    ckpt_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_name": "convnext_tiny.in12k_ft_in1k",
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    ret = run_prediction(str(temp_image_file), checkpoint_path=str(ckpt_path))
    assert ret == 0


def test_youden_threshold_optimization():
    """Verifies Youden's J statistic threshold selection."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90])

    best_thresh, best_j = find_optimal_threshold_youden(y_true, y_scores)
    assert 0.40 <= best_thresh <= 0.65
    assert best_j > 0.8  # Perfect separation gives J = 1.0


def test_deterministic_validation_evaluation():
    """Verifies that metrics calculation is perfectly deterministic."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_scores = np.array([0.2, 0.3, 0.8, 0.7, 0.1, 0.9, 0.4, 0.6])

    metrics1 = calculate_metrics(y_true, y_scores, operating_threshold=0.50)
    metrics2 = calculate_metrics(y_true, y_scores, operating_threshold=0.50)

    assert metrics1["roc_auc"] == metrics2["roc_auc"]
    assert metrics1["macro_f1"] == metrics2["macro_f1"]
    assert metrics1["accuracy"] == metrics2["accuracy"]
    assert metrics1["fpr"] == metrics2["fpr"]
