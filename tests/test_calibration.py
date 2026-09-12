"""Tests for Phase 5 Temperature Scaling calibration module."""

from pathlib import Path
import numpy as np
import pytest
import torch

from model.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
)


def test_brier_score_bounds_and_values():
    """Verifies Brier score computation."""
    # Perfect predictions: 0.0 error
    y_true = np.array([1, 0, 1, 0])
    y_prob_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_brier_score(y_true, y_prob_perfect) == 0.0

    # Completely wrong: 1.0 error
    y_prob_wrong = np.array([0.0, 1.0, 0.0, 1.0])
    assert compute_brier_score(y_true, y_prob_wrong) == 1.0

    # Intermediate known error:
    # (0.8 - 1)^2 = 0.04; (0.2 - 0)^2 = 0.04 -> mean = 0.04
    y_prob_mid = np.array([0.8, 0.2, 0.8, 0.2])
    assert pytest.approx(compute_brier_score(y_true, y_prob_mid), rel=1e-4) == 0.04


def test_ece_computation():
    """Verifies Expected Calibration Error (ECE) and MCE computation."""
    y_true = np.array([1, 1, 0, 0])
    y_probs = np.array([0.9, 0.8, 0.1, 0.2])
    ece, mce, details = compute_ece(y_true, y_probs, n_bins=5)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert details["n_bins"] == 5
    assert len(details["bins"]) == 5


def test_temperature_scaler_optimization_and_parity(tmp_path: Path):
    """Verifies TemperatureScaler parameter fitting, calibration, and serialization."""
    scaler = TemperatureScaler()
    assert pytest.approx(scaler.temperature, rel=1e-3) == 1.0

    # Simulate overconfident logits
    torch.manual_seed(42)
    logits = torch.randn(500, 1) * 5.0
    targets = (torch.sigmoid(logits * 0.5) > 0.5).float()

    t_fitted = scaler.fit(logits, targets, max_iter=20)
    assert t_fitted > 0.0

    cal_probs = scaler.calibrate(logits)
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    # Save and Load parity
    save_path = tmp_path / "scaler_test.json"
    scaler.save(save_path)
    assert save_path.exists()

    new_scaler = TemperatureScaler()
    new_scaler.load(save_path)
    assert pytest.approx(new_scaler.temperature, rel=1e-4) == t_fitted

    test_logit = torch.tensor([[2.5]])
    assert torch.allclose(scaler.calibrate(test_logit), new_scaler.calibrate(test_logit))
