"""Regression tests for production model version selection and safe failure behavior."""

import os
import pytest

from app.inference.artifacts import ModelArtifactManager, MODEL_REGISTRY, DEFAULT_MODEL_VERSION
from app.inference.engine import SignalScopeInferenceEngine


def test_default_production_model_version():
    """Confirms default production model is signalscope-v2 with canonical v2 checksum."""
    mgr = ModelArtifactManager()
    assert mgr.model_version == "signalscope-v2"
    assert mgr.expected_checkpoint_sha256 == "47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d"
    info = ModelArtifactManager.get_version_info("signalscope-v2")
    assert info["model_version"] == "signalscope-v2"
    assert info["weights_sha256"] == "47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d"
    assert info["calibration_temperature"] == 0.9986


def test_explicit_v2_model_version_selection(monkeypatch):
    """Proves MODEL_VERSION=signalscope-v2 explicitly selects v2."""
    monkeypatch.setenv("MODEL_VERSION", "signalscope-v2")
    mgr = ModelArtifactManager()
    assert mgr.model_version == "signalscope-v2"
    assert mgr.expected_checkpoint_sha256 == "47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d"


def test_rollback_v1_model_version_selection(monkeypatch):
    """Proves MODEL_VERSION=signalscope-baseline-v1 selects baseline v1 with canonical v1 checksum."""
    monkeypatch.setenv("MODEL_VERSION", "signalscope-baseline-v1")
    mgr = ModelArtifactManager()
    assert mgr.model_version == "signalscope-baseline-v1"
    assert mgr.expected_checkpoint_sha256 == "c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc"
    info = ModelArtifactManager.get_version_info("signalscope-baseline-v1")
    assert info["model_version"] == "signalscope-baseline-v1"
    assert info["weights_sha256"] == "c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc"
    assert info["calibration_temperature"] == 0.99953


def test_invalid_model_version_fails_safely(monkeypatch):
    """Proves invalid model versions fail safely rather than silently selecting v1 or v2."""
    monkeypatch.setenv("MODEL_VERSION", "nonexistent-model-xyz")
    mgr = ModelArtifactManager()
    assert mgr.model_version == "nonexistent-model-xyz"
    assert mgr.spec is None
    assert mgr.expected_checkpoint_sha256 is None
    assert mgr.resolve_checkpoint() is None
    assert mgr.resolve_scaler() is None

    # Version info must raise ValueError rather than returning wrong model metadata
    with pytest.raises(ValueError, match="Unknown or invalid model_version"):
        ModelArtifactManager.get_version_info("nonexistent-model-xyz")


def test_inference_engine_fails_safely_on_invalid_version():
    """Inference engine refuses to boot into ready state with invalid model version."""
    engine = SignalScopeInferenceEngine(model_version="invalid-version-12345")
    assert engine.model_version == "invalid-version-12345"
    assert engine.is_ready is False
    assert engine.has_trained_weights is False


def test_inference_engine_accepts_explicit_version():
    """Inference engine accepts explicit model_version argument for rollback and testing."""
    engine_v1 = SignalScopeInferenceEngine(
        model_version="signalscope-baseline-v1",
        checkpoint_path="checkpoints/baseline_convnext/best_model.pt",
    )
    assert engine_v1.model_version == "signalscope-baseline-v1"
    assert engine_v1.has_trained_weights is True
    assert engine_v1.is_ready is True
