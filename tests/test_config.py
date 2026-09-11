"""Tests for YAML configuration loading and structure."""

from pathlib import Path
import pytest
import yaml

from model.train import load_config


def test_default_config_exists_and_loads():
    config_path = Path("model/configs/default.yaml")
    assert config_path.exists(), "default.yaml must exist"

    cfg = load_config(str(config_path))
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "dataset" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "evaluation" in cfg
    assert "stability" in cfg

    # Verify key properties
    assert cfg["project"]["name"] == "SignalScope"
    assert cfg["dataset"]["image_size"] == 224
    assert cfg["model"]["num_classes"] == 1
    assert cfg["evaluation"]["operating_threshold"] == 0.50


def test_baseline_convnext_config():
    config_path = Path("model/configs/baseline_convnext.yaml")
    assert config_path.exists(), "baseline_convnext.yaml must exist"

    cfg = load_config(str(config_path))
    assert isinstance(cfg, dict)
    assert cfg["model"]["name"] == "convnext_tiny"
