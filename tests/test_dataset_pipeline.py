"""Tests for dataset discovery, sample loading, generator tracking, and split logic."""

from pathlib import Path
import pytest
import torch
from PIL import Image

from model.dataset import (
    SignalScopeDataset,
    create_honest_splits,
    get_default_transforms,
    scan_dataset_directory,
)


@pytest.fixture
def mock_dataset_directory(tmp_path: Path) -> Path:
    """Creates a mock directory hierarchy with real and generator-labeled synthetic images."""
    data_dir = tmp_path / "mock_data"
    real_dir = data_dir / "real"
    real_dir.mkdir(parents=True)

    synth_mj = data_dir / "synthetic" / "Midjourney"
    synth_sd = data_dir / "synthetic" / "StableDiffusion"
    synth_mj.mkdir(parents=True)
    synth_sd.mkdir(parents=True)

    # Create dummy images
    for i in range(6):
        img = Image.new("RGB", (64, 64), color=(i * 20, 100, 100))
        img.save(real_dir / f"real_{i}.jpg")

    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(100, i * 30, 50))
        img.save(synth_mj / f"mj_{i}.png")

    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(50, 50, i * 40))
        img.save(synth_sd / f"sd_{i}.webp")

    return data_dir


def test_scan_dataset_directory(mock_dataset_directory: Path):
    samples = scan_dataset_directory(mock_dataset_directory)

    assert len(samples) == 14  # 6 real + 4 MJ + 4 SD
    real_samples = [s for s in samples if s[1] == 0]
    synth_samples = [s for s in samples if s[1] == 1]

    assert len(real_samples) == 6
    assert len(synth_samples) == 8

    # Check generator tracking
    generators = set(s[2] for s in synth_samples)
    assert "Midjourney" in generators
    assert "StableDiffusion" in generators


def test_dataset_sample_loading(mock_dataset_directory: Path):
    samples = scan_dataset_directory(mock_dataset_directory)
    dataset = SignalScopeDataset(samples, transform=get_default_transforms(image_size=224))

    assert len(dataset) == 14
    tensor, target, meta = dataset[0]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(target, torch.Tensor)
    assert target.item() in {0.0, 1.0}
    assert "generator" in meta
    assert "path" in meta


def test_create_honest_splits_no_leakage(mock_dataset_directory: Path):
    samples = scan_dataset_directory(mock_dataset_directory)
    train, val, test = create_honest_splits(samples, val_ratio=0.2, test_ratio=0.2, seed=42)

    total_len = len(train) + len(val) + len(test)
    assert total_len == len(samples)

    # Disjoint paths test
    train_p = set(s[0] for s in train)
    val_p = set(s[0] for s in val)
    test_p = set(s[0] for s in test)

    assert len(train_p.intersection(val_p)) == 0
    assert len(train_p.intersection(test_p)) == 0
    assert len(val_p.intersection(test_p)) == 0


def test_create_honest_splits_generator_holdout(mock_dataset_directory: Path):
    samples = scan_dataset_directory(mock_dataset_directory)
    # Hold out Midjourney completely to simulate unseen-generator test
    train, val, test = create_honest_splits(
        samples,
        val_ratio=0.2,
        seed=42,
        holdout_generators=["Midjourney"],
    )

    train_generators = set(s[2].lower() for s in train)
    val_generators = set(s[2].lower() for s in val)
    test_generators = set(s[2].lower() for s in test)

    # Midjourney must NEVER appear in train or val!
    assert "midjourney" not in train_generators
    assert "midjourney" not in val_generators
    assert "midjourney" in test_generators
