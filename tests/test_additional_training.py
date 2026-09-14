"""Unit and integration tests for Additional Training dataset splits and pipelines."""

from pathlib import Path
import pytest
import torch
from PIL import Image

from model.dataset_splits import (
    get_official_splits,
    get_v2_splits,
    get_v2_real_samples,
    verify_split_integrity,
)
from model.train_experiments import (
    JPEGCompressionSim,
    get_experiment3_transforms,
)


def test_v2_splits_integrity_and_disjointness():
    train_v2, val_v2 = get_v2_splits()
    assert len(train_v2) == 9758
    assert len(val_v2) == 1118

    # Verify no path overlap
    train_paths = {p for p, _, _ in train_v2}
    val_paths = {p for p, _, _ in val_v2}
    assert len(train_paths.intersection(val_paths)) == 0


def test_split_integrity_verifier():
    # Test valid disjoint sets
    train = [(Path("a.jpg"), 0, "off"), (Path("b.jpg"), 1, "off")]
    val_old = [(Path("c.jpg"), 0, "off")]
    val_new = [(Path("d.jpg"), 1, "v2")]
    assert verify_split_integrity(train, val_old, val_new) is True

    # Test leakage detection
    leaked_train = [(Path("a.jpg"), 0, "off"), (Path("c.jpg"), 0, "off")]
    with pytest.raises(ValueError, match="Leakage detected"):
        verify_split_integrity(leaked_train, val_old, val_new)


def test_v2_real_sampling():
    samples = get_v2_real_samples(limit_per_category=10, seed=42)
    assert len(samples) > 0
    # Verify all sampled are real (label == 0)
    for p, lbl, cat in samples:
        assert lbl == 0
        assert p.exists()


def test_jpeg_compression_sim_transform():
    sim = JPEGCompressionSim(p=1.0, quality_range=(75, 75))
    dummy_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    compressed = sim(dummy_img)
    assert compressed.size == (64, 64)
    assert compressed.mode == "RGB"


def test_experiment3_transforms_shape():
    tf = get_experiment3_transforms(image_size=224)
    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = tf(dummy_img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
