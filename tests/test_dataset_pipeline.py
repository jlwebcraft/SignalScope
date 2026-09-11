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


def test_official_dataset_discovery_and_class_mapping():
    data_root = Path(r"C:\Programming\SignalScope-data")
    if not data_root.exists():
        pytest.skip("Official dataset not present at C:\\Programming\\SignalScope-data")

    train_dir = data_root / "train"
    assert (train_dir / "REAL").exists(), "REAL directory missing in train"
    assert (train_dir / "FAKE").exists(), "FAKE directory missing in train"

    # Scan train directory
    samples = scan_dataset_directory(train_dir)
    assert len(samples) == 100000, f"Expected 100,000 samples in train, got {len(samples)}"

    real_count = sum(1 for s in samples if s[1] == 0)
    fake_count = sum(1 for s in samples if s[1] == 1)
    assert real_count == 50000, f"Expected 50,000 REAL, got {real_count}"
    assert fake_count == 50000, f"Expected 50,000 FAKE, got {fake_count}"


def test_official_train_sample_loading_and_smoke_pass():
    data_root = Path(r"C:\Programming\SignalScope-data")
    if not data_root.exists():
        pytest.skip("Official dataset not present at C:\\Programming\\SignalScope-data")

    train_dir = data_root / "train"
    samples = scan_dataset_directory(train_dir)

    # Take tiny balanced batch of 4 real and 4 fake samples
    real_subset = [s for s in samples if s[1] == 0][:4]
    fake_subset = [s for s in samples if s[1] == 1][:4]
    batch_samples = real_subset + fake_subset

    transform = get_default_transforms(image_size=224, is_training=False)
    dataset = SignalScopeDataset(batch_samples, transform=transform)
    assert len(dataset) == 8

    # Load items and stack
    tensors, targets = [], []
    for i in range(len(dataset)):
        t, y, _ = dataset[i]
        assert t.shape == (3, 224, 224)
        tensors.append(t)
        targets.append(y)

    batch_tensors = torch.stack(tensors)
    batch_targets = torch.stack(targets)

    # Model smoke pass
    from model.architectures.convnext import build_convnext_tiny
    import torch.nn.functional as F

    model = build_convnext_tiny(pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(batch_tensors)
        loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
        probs = torch.sigmoid(logits)

    assert logits.shape == (8, 1)
    assert probs.shape == (8, 1)
    assert loss.item() >= 0.0

