"""Unit tests for SignalScope frequency representation and lightweight Frequency CNN Branch."""

from pathlib import Path
import pytest
import torch
from PIL import Image

from model.architectures.frequency import (
    FrequencyCNNBranch,
    compute_azimuthal_average,
    compute_dct_2d,
    compute_fft_2d,
    get_dct2_basis_matrix,
)
from model.dataset import SignalScopeDualDataset, get_default_transforms, get_native_transforms


def test_fft_2d_shape_and_range():
    """Verifies 2D FFT computation produces expected shape, DC centering, and [0, 1] range."""
    dummy_batch = torch.randn(4, 3, 32, 32)

    fft_map = compute_fft_2d(dummy_batch, shift=True, normalize=True)

    assert fft_map.shape == (4, 1, 32, 32)
    assert fft_map.min() >= 0.0
    assert fft_map.max() <= 1.0 + 1e-6

    # Test single 3D tensor input
    dummy_single = torch.randn(3, 32, 32)
    fft_single = compute_fft_2d(dummy_single, shift=True, normalize=True)
    assert fft_single.shape == (1, 32, 32)


def test_dct_2d_basis_and_computation():
    """Verifies orthonormal DCT-II basis matrix and 2D DCT spectrum output."""
    n = 32
    d = get_dct2_basis_matrix(n)
    assert d.shape == (n, n)

    # Orthonormality check: D @ D.T == Identity
    identity = torch.eye(n)
    product = torch.matmul(d, d.t())
    assert torch.allclose(product, identity, atol=1e-5)

    # Compute 2D DCT on batch
    dummy_batch = torch.randn(4, 3, 32, 32)
    dct_map = compute_dct_2d(dummy_batch, normalize=True)

    assert dct_map.shape == (4, 1, 32, 32)
    assert dct_map.min() >= 0.0
    assert dct_map.max() <= 1.0 + 1e-6


def test_azimuthal_average():
    """Verifies radial azimuthal profile extraction."""
    spectrum = torch.rand(32, 32)
    profile = compute_azimuthal_average(spectrum, num_bins=16)

    assert profile.shape == (16,)
    assert (profile >= 0.0).all()


def test_frequency_cnn_branch_cpu():
    """Verifies lightweight FrequencyCNNBranch feature extraction and forward pass."""
    model = FrequencyCNNBranch(in_channels=1, feature_dim=128, dropout_rate=0.2)
    model.eval()

    freq_input = torch.rand(4, 1, 32, 32)

    # Feature extraction (embedding)
    with torch.no_grad():
        embedding = model.extract_features(freq_input)
        logits = model(freq_input)

    assert embedding.shape == (4, 128)
    assert logits.shape == (4, 1)

    # Parameter count verification: should be lightweight (< 0.5M params)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 500_000, f"Frequency branch exceeds budget: {param_count} parameters"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_frequency_cnn_branch_cuda():
    """Verifies FrequencyCNNBranch execution on GPU."""
    device = torch.device("cuda")
    model = FrequencyCNNBranch(in_channels=1, feature_dim=128).to(device)
    model.eval()

    freq_input = torch.rand(4, 1, 32, 32, device=device)
    with torch.no_grad():
        embedding = model.extract_features(freq_input)
        logits = model(freq_input)

    assert embedding.device.type == "cuda"
    assert logits.device.type == "cuda"
    assert embedding.shape == (4, 128)
    assert logits.shape == (4, 1)


def test_dual_branch_dataset(tmp_path: Path):
    """Verifies SignalScopeDualDataset yields synchronized 224x224 and native 32x32 tensors."""
    # Create sample image
    sample_path = tmp_path / "test_sample.jpg"
    img = Image.new("RGB", (32, 32), color=(120, 150, 200))
    img.save(sample_path)

    samples = [(sample_path, 1, "MockGenerator")]
    dataset = SignalScopeDualDataset(samples)

    assert len(dataset) == 1
    spatial_tensor, native_tensor, target, meta = dataset[0]

    assert spatial_tensor.shape == (3, 224, 224)
    assert native_tensor.shape == (3, 32, 32)
    assert target.item() == 1.0
    assert meta["generator"] == "MockGenerator"


def test_dual_branch_fusion_forward_pass():
    """Verifies DualBranchFusionDetector forward pass, output shape, and probability range."""
    from model.architectures.fusion import DualBranchFusionDetector

    model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128)
    model.eval()

    spatial_input = torch.randn(2, 3, 224, 224)
    native_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        fused_logits = model(spatial_input, x_native=native_input)
        probs = model.predict_probability(spatial_input, x_native=native_input)
        fused_aux, sp_aux, fr_aux = model(spatial_input, x_native=native_input, return_aux=True)

    assert fused_logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert sp_aux.shape == (2, 1)
    assert fr_aux.shape == (2, 1)


def test_frequency_only_checkpoint_save_and_load(tmp_path: Path):
    """Verifies FrequencyCNNBranch checkpoint saving and loading."""
    model = FrequencyCNNBranch(in_channels=1, feature_dim=128)
    ckpt_path = tmp_path / "freq_test.pt"

    torch.save(
        {
            "model_name": "FrequencyCNNBranch (FFT 32x32)",
            "state_dict": model.state_dict(),
            "val_auc": 0.937,
        },
        ckpt_path,
    )

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["val_auc"] == 0.937

    new_model = FrequencyCNNBranch(in_channels=1, feature_dim=128)
    new_model.load_state_dict(loaded["state_dict"])
    model.eval()
    new_model.eval()

    dummy = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out1 = model(dummy)
        out2 = new_model(dummy)
    assert torch.allclose(out1, out2)


def test_fusion_checkpoint_save_and_load(tmp_path: Path):
    """Verifies DualBranchFusionDetector checkpoint saving and loading."""
    from model.architectures.fusion import DualBranchFusionDetector

    model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128)
    ckpt_path = tmp_path / "fusion_test.pt"

    torch.save(
        {
            "model_name": "DualBranchFusionDetector",
            "state_dict": model.state_dict(),
            "val_auc": 0.999,
        },
        ckpt_path,
    )

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["val_auc"] == 0.999

    new_model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128)
    new_model.load_state_dict(loaded["state_dict"])
    model.eval()
    new_model.eval()

    sp = torch.randn(2, 3, 224, 224)
    nat = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out1 = model(sp, x_native=nat)
        out2 = new_model(sp, x_native=nat)
    assert torch.allclose(out1, out2)
