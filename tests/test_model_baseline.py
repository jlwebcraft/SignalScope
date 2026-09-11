"""Tests for ConvNeXt-Tiny baseline model instantiation, preprocessing, and forward pass."""

import pytest
import torch
from PIL import Image

from model.architectures.convnext import ConvNeXtTinyDetector, build_convnext_tiny
from model.dataset import get_default_transforms


def test_convnext_tiny_instantiation():
    # Test instantiation (uninitialized or pretrained)
    model = build_convnext_tiny(pretrained=False)
    assert isinstance(model, ConvNeXtTinyDetector)
    assert model.num_classes == 1


def test_convnext_tiny_preprocessing_tensor_shape(sample_pil_image: Image.Image):
    transform = get_default_transforms(image_size=224, is_training=False)
    tensor = transform(sample_pil_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_convnext_tiny_forward_pass_cpu():
    model = build_convnext_tiny(pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_probability(dummy_input)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_convnext_tiny_forward_pass_cuda():
    device = torch.device("cuda")
    model = build_convnext_tiny(pretrained=False).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_probability(dummy_input)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert logits.device.type == "cuda"
    assert probs.device.type == "cuda"
