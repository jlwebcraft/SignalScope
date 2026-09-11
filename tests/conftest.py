"""Pytest fixtures for SignalScope testing."""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Creates a sample 256x256 synthetic gradient RGB image."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        arr[i, :, 0] = i
        arr[:, i, 1] = 255 - i
        arr[i, i, 2] = (i * 2) % 256
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_image_bytes(sample_pil_image: Image.Image) -> bytes:
    """Returns JPEG encoded bytes of sample image."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def temp_image_file(tmp_path: Path, sample_pil_image: Image.Image) -> Path:
    """Saves sample image to temporary directory and returns path."""
    file_path = tmp_path / "test_image.jpg"
    sample_pil_image.save(file_path, format="JPEG", quality=90)
    return file_path
