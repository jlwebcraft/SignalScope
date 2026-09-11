"""Tests for image loading, validation, and preprocessing utilities."""

from pathlib import Path
import pytest
from PIL import Image

from app.utils.image import (
    ImageValidationError,
    get_image_dimensions,
    load_image_safely,
)


def test_load_image_from_path(temp_image_file: Path):
    img = load_image_safely(temp_image_file)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (256, 256)


def test_load_image_from_bytes(sample_image_bytes: bytes):
    img = load_image_safely(sample_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (256, 256)


def test_load_nonexistent_path():
    with pytest.raises(ImageValidationError):
        load_image_safely("nonexistent_path_to_image_12345.jpg")


def test_load_invalid_bytes():
    corrupt_bytes = b"NOT_A_VALID_IMAGE_BYTE_SEQUENCE"
    with pytest.raises(ImageValidationError):
        load_image_safely(corrupt_bytes)


def test_get_image_dimensions(sample_pil_image: Image.Image):
    dims = get_image_dimensions(sample_pil_image)
    assert dims == (256, 256)
