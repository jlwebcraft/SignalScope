"""SignalScope image loading, validation, and format handling utilities."""

import io
from pathlib import Path
from typing import Tuple, Union
from PIL import Image, ImageOps

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class ImageValidationError(ValueError):
    """Raised when an input image fails validation checks."""
    pass


def load_image_safely(
    image_input: Union[str, Path, bytes, io.BytesIO],
    max_dimension: int = 4096,
) -> Image.Image:
    """Safely loads an image, validates format, applies EXIF orientation, and converts to RGB.

    Args:
        image_input: File path, bytes, or BytesIO buffer.
        max_dimension: Maximum allowed pixel width or height to prevent decompression bombs.

    Returns:
        Validated PIL Image in RGB mode.

    Raises:
        ImageValidationError: If the image cannot be decoded, exceeds size limits, or has unsupported format.
    """
    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise ImageValidationError(f"Image file does not exist: {path}")
            if path.stat().st_size > MAX_IMAGE_SIZE_BYTES:
                raise ImageValidationError(f"File size exceeds 25MB limit: {path.stat().st_size} bytes")
            pil_img = Image.open(path)
        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > MAX_IMAGE_SIZE_BYTES:
                raise ImageValidationError(f"Payload size exceeds 25MB limit: {len(image_input)} bytes")
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            pil_img = Image.open(image_input)
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(image_input)}")

        # Verify format
        img_format = (pil_img.format or "").upper()
        if img_format not in SUPPORTED_FORMATS and img_format != "":
            raise ImageValidationError(f"Unsupported image format: {img_format}. Supported: {SUPPORTED_FORMATS}")

        # Dimension checks
        width, height = pil_img.size
        if width <= 0 or height <= 0:
            raise ImageValidationError(f"Invalid image dimensions: {width}x{height}")
        if width < 16 or height < 16:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) are smaller than minimum allowed (16x16px)"
            )
        if width > max_dimension or height > max_dimension:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dimension}px)"
            )

        # Handle EXIF orientation safely if present
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass

        # Convert to RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return pil_img

    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"Failed to decode image: {str(exc)}") from exc


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Returns (width, height) of PIL Image."""
    return image.size
