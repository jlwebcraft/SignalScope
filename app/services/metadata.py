"""Metadata and provenance extraction service.

Analyzes EXIF data, camera hardware signatures, software tags,
and C2PA content credential markers.
"""

from typing import Any, Dict, List, Optional
from PIL import Image
from PIL.ExifTags import TAGS

from app.api.schemas import MetadataInfo
from app.utils.logger import logger

# Known software or metadata signatures common in synthetic media generators
AI_SOFTWARE_SIGNATURES = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "comfyui",
    "automatic1111",
    "adobe firefly",
    "novelai",
    "invokeai",
    "flux.1",
    "bing image creator",
]


def extract_metadata(image: Image.Image) -> MetadataInfo:
    """Extracts provenance, EXIF tags, and scans for synthetic generator signatures.

    Args:
        image: PIL Image instance.

    Returns:
        MetadataInfo schema containing structured metadata and anomaly flags.
    """
    raw_tags: Dict[str, Any] = {}
    anomalies: List[str] = []
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    c2pa_detected = False

    try:
        exif_data = image.getexif()
        if exif_data and len(exif_data) > 0:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                # Store string representation of printable tags
                if isinstance(value, (str, int, float)):
                    raw_tags[tag_name] = value
                elif isinstance(value, bytes):
                    try:
                        raw_tags[tag_name] = value.decode("utf-8", errors="ignore")
                    except Exception:
                        raw_tags[tag_name] = f"<bytes len={len(value)}>"

            camera_make = raw_tags.get("Make")
            camera_model = raw_tags.get("Model")
            software = raw_tags.get("Software")

            # Check for AI generation signatures in software or description
            software_str = str(software or "").lower()
            description_str = str(raw_tags.get("ImageDescription", "")).lower()

            for sig in AI_SOFTWARE_SIGNATURES:
                if sig in software_str or sig in description_str:
                    anomalies.append(f"Known AI generator software marker detected: '{sig}'")

            # Check for C2PA marker presence
            for tag_k, tag_v in raw_tags.items():
                if "c2pa" in str(tag_k).lower() or "c2pa" in str(tag_v).lower():
                    c2pa_detected = True
        else:
            anomalies.append("No EXIF metadata present (common in stripped or direct synthetic generation)")

        # Check raw image.info dictionary for PNG/WebP text chunks
        if hasattr(image, "info") and isinstance(image.info, dict):
            for info_key, info_val in image.info.items():
                info_key_lower = str(info_key).lower()
                info_val_str = str(info_val).lower()
                if "parameters" in info_key_lower or "prompt" in info_key_lower:
                    anomalies.append(f"AI generation prompt metadata chunk detected: '{info_key}'")
                for sig in AI_SOFTWARE_SIGNATURES:
                    if sig in info_val_str:
                        anomalies.append(f"AI generator reference detected in image chunks: '{sig}'")

    except Exception as exc:
        logger.debug(f"Error during metadata parsing: {exc}")
        anomalies.append(f"Metadata parsing exception: {str(exc)}")

    has_exif = bool(camera_make or camera_model or len(raw_tags) > 2)

    return MetadataInfo(
        has_exif=has_exif,
        camera_make=str(camera_make) if camera_make else None,
        camera_model=str(camera_model) if camera_model else None,
        software=str(software) if software else None,
        c2pa_detected=c2pa_detected,
        anomalies=anomalies,
        raw_tags={k: str(v)[:100] for k, v in list(raw_tags.items())[:20]},
    )
