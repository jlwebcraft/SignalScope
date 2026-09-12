"""API v1 endpoints for SignalScope."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image

from app.api.schemas import HealthResponse, PredictionResponse
from app.inference.engine import SignalScopeInferenceEngine
from app.utils.image import ImageValidationError, load_image_safely
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["authenticity"])

# Module-level inference engine instance
inference_engine = SignalScopeInferenceEngine()


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.get("/health", response_model=HealthResponse, summary="API Health Check")
async def health_check() -> HealthResponse:
    """Returns service health status and device readiness."""
    return HealthResponse(
        status="healthy",
        service="SignalScope Authenticity API",
        version="0.1.0",
        model_loaded=inference_engine.is_ready,
        device=inference_engine.device,
    )


@router.get("/info", summary="System Information & Guidelines")
async def get_system_info():
    """Returns system parameters, ethical scope, and supported modalities."""
    return {
        "name": "SignalScope",
        "description": "Telling Real From Synthetic in the Age of Generative Media",
        "hackathon": "SIH 2026 Internal Hackathon",
        "schema_version": "1.0",
        "modalities": ["rgb_spatial", "frequency_fft", "metadata_provenance", "authenticity_stability"],
        "supported_formats": ["JPEG", "PNG", "WEBP", "BMP"],
        "max_upload_size_mb": 25,
        "primary_model": inference_engine.model_name,
        "operating_threshold": inference_engine.operating_threshold,
        "uncertainty_band": inference_engine.uncertainty_band,
        "ethical_scope": {
            "is_identity_system": False,
            "claims_about_identifiable_people": False,
            "verdict_types": ["likely_ai_generated", "likely_real", "uncertain"],
            "disclaimer": "SignalScope outputs are statistical authenticity estimates and must not be used as definitive proof or accusations."
        },
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Image Authenticity",
)
async def predict_authenticity(
    file: UploadFile = File(..., description="Single image file (JPEG, PNG, WEBP, BMP)"),
) -> PredictionResponse:
    """Analyzes a single image and produces an evidence-grounded authenticity assessment.

    - **file**: Uploaded image file (max 25MB).
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    # Validate MIME type if provided
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. Allowed types: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds the 25MB maximum upload limit.",
            )

        # Safely parse and validate image
        pil_img: Image.Image = load_image_safely(content)

    except ImageValidationError as err:
        logger.warning(f"Image validation rejected: {err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Image validation failed: {str(err)}",
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Unexpected error loading image: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during image ingestion.",
        )

    # Perform multimodal evidence fusion inference
    try:
        result = inference_engine.analyze(pil_img)
        return result
    except Exception as err:
        logger.error(f"Inference error: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authenticity evaluation encountered an internal error. Please try again with a valid image.",
        )

