"""Pydantic schemas for API requests, responses, and structured evidence."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    """Calibrated authenticity verdict following SIH 2026 ethical guidelines."""
    LIKELY_AI_GENERATED = "likely_ai_generated"
    LIKELY_REAL = "likely_real"
    UNCERTAIN = "uncertain"


class ConfidenceLevelEnum(str, Enum):
    """Categorical confidence level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceItem(BaseModel):
    """Individual piece of modality or branch evidence."""
    source: str = Field(..., description="Evidence source (e.g., 'rgb_spatial', 'frequency_fft', 'metadata', 'stability')")
    metric: str = Field(..., description="Metric or test name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized synthetic score from 0.0 (real) to 1.0 (AI)")
    weight: float = Field(default=1.0, ge=0.0, description="Evidence weight in fusion decision")
    description: str = Field(..., description="Human-readable interpretation of the evidence")
    supports_synthetic: bool = Field(..., description="True if this evidence signals AI generation")


class StabilityTransformResult(BaseModel):
    """Results from testing image under controlled degradation transforms."""
    transform_name: str = Field(..., description="Name of transformation (e.g., 'original', 'jpeg_recompression', 'resize_down_up', 'screenshot_simulation')")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted synthetic probability under transform")
    delta_from_original: float = Field(..., description="Absolute change in prediction compared to pristine original")


class AuthenticityStabilityInfo(BaseModel):
    """Authenticity Stability Score summary."""
    stability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall stability score (1.0 = completely invariant across transforms, 0.0 = volatile)"
    )
    is_stable: bool = Field(..., description="Whether stability meets the minimum operating threshold")
    transform_results: List[StabilityTransformResult] = Field(default_factory=list)
    degradation_impact: str = Field(
        default="minimal",
        description="Impact of common degradations: 'minimal', 'moderate', 'severe'"
    )


class MetadataInfo(BaseModel):
    """Extracted provenance and image metadata."""
    has_exif: bool = Field(default=False, description="Whether EXIF metadata is present")
    camera_make: Optional[str] = Field(default=None, description="Camera manufacturer if present")
    camera_model: Optional[str] = Field(default=None, description="Camera hardware model if present")
    software: Optional[str] = Field(default=None, description="Editing/generation software tag")
    c2pa_detected: bool = Field(default=False, description="Whether C2PA Content Credentials were found")
    anomalies: List[str] = Field(default_factory=list, description="Suspicious metadata indicators or absences")
    raw_tags: Dict[str, Any] = Field(default_factory=dict, description="Sanitized key-value tags")


class SpatialEvidence(BaseModel):
    """Spatial attribution evidence via Grad-CAM."""
    available: bool = Field(default=False, description="Whether spatial Grad-CAM attribution was computed")
    heatmap: Optional[str] = Field(default=None, description="Base64 data URI (image/png) of Grad-CAM visual overlay")
    attribution_concentration: Optional[float] = Field(default=None, description="Normalized attribution concentration metric")
    target_layer: Optional[str] = Field(default=None, description="Inspected feature extraction layer")


class SpectralEvidence(BaseModel):
    """Frequency-domain spectral evidence via 2D FFT."""
    available: bool = Field(default=True, description="Whether 2D FFT spectral cues were computed")
    spectrum: Optional[str] = Field(default=None, description="Base64 data URI (image/png) of 2D FFT log-magnitude spectrum")
    high_frequency_energy_ratio: Optional[float] = Field(default=None, description="High-frequency energy ratio in spatial spectrum")
    representation: str = Field(default="2D Log-Magnitude Fast Fourier Transform", description="Spectral representation identifier")


class RobustnessEvidence(BaseModel):
    """Authenticity stability evidence under controlled transformations."""
    available: bool = Field(default=True, description="Whether stability probes were executed")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Overall stability score in [0, 1]")
    prediction_flip_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of transforms causing decision flip")
    mean_probability_drift: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean absolute probability change across transforms")
    is_stable: bool = Field(default=True, description="Whether stability meets threshold")
    degradation_impact: str = Field(default="minimal", description="'minimal', 'moderate', or 'severe'")
    transform_results: List[StabilityTransformResult] = Field(default_factory=list, description="Per-transformation probe results")


class MetadataEvidence(BaseModel):
    """Provenance and file metadata evidence."""
    available: bool = Field(default=True, description="Whether metadata inspection was performed")
    has_exif: bool = Field(default=False, description="Whether camera EXIF tags were present")
    c2pa_present: bool = Field(default=False, description="Whether C2PA Content Credentials were detected")
    camera_make: Optional[str] = Field(default=None, description="Camera manufacturer if present")
    camera_model: Optional[str] = Field(default=None, description="Camera hardware model if present")
    software: Optional[str] = Field(default=None, description="Software tag if present")
    anomalies: List[str] = Field(default_factory=list, description="Anomalies or generator markers")


class PredictionResponse(BaseModel):
    """Comprehensive, evidence-grounded authenticity assessment response."""
    schema_version: str = Field(default="1.0", description="API Response Contract Schema Version")
    verdict: VerdictEnum = Field(..., description="Authenticity verdict ('likely_ai_generated', 'likely_real', 'uncertain')")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated synthetic probability (0.0=real, 1.0=synthetic)")
    raw_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Uncalibrated sigmoid probability")
    calibrated_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Temperature-scaled calibrated probability")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level based on probability margin and stability")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Authenticity stability score under transformations")
    evidence_disagreement: bool = Field(..., description="True if evidence sources conflict")
    uncertain: bool = Field(default=False, description="Whether prediction is flagged as uncertain due to boundary corridor or volatility")
    is_development_placeholder: bool = Field(
        default=True,
        description="True if inference ran without trained checkpoint weights (heuristic/development scaffolding)"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured multimodal evidence containing spatial, spectral, robustness, and metadata details"
    )
    structured_evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Alias for structured evidence dictionary"
    )
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Legacy flat evidence list for backwards compatibility"
    )

    stability: AuthenticityStabilityInfo = Field(..., description="Detailed stability breakdown across transformations")
    metadata: MetadataInfo = Field(..., description="Extracted metadata and provenance")
    explanation: str = Field(..., description="Faithful natural-language explanation grounded in evidence")
    heatmap_available: bool = Field(default=False, description="Whether a spatial saliency heatmap was computed")
    disclaimer: str = Field(
        default=(
            "SignalScope outputs are statistical estimates of authenticity signals. "
            "They do not constitute legal proof or personal accusations. "
            "Outputs should be evaluated in context with supporting human review."
        ),
        description="Mandatory ethical disclaimer"
    )

    def to_evidence_object(self) -> Dict[str, Any]:
        """Converts response into the Phase 5/6 standard structured evidence dictionary."""
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict.value if hasattr(self.verdict, "value") else str(self.verdict),
            "probability": self.probability,
            "raw_probability": self.raw_probability if self.raw_probability is not None else self.probability,
            "calibrated_probability": self.calibrated_probability if self.calibrated_probability is not None else self.probability,
            "confidence_level": self.confidence_level.value if hasattr(self.confidence_level, "value") else str(self.confidence_level),
            "evidence": self.evidence,
            "explanation": self.explanation,
            "uncertain": self.uncertain,
            "disclaimer": self.disclaimer,
        }




class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="healthy")
    service: str = Field(default="SignalScope Authenticity API")
    version: str = Field(default="0.1.0")
    model_loaded: bool = Field(default=False)
    device: str = Field(default="cpu")


class ReadyResponse(BaseModel):
    """Readiness probe response schema."""
    status: str = Field(default="ready")
    model_version: str = Field(default="signalscope-baseline-v1")
    model_loaded: bool = Field(default=True)
    has_trained_weights: bool = Field(default=True)
    device: str = Field(default="cpu")
    message: str = Field(default="Inference engine is ready.")

