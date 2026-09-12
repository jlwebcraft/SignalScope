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


class PredictionResponse(BaseModel):
    """Comprehensive, evidence-grounded authenticity assessment response."""
    verdict: VerdictEnum = Field(..., description="Authenticity verdict ('likely_ai_generated', 'likely_real', 'uncertain')")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated synthetic probability (0.0=real, 1.0=synthetic)")
    raw_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Uncalibrated sigmoid probability")
    calibrated_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Temperature-scaled calibrated probability")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level based on probability margin and stability")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Authenticity stability score under transformations")
    evidence_disagreement: bool = Field(..., description="True if evidence sources (RGB vs Frequency vs Metadata) conflict")
    uncertain: bool = Field(default=False, description="Whether prediction is flagged as uncertain due to boundary corridor or volatility")
    is_development_placeholder: bool = Field(
        default=True,
        description="True if inference ran without trained checkpoint weights (heuristic/development scaffolding)"
    )
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Structured multimodal evidence items")
    structured_evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Modality-grouped evidence structure containing spatial, frequency, robustness, and metadata details"
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
        """Converts response into the Phase 5 standard structured evidence dictionary."""
        return {
            "verdict": self.verdict.value if hasattr(self.verdict, "value") else str(self.verdict),
            "raw_probability": self.raw_probability if self.raw_probability is not None else self.probability,
            "calibrated_probability": self.calibrated_probability if self.calibrated_probability is not None else self.probability,
            "confidence_level": self.confidence_level.value if hasattr(self.confidence_level, "value") else str(self.confidence_level),
            "evidence": self.structured_evidence or {
                "spatial": {"available": self.heatmap_available},
                "frequency": {"available": True},
                "robustness": {"stability_score": self.stability_score},
                "metadata": self.metadata.model_dump() if hasattr(self.metadata, "model_dump") else {},
            },
            "explanation": self.explanation,
            "uncertain": self.uncertain,
        }



class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="healthy")
    service: str = Field(default="SignalScope Authenticity API")
    version: str = Field(default="0.1.0")
    model_loaded: bool = Field(default=False)
    device: str = Field(default="cpu")
