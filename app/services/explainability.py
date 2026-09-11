"""Explainability service for SignalScope.

Provides saliency map computation interfaces and evidence-grounded
explanation generation without hallucination.
"""

from typing import List, Optional
from PIL import Image

from app.api.schemas import (
    ConfidenceLevelEnum,
    EvidenceItem,
    MetadataInfo,
    VerdictEnum,
)
from app.utils.logger import logger


def generate_grounded_explanation(
    verdict: VerdictEnum,
    probability: float,
    confidence_level: ConfidenceLevelEnum,
    stability_score: float,
    evidence_items: List[EvidenceItem],
    metadata_info: MetadataInfo,
    evidence_disagreement: bool,
) -> str:
    """Synthesizes an honest, factual natural-language explanation grounded strictly in evidence.

    Ethical constraint: Never present results as absolute proof or an accusation.
    Use terms like 'Likely AI-generated', 'Likely real', 'Uncertain'.
    """
    sentences: List[str] = []

    # Verdict statement
    percentage = round(probability * 100, 1)
    if verdict == VerdictEnum.LIKELY_AI_GENERATED:
        sentences.append(
            f"The image exhibits statistical artifacts characteristic of generative models "
            f"(estimated synthetic probability: {percentage}%, confidence: {confidence_level.value})."
        )
    elif verdict == VerdictEnum.LIKELY_REAL:
        sentences.append(
            f"The image does not show typical generative artifacts and aligns with authentic natural capture "
            f"(estimated synthetic probability: {percentage}%, confidence: {confidence_level.value})."
        )
    else:
        sentences.append(
            f"Authenticity assessment is indeterminate (estimated synthetic probability: {percentage}%). "
            f"Signals are ambiguous or close to the operating decision threshold."
        )

    # Stability statement
    stability_pct = round(stability_score * 100, 1)
    if stability_score >= 0.80:
        sentences.append(
            f"Authenticity stability is high ({stability_pct}%), indicating the prediction remains consistent "
            f"under standard compression, scaling, and re-encoding."
        )
    elif stability_score >= 0.60:
        sentences.append(
            f"Authenticity stability is moderate ({stability_pct}%); common web compressions cause minor variations."
        )
    else:
        sentences.append(
            f"Authenticity stability is low ({stability_pct}%); transformations like resizing or JPEG compression "
            f"significantly shift the prediction, indicating sensitive or fragile visual cues."
        )

    # Evidence conflict / agreement
    if evidence_disagreement:
        sentences.append(
            "Notice: Conflicting signals were observed between modality branches (e.g., spatial RGB vs frequency domain or metadata), "
            "which reduces overall certainty."
        )

    # Metadata context
    if metadata_info.has_exif and metadata_info.camera_make:
        sentences.append(
            f"Provenance data records camera hardware ({metadata_info.camera_make} {metadata_info.camera_model or ''})."
        )
    elif any("AI" in a or "generator" in a.lower() for a in metadata_info.anomalies):
        sentences.append("Metadata inspection revealed synthetic generator tags or parameter blocks.")

    # Branch-specific highlights
    key_signals = [item.description for item in evidence_items if item.supports_synthetic]
    if key_signals and verdict == VerdictEnum.LIKELY_AI_GENERATED:
        sentences.append(f"Key contributing factors: {'; '.join(key_signals[:2])}.")

    return " ".join(sentences)
