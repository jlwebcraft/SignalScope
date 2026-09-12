"""SignalScope Explainability and Evidence Module."""

from model.explainability.gradcam import GradCAM, compute_spatial_attribution
from model.explainability.spectral import extract_spectral_features, render_multimodal_evidence_panel
from model.explainability.evidence import EvidenceExtractor

__all__ = [
    "GradCAM",
    "compute_spatial_attribution",
    "extract_spectral_features",
    "render_multimodal_evidence_panel",
    "EvidenceExtractor",
]
