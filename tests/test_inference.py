"""Tests for SignalScope inference engine and multimodal fusion."""

import pytest
from PIL import Image

from app.api.schemas import PredictionResponse, VerdictEnum
from app.inference.engine import SignalScopeInferenceEngine


def test_inference_engine_analysis(sample_pil_image: Image.Image):
    engine = SignalScopeInferenceEngine()
    assert engine.is_ready is True

    result = engine.analyze(sample_pil_image)

    assert isinstance(result, PredictionResponse)
    assert result.verdict in {
        VerdictEnum.LIKELY_AI_GENERATED,
        VerdictEnum.LIKELY_REAL,
        VerdictEnum.UNCERTAIN,
    }
    assert 0.0 <= result.probability <= 1.0
    assert result.raw_probability is not None and 0.0 <= result.raw_probability <= 1.0
    assert result.calibrated_probability is not None and 0.0 <= result.calibrated_probability <= 1.0
    assert isinstance(result.uncertain, bool)
    assert 0.0 <= result.stability_score <= 1.0
    assert len(result.evidence) >= 2
    assert result.structured_evidence is not None
    assert "spatial" in result.structured_evidence
    assert "frequency" in result.structured_evidence
    assert "robustness" in result.structured_evidence
    assert len(result.explanation) > 10
    assert "SignalScope" in result.disclaimer

    # Test Section 14 evidence dictionary serialization
    evidence_obj = result.to_evidence_object()
    assert "verdict" in evidence_obj
    assert "raw_probability" in evidence_obj
    assert "calibrated_probability" in evidence_obj
    assert "confidence_level" in evidence_obj
    assert "evidence" in evidence_obj
    assert "explanation" in evidence_obj
    assert "uncertain" in evidence_obj


def test_inference_engine_calibrated_predict(sample_pil_image: Image.Image):
    engine = SignalScopeInferenceEngine()
    prob = engine.predict_calibrated_probability(sample_pil_image)
    assert 0.0 <= prob <= 1.0

