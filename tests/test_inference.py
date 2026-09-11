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
    assert 0.0 <= result.stability_score <= 1.0
    assert len(result.evidence) >= 2
    assert len(result.explanation) > 10
    assert "SignalScope" in result.disclaimer
