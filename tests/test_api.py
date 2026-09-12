"""Tests for FastAPI endpoints."""

import io
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "SignalScope API" in data["service"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_version" in data


def test_api_v1_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_api_v1_ready():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_api_v1_info():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SignalScope"
    assert "model_metadata" in data
    assert data["model_metadata"]["model_version"] == "signalscope-baseline-v1"
    assert "ethical_scope" in data
    assert data["ethical_scope"]["is_identity_system"] is False



def test_predict_endpoint_valid_image(sample_image_bytes: bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "probability" in data
    assert "stability_score" in data
    assert "evidence" in data
    assert len(data["evidence"]) >= 2
    assert "explanation" in data


def test_predict_endpoint_empty_file():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_endpoint_corrupt_file():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("bad.jpg", b"corrupted_data_not_an_image", "image/jpeg")},
    )
    assert response.status_code == 422


def test_predict_endpoint_unsupported_media_type(sample_image_bytes: bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("document.pdf", sample_image_bytes, "application/pdf")},
    )
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_predict_endpoint_schema_contract_completeness(sample_image_bytes: bytes):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("valid.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    # Section 7 Schema verification
    assert data["schema_version"] == "1.0"
    assert "verdict" in data
    assert data["verdict"] in {"likely_ai_generated", "likely_real", "uncertain"}
    assert "probability" in data
    assert "confidence_level" in data
    assert "uncertain" in data
    assert isinstance(data["uncertain"], bool)

    # Evidence contract
    assert "evidence" in data
    evidence = data["evidence"]
    assert "spatial" in evidence
    assert "spectral" in evidence
    assert "robustness" in evidence
    assert "metadata" in evidence

    assert "available" in evidence["spatial"]
    assert "available" in evidence["spectral"]
    assert "stability_score" in evidence["robustness"]
    assert "has_exif" in evidence["metadata"]

    assert "explanation" in data
    assert "disclaimer" in data

