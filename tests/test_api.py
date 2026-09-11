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


def test_api_v1_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_api_v1_info():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SignalScope"
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
