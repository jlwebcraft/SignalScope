"""SignalScope Production Smoke Test & Reproducibility Verification.

Validates:
1. Production environment & dependencies.
2. Model artifact resolution (local or remote via ModelArtifactManager).
3. Model engine loading and readiness.
4. Liveness (/health) and Readiness (/ready) HTTP API endpoints.
5. System metadata and model versioning (/info).
6. Full inference prediction and structured response contract (/predict)
   on multiple representative samples without private path dependencies.
"""

import io
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_synthetic_test_image() -> bytes:
    """Generates an in-memory 32x32 test JPEG to test prediction without disk paths."""
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run_smoke_test():
    print("==================================================")
    print(" SIGNALSCOPE PRODUCTION SMOKE TEST")
    print("==================================================")

    # 1. Environment Check
    print("[1/6] Checking Python & core runtime...")
    import torch
    import timm
    import fastapi
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"  Timm:    {timm.__version__}")
    print(f"  FastAPI: {fastapi.__version__}")

    # 2. Model Artifact Resolution
    print("\n[2/6] Verifying model artifact resolution...")
    from app.inference.artifacts import ModelArtifactManager
    manager = ModelArtifactManager()
    ckpt_path = manager.resolve_checkpoint()
    scaler_path = manager.resolve_scaler()
    version_info = manager.get_version_info()
    print(f"  Model Version:  {version_info['model_version']}")
    print(f"  Resolved Ckpt:  {ckpt_path}")
    print(f"  Resolved Scaler:{scaler_path}")
    assert ckpt_path is not None, "Failed to resolve model checkpoint!"

    # 3. FastAPI Client Initialization
    print("\n[3/6] Initializing FastAPI test client & warming up engine...")
    t0 = time.perf_counter()
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    t_boot = (time.perf_counter() - t0) * 1000
    print(f"  App boot time: {t_boot:.2f} ms")

    # 4. Health and Readiness Probes
    print("\n[4/6] Testing liveness and readiness probes...")
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
    health_data = r_health.json()
    assert health_data["status"] == "healthy"
    print(f"  GET /health: {health_data}")

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200, f"Readiness check failed: {r_ready.text}"
    ready_data = r_ready.json()
    assert ready_data["status"] == "ready"
    print(f"  GET /ready:  {ready_data}")

    r_info = client.get("/api/v1/info")
    assert r_info.status_code == 200, f"Info check failed: {r_info.text}"
    info_data = r_info.json()
    assert info_data["model_metadata"]["model_version"] == "signalscope-baseline-v1"
    print(f"  GET /info:   Model '{info_data['model_metadata']['model_version']}' on device '{info_data['model_metadata']['device']}'")

    # 5. Prediction with In-Memory Clean Bytes
    print("\n[5/6] Testing prediction on synthetic in-memory payload...")
    test_bytes = generate_synthetic_test_image()
    t0 = time.perf_counter()
    r_pred = client.post(
        "/api/v1/predict",
        files={"file": ("in_memory_test.jpg", test_bytes, "image/jpeg")},
    )
    t_pred = (time.perf_counter() - t0) * 1000
    assert r_pred.status_code == 200, f"Prediction failed: {r_pred.text}"
    p_data = r_pred.json()
    assert p_data["schema_version"] == "1.0"
    assert "verdict" in p_data
    assert "probability" in p_data
    assert "evidence" in p_data
    assert p_data["evidence"]["spatial"]["available"] is True
    assert p_data["evidence"]["spectral"]["available"] is True
    assert p_data["evidence"]["robustness"]["available"] is True
    print(f"  Prediction Latency: {t_pred:.2f} ms")
    print(f"  Verdict: {p_data['verdict']}, Probability: {p_data['probability']:.4f}")

    # 6. Prediction on Curated Validation Samples
    print("\n[6/6] Testing prediction across representative validation samples...")
    sample_files = [
        ("Authentic Real", Path("frontend/public/samples/authentic_real.jpg")),
        ("Synthetic AI", Path("frontend/public/samples/synthetic_ai.jpg")),
        ("Borderline Volatile", Path("frontend/public/samples/borderline_uncertain.jpg")),
    ]

    for label, s_path in sample_files:
        if s_path.exists():
            with open(s_path, "rb") as f:
                img_bytes = f.read()
            t0 = time.perf_counter()
            resp = client.post(
                "/api/v1/predict",
                files={"file": (s_path.name, img_bytes, "image/jpeg")},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            data = resp.json()
            stab = data["evidence"]["robustness"]["stability_score"]
            print(f"  [{label}] Latency: {elapsed:.1f}ms | Verdict: {data['verdict']:<20} | Prob: {data['probability']:.4f} | Stability: {stab:.3f} | Uncertain: {data['uncertain']}")
        else:
            print(f"  [{label}] File {s_path} not found (skipped)")

    print("\n==================================================")
    print(" ALL PRODUCTION SMOKE TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
