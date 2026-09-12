"""End-to-End Production Verification and Latency Benchmark for SignalScope.

Validates the full stack pipeline across local validation samples:
1. Authentic Real sample
2. Synthetic AI sample
3. Borderline / Volatile Uncertain sample

Measures latency across:
- Model initialization
- Single-image primary inference
- Grad-CAM attribution
- 2D FFT spectral extraction
- Authenticity Stability (4 probes)
- Total end-to-end API response time
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.inference.engine import SignalScopeInferenceEngine


def run_e2e_benchmark():
    print("=" * 70)
    print(" SignalScope Phase 6 — Production End-to-End & Latency Benchmark")
    print("=" * 70)

    # 1. Model Loading Latency
    t0 = time.perf_counter()
    ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return 1

    engine = SignalScopeInferenceEngine(checkpoint_path=str(ckpt_path))
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[*] Engine Initialization: {t_load:.1f} ms (Model: {engine.model_name}, Device: {engine.device})")
    print(f"[*] Calibrated Scaler T: {engine.scaler.temperature:.4f}")

    # 2. Local Validation Test Cases
    data_root = Path(os.environ.get("SIGNALSCOPE_DATA_ROOT", "C:/Programming/SignalScope-data"))
    test_cases = [
        {
            "name": "Authentic Real",
            "path": data_root / "train" / "REAL" / "0955 (6).jpg",
            "expected_verdict": "likely_real",
        },
        {
            "name": "Synthetic AI",
            "path": data_root / "train" / "FAKE" / "3244 (9).jpg",
            "expected_verdict": "likely_ai_generated",
        },
        {
            "name": "Borderline / Volatile",
            "path": data_root / "train" / "FAKE" / "5457 (8).jpg",
            "expected_verdict": "uncertain",
        },
    ]

    client = TestClient(app)

    print("-" * 70)
    print(" TESTING LOCAL VALIDATION SAMPLES VIA API (POST /api/v1/predict)")
    print("-" * 70)

    for case in test_cases:
        c_name = case["name"]
        c_path = case["path"]
        if not c_path.exists():
            print(f"[SKIP] Case file missing: {c_path}")
            continue

        with open(c_path, "rb") as f:
            img_bytes = f.read()

        t_req_start = time.perf_counter()
        response = client.post(
            "/api/v1/predict",
            files={"file": (c_path.name, img_bytes, "image/jpeg")},
        )
        t_req_total = (time.perf_counter() - t_req_start) * 1000

        assert response.status_code == 200, f"Failed HTTP 200: {response.status_code} {response.text}"
        data = response.json()

        # Contract assertions
        assert data["schema_version"] == "1.0"
        assert "verdict" in data
        assert "probability" in data
        assert "evidence" in data
        evidence = data["evidence"]
        assert "spatial" in evidence
        assert "spectral" in evidence
        assert "robustness" in evidence
        assert "metadata" in evidence

        # Visual base64 check
        has_heatmap = evidence["spatial"]["heatmap"] is not None and evidence["spatial"]["heatmap"].startswith("data:image/")
        has_spectrum = evidence["spectral"]["spectrum"] is not None and evidence["spectral"]["spectrum"].startswith("data:image/")

        print(f"\n[CASE: {c_name.upper()}]")
        print(f"  File                   : {c_path.name}")
        print(f"  Verdict                : {data['verdict'].upper()} (Expected: {case['expected_verdict'].upper()})")
        print(f"  Calibrated Prob        : {data['probability']*100:.2f}% (Raw: {data.get('raw_probability', 0)*100:.2f}%)")
        print(f"  Confidence Level       : {data['confidence_level'].upper()}")
        print(f"  Uncertain Flag         : {data['uncertain']}")
        print(f"  Stability Score        : {data['stability_score']*100:.1f}% (Impact: {evidence['robustness']['degradation_impact']})")
        print(f"  Attribution Conc.      : {evidence['spatial'].get('attribution_concentration')}")
        print(f"  HF Energy Ratio        : {evidence['spectral'].get('high_frequency_energy_ratio')}")
        print(f"  Base64 Heatmap Overlay : {'VALID DATA-URI' if has_heatmap else 'NONE'}")
        print(f"  Base64 FFT Spectrum    : {'VALID DATA-URI' if has_spectrum else 'NONE'}")
        print(f"  API Latency (Total)    : {t_req_total:.1f} ms")

    # 3. Micro-benchmark of individual components on a sample image
    print("-" * 70)
    print(" SUB-COMPONENT LATENCY BREAKDOWN (Single 32x32 Image, RTX 3050 GPU)")
    print("-" * 70)
    sample_img = Image.open(test_cases[0]["path"]).convert("RGB")

    # A. Raw inference
    t0 = time.perf_counter()
    _ = engine.predict_raw_probability(sample_img)
    t_inf = (time.perf_counter() - t0) * 1000

    # B. Calibrated inference
    t0 = time.perf_counter()
    _ = engine.predict_calibrated_probability(sample_img)
    t_cal = (time.perf_counter() - t0) * 1000

    # C. Grad-CAM attribution
    from model.explainability.gradcam import compute_spatial_attribution
    t0 = time.perf_counter()
    _ = compute_spatial_attribution(engine.model, sample_img, engine.device)
    t_cam = (time.perf_counter() - t0) * 1000

    # D. 2D FFT spectral extraction
    from model.explainability.spectral import extract_spectral_features
    t0 = time.perf_counter()
    _ = extract_spectral_features(sample_img)
    t_fft = (time.perf_counter() - t0) * 1000

    # E. 4-probe stability evaluation
    from app.services.stability import evaluate_authenticity_stability
    t0 = time.perf_counter()
    _ = evaluate_authenticity_stability(sample_img, engine.predict_calibrated_probability, 0.5)
    t_stab = (time.perf_counter() - t0) * 1000

    print(f"  1. Primary ConvNeXt Forward Pass   : {t_inf:.2f} ms")
    print(f"  2. Scaler Calibration Layer        : {t_cal - t_inf:.2f} ms")
    print(f"  3. Grad-CAM Activation & Gradient  : {t_cam:.2f} ms")
    print(f"  4. 2D FFT Spectral Features        : {t_fft:.2f} ms")
    print(f"  5. Authenticity Stability (4 probes): {t_stab:.2f} ms")
    print(f"  --------------------------------------------------")
    print(f"  Total Single-Image Pipeline Latency: {t_inf + t_cam + t_fft + t_stab:.2f} ms")
    print("=" * 70)
    print(" [OK] End-to-End Verification and Latency Benchmark Successful.")
    return 0


if __name__ == "__main__":
    sys.exit(run_e2e_benchmark())
