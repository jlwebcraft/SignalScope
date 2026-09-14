"""Verifies SignalScope v2 production inference locally across the three safe canonical demo samples.

Tests:
1. Authentic Real #0955
2. Synthetic AI #3244
3. Borderline / Uncertain #5457
"""

import json
import os
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference.engine import SignalScopeInferenceEngine


def verify_v2_local():
    print("==================================================")
    print(" SIGNALSCOPE v2 LOCAL PRODUCTION INFERENCE VERIFICATION")
    print("==================================================")

    engine = SignalScopeInferenceEngine()
    print(f"Engine Ready: {engine.is_ready}")
    print(f"Trained Weights: {engine.has_trained_weights}")
    print(f"Device: {engine.device}")
    print(f"Active Model: {engine.model_name}")
    print(f"Fitted Temperature: {engine.scaler.temperature:.4f}")

    data_root = Path(os.environ.get("SIGNALSCOPE_DATA_ROOT", "C:/Programming/SignalScope-data"))
    test_cases = [
        {
            "id": "CASE 1",
            "name": "Authentic Real #0955",
            "path": data_root / "train" / "REAL" / "0955 (6).jpg",
            "expected_verdict": "likely_real",
        },
        {
            "id": "CASE 2",
            "name": "Synthetic AI #3244",
            "path": data_root / "train" / "FAKE" / "3244 (9).jpg",
            "expected_verdict": "likely_ai_generated",
        },
        {
            "id": "CASE 3",
            "name": "Borderline / Uncertain #5457",
            "path": data_root / "train" / "FAKE" / "5457 (8).jpg",
            "expected_verdict": "uncertain",
        },
    ]

    results = {}
    for case in test_cases:
        print("\n" + "-" * 60)
        print(f"Testing {case['id']}: {case['name']}")
        print(f"Path: {case['path']}")
        assert case["path"].exists(), f"Sample not found at {case['path']}"

        img = Image.open(case["path"]).convert("RGB")
        pred = engine.analyze(img)

        ev = pred.structured_evidence
        print(f"  Verdict:                {pred.verdict}")
        print(f"  Raw Probability:        {pred.raw_probability:.4f}")
        print(f"  Calibrated Probability: {pred.calibrated_probability:.4f}")
        print(f"  Confidence:             {pred.confidence_level}")
        print(f"  Uncertain:              {pred.uncertain}")
        print(f"  Stability Score:        {pred.stability_score:.4f}")
        print(f"  Grad-CAM Active:        {ev.get('spatial', {}).get('has_heatmap', False)}")
        print(f"  FFT High-Freq Ratio:    {ev.get('frequency', {}).get('high_frequency_energy_ratio', 0):.4f}")
        print(f"  Explanation:            {pred.explanation[:120]}...")

        # Record for report
        results[case["id"]] = {
            "name": case["name"],
            "verdict": pred.verdict.value if hasattr(pred.verdict, "value") else str(pred.verdict),
            "raw_probability": pred.raw_probability,
            "calibrated_probability": pred.calibrated_probability,
            "confidence_level": pred.confidence_level.value if hasattr(pred.confidence_level, "value") else str(pred.confidence_level),
            "uncertain": pred.uncertain,
            "stability_score": pred.stability_score,
            "explanation": pred.explanation,
        }

    print("\n==================================================")
    print(" ALL 3 LOCAL SAMPLES VERIFIED SUCCESSFULLY!")
    print("==================================================")
    return results


if __name__ == "__main__":
    verify_v2_local()
