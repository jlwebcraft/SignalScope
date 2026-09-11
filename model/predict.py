"""CLI prediction runner for SignalScope.

Usage:
    python model/predict.py --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference.engine import SignalScopeInferenceEngine
from app.utils.image import load_image_safely
from app.utils.logger import logger


def run_prediction(image_path: str, format_json: bool = False) -> int:
    """Loads image, runs inference engine, and displays results."""
    p = Path(image_path)
    if not p.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    try:
        pil_img = load_image_safely(p)
    except Exception as exc:
        logger.error(f"Failed to load image safely: {exc}")
        return 1

    engine = SignalScopeInferenceEngine()
    result = engine.analyze(pil_img)

    if format_json:
        print(result.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 55)
        print(" SIGNALSCOPE AUTHENTICITY ASSESSMENT")
        print(" 'Telling Real From Synthetic in the Age of Generative Media'")
        print("=" * 55)
        if result.is_development_placeholder:
            print(" [!] NOTICE: DEVELOPMENT PLACEHOLDER — NOT A VALID MODEL PREDICTION")
            print("     Trained model weights have not been loaded. Output is heuristic scaffolding.")
            print("-" * 55)
        print(f" Image                  : {image_path}")
        print(f" Verdict                : {result.verdict.value.upper()}")
        print(f" Synthetic Probability  : {result.probability * 100:.1f}%")
        print(f" Confidence Level       : {result.confidence_level.value.upper()}")
        print(f" Stability Score        : {result.stability_score * 100:.1f}% ({result.stability.degradation_impact} impact)")
        print(f" Evidence Conflict      : {'YES' if result.evidence_disagreement else 'NO'}")
        print("-" * 55)
        print(" EVIDENCE BREAKDOWN:")
        for item in result.evidence:
            marker = "[AI]" if item.supports_synthetic else "[REAL]"
            print(f"   * {marker} {item.source} ({item.metric}): score={item.score:.2f} -> {item.description}")
        print("-" * 55)
        print(" GROUNDED EXPLANATION:")
        print(f"   {result.explanation}")
        print("-" * 55)
        print(f" DISCLAIMER: {result.disclaimer}")
        print("=" * 55 + "\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope CLI Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    args = parser.parse_args()

    sys.exit(run_prediction(args.image, format_json=args.json))
