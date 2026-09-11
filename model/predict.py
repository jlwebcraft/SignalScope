"""CLI prediction runner for SignalScope.

Usage:
    python model/predict.py --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference.engine import SignalScopeInferenceEngine
from app.utils.image import load_image_safely
from app.utils.logger import logger


def run_prediction(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    format_json: bool = False,
) -> int:
    """Loads image, runs inference engine with trained checkpoint, and displays results."""
    p = Path(image_path)
    if not p.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    # Checkpoint resolution: specified path or default baseline location
    default_ckpt = Path("checkpoints/baseline_convnext/best_model.pt")
    resolved_ckpt: Optional[Path] = None

    if checkpoint_path:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            logger.error(f"Specified checkpoint not found: {checkpoint_path}")
            print(f"\n[ERROR] Specified checkpoint not found: {checkpoint_path}")
            return 1
        resolved_ckpt = ckpt_p
    elif default_ckpt.exists():
        resolved_ckpt = default_ckpt
    else:
        logger.warning("No trained checkpoint supplied or found at default location.")
        print("\n" + "=" * 65)
        print(" [!] NO TRAINED MODEL CHECKPOINT AVAILABLE")
        print("=" * 65)
        print(" A valid trained model checkpoint is required to perform authenticity detection.")
        print(" Please specify --checkpoint <path_to_model.pt> or train a baseline model using:")
        print("     python model/train.py --config model/configs/baseline_convnext.yaml")
        print(" (Untrained or random weights will not be presented as a meaningful detector.)")
        print("=" * 65 + "\n")
        return 1

    try:
        pil_img = load_image_safely(p)
    except Exception as exc:
        logger.error(f"Failed to load image safely: {exc}")
        return 1

    engine = SignalScopeInferenceEngine(checkpoint_path=str(resolved_ckpt))
    if not engine.has_trained_weights:
        print(f"\n[ERROR] Failed to load model weights from {resolved_ckpt}")
        return 1

    result = engine.analyze(pil_img)

    if format_json:
        payload = result.model_dump()
        payload["model_identifier"] = engine.model_name
        payload["checkpoint_path"] = str(resolved_ckpt)
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("\n" + "=" * 65)
        print(" SIGNALSCOPE AUTHENTICITY ASSESSMENT")
        print(" 'Telling Real From Synthetic in the Age of Generative Media'")
        print("=" * 65)
        print(f" Model Identifier       : {engine.model_name}")
        print(f" Checkpoint Source      : {resolved_ckpt}")
        print(f" Image                  : {image_path}")
        print(f" Verdict / Class        : {result.verdict.value.upper()}")
        print(f" Synthetic Probability  : {result.probability * 100:.2f}%")
        print(f" Operating Threshold    : {engine.operating_threshold:.2f}")
        print(f" Confidence Level       : {result.confidence_level.value.upper()}")
        print(f" Stability Score        : {result.stability_score * 100:.1f}% ({result.stability.degradation_impact} impact)")
        print(f" Evidence Conflict      : {'YES' if result.evidence_disagreement else 'NO'}")
        print("-" * 65)
        print(" EVIDENCE BREAKDOWN:")
        for item in result.evidence:
            marker = "[AI]" if item.supports_synthetic else "[REAL]"
            print(f"   * {marker} {item.source} ({item.metric}): score={item.score:.2f} -> {item.description}")
        print("-" * 65)
        print(" GROUNDED EXPLANATION:")
        print(f"   {result.explanation}")
        print("-" * 65)
        print(f" DISCLAIMER: {result.disclaimer}")
        print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope CLI Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    args = parser.parse_args()

    sys.exit(run_prediction(args.image, checkpoint_path=args.checkpoint, format_json=args.json))
