"""Universal Evaluation Runner across Validation Domains for SignalScope.

Evaluates any candidate or baseline checkpoint on:
1. LOCAL VALIDATION (val_old):
   - 15,000 samples from C:\\Programming\\SignalScope-data\\train (seed=42)
2. NEW-DATA VALIDATION (val_new):
   - 1,118 samples from C:\\Programming\\SignalScope-data\\version-2\\AIGenImages2026\\val (559 real, 559 synthetic across 19 generators)

Metrics evaluated:
- ROC-AUC
- Macro-F1
- Accuracy
- Precision
- Recall
- FPR (at 0.50 and Youden threshold)
- Optimal Threshold (Youden's J)
- Confusion Matrix
- Brier Score
- ECE (Expected Calibration Error)
- Generator-by-generator accuracy & AUC breakdown on val_new

GUARANTEE:
STRICTLY EXCLUDES and NEVER TOUCHES C:\\Programming\\SignalScope-data\\test.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import logger
from model.architectures.convnext import build_convnext_tiny
from model.calibration import compute_brier_score, compute_ece
from model.dataset import SignalScopeDataset, get_default_transforms
from model.dataset_splits import get_official_splits, get_v2_splits
from model.evaluate import calculate_metrics, find_optimal_threshold_youden


def evaluate_dataset_partition(
    model: nn.Module,
    samples: List[Tuple[Path, int, str]],
    batch_size: int = 64,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_workers: int = 2,
    domain_name: str = "VALIDATION",
) -> Dict[str, Any]:
    """Runs inference across all samples and computes comprehensive metrics."""
    logger.info(f"Evaluating on {domain_name} ({len(samples):,} samples)...")
    
    transform = get_default_transforms(image_size=224, is_training=False)
    dataset = SignalScopeDataset(samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    all_targets: List[int] = []
    all_probs: List[float] = []
    all_generators: List[str] = []

    with torch.no_grad():
        for step, (images, targets, meta) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
            all_generators.extend(meta["generator"])

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                logger.info(f"  [{domain_name}] Step [{step + 1}/{len(loader)}] evaluated")

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)

    # Operating metrics at threshold 0.50
    metrics_050 = calculate_metrics(y_true, y_scores, operating_threshold=0.50)

    # Optimal threshold via Youden's J statistic
    optimal_th, best_j = find_optimal_threshold_youden(y_true, y_scores)
    metrics_opt = calculate_metrics(y_true, y_scores, operating_threshold=optimal_th)

    # Calibration metrics
    brier = compute_brier_score(y_true, y_scores)
    ece_val, mce_val, _ = compute_ece(y_true, y_scores, n_bins=15)

    # Breakdown by generator
    generator_breakdown = {}
    unique_generators = sorted(list(set(all_generators)))
    for gen in unique_generators:
        gen_mask = np.array([g == gen for g in all_generators])
        if np.sum(gen_mask) > 0:
            gen_true = y_true[gen_mask]
            gen_scores = y_scores[gen_mask]
            gen_pred_050 = (gen_scores >= 0.50).astype(int)
            gen_acc = float(np.mean(gen_pred_050 == gen_true))
            gen_mean_score = float(np.mean(gen_scores))
            generator_breakdown[gen] = {
                "sample_count": int(np.sum(gen_mask)),
                "accuracy_at_050": round(gen_acc, 4),
                "mean_predicted_prob": round(gen_mean_score, 4),
            }

    results = {
        "domain": domain_name,
        "sample_count": len(samples),
        "metrics_at_050": {
            "roc_auc": metrics_050["roc_auc"],
            "macro_f1": metrics_050["macro_f1"],
            "accuracy": metrics_050["accuracy"],
            "precision": metrics_050["precision"],
            "recall": metrics_050["recall"],
            "fpr": metrics_050["fpr"],
            "confusion_matrix": metrics_050["confusion_matrix"],
        },
        "optimal_threshold_metrics": {
            "optimal_threshold": round(float(optimal_th), 4),
            "youden_j": round(float(best_j), 4),
            "roc_auc": metrics_opt["roc_auc"],
            "macro_f1": metrics_opt["macro_f1"],
            "accuracy": metrics_opt["accuracy"],
            "fpr": metrics_opt["fpr"],
            "confusion_matrix": metrics_opt["confusion_matrix"],
        },
        "calibration": {
            "brier_score": round(float(brier), 4),
            "ece": round(float(ece_val), 4),
            "mce": round(float(mce_val), 4),
        },
        "generator_breakdown": generator_breakdown,
    }

    logger.info(
        f"[{domain_name}] ROC-AUC: {metrics_050['roc_auc']:.4f} | "
        f"Macro-F1 (0.50): {metrics_050['macro_f1']:.4f} | "
        f"Accuracy: {metrics_050['accuracy']:.4f} | "
        f"FPR: {metrics_050['fpr']:.4f} | "
        f"ECE: {ece_val:.4f}"
    )
    return results


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str = "convnext_tiny",
    output_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from: {ckpt_path} on {device}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Strip any potential prefix
    cleaned_state = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "")
        cleaned_state[k_clean] = v

    model = build_convnext_tiny(pretrained=False).to(device)
    model.load_state_dict(cleaned_state)
    model.eval()

    # Load splits
    _, val_old_samples = get_official_splits(seed=42)
    _, val_new_samples = get_v2_splits()

    val_old_results = evaluate_dataset_partition(
        model,
        val_old_samples,
        batch_size=128,
        device=device,
        domain_name="LOCAL VALIDATION",
    )

    val_new_results = evaluate_dataset_partition(
        model,
        val_new_samples,
        batch_size=128,
        device=device,
        domain_name="NEW-DATA VALIDATION",
    )

    overall_results = {
        "checkpoint": str(ckpt_path),
        "model_architecture": model_name,
        "evaluated_at": str(np.datetime64("now")),
        "local_validation": val_old_results,
        "new_data_validation": val_new_results,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(overall_results, f, indent=2)
        logger.info(f"Results written to {out_path}")

    return overall_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt checkpoint across validation domains")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_convnext/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/evaluation/baseline_dual_eval.json")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, output_json=args.output)
