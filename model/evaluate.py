"""Evaluation metrics and evaluation runner for SignalScope.

Computes mandatory SIH 2026 challenge metrics:
- ROC-AUC
- Macro-F1
- Confusion Matrix
- Accuracy, Precision, Recall
- False Positive Rate (FPR) at operating threshold
- Unseen generator breakdown
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.utils.logger import logger


def calculate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    operating_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Computes comprehensive authenticity metrics.

    Args:
        y_true: Ground truth binary array (0 = real, 1 = synthetic).
        y_scores: Predicted probabilities of synthetic class.
        operating_threshold: Operating decision threshold for binary classification.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= operating_threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Accuracy, Precision, Recall, Macro-F1
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.5

    # False Positive Rate (FP / (FP + TN))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # True Negative Rate / Specificity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "operating_threshold": operating_threshold,
        "roc_auc": round(roc_auc, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fpr": round(fpr, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "sample_count": {
            "total": len(y_true),
            "real": int(np.sum(y_true == 0)),
            "synthetic": int(np.sum(y_true == 1)),
        },
    }


def find_threshold_for_target_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> Tuple[float, float]:
    """Finds the operating threshold that achieves a target False Positive Rate (e.g. 5%).

    Returns:
        Tuple of (optimal_threshold, actual_fpr_at_threshold).
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fprs <= target_fpr)[0]
    if len(idx) > 0:
        best_idx = idx[-1]
        return float(thresholds[best_idx]), float(fprs[best_idx])
    return 0.50, float(fprs[0])


def print_metrics_report(metrics: Dict[str, Any], title: str = "Evaluation Report") -> None:
    """Prints a clean, formatted metrics summary."""
    cm = metrics["confusion_matrix"]
    logger.info("==================================================")
    logger.info(f" {title} ")
    logger.info("==================================================")
    logger.info(f" Operating Threshold : {metrics['operating_threshold']:.2f}")
    logger.info(f" ROC-AUC             : {metrics['roc_auc']:.4f}")
    logger.info(f" Macro-F1            : {metrics['macro_f1']:.4f}")
    logger.info(f" Accuracy            : {metrics['accuracy'] * 100:.2f}%")
    logger.info(f" Precision           : {metrics['precision'] * 100:.2f}%")
    logger.info(f" Recall              : {metrics['recall'] * 100:.2f}%")
    logger.info(f" False Positive Rate : {metrics['fpr'] * 100:.2f}%")
    logger.info(f" Specificity         : {metrics['specificity'] * 100:.2f}%")
    logger.info(" Confusion Matrix:")
    logger.info(f"   TN: {cm['tn']:<6} | FP: {cm['fp']:<6}")
    logger.info(f"   FN: {cm['fn']:<6} | TP: {cm['tp']:<6}")
    logger.info("==================================================")


if __name__ == "__main__":
    # Test metric calculation on synthetic mock data
    y_t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_s = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9])
    res = calculate_metrics(y_t, y_s, operating_threshold=0.5)
    print_metrics_report(res, title="Verification Test Metrics")
