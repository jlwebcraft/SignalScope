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
from typing import Any, Dict, List, Optional, Tuple, Union

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


def find_optimal_threshold_youden(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """Finds optimal operating threshold maximizing Youden's J statistic (TPR - FPR).

    Returns:
        Tuple of (optimal_threshold, max_j_statistic).
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_scores)
    j_scores = tprs - fprs
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    # Clip between 0.01 and 0.99 for numerical stability
    best_threshold = float(np.clip(best_threshold, 0.01, 0.99))
    return best_threshold, float(j_scores[best_idx])


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: Union[str, Path],
    title: str = "SignalScope Baseline ROC Curve",
) -> None:
    """Generates and saves high-resolution ROC curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fprs, tprs, _ = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fprs, tprs, color="#2563eb", lw=2, label=f"ConvNeXt Baseline (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()
    logger.info(f"Saved ROC curve plot to {out_p}")


def plot_confusion_matrix_figure(
    cm_matrix: List[List[int]],
    output_path: Union[str, Path],
    class_names: Tuple[str, str] = ("Real", "Synthetic"),
    title: str = "Baseline Confusion Matrix",
) -> None:
    """Generates and saves annotated confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm_matrix)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    plt.ylabel("Ground Truth", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {out_p}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Baseline Training & Validation Curves",
) -> None:
    """Plots training/validation loss and validation AUC curves over epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#dc2626", lw=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", color="#2563eb", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (BCE With Logits)", fontsize=11)
    ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Validation AUC & F1 curves
    ax2.plot(epochs, history["val_auc"], "o-", color="#16a34a", lw=2, label="Val ROC-AUC")
    ax2.plot(epochs, history["val_f1"], "s-", color="#9333ea", lw=2, label="Val Macro-F1")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_title("Validation ROC-AUC & Macro-F1", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()
    logger.info(f"Saved training history curves to {out_p}")


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
