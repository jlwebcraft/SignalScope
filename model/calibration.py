"""Probability Calibration module for SignalScope using Temperature Scaling.

Implements post-hoc probability calibration:
    p_calibrated = sigmoid(logit / T)

Methodology:
- Calibration set: 5,000 samples carved deterministically from the 85,000 training partition (seed 42).
- Evaluation set: Untouched 15,000 local validation partition.
- Optimizer: L-BFGS minimizing Negative Log-Likelihood (Binary Cross Entropy).
- Metrics: Brier Score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Reliability Diagrams.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.utils.logger import logger
from model.dataset import SignalScopeDataset, get_default_transforms, scan_dataset_directory
from model.evaluate import calculate_metrics


# =====================================================================
# 1. Calibration Metrics: Brier Score & ECE
# =====================================================================

def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Computes mean squared error between probabilities and binary ground truth.

    Brier = (1/N) * sum((p_i - y_i)^2) in [0, 1]. Lower is better.
    """
    return float(np.mean((y_probs - y_true) ** 2))


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, Dict[str, Any]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    For binary classification:
        confidence = max(p, 1 - p)
        accuracy = 1 if predicted_class == true_class else 0
        ECE = sum((|B_m| / N) * |acc(B_m) - conf(B_m)|)

    Returns:
        (ece, mce, bin_details_dict)
    """
    preds = (y_probs >= 0.50).astype(int)
    confidences = np.where(preds == 1, y_probs, 1.0 - y_probs)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_details = []

    n_samples = len(y_true)

    for bin_idx, (b_low, b_high) in enumerate(zip(bin_lowers, bin_uppers)):
        # Include upper boundary in last bin
        if bin_idx == n_bins - 1:
            in_bin = (confidences >= b_low) & (confidences <= b_high)
        else:
            in_bin = (confidences >= b_low) & (confidences < b_high)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(accuracies[in_bin]))
            gap = abs(avg_accuracy - avg_confidence)
            ece += float(prop_in_bin * gap)
            mce = max(mce, float(gap))

            bin_details.append({
                "bin_idx": bin_idx,
                "range": [round(b_low, 4), round(b_high, 4)],
                "count": int(np.sum(in_bin)),
                "proportion": round(float(prop_in_bin), 4),
                "avg_confidence": round(avg_confidence, 4),
                "avg_accuracy": round(avg_accuracy, 4),
                "gap": round(float(gap), 4),
            })
        else:
            bin_details.append({
                "bin_idx": bin_idx,
                "range": [round(b_low, 4), round(b_high, 4)],
                "count": 0,
                "proportion": 0.0,
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
                "gap": 0.0,
            })

    return round(float(ece), 4), round(float(mce), 4), {"n_bins": n_bins, "bins": bin_details}


# =====================================================================
# 2. Temperature Scaler Module
# =====================================================================

class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling model for binary logit calibration.

    Fits scalar T > 0 such that calibrated_prob = sigmoid(logit / T).
    Preserves class ranking and ROC-AUC identically while calibrating confidence.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize T = 1.0 (log(T) = 0.0) for unconstrained optimization
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature."""
        t = torch.exp(self.log_temperature).to(device=logits.device, dtype=logits.dtype)
        return logits / t

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities via sigmoid(logits / T)."""
        scaled = self.forward(logits)
        return torch.sigmoid(scaled)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Fits temperature T on calibration logits using L-BFGS to minimize NLL."""
        self.train()
        device = logits.device
        self.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        fitted_t = self.temperature
        logger.info(f"Fitted optimal calibration temperature T = {fitted_t:.4f}")
        return fitted_t

    def save(self, path: Union[str, Path]) -> None:
        """Saves calibrated temperature to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": round(self.temperature, 5),
            "log_temperature": float(self.log_temperature.item()),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved temperature scaler (T = {self.temperature:.4f}) to {p}")

    def load(self, path: Union[str, Path]) -> None:
        """Loads temperature parameter from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Temperature scaler file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        with torch.no_grad():
            self.log_temperature.copy_(torch.tensor([data["log_temperature"]]))
        logger.info(f"Loaded temperature scaler (T = {self.temperature:.4f}) from {p}")


# =====================================================================
# 3. Data Partitioning for Calibration
# =====================================================================

def get_calibration_splits(
    train_dir: Path,
    calibration_size: int = 5000,
    val_size: int = 15000,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int, str]], List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    """Partitions official train/ (100,000 samples) into:

    - 80,000 Actual Training
    - 5,000 Dedicated Calibration (2,500 real, 2,500 synthetic)
    - 15,000 Untouched Local Validation (7,500 real, 7,500 synthetic)

    Never touches C:\\Programming\\SignalScope-data\\test.
    """
    all_samples = scan_dataset_directory(train_dir)
    rng = np.random.RandomState(seed)

    real_samples = [s for s in all_samples if s[1] == 0]
    fake_samples = [s for s in all_samples if s[1] == 1]

    n_val_per_class = val_size // 2
    n_cal_per_class = calibration_size // 2

    # Shuffle deterministically
    idx_real = rng.permutation(len(real_samples))
    idx_fake = rng.permutation(len(fake_samples))

    # 1. Untouched validation split (exact same first 7,500 per class as standard honest split)
    val_real = [real_samples[i] for i in idx_real[:n_val_per_class]]
    val_fake = [fake_samples[i] for i in idx_fake[:n_val_per_class]]
    val_samples = val_real + val_fake

    # 2. Carve dedicated calibration split from the training pool
    train_pool_real = [real_samples[i] for i in idx_real[n_val_per_class:]]
    train_pool_fake = [fake_samples[i] for i in idx_fake[n_val_per_class:]]

    cal_real = train_pool_real[:n_cal_per_class]
    cal_fake = train_pool_fake[:n_cal_per_class]
    cal_samples = cal_real + cal_fake

    # 3. Remaining training pool
    actual_train_real = train_pool_real[n_cal_per_class:]
    actual_train_fake = train_pool_fake[n_cal_per_class:]
    train_samples = actual_train_real + actual_train_fake

    # Verify zero leakage
    train_paths = set(str(s[0]) for s in train_samples)
    cal_paths = set(str(s[0]) for s in cal_samples)
    val_paths = set(str(s[0]) for s in val_samples)

    assert len(cal_paths.intersection(val_paths)) == 0, "Leakage: Calibration and Validation overlap!"
    assert len(train_paths.intersection(cal_paths)) == 0, "Leakage: Train and Calibration overlap!"
    assert len(train_paths.intersection(val_paths)) == 0, "Leakage: Train and Validation overlap!"

    rng.shuffle(val_samples)
    rng.shuffle(cal_samples)
    rng.shuffle(train_samples)

    logger.info(
        f"Calibration Splits: Train={len(train_samples):,} | "
        f"Calibration={len(cal_samples):,} | Validation={len(val_samples):,}"
    )
    return train_samples, cal_samples, val_samples


# =====================================================================
# 4. Logit Extraction Runner
# =====================================================================

@torch.no_grad()
def extract_model_logits(
    model: nn.Module,
    samples: List[Tuple[Path, int, str]],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts raw unscaled binary classification logits and targets."""
    model.eval()
    val_tf = get_default_transforms(image_size=224, is_training=False)
    ds = SignalScopeDataset(samples, transform=val_tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.float().cpu())

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    return cat_logits, cat_targets


# =====================================================================
# 5. Calibration Visualizations
# =====================================================================

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs_pre: np.ndarray,
    y_probs_post: np.ndarray,
    save_path: Path,
    n_bins: int = 15,
) -> None:
    """Plots comparative reliability diagram (Accuracy vs. Confidence) before and after calibration."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    for ax, probs, title, color in [
        (ax1, y_probs_pre, "Before Calibration (Raw Sigmoid)", "#d62728"),
        (ax2, y_probs_post, "After Calibration (Temperature Scaled)", "#2ca02c"),
    ]:
        ece, mce, details = compute_ece(y_true, probs, n_bins=n_bins)
        bins = details["bins"]
        confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accs = [b["avg_accuracy"] for b in bins if b["count"] > 0]
        widths = [b["range"][1] - b["range"][0] for b in bins if b["count"] > 0]
        centers = [(b["range"][0] + b["range"][1]) / 2 for b in bins if b["count"] > 0]

        # Perfect calibration line
        ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", label="Perfect Calibration")

        # Bars
        ax.bar(centers, accs, width=widths, alpha=0.6, color=color, edgecolor="black", label="Outputs")

        # Gap markers
        for c, a in zip(confs, accs):
            ax.plot([c, c], [c, a], color="black", linestyle=":", alpha=0.7)

        ax.set_xlim([0.5, 1.0])
        ax.set_ylim([0.45, 1.02])
        ax.set_xlabel(r"Confidence $\max(p, 1-p)$")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE: {ece:.4f} | MCE: {mce:.4f}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved reliability diagram to {save_path}")


def plot_confidence_histogram(
    y_probs_pre: np.ndarray,
    y_probs_post: np.ndarray,
    save_path: Path,
) -> None:
    """Plots distribution of predicted confidence max(p, 1-p) before and after."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    conf_pre = np.maximum(y_probs_pre, 1.0 - y_probs_pre)
    conf_post = np.maximum(y_probs_post, 1.0 - y_probs_post)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bins = np.linspace(0.5, 1.0, 26)

    ax.hist(conf_pre, bins=bins, alpha=0.5, label=f"Pre-Calibration (Mean: {np.mean(conf_pre):.4f})", color="#1f77b4")
    ax.hist(conf_post, bins=bins, alpha=0.5, label=f"Post-Calibration (Mean: {np.mean(conf_post):.4f})", color="#2ca02c")

    ax.set_xlabel(r"Predicted Confidence $\max(p, 1-p)$")
    ax.set_ylabel("Sample Count")
    ax.set_title("Prediction Confidence Distribution (Validation Split: 15,000 samples)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confidence histogram to {save_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_probs_pre: np.ndarray,
    y_probs_post: np.ndarray,
    save_path: Path,
) -> None:
    """Plots standard calibration curve: fraction of positives vs mean predicted probability."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    ax.plot([0, 1], [0, 1], "k--", label="Ideal Calibration")

    for probs, label, color in [
        (y_probs_pre, "Raw Sigmoid (Pre-Calibration)", "#d62728"),
        (y_probs_post, "Temperature Scaled (Post-Calibration)", "#2ca02c"),
    ]:
        bins = np.linspace(0.0, 1.0, 11)
        prob_means = []
        emp_accs = []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if np.sum(mask) > 0:
                prob_means.append(float(np.mean(probs[mask])))
                emp_accs.append(float(np.mean(y_true[mask])))

        ax.plot(prob_means, emp_accs, "s-", color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Mean Predicted Probability (Synthetic)")
    ax.set_ylabel("Fraction of Positives (Actual Synthetic)")
    ax.set_title("Calibration Curve (Local Validation Set: 15,000 samples)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved calibration curve to {save_path}")
