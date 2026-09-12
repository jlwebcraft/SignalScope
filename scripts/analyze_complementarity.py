"""Complementarity and Error Analysis for SignalScope Phase 3.

Evaluates predictions on the 15,000 validation samples across:
1. Baseline Spatial ConvNeXt-Tiny
2. Frequency-Only Model (2D FFT 32x32)
3. RGB + FFT Dual-Branch Fusion Model

Computes:
- Score correlation (Pearson and Spearman)
- 2x2 Contingency Table:
    * Both correct
    * RGB correct / Frequency wrong
    * RGB wrong / Frequency correct
    * Both wrong
- Disagreement rate
- Representative error cases for qualitative failure mode inspection
- Saves lightweight structured JSON to report/experiments/complementarity_analysis.json
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from app.utils.logger import logger
from model.architectures.convnext import build_convnext_tiny
from model.architectures.frequency import FrequencyCNNBranch, compute_fft_2d
from model.architectures.fusion import DualBranchFusionDetector
from model.dataset import (
    SignalScopeDataset,
    SignalScopeDualDataset,
    create_honest_splits,
    get_default_transforms,
    get_native_transforms,
    scan_dataset_directory,
)


def get_validation_samples(seed: int = 42) -> List[Tuple[Path, int, str]]:
    """Loads the exact 15,000 validation samples from train/."""
    train_dir = Path(r"C:\Programming\SignalScope-data\train")
    all_samples = scan_dataset_directory(train_dir)
    _, val_samples, _ = create_honest_splits(
        all_samples,
        val_ratio=0.15,
        test_ratio=0.0,
        seed=seed,
    )
    return val_samples


def get_or_compute_baseline_predictions(
    val_samples: List[Tuple[Path, int, str]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves or computes predictions from baseline ConvNeXt-Tiny."""
    cache_path = PROJECT_ROOT / "report" / "experiments" / "baseline_val_predictions.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["y_true"], data["y_scores"]

    logger.info("Computing baseline predictions on validation set...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model = build_convnext_tiny(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    val_tf = get_default_transforms(image_size=224, is_training=False)
    ds = SignalScopeDataset(val_samples, transform=val_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    y_true_list, y_scores_list = [], []
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            y_true_list.extend(targets.squeeze(-1).cpu().numpy().tolist())
            y_scores_list.extend(probs.squeeze(-1).cpu().numpy().tolist())

    y_true = np.array(y_true_list, dtype=int)
    y_scores = np.array(y_scores_list, dtype=float)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, y_true=y_true, y_scores=y_scores)
    return y_true, y_scores


def get_or_compute_frequency_predictions(
    val_samples: List[Tuple[Path, int, str]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves or computes predictions from Frequency-Only model."""
    cache_path = PROJECT_ROOT / "report" / "experiments" / "frequency_only_val_predictions.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["y_true"], data["y_scores"]

    logger.info("Computing frequency-only predictions on validation set...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "frequency_only" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = FrequencyCNNBranch(in_channels=1, feature_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_tf = get_native_transforms(image_size=32, is_training=False)
    ds = SignalScopeDataset(val_samples, transform=val_tf)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    y_true_list, y_scores_list = [], []
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            freq_map = compute_fft_2d(imgs, shift=True, normalize=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(freq_map)
                probs = torch.sigmoid(logits)
            y_true_list.extend(targets.squeeze(-1).cpu().numpy().tolist())
            y_scores_list.extend(probs.squeeze(-1).cpu().numpy().tolist())

    y_true = np.array(y_true_list, dtype=int)
    y_scores = np.array(y_scores_list, dtype=float)
    np.savez_compressed(cache_path, y_true=y_true, y_scores=y_scores)
    return y_true, y_scores


def get_or_compute_fusion_predictions(
    val_samples: List[Tuple[Path, int, str]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves or computes predictions from RGB + FFT Fusion model."""
    cache_path = PROJECT_ROOT / "report" / "experiments" / "rgb_frequency_fusion_val_predictions.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["y_true"], data["y_scores"]

    logger.info("Computing fusion predictions on validation set...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fusion" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "rgb_frequency_fusion" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    spatial_tf = get_default_transforms(image_size=224, is_training=False)
    native_tf = get_native_transforms(image_size=32, is_training=False)
    ds = SignalScopeDualDataset(val_samples, spatial_transform=spatial_tf, native_transform=native_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    y_true_list, y_scores_list = [], []
    with torch.no_grad():
        for spatial_imgs, native_imgs, targets, _ in loader:
            spatial_imgs = spatial_imgs.to(device, non_blocking=True)
            native_imgs = native_imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(spatial_imgs, x_native=native_imgs)
                probs = torch.sigmoid(logits)
            y_true_list.extend(targets.squeeze(-1).cpu().numpy().tolist())
            y_scores_list.extend(probs.squeeze(-1).cpu().numpy().tolist())

    y_true = np.array(y_true_list, dtype=int)
    y_scores = np.array(y_scores_list, dtype=float)
    np.savez_compressed(cache_path, y_true=y_true, y_scores=y_scores)
    return y_true, y_scores


def run_complementarity_analysis() -> Dict[str, Any]:
    """Performs comprehensive complementarity, correlation, and error analysis."""
    logger.info("=" * 65)
    logger.info(" SignalScope Phase 3: Complementarity & Error Analysis")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_samples = get_validation_samples(seed=42)
    n_samples = len(val_samples)
    logger.info(f"Loaded {n_samples:,} validation samples.")

    # Get predictions
    y_true, scores_rgb = get_or_compute_baseline_predictions(val_samples, device)
    _, scores_freq = get_or_compute_frequency_predictions(val_samples, device)
    _, scores_fusion = get_or_compute_fusion_predictions(val_samples, device)

    # 1. Score Correlation Analysis
    pearson_corr_rf, _ = stats.pearsonr(scores_rgb, scores_freq)
    spearman_corr_rf, _ = stats.spearmanr(scores_rgb, scores_freq)

    pearson_corr_rfusion, _ = stats.pearsonr(scores_rgb, scores_fusion)
    pearson_corr_ffusion, _ = stats.pearsonr(scores_freq, scores_fusion)

    logger.info("\n--- Prediction Score Correlation ---")
    logger.info(f"  RGB vs Frequency: Pearson r = {pearson_corr_rf:.4f} | Spearman rho = {spearman_corr_rf:.4f}")
    logger.info(f"  RGB vs Fusion   : Pearson r = {pearson_corr_rfusion:.4f}")
    logger.info(f"  Freq vs Fusion  : Pearson r = {pearson_corr_ffusion:.4f}")

    # 2. Binary Decisions at threshold 0.50
    preds_rgb = (scores_rgb >= 0.50).astype(int)
    preds_freq = (scores_freq >= 0.50).astype(int)
    preds_fusion = (scores_fusion >= 0.50).astype(int)

    rgb_correct = (preds_rgb == y_true)
    freq_correct = (preds_freq == y_true)
    fusion_correct = (preds_fusion == y_true)

    # 2x2 Contingency Table (RGB vs Frequency)
    both_correct = int(np.sum(rgb_correct & freq_correct))
    rgb_corr_freq_wrong = int(np.sum(rgb_correct & ~freq_correct))
    rgb_wrong_freq_corr = int(np.sum(~rgb_correct & freq_correct))
    both_wrong = int(np.sum(~rgb_correct & ~freq_correct))

    disagreements = int(np.sum(preds_rgb != preds_freq))
    disagreement_rate = round(disagreements / n_samples * 100, 2)

    logger.info("\n--- 2x2 Contingency Table (RGB vs Frequency) ---")
    logger.info(f"  Both Correct                   : {both_correct:,} ({both_correct/n_samples*100:.2f}%)")
    logger.info(f"  RGB Correct / Frequency Wrong  : {rgb_corr_freq_wrong:,} ({rgb_corr_freq_wrong/n_samples*100:.2f}%)")
    logger.info(f"  RGB Wrong / Frequency Correct  : {rgb_wrong_freq_corr:,} ({rgb_wrong_freq_corr/n_samples*100:.2f}%)")
    logger.info(f"  Both Wrong                     : {both_wrong:,} ({both_wrong/n_samples*100:.2f}%)")
    logger.info(f"  Total Disagreements            : {disagreements:,} ({disagreement_rate}%)")

    # Impact of Fusion
    fusion_fixes_rgb = int(np.sum(~rgb_correct & fusion_correct))
    fusion_breaks_rgb = int(np.sum(rgb_correct & ~fusion_correct))
    net_gain = fusion_fixes_rgb - fusion_breaks_rgb

    logger.info("\n--- Fusion Impact Relative to RGB Baseline ---")
    logger.info(f"  Samples where Fusion fixed RGB error  : {fusion_fixes_rgb:,}")
    logger.info(f"  Samples where Fusion caused new error : {fusion_breaks_rgb:,}")
    logger.info(f"  Net Correctness Gain                  : {net_gain:+,}")

    # 3. Representative Error Analysis Cases
    # Focus on:
    # A. RGB wrong, Frequency correct
    # B. RGB correct, Frequency wrong
    # C. Both confident but wrong
    error_cases = []

    # Find cases where RGB wrong but Frequency correct
    cases_rgb_wrong_freq_corr = np.where(~rgb_correct & freq_correct)[0]
    for idx in cases_rgb_wrong_freq_corr[:3]:
        p, lbl, gen = val_samples[idx]
        error_cases.append({
            "category": "RGB_wrong_Frequency_correct",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "rgb_score": round(float(scores_rgb[idx]), 4),
            "freq_score": round(float(scores_freq[idx]), 4),
            "fusion_score": round(float(scores_fusion[idx]), 4),
            "observation": (
                f"RGB spatial branch incorrectly predicted {'Real' if lbl==1 else 'Synthetic'} "
                f"(score {scores_rgb[idx]:.2f}), while frequency spectral branch correctly identified "
                f"the sample (score {scores_freq[idx]:.2f})."
            ),
        })

    # Find cases where both confident but wrong
    both_wrong_indices = np.where(~rgb_correct & ~freq_correct)[0]
    for idx in both_wrong_indices[:2]:
        p, lbl, gen = val_samples[idx]
        error_cases.append({
            "category": "both_wrong",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "rgb_score": round(float(scores_rgb[idx]), 4),
            "freq_score": round(float(scores_freq[idx]), 4),
            "fusion_score": round(float(scores_fusion[idx]), 4),
            "observation": (
                f"Both spatial and spectral modalities failed on this sample. "
                f"True label is {'Synthetic' if lbl==1 else 'Real'}, but both models assigned scores "
                f"on the opposite side of 0.50 (RGB: {scores_rgb[idx]:.2f}, Freq: {scores_freq[idx]:.2f})."
            ),
        })

    # 4. Save Structured Report
    report = {
        "analysis_name": "complementarity_and_error_analysis",
        "validation_samples_count": n_samples,
        "score_correlations": {
            "rgb_vs_frequency_pearson_r": round(float(pearson_corr_rf), 4),
            "rgb_vs_frequency_spearman_rho": round(float(spearman_corr_rf), 4),
            "rgb_vs_fusion_pearson_r": round(float(pearson_corr_rfusion), 4),
            "frequency_vs_fusion_pearson_r": round(float(pearson_corr_ffusion), 4),
        },
        "contingency_table": {
            "both_correct": both_correct,
            "rgb_correct_frequency_wrong": rgb_corr_freq_wrong,
            "rgb_wrong_frequency_correct": rgb_wrong_freq_corr,
            "both_wrong": both_wrong,
            "total_disagreements": disagreements,
            "disagreement_rate_percent": disagreement_rate,
        },
        "fusion_impact": {
            "fusion_fixes_rgb_errors": fusion_fixes_rgb,
            "fusion_breaks_rgb_correct": fusion_breaks_rgb,
            "net_correctness_change": net_gain,
        },
        "representative_error_cases": error_cases,
        "methodological_notes": (
            "In the sampled training data, synthetic examples showed different spectral distributions "
            "from real examples, including localized high-frequency structure. The cause and generality "
            "of these patterns remain to be established experimentally. Local validation only; "
            "unseen-generator generalization remains pending organizer test evaluation."
        ),
    }

    report_path = PROJECT_ROOT / "report" / "experiments" / "complementarity_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nSaved complementarity analysis report to {report_path}")

    return report


if __name__ == "__main__":
    run_complementarity_analysis()
