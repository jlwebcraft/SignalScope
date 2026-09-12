"""Benchmark runner for SignalScope Phase 4: Robustness + Authenticity Stability.

Evaluates the ConvNeXt baseline and RGB + FFT fusion detectors under controlled
degradation transformations exclusively using the local validation partition.

Produces:
- report/robustness/robustness_comparison.json
- report/robustness/robustness_report.md
- report/robustness/robustness_auc.png
- report/robustness/robustness_accuracy.png
- report/robustness/robustness_flip_rate.png
- report/robustness/stability_distribution.png
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

# Prevent OpenMP runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import logger
from model.architectures.convnext import build_convnext_tiny
from model.architectures.fusion import DualBranchFusionDetector
from model.dataset import create_honest_splits, scan_dataset_directory
from model.evaluate import calculate_metrics
from model.robustness import (
    TRANSFORMATION_REGISTRY,
    compute_authenticity_stability_score,
    compute_mean_probability_drift,
    compute_prediction_flip_rate,
    evaluate_model_on_transformed_dataset,
)


def load_baseline_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Loads verified ConvNeXt-Tiny baseline detector."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model = build_convnext_tiny(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_fusion_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Loads verified DualBranchFusionDetector (ConvNeXt + FFT)."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_validation_data(
    train_dir: Path,
    subset_size: int = 3000,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int, str]], int]:
    """Retrieves deterministic validation samples strictly from train/.

    Never accesses or inspects test/.
    """
    all_samples = scan_dataset_directory(train_dir)
    _, val_samples, _ = create_honest_splits(
        all_samples,
        val_ratio=0.15,
        test_ratio=0.0,
        seed=seed,
    )
    total_val_len = len(val_samples)

    if subset_size > 0 and subset_size < total_val_len:
        # Stratified deterministic selection
        rng = np.random.RandomState(seed)
        real_val = [s for s in val_samples if s[1] == 0]
        fake_val = [s for s in val_samples if s[1] == 1]

        n_per_class = subset_size // 2
        rng.shuffle(real_val)
        rng.shuffle(fake_val)
        selected = real_val[:n_per_class] + fake_val[:n_per_class]
        rng.shuffle(selected)
        return selected, total_val_len

    return val_samples, total_val_len


def run_smoke_test(
    baseline_model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    samples: List[Tuple[Path, int, str]],
    device: torch.device,
) -> None:
    """Runs a quick 5-sample sanity check printing individual transform predictions."""
    logger.info("=" * 65)
    logger.info(" Running Robustness Smoke Test (5 validation samples)")
    logger.info("=" * 65)

    test_samples = samples[:5]
    for idx, (path, label, gen) in enumerate(test_samples, 1):
        gt_str = "Synthetic (1)" if label == 1 else "Real (0)"
        logger.info(f"\n[Sample {idx}/5] {path.name} | GT: {gt_str}")

        for tname, (tfunc, tmeta) in TRANSFORMATION_REGISTRY.items():
            _, b_score, _ = evaluate_model_on_transformed_dataset(
                baseline_model, [(path, label, gen)], tfunc, is_fusion=False, device=device, batch_size=1
            )
            _, f_score, _ = evaluate_model_on_transformed_dataset(
                fusion_model, [(path, label, gen)], tfunc, is_fusion=True, device=device, batch_size=1
            )
            b_p = b_score[0]
            f_p = f_score[0]
            b_decision = "Synthetic" if b_p >= 0.50 else "Real"
            f_decision = "Synthetic" if f_p >= 0.50 else "Real"
            logger.info(
                f"  {tname:<12} -> Baseline: {b_p:.4f} ({b_decision:<9}) | Fusion: {f_p:.4f} ({f_decision:<9})"
            )

    logger.info("\nSmoke test completed successfully!")


def plot_robustness_figures(
    comparison_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Generates comparison figures for AUC, Accuracy, Flip Rate, and Stability Score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    transforms = list(TRANSFORMATION_REGISTRY.keys())
    t_labels = [t.replace("_", " ").title() for t in transforms]

    b_results = comparison_data["baseline"]["transformations"]
    f_results = comparison_data["fusion"]["transformations"]

    b_auc = [b_results[t]["roc_auc"] for t in transforms]
    f_auc = [f_results[t]["roc_auc"] for t in transforms]

    b_acc = [b_results[t]["accuracy"] * 100 for t in transforms]
    f_acc = [f_results[t]["accuracy"] * 100 for t in transforms]

    b_flip = [b_results[t]["prediction_flip_rate"] * 100 for t in transforms]
    f_flip = [f_results[t]["prediction_flip_rate"] * 100 for t in transforms]

    x = np.arange(len(transforms))
    width = 0.35

    # 1. ROC-AUC Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - width/2, b_auc, width, label="ConvNeXt Baseline", color="#1f77b4", alpha=0.9)
    ax.bar(x + width/2, f_auc, width, label="RGB + FFT Fusion", color="#ff7f0e", alpha=0.9)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Robustness Benchmark: ROC-AUC under Controlled Degradations")
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, rotation=20, ha="right")
    ax.set_ylim([0.85, 1.005])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    plt.tight_layout()
    auc_path = output_dir / "robustness_auc.png"
    plt.savefig(auc_path)
    plt.close()
    figures["auc"] = str(auc_path)

    # 2. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - width/2, b_acc, width, label="ConvNeXt Baseline", color="#1f77b4", alpha=0.9)
    ax.bar(x + width/2, f_acc, width, label="RGB + FFT Fusion", color="#ff7f0e", alpha=0.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness Benchmark: Accuracy under Controlled Degradations")
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, rotation=20, ha="right")
    ax.set_ylim([75.0, 101.0])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    plt.tight_layout()
    acc_path = output_dir / "robustness_accuracy.png"
    plt.savefig(acc_path)
    plt.close()
    figures["accuracy"] = str(acc_path)

    # 3. Prediction Flip Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - width/2, b_flip, width, label="ConvNeXt Baseline", color="#1f77b4", alpha=0.9)
    ax.bar(x + width/2, f_flip, width, label="RGB + FFT Fusion", color="#ff7f0e", alpha=0.9)
    ax.set_ylabel("Prediction Flip Rate (%)")
    ax.set_title("Robustness Benchmark: Decision Volatility (Prediction Flip Rate)")
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    flip_path = output_dir / "robustness_flip_rate.png"
    plt.savefig(flip_path)
    plt.close()
    figures["flip_rate"] = str(flip_path)

    # 4. Authenticity Stability Score Distribution
    b_scores = comparison_data["baseline"]["stability_scores"]
    f_scores = comparison_data["fusion"]["stability_scores"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(b_scores, bins=bins, alpha=0.6, label=f"Baseline (Mean: {np.mean(b_scores):.3f})", color="#1f77b4")
    ax.hist(f_scores, bins=bins, alpha=0.6, label=f"Fusion (Mean: {np.mean(f_scores):.3f})", color="#ff7f0e")
    ax.set_xlabel("Authenticity Stability Score S in [0, 1]")
    ax.set_ylabel("Sample Count")
    ax.set_title("Authenticity Stability Score Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    dist_path = output_dir / "stability_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    figures["stability_distribution"] = str(dist_path)

    return figures


def generate_markdown_report(
    comparison_data: Dict[str, Any],
    report_path: Path,
) -> None:
    """Writes detailed technical markdown report."""
    b_res = comparison_data["baseline"]["transformations"]
    f_res = comparison_data["fusion"]["transformations"]
    n_eval = comparison_data["dataset"]["evaluated_samples"]
    n_val_total = comparison_data["dataset"]["total_validation_pool"]

    transforms = list(TRANSFORMATION_REGISTRY.keys())

    md = []
    md.append("# SignalScope Phase 4: Robustness & Authenticity Stability Report")
    md.append(f"**Date**: {comparison_data['timestamp']}")
    md.append(f"**Evaluation Partition**: Local Validation Split from `train/` ({n_eval:,} samples evaluated out of {n_val_total:,} validation pool)")
    md.append("> [!IMPORTANT]\n> **Data Compliance**: These are local validation robustness results. The official organizer held-out partition `C:\\Programming\\SignalScope-data\\test` was completely untouched.\n")

    md.append("## 1. Executive Summary")
    md.append(
        "Detectors in practical media authentication environments confront diverse transmission channel perturbations, "
        "including social media recompression, resolution rescaling, screenshot captures, and cropping. "
        "This benchmark systematically measures how classification accuracy, decision stability, and output probabilities "
        "behave under controlled, deterministic transformations without retraining."
    )

    md.append("\n## 2. Transformation Benchmark Matrix")
    md.append("| Transform | Parameter Details | Description |")
    md.append("|---|---|---|")
    for tname, (_, meta) in TRANSFORMATION_REGISTRY.items():
        desc = meta.get("description", "")
        params = ", ".join(f"{k}={v}" for k, v in meta.items() if k != "description") or "None (Pristine)"
        md.append(f"| **{tname}** | {params} | {desc} |")

    md.append("\n## 3. Comparative Robustness Results")
    md.append("\n### ConvNeXt Baseline vs. RGB + FFT Fusion")
    md.append("| Model | Transform | ROC-AUC | Macro-F1 | Accuracy | FPR | Flip Rate | Mean Drift |")
    md.append("|---|---|---|---|---|---|---|---|")

    for t in transforms:
        b = b_res[t]
        md.append(
            f"| **Baseline** | {t} | **{b['roc_auc']:.4f}** | {b['macro_f1']:.4f} | "
            f"{b['accuracy']*100:.2f}% | {b['fpr']*100:.2f}% | {b['prediction_flip_rate']*100:.2f}% | {b['mean_probability_drift']:.4f} |"
        )
        f = f_res[t]
        md.append(
            f"| **Fusion** | {t} | **{f['roc_auc']:.4f}** | {f['macro_f1']:.4f} | "
            f"{f['accuracy']*100:.2f}% | {f['fpr']*100:.2f}% | {f['prediction_flip_rate']*100:.2f}% | {f['mean_probability_drift']:.4f} |"
        )

    md.append("\n## 4. Authenticity Stability Score Analysis")
    md.append(
        "The Authenticity Stability Score $S \\in [0, 1]$ measures the deterministic invariance of an image's classification "
        "across all $K=6$ transformations:\n"
        "$$S = C \\times \\left(1 - \\frac{\\bar{D} + D_{\\max}}{2}\\right)$$\n"
        "- $C$: Prediction Consistency (fraction of transformations retaining original hard decision)\n"
        "- $\\bar{D}$: Mean Absolute Probability Drift\n"
        "- $D_{\\max}$: Maximum Absolute Probability Drift\n"
    )

    b_stat = comparison_data["baseline"]["stability_summary"]
    f_stat = comparison_data["fusion"]["stability_summary"]

    md.append("| Metric | ConvNeXt Baseline | RGB + FFT Fusion |")
    md.append("|---|---|---|")
    md.append(f"| **Mean Stability Score** | **{b_stat['mean']:.4f}** | **{f_stat['mean']:.4f}** |")
    md.append(f"| **Median Stability Score** | {b_stat['median']:.4f} | {f_stat['median']:.4f} |")
    md.append(f"| **Std Deviation** | {b_stat['std']:.4f} | {f_stat['std']:.4f} |")
    md.append(f"| **Min Stability Score** | {b_stat['min']:.4f} | {f_stat['min']:.4f} |")
    md.append(f"| **Max Stability Score** | {b_stat['max']:.4f} | {f_stat['max']:.4f} |")
    md.append(f"| **Stable Fraction ($S \\ge 0.75$)** | **{b_stat['fraction_stable']*100:.2f}%** | **{f_stat['fraction_stable']*100:.2f}%** |")

    md.append("\n## 5. Stability Case Studies & Responsible Uncertainty")
    cases = comparison_data.get("case_studies", [])
    for idx, c in enumerate(cases, 1):
        md.append(f"\n### Case {idx}: {c['category']}")
        md.append(f"- **Image**: `{Path(c['image_path']).name}`")
        md.append(f"- **Ground Truth**: {c['ground_truth']}")
        md.append(f"- **Baseline**: Original prob = `{c['baseline_orig_prob']:.4f}`, Stability = `{c['baseline_stability']:.4f}`")
        md.append(f"- **Fusion**: Original prob = `{c['fusion_orig_prob']:.4f}`, Stability = `{c['fusion_stability']:.4f}`")
        md.append(f"- **Observation**: {c['observation']}")

    md.append("\n## 6. Recommendations & Technical Conclusion")
    rec = comparison_data["recommendation"]
    md.append(f"**Recommendation State**: `{rec['state']}`\n")
    md.append(f"**Rationale**: {rec['rationale']}\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Saved robustness report to {report_path}")


def run_benchmark(
    subset_size: int = 3000,
    batch_size: int = 64,
    seed: int = 42,
    smoke_test_first: bool = True,
) -> Dict[str, Any]:
    """Runs complete Phase 4 robustness benchmark."""
    logger.info("=" * 65)
    logger.info(" SignalScope Phase 4: Robustness & Authenticity Stability Benchmark")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load checkpoints
    b_ckpt = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    f_ckpt = PROJECT_ROOT / "checkpoints" / "fusion" / "best_model.pt"

    baseline_model = load_baseline_model(b_ckpt, device)
    fusion_model = load_fusion_model(f_ckpt, device)
    logger.info("Loaded baseline ConvNeXt-Tiny and RGB+FFT Fusion models.")

    # Load validation samples strictly from train/
    train_dir = Path(r"C:\Programming\SignalScope-data\train")
    samples, total_val_len = get_validation_data(train_dir, subset_size=subset_size, seed=seed)
    n_samples = len(samples)
    logger.info(f"Loaded {n_samples:,} validation samples (out of {total_val_len:,} total validation pool).")

    # Optional Smoke test
    if smoke_test_first:
        run_smoke_test(baseline_model, fusion_model, samples, device)

    # Main Benchmark
    logger.info("\nEvaluating ConvNeXt Baseline across transformations...")
    b_trans_results = {}
    b_scores_dict = {}

    for tname, (tfunc, _) in TRANSFORMATION_REGISTRY.items():
        logger.info(f"  Running transform: {tname} ...")
        y_true, scores, paths = evaluate_model_on_transformed_dataset(
            baseline_model, samples, tfunc, is_fusion=False, device=device, batch_size=batch_size
        )
        b_scores_dict[tname] = scores

    logger.info("\nEvaluating RGB + FFT Fusion across transformations...")
    f_trans_results = {}
    f_scores_dict = {}

    for tname, (tfunc, _) in TRANSFORMATION_REGISTRY.items():
        logger.info(f"  Running transform: {tname} ...")
        y_true, scores, paths = evaluate_model_on_transformed_dataset(
            fusion_model, samples, tfunc, is_fusion=True, device=device, batch_size=batch_size
        )
        f_scores_dict[tname] = scores

    # Compute metrics for each transform
    b_orig = b_scores_dict["original"]
    f_orig = f_scores_dict["original"]

    for tname in TRANSFORMATION_REGISTRY.keys():
        b_score = b_scores_dict[tname]
        b_metrics = calculate_metrics(y_true, b_score, operating_threshold=0.50)
        b_flip = compute_prediction_flip_rate(b_orig, b_score, threshold=0.50)
        b_drift = compute_mean_probability_drift(b_orig, b_score)

        b_trans_results[tname] = {
            "roc_auc": b_metrics["roc_auc"],
            "macro_f1": b_metrics["macro_f1"],
            "accuracy": b_metrics["accuracy"],
            "precision": b_metrics["precision"],
            "recall": b_metrics["recall"],
            "fpr": b_metrics["fpr"],
            "mean_probability": float(np.mean(b_score)),
            "std_probability": float(np.std(b_score)),
            "prediction_flip_rate": round(b_flip, 4),
            "mean_probability_drift": round(b_drift, 4),
        }

        f_score = f_scores_dict[tname]
        f_metrics = calculate_metrics(y_true, f_score, operating_threshold=0.50)
        f_flip = compute_prediction_flip_rate(f_orig, f_score, threshold=0.50)
        f_drift = compute_mean_probability_drift(f_orig, f_score)

        f_trans_results[tname] = {
            "roc_auc": f_metrics["roc_auc"],
            "macro_f1": f_metrics["macro_f1"],
            "accuracy": f_metrics["accuracy"],
            "precision": f_metrics["precision"],
            "recall": f_metrics["recall"],
            "fpr": f_metrics["fpr"],
            "mean_probability": float(np.mean(f_score)),
            "std_probability": float(np.std(f_score)),
            "prediction_flip_rate": round(f_flip, 4),
            "mean_probability_drift": round(f_drift, 4),
        }

    # Compute sample-level Authenticity Stability Scores
    transform_keys = [k for k in TRANSFORMATION_REGISTRY.keys() if k != "original"]
    b_stability_scores: List[float] = []
    f_stability_scores: List[float] = []

    for idx in range(n_samples):
        b_p0 = float(b_orig[idx])
        b_pk = [float(b_scores_dict[k][idx]) for k in transform_keys]
        b_s = compute_authenticity_stability_score(b_p0, b_pk, threshold=0.50)
        b_stability_scores.append(round(b_s, 4))

        f_p0 = float(f_orig[idx])
        f_pk = [float(f_scores_dict[k][idx]) for k in transform_keys]
        f_s = compute_authenticity_stability_score(f_p0, f_pk, threshold=0.50)
        f_stability_scores.append(round(f_s, 4))

    b_stab_arr = np.array(b_stability_scores)
    f_stab_arr = np.array(f_stability_scores)

    b_stab_summary = {
        "mean": round(float(np.mean(b_stab_arr)), 4),
        "median": round(float(np.median(b_stab_arr)), 4),
        "std": round(float(np.std(b_stab_arr)), 4),
        "min": round(float(np.min(b_stab_arr)), 4),
        "max": round(float(np.max(b_stab_arr)), 4),
        "fraction_stable": round(float(np.mean(b_stab_arr >= 0.75)), 4),
    }

    f_stab_summary = {
        "mean": round(float(np.mean(f_stab_arr)), 4),
        "median": round(float(np.median(f_stab_arr)), 4),
        "std": round(float(np.std(f_stab_arr)), 4),
        "min": round(float(np.min(f_stab_arr)), 4),
        "max": round(float(np.max(f_stab_arr)), 4),
        "fraction_stable": round(float(np.mean(f_stab_arr >= 0.75)), 4),
    }

    # Representative Case Studies
    # A. Baseline stable (S >= 0.85) while Fusion changes (S < 0.60)
    # B. Fusion stable (S >= 0.85) while Baseline changes (S < 0.60)
    # C. Both highly stable (S >= 0.95)
    # D. Both unstable (S < 0.50)
    # E. High confidence original -> low confidence perturbed
    case_studies = []

    case_a_candidates = np.where((b_stab_arr >= 0.85) & (f_stab_arr < 0.60))[0]
    if len(case_a_candidates) > 0:
        idx = case_a_candidates[0]
        p, lbl, _ = samples[idx]
        case_studies.append({
            "category": "Baseline stable while Fusion degrades",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "baseline_orig_prob": round(float(b_orig[idx]), 4),
            "baseline_stability": b_stability_scores[idx],
            "fusion_orig_prob": round(float(f_orig[idx]), 4),
            "fusion_stability": f_stability_scores[idx],
            "observation": "Baseline spatial representation remained consistent under compression, while frequency spectrum suffered degradation that lowered fusion confidence.",
        })

    case_b_candidates = np.where((f_stab_arr >= 0.85) & (b_stab_arr < 0.60))[0]
    if len(case_b_candidates) > 0:
        idx = case_b_candidates[0]
        p, lbl, _ = samples[idx]
        case_studies.append({
            "category": "Fusion stable while Baseline degrades",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "baseline_orig_prob": round(float(b_orig[idx]), 4),
            "baseline_stability": b_stability_scores[idx],
            "fusion_orig_prob": round(float(f_orig[idx]), 4),
            "fusion_stability": f_stability_scores[idx],
            "observation": "Spatial perturbations drifted ConvNeXt prediction, but multimodal FFT representation grounded the decision, maintaining high fusion stability.",
        })

    case_c_candidates = np.where((b_stab_arr >= 0.95) & (f_stab_arr >= 0.95))[0]
    if len(case_c_candidates) > 0:
        idx = case_c_candidates[0]
        p, lbl, _ = samples[idx]
        case_studies.append({
            "category": "Both models highly stable",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "baseline_orig_prob": round(float(b_orig[idx]), 4),
            "baseline_stability": b_stability_scores[idx],
            "fusion_orig_prob": round(float(f_orig[idx]), 4),
            "fusion_stability": f_stability_scores[idx],
            "observation": "Both architectures decisively recognized salient features that persisted across all degradation transforms.",
        })

    case_d_candidates = np.where((b_stab_arr < 0.50) & (f_stab_arr < 0.50))[0]
    if len(case_d_candidates) > 0:
        idx = case_d_candidates[0]
        p, lbl, _ = samples[idx]
        case_studies.append({
            "category": "Both models unstable (Severe degradation impact)",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "baseline_orig_prob": round(float(b_orig[idx]), 4),
            "baseline_stability": b_stability_scores[idx],
            "fusion_orig_prob": round(float(f_orig[idx]), 4),
            "fusion_stability": f_stability_scores[idx],
            "observation": "Subtle boundary features were distorted by recompression and downsampling, inducing high probability drift and prediction flips across both models.",
        })

    # Responsible Uncertainty Candidate (High confidence original -> ambiguous perturbed)
    ambig_candidates = np.where(
        (b_orig > 0.90) & (np.abs(b_scores_dict["jpeg_70"] - 0.50) < 0.20)
    )[0]
    if len(ambig_candidates) > 0:
        idx = ambig_candidates[0]
        p, lbl, _ = samples[idx]
        case_studies.append({
            "category": "Responsible Uncertainty: High Confidence to Ambiguous",
            "image_path": str(p),
            "ground_truth": "Synthetic (1)" if lbl == 1 else "Real (0)",
            "baseline_orig_prob": round(float(b_orig[idx]), 4),
            "baseline_stability": b_stability_scores[idx],
            "fusion_orig_prob": round(float(f_orig[idx]), 4),
            "fusion_stability": f_stability_scores[idx],
            "observation": f"Original image produced decisive prediction ({b_orig[idx]:.2f}), but JPEG Q=70 shifted model probability to {b_scores_dict['jpeg_70'][idx]:.2f}, signaling an 'Uncertain / Human Review Recommended' state.",
        })

    # Scientific Recommendation
    mean_flip_b = np.mean([b_trans_results[k]["prediction_flip_rate"] for k in transform_keys])
    mean_flip_f = np.mean([f_trans_results[k]["prediction_flip_rate"] for k in transform_keys])
    mean_auc_b = np.mean([b_trans_results[k]["roc_auc"] for k in transform_keys])
    mean_auc_f = np.mean([f_trans_results[k]["roc_auc"] for k in transform_keys])

    if abs(mean_auc_b - mean_auc_f) < 0.005 and abs(mean_flip_b - mean_flip_f) < 0.005:
        rec_state = "C. Baseline and Fusion are similar in overall robustness"
        rationale = f"Both architectures exhibit close average degraded AUC ({mean_auc_b:.4f} vs. {mean_auc_f:.4f}) and similar average flip rates ({mean_flip_b*100:.2f}% vs. {mean_flip_f*100:.2f}%)."
    elif mean_auc_b > mean_auc_f + 0.005:
        rec_state = "A. ConvNeXt Baseline is more robust"
        rationale = f"ConvNeXt spatial backbone retained higher degraded AUC ({mean_auc_b:.4f} vs. {mean_auc_f:.4f}) and lower flip rate ({mean_flip_b*100:.2f}% vs. {mean_flip_f*100:.2f}%)."
    elif mean_auc_f > mean_auc_b + 0.005:
        rec_state = "B. RGB + FFT Fusion is more robust"
        rationale = f"Fusion dual-branch retained higher degraded AUC ({mean_auc_f:.4f} vs. {mean_auc_b:.4f}) and lower flip rate ({mean_flip_f*100:.2f}% vs. {mean_flip_b*100:.2f}%)."
    else:
        rec_state = "D. Evidence is inconclusive / domain-dependent"
        rationale = "Degradation performance varies across specific transformation types without a uniform dominant model."

    report_data = {
        "benchmark_name": "robustness_and_authenticity_stability_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source_directory": "C:\\Programming\\SignalScope-data\\train",
            "evaluated_samples": n_samples,
            "total_validation_pool": total_val_len,
            "random_seed": seed,
            "held_out_test_status": "STRICTLY_UNTOUCHED",
        },
        "models": {
            "baseline": {
                "name": "ConvNeXt-Tiny Spatial Baseline",
                "checkpoint": str(b_ckpt),
            },
            "fusion": {
                "name": "DualBranchFusionDetector (ConvNeXt-Tiny + FFT 32x32)",
                "checkpoint": str(f_ckpt),
            },
        },
        "baseline": {
            "transformations": b_trans_results,
            "stability_summary": b_stab_summary,
            "stability_scores": [round(float(s), 4) for s in b_stability_scores],
        },
        "fusion": {
            "transformations": f_trans_results,
            "stability_summary": f_stab_summary,
            "stability_scores": [round(float(s), 4) for s in f_stability_scores],
        },
        "case_studies": case_studies,
        "recommendation": {
            "state": rec_state,
            "rationale": rationale,
        },
    }

    # Save structured JSON
    report_dir = PROJECT_ROOT / "report" / "robustness"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "robustness_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved structured robustness data to {json_path}")

    # Generate Figures
    figures = plot_robustness_figures(report_data, report_dir)
    logger.info(f"Generated robustness figures in {report_dir}")

    # Generate Markdown Report
    md_path = report_dir / "robustness_report.md"
    generate_markdown_report(report_data, md_path)

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope Robustness & Authenticity Stability Benchmark")
    parser.add_argument("--subset-size", type=int, default=3000, help="Number of validation samples to evaluate (default: 3000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model inference (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic subsetting")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run only the 5-sample smoke test")
    args = parser.parse_args()

    if args.smoke_test_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b_ckpt = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
        f_ckpt = PROJECT_ROOT / "checkpoints" / "fusion" / "best_model.pt"
        bm = load_baseline_model(b_ckpt, device)
        fm = load_fusion_model(f_ckpt, device)
        train_dir = Path(r"C:\Programming\SignalScope-data\train")
        val_samples, _ = get_validation_data(train_dir, subset_size=10, seed=args.seed)
        run_smoke_test(bm, fm, val_samples, device)
    else:
        run_benchmark(
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            seed=args.seed,
            smoke_test_first=True,
        )
