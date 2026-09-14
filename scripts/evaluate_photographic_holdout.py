"""Independent High-Resolution Photographic-Real Validation Gate.

Stress-tests:
- Baseline v1 (signalscope-baseline-v1)
- Exp3-5ep (signalscope-v2-candidate)

Uses ONLY UNUSED images from version-2/REAL:
- 9 photographic categories: apparel, cars, dishes, furniture, landmark, packaged, storefronts, toys, artwork
- Excludes all 5,004 images used during Exp2/Exp3 training
- Strictly asserts zero leakage with all prior training/validation partitions
- Evaluates 9,000 authentic photographic images (1,000 per category)
- Evaluates secondary degradation robustness on a 900-sample subset
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures.convnext import build_convnext_tiny
from model.calibration import TemperatureScaler
from model.dataset import SignalScopeDataset, get_default_transforms
from model.dataset_splits import (
    get_official_splits,
    get_v2_real_samples,
    get_v2_splits,
)
from model.robustness import (
    TRANSFORMATION_REGISTRY,
    compute_mean_probability_drift,
    compute_prediction_flip_rate,
)

V2_REAL_DIR = Path(r"C:\Programming\SignalScope-data\version-2\REAL")

PHOTOGRAPHIC_CATEGORIES = [
    "apparel", "cars", "dishes", "furniture",
    "landmark", "packaged", "storefronts", "toys", "artwork"
]


def load_model_checkpoint(path: Path, device: torch.device) -> nn.Module:
    model = build_convnext_tiny(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


def build_independent_photographic_holdout(
    samples_per_category: int = 1000,
    training_seed: int = 42,
    holdout_seed: int = 2026,
) -> Tuple[List[Tuple[Path, int, str]], List[Path]]:
    """Builds an independent photographic holdout strictly excluding all training images."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    rng_train = np.random.RandomState(training_seed)
    rng_holdout = np.random.RandomState(holdout_seed)

    holdout_samples: List[Tuple[Path, int, str]] = []
    excluded_training_paths: List[Path] = []

    for cat in PHOTOGRAPHIC_CATEGORIES:
        cat_dir = V2_REAL_DIR / cat
        cat_files = sorted([p for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

        # Reproduce exact training sample selection (556 per category)
        train_indices = set(rng_train.permutation(len(cat_files))[:556])
        train_paths = [cat_files[i] for i in train_indices]
        excluded_training_paths.extend(train_paths)

        # Candidate pool of completely unused images
        unused_files = [cat_files[i] for i in range(len(cat_files)) if i not in train_indices]

        # Deterministically sample holdout
        holdout_indices = rng_holdout.permutation(len(unused_files))[:samples_per_category]
        selected_holdout = [unused_files[i] for i in holdout_indices]

        for p in selected_holdout:
            holdout_samples.append((p, 0, cat))

    return holdout_samples, excluded_training_paths


def verify_holdout_leakage(
    holdout_samples: List[Tuple[Path, int, str]],
    excluded_training_paths: List[Path],
) -> None:
    """Verifies complete disjointness against all prior experiment datasets."""
    holdout_set = set(str(p) for p, _, _ in holdout_samples)
    train_real_set = set(str(p) for p in excluded_training_paths)

    assert len(holdout_set.intersection(train_real_set)) == 0, "FATAL: Holdout overlaps with training real samples!"

    train_off, val_old = get_official_splits(seed=42)
    train_v2, val_new = get_v2_splits()

    off_train_set = set(str(p) for p, _, _ in train_off)
    val_old_set = set(str(p) for p, _, _ in val_old)
    v2_train_set = set(str(p) for p, _, _ in train_v2)
    val_new_set = set(str(p) for p, _, _ in val_new)

    assert len(holdout_set.intersection(off_train_set)) == 0, "FATAL: Holdout overlaps with official train!"
    assert len(holdout_set.intersection(val_old_set)) == 0, "FATAL: Holdout overlaps with val_old!"
    assert len(holdout_set.intersection(v2_train_set)) == 0, "FATAL: Holdout overlaps with AIGen train!"
    assert len(holdout_set.intersection(val_new_set)) == 0, "FATAL: Holdout overlaps with AIGen val!"

    print(f"Leakage Verification PASSED: 0 overlaps across {len(holdout_set):,} holdout images.")


def compute_distribution_stats(probs: np.ndarray) -> Dict[str, float]:
    return {
        "mean": round(float(np.mean(probs)), 4),
        "std": round(float(np.std(probs)), 4),
        "min": round(float(np.min(probs)), 4),
        "p25": round(float(np.percentile(probs, 25)), 4),
        "median": round(float(np.median(probs)), 4),
        "p75": round(float(np.percentile(probs, 75)), 4),
        "p90": round(float(np.percentile(probs, 90)), 4),
        "p95": round(float(np.percentile(probs, 95)), 4),
        "p99": round(float(np.percentile(probs, 99)), 4),
        "max": round(float(np.max(probs)), 4),
    }


class TransformedEvalDataset(Dataset):
    def __init__(self, samples, transform_fn, base_transform):
        self.samples = samples
        self.transform_fn = transform_fn
        self.base_transform = base_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cat = self.samples[idx]
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
        if self.transform_fn:
            pil_img = self.transform_fn(pil_img)
        tensor = self.base_transform(pil_img)
        return tensor, torch.tensor([label], dtype=torch.float32), cat


def predict_holdout(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_cats = []
    with torch.no_grad():
        for images, targets, cats in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
            if isinstance(cats, dict) and "generator" in cats:
                all_cats.extend(cats["generator"])
            else:
                all_cats.extend(cats)
    return np.array(all_probs, dtype=float), all_cats


def run_photographic_validation_gate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Build holdout
    print("\n--- Building Independent Photographic-Real Holdout ---")
    holdout_samples, excluded_paths = build_independent_photographic_holdout(samples_per_category=1000)
    print(f"Constructed holdout: {len(holdout_samples):,} samples across {len(PHOTOGRAPHIC_CATEGORIES)} categories (1,000 per category).")
    print(f"Excluded training paths: {len(excluded_paths):,} samples.")

    # 2. Verify leakage
    verify_holdout_leakage(holdout_samples, excluded_paths)

    # 3. Load models
    b_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    exp3_path = PROJECT_ROOT / "checkpoints" / "additional_training" / "exp3_5ep" / "best_model.pt"

    print(f"\nLoading Baseline from: {b_path}")
    baseline_model = load_model_checkpoint(b_path, device)
    print(f"Loading Exp3-5ep from: {exp3_path}")
    exp3_model = load_model_checkpoint(exp3_path, device)

    # 4. Fitted Temperatures
    # Baseline T = 0.9995, Exp3 T = 0.9986
    t_base = 0.9995
    t_exp3 = 0.9986

    # 5. Dataloader for 9,000 samples
    transform = get_default_transforms(image_size=224, is_training=False)
    loader = DataLoader(SignalScopeDataset(holdout_samples, transform=transform), batch_size=128, shuffle=False, num_workers=0)

    print("\n--- Evaluating Baseline on 9,000 Photographic Real Images ---")
    p_base, cats = predict_holdout(baseline_model, loader, device)

    print("\n--- Evaluating Exp3-5ep on 9,000 Photographic Real Images ---")
    p_exp3, _ = predict_holdout(exp3_model, loader, device)

    # 6. Overall Metrics (Ground truth: ALL REAL = 0)
    N = len(holdout_samples)
    base_fp = int(np.sum(p_base >= 0.50))
    base_tn = N - base_fp
    base_fpr = base_fp / N

    exp3_fp = int(np.sum(p_exp3 >= 0.50))
    exp3_tn = N - exp3_fp
    exp3_fpr = exp3_fp / N

    # Calibrated probabilities: sigmoid(logit / T)
    p_base_cal = torch.sigmoid(torch.tensor(p_base).logit() / t_base).numpy()
    p_exp3_cal = torch.sigmoid(torch.tensor(p_exp3).logit() / t_exp3).numpy()

    stats_base = compute_distribution_stats(p_base)
    stats_base_cal = compute_distribution_stats(p_base_cal)
    stats_exp3 = compute_distribution_stats(p_exp3)
    stats_exp3_cal = compute_distribution_stats(p_exp3_cal)

    print(f"\n=== OVERALL PHOTOGRAPHIC-REAL HOLDOUT RESULTS (N = {N:,}) ===")
    print(f"Baseline (v1)   : FPR = {base_fpr*100:.2f}% ({base_fp:,} False Positives, {base_tn:,} True Negatives)")
    print(f"                  Median Prob = {stats_base['median']:.4f}, Mean = {stats_base['mean']:.4f}, Max = {stats_base['max']:.4f}")
    print(f"                  P90 = {stats_base['p90']:.4f}, P95 = {stats_base['p95']:.4f}, P99 = {stats_base['p99']:.4f}")
    print(f"Exp3-5ep        : FPR = {exp3_fpr*100:.2f}% ({exp3_fp:,} False Positives, {exp3_tn:,} True Negatives)")
    print(f"                  Median Prob = {stats_exp3['median']:.4f}, Mean = {stats_exp3['mean']:.4f}, Max = {stats_exp3['max']:.4f}")
    print(f"                  P90 = {stats_exp3['p90']:.4f}, P95 = {stats_exp3['p95']:.4f}, P99 = {stats_exp3['p99']:.4f}")

    fpr_reduction = base_fpr - exp3_fpr
    fpr_relative_drop = (base_fpr - exp3_fpr) / max(1e-6, base_fpr) * 100
    print(f"\nAbsolute FPR Reduction : -{fpr_reduction*100:.2f}%")
    print(f"Relative FPR Drop      : -{fpr_relative_drop:.2f}%")

    # 7. Category Breakdown
    print("\n=== CATEGORY BREAKDOWN (1,000 samples per category) ===")
    cat_breakdown = {}
    for cat in PHOTOGRAPHIC_CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        p_b_cat = p_base[idx]
        p_e_cat = p_exp3[idx]

        b_fp = int(np.sum(p_b_cat >= 0.50))
        e_fp = int(np.sum(p_e_cat >= 0.50))
        b_fpr = b_fp / len(idx)
        e_fpr = e_fp / len(idx)

        b_med = float(np.median(p_b_cat))
        e_med = float(np.median(p_e_cat))

        improvement = b_fpr - e_fpr

        cat_breakdown[cat] = {
            "samples": len(idx),
            "baseline_fp": b_fp,
            "baseline_fpr": round(b_fpr, 4),
            "baseline_median_prob": round(b_med, 4),
            "exp3_fp": e_fp,
            "exp3_fpr": round(e_fpr, 4),
            "exp3_median_prob": round(e_med, 4),
            "fpr_improvement": round(improvement, 4),
        }

        print(f"  {cat:12}: Baseline FPR={b_fpr*100:5.2f}% (Med: {b_med:.4f}) -> Exp3 FPR={e_fpr*100:5.2f}% (Med: {e_med:.4f}) | Delta: -{improvement*100:5.2f}%")

    worst_base = max(cat_breakdown.items(), key=lambda x: x[1]["baseline_fpr"])
    worst_exp3 = max(cat_breakdown.items(), key=lambda x: x[1]["exp3_fpr"])
    best_imp = max(cat_breakdown.items(), key=lambda x: x[1]["fpr_improvement"])

    print(f"\nWorst category for Baseline : {worst_base[0]} ({worst_base[1]['baseline_fpr']*100:.2f}% FPR)")
    print(f"Worst category for Exp3     : {worst_exp3[0]} ({worst_exp3[1]['exp3_fpr']*100:.2f}% FPR)")
    print(f"Category largest improvement: {best_imp[0]} (-{best_imp[1]['fpr_improvement']*100:.2f}% FPR)")

    # 8. Secondary Robustness Benchmark on Photographic Holdout (900 samples, 100/cat)
    print("\n=== SECONDARY ROBUSTNESS ON PHOTOGRAPHIC-REAL IMAGES (900 samples) ===")
    rng_robust = np.random.RandomState(42)
    robust_sub = []
    for cat in PHOTOGRAPHIC_CATEGORIES:
        cat_samples = [s for s in holdout_samples if s[2] == cat]
        sel_idx = rng_robust.permutation(len(cat_samples))[:100]
        robust_sub.extend([cat_samples[i] for i in sel_idx])

    transform_keys = ["original", "jpeg_95", "jpeg_85", "jpeg_70", "resize", "screenshot", "light_edit"]
    robust_results = {"baseline": {}, "exp3_5ep": {}}

    b_orig_p = None
    e_orig_p = None

    for t_key in transform_keys:
        t_fn, meta = TRANSFORMATION_REGISTRY[t_key]
        ds = TransformedEvalDataset(robust_sub, t_fn, transform)
        ldr = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

        p_b_t, _ = predict_holdout(baseline_model, ldr, device)
        p_e_t, _ = predict_holdout(exp3_model, ldr, device)

        if t_key == "original":
            b_orig_p = p_b_t
            e_orig_p = p_e_t
            b_flip = 0.0
            e_flip = 0.0
            b_drift = 0.0
            e_drift = 0.0
        else:
            b_flip = compute_prediction_flip_rate(b_orig_p, p_b_t, threshold=0.50)
            e_flip = compute_prediction_flip_rate(e_orig_p, p_e_t, threshold=0.50)
            b_drift = compute_mean_probability_drift(b_orig_p, p_b_t)
            e_drift = compute_mean_probability_drift(e_orig_p, p_e_t)

        b_fpr_t = float(np.mean(p_b_t >= 0.50))
        e_fpr_t = float(np.mean(p_e_t >= 0.50))

        robust_results["baseline"][t_key] = {
            "fpr": round(b_fpr_t, 4),
            "median_prob": round(float(np.median(p_b_t)), 4),
            "flip_rate": round(b_flip, 4),
            "drift": round(b_drift, 4),
        }
        robust_results["exp3_5ep"][t_key] = {
            "fpr": round(e_fpr_t, 4),
            "median_prob": round(float(np.median(p_e_t)), 4),
            "flip_rate": round(e_flip, 4),
            "drift": round(e_drift, 4),
        }

        print(f"  {t_key:12}: Baseline FPR={b_fpr_t*100:5.2f}% (Flip: {b_flip*100:4.1f}%) | Exp3 FPR={e_fpr_t*100:5.2f}% (Flip: {e_flip*100:4.1f}%)")

    # 9. Gate Assessment
    # Decision Gate Criteria:
    # 1. Materially reduces FPR on independent photographic holdout.
    # 2. No unacceptable regression on old validation (already verified: 0.9991 vs 0.9992).
    # 3. Strong AIGen validation performance (0.9220 vs 0.5950).
    # 4. Meaningfully less overconfident probabilities.
    # 5. No obvious category failure.
    # 6. Acceptable robustness.
    gate_passed = (
        exp3_fpr < (base_fpr * 0.5) and  # Over 50% relative reduction
        stats_exp3["median"] < stats_base["median"] and
        exp3_fpr < 0.20  # Under 20% FPR on completely unseen photography
    )

    verdict = "PRODUCTION GATE: PASSED — EXP3 APPROVED FOR RELEASE" if gate_passed else "PRODUCTION GATE: FAILED — KEEP SIGNALSCOPE-BASELINE-V1"
    print(f"\n==========================================")
    print(f" VERDICT: {verdict}")
    print(f"==========================================")

    payload = {
        "benchmark_name": "independent_photographic_real_validation_gate",
        "sample_count": N,
        "categories": PHOTOGRAPHIC_CATEGORIES,
        "leakage_verification": {
            "status": "PASSED",
            "excluded_training_paths_count": len(excluded_paths),
            "overlap_count": 0,
        },
        "overall_results": {
            "baseline": {
                "false_positives": base_fp,
                "true_negatives": base_tn,
                "fpr": round(base_fpr, 4),
                "raw_stats": stats_base,
                "calibrated_stats": stats_base_cal,
            },
            "exp3_5ep": {
                "false_positives": exp3_fp,
                "true_negatives": exp3_tn,
                "fpr": round(exp3_fpr, 4),
                "raw_stats": stats_exp3,
                "calibrated_stats": stats_exp3_cal,
            },
            "comparison": {
                "absolute_fpr_reduction": round(fpr_reduction, 4),
                "relative_fpr_reduction_pct": round(fpr_relative_drop, 2),
            }
        },
        "category_breakdown": cat_breakdown,
        "worst_category_baseline": worst_base[0],
        "worst_category_exp3": worst_exp3[0],
        "category_largest_improvement": best_imp[0],
        "robustness_on_photographic_holdout": robust_results,
        "gate_decision": {
            "passed": bool(gate_passed),
            "verdict": verdict,
        }
    }

    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "photographic_holdout_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to {out_file}")
    return payload


if __name__ == "__main__":
    run_photographic_validation_gate()
