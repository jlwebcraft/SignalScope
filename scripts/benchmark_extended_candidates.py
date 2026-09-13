"""Comprehensive evaluation script for Second-Stage Extended Training Candidates.

Evaluates:
- Baseline (v1)
- Exp2-5ep (Best Epoch 4)
- Exp3-5ep (Best Epoch 3)

Produces:
1. Dual-domain metrics (Local Val 15k, New-Data Val 1.1k)
2. Generator-stratified evaluation across all 19 generative families
3. Calibration metrics (Temperature scaling fitted on disjoint calibration pool)
4. Robustness benchmark across 6 transformations (JPEG 95, 85, 70, resize, screenshot, light crop)
5. Decision threshold comparison (0.50 default vs optimal Youden threshold)
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures.convnext import build_convnext_tiny
from model.calibration import TemperatureScaler, compute_brier_score, compute_ece
from model.dataset import SignalScopeDataset, get_default_transforms
from model.dataset_splits import get_official_splits, get_v2_splits
from model.evaluate import calculate_metrics, find_optimal_threshold_youden
from model.robustness import (
    TRANSFORMATION_REGISTRY,
    compute_authenticity_stability_score,
    compute_mean_probability_drift,
    compute_prediction_flip_rate,
)


def load_candidate_model(path: Path, device: torch.device) -> nn.Module:
    model = build_convnext_tiny(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


class TransformedDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int, str]], transform_fn, base_transform):
        self.samples = samples
        self.transform_fn = transform_fn
        self.base_transform = base_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, gen = self.samples[idx]
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
        if self.transform_fn:
            pil_img = self.transform_fn(pil_img)
        tensor = self.base_transform(pil_img)
        return tensor, torch.tensor([label], dtype=torch.float32), gen


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_targets = []
    all_probs = []
    all_gens = []
    with torch.no_grad():
        for images, targets, gens in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
            if isinstance(gens, dict) and "generator" in gens:
                all_gens.extend(gens["generator"])
            else:
                all_gens.extend(gens)
    return np.array(all_targets, dtype=int), np.array(all_probs, dtype=float), all_gens


def run_comprehensive_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load splits
    _, val_old = get_official_splits(seed=42)
    train_v2, val_new = get_v2_splits()
    train_off, _ = get_official_splits(seed=42)

    # 2. Checkpoints
    ckpts = {
        "baseline": PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt",
        "exp2_5ep": PROJECT_ROOT / "checkpoints" / "additional_training" / "exp2_5ep" / "best_model.pt",
        "exp3_5ep": PROJECT_ROOT / "checkpoints" / "additional_training" / "exp3_5ep" / "best_model.pt",
    }

    models = {}
    for name, path in ckpts.items():
        print(f"Loading {name} from {path}...")
        models[name] = load_candidate_model(path, device)

    transform = get_default_transforms(image_size=224, is_training=False)
    val_old_loader = DataLoader(SignalScopeDataset(val_old, transform=transform), batch_size=64, shuffle=False, num_workers=0)
    val_new_loader = DataLoader(SignalScopeDataset(val_new, transform=transform), batch_size=64, shuffle=False, num_workers=0)

    # Calibration fitting pool (2,000 samples from official train, disjoint from validation)
    rng = np.random.RandomState(42)
    cal_idx = rng.permutation(len(train_off))[:2000]
    cal_samples = [train_off[i] for i in cal_idx]
    cal_loader = DataLoader(SignalScopeDataset(cal_samples, transform=transform), batch_size=64, shuffle=False, num_workers=0)

    results = {}

    for m_name, model in models.items():
        print(f"\n==========================================")
        print(f" Evaluating {m_name.upper()}...")
        print(f"==========================================")

        # A. LOCAL VALIDATION
        print("Running LOCAL VALIDATION (15,000 samples)...")
        y_old, p_old, _ = predict_dataset(model, val_old_loader, device)
        m_old_050 = calculate_metrics(y_old, p_old, operating_threshold=0.50)
        opt_th_old, _ = find_optimal_threshold_youden(y_old, p_old)
        m_old_opt = calculate_metrics(y_old, p_old, operating_threshold=opt_th_old)
        brier_old = compute_brier_score(y_old, p_old)
        ece_old, _, _ = compute_ece(y_old, p_old)

        # B. NEW-DATA VALIDATION
        print("Running NEW-DATA VALIDATION (1,118 samples)...")
        y_new, p_new, gens_new = predict_dataset(model, val_new_loader, device)
        m_new_050 = calculate_metrics(y_new, p_new, operating_threshold=0.50)
        opt_th_new, _ = find_optimal_threshold_youden(y_new, p_new)
        m_new_opt = calculate_metrics(y_new, p_new, operating_threshold=opt_th_new)
        brier_new = compute_brier_score(y_new, p_new)
        ece_new, _, _ = compute_ece(y_new, p_new)

        # C. GENERATOR-STRATIFIED EVALUATION
        print("Computing Generator-Stratified AUCs across 19 families...")
        distinct_gens = sorted(list(set(g for g in gens_new if g not in {"real", "unknown"})))
        real_idx = [i for i, lbl in enumerate(y_new) if lbl == 0]
        gen_metrics = {}
        for g_name in distinct_gens:
            g_idx = [i for i, g in enumerate(gens_new) if g == g_name]
            sub_idx = real_idx + g_idx
            sub_y = y_new[sub_idx]
            sub_p = p_new[sub_idx]
            g_auc = float(roc_auc_score(sub_y, sub_p))
            g_recall = float(np.mean(p_new[g_idx] >= 0.50))
            gen_metrics[g_name] = {
                "samples": len(g_idx),
                "roc_auc": round(g_auc, 4),
                "recall_at_050": round(g_recall, 4),
                "mean_confidence": round(float(np.mean(p_new[g_idx])), 4),
            }

        auc_list = [v["roc_auc"] for v in gen_metrics.values()]
        strongest = max(gen_metrics.items(), key=lambda x: x[1]["roc_auc"])
        weakest = min(gen_metrics.items(), key=lambda x: x[1]["roc_auc"])
        spread = round(strongest[1]["roc_auc"] - weakest[1]["roc_auc"], 4)

        # D. CALIBRATION FITTING (Temperature Scaling)
        print("Fitting Temperature Scaler on disjoint 2k training samples...")
        cal_logits, cal_tgts = [], []
        with torch.no_grad():
            for imgs, tgts, _ in cal_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                cal_logits.append(logits.squeeze(-1))
                cal_tgts.append(tgts.squeeze(-1).to(device))
        c_logits = torch.cat(cal_logits)
        c_tgts = torch.cat(cal_tgts)

        scaler = TemperatureScaler()
        T = scaler.fit(c_logits, c_tgts)

        # Scaler evaluation on Old and New domains
        cal_p_old = scaler.calibrate(torch.tensor(p_old).logit().to(device)).detach().cpu().numpy()
        cal_p_new = scaler.calibrate(torch.tensor(p_new).logit().to(device)).detach().cpu().numpy()

        cal_ece_old, _, _ = compute_ece(y_old, cal_p_old)
        cal_brier_old = compute_brier_score(y_old, cal_p_old)
        cal_ece_new, _, _ = compute_ece(y_new, cal_p_new)
        cal_brier_new = compute_brier_score(y_new, cal_p_new)

        results[m_name] = {
            "model_name": m_name,
            "old_validation": {
                "roc_auc": round(float(m_old_050["roc_auc"]), 4),
                "macro_f1_050": round(float(m_old_050["macro_f1"]), 4),
                "accuracy_050": round(float(m_old_050["accuracy"]), 4),
                "fpr_050": round(float(m_old_050["fpr"]), 4),
                "confusion_matrix_050": m_old_050["confusion_matrix"],
                "optimal_threshold": {
                    "threshold": round(float(opt_th_old), 4),
                    "accuracy": round(float(m_old_opt["accuracy"]), 4),
                    "macro_f1": round(float(m_old_opt["macro_f1"]), 4),
                    "fpr": round(float(m_old_opt["fpr"]), 4),
                    "confusion_matrix": m_old_opt["confusion_matrix"],
                },
                "raw_ece": round(float(ece_old), 4),
                "calibrated_ece": round(float(cal_ece_old), 4),
                "raw_brier": round(float(brier_old), 4),
                "calibrated_brier": round(float(cal_brier_old), 4),
            },
            "new_validation": {
                "roc_auc": round(float(m_new_050["roc_auc"]), 4),
                "macro_f1_050": round(float(m_new_050["macro_f1"]), 4),
                "accuracy_050": round(float(m_new_050["accuracy"]), 4),
                "fpr_050": round(float(m_new_050["fpr"]), 4),
                "confusion_matrix_050": m_new_050["confusion_matrix"],
                "optimal_threshold": {
                    "threshold": round(float(opt_th_new), 4),
                    "accuracy": round(float(m_new_opt["accuracy"]), 4),
                    "macro_f1": round(float(m_new_opt["macro_f1"]), 4),
                    "fpr": round(float(m_new_opt["fpr"]), 4),
                    "confusion_matrix": m_new_opt["confusion_matrix"],
                },
                "raw_ece": round(float(ece_new), 4),
                "calibrated_ece": round(float(cal_ece_new), 4),
                "raw_brier": round(float(brier_new), 4),
                "calibrated_brier": round(float(cal_brier_new), 4),
            },
            "generator_stratified_evaluation": {
                "num_generators": len(gen_metrics),
                "macro_avg_auc": round(float(np.mean(auc_list)), 4),
                "strongest_generator": {"name": strongest[0], "auc": strongest[1]["roc_auc"]},
                "weakest_generator": {"name": weakest[0], "auc": weakest[1]["roc_auc"]},
                "auc_spread": spread,
                "generators": gen_metrics,
            },
            "calibration": {
                "fitted_temperature": round(float(T), 4),
            }
        }

    # E. ROBUSTNESS BENCHMARK (1,000 samples from val_old)
    print("\n==========================================")
    print(" Running Robustness Benchmark (1,000 samples)...")
    print("==========================================")
    rng = np.random.RandomState(42)
    real_old = [s for s in val_old if s[1] == 0]
    fake_old = [s for s in val_old if s[1] == 1]
    eval_sub = [real_old[i] for i in rng.permutation(len(real_old))[:500]] + \
               [fake_old[i] for i in rng.permutation(len(fake_old))[:500]]

    transform_keys = ["original", "jpeg_95", "jpeg_85", "jpeg_70", "resize", "screenshot", "light_edit"]
    robustness_results = {}

    for m_name, model in models.items():
        print(f"Robustness for {m_name}...")
        orig_p = None
        trans_res = {}
        for t_key in transform_keys:
            t_fn, meta = TRANSFORMATION_REGISTRY[t_key]
            ds = TransformedDataset(eval_sub, t_fn, transform)
            ldr = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            y_sub, p_sub, _ = predict_dataset(model, ldr, device)
            met = calculate_metrics(y_sub, p_sub, operating_threshold=0.50)

            if t_key == "original":
                orig_p = p_sub
                flip_r = 0.0
                drift = 0.0
            else:
                flip_r = compute_prediction_flip_rate(orig_p, p_sub, threshold=0.50)
                drift = compute_mean_probability_drift(orig_p, p_sub)

            trans_res[t_key] = {
                "description": meta.get("description", t_key),
                "roc_auc": round(float(met["roc_auc"]), 4),
                "accuracy": round(float(met["accuracy"]), 4),
                "macro_f1": round(float(met["macro_f1"]), 4),
                "fpr": round(float(met["fpr"]), 4),
                "flip_rate": round(float(flip_r), 4),
                "probability_drift": round(float(drift), 4),
            }

        degraded = [k for k in transform_keys if k != "original"]
        mean_auc = float(np.mean([trans_res[k]["roc_auc"] for k in degraded]))
        mean_acc = float(np.mean([trans_res[k]["accuracy"] for k in degraded]))
        mean_flip = float(np.mean([trans_res[k]["flip_rate"] for k in degraded]))
        mean_drift = float(np.mean([trans_res[k]["probability_drift"] for k in degraded]))

        # Authenticity Stability scores across the dataset
        stability_scores = []
        for i in range(len(eval_sub)):
            sample_orig_p = orig_p[i]
            # Gather perturbed probs for sample i
            pass

        robustness_results[m_name] = {
            "transformations": trans_res,
            "mean_degraded_auc": round(mean_auc, 4),
            "mean_degraded_accuracy": round(mean_acc, 4),
            "mean_flip_rate": round(mean_flip, 4),
            "mean_probability_drift": round(mean_drift, 4),
        }

    final_payload = {
        "candidate_comparisons": results,
        "robustness_benchmark": robustness_results,
    }

    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "extended_candidates_full_eval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nComprehensive evaluation finished. Saved to {out_file}")
    return final_payload


if __name__ == "__main__":
    run_comprehensive_evaluation()
