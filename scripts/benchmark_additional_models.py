"""Comprehensive Robustness, Calibration, and Explainability Benchmark for Additional Training Candidates."""

import json
import os
from pathlib import Path
import random
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
from model.calibration import TemperatureScaler, compute_brier_score, compute_ece
from model.dataset import SignalScopeDataset, get_default_transforms, get_native_transforms
from model.dataset_splits import get_official_splits, get_v2_splits
from model.evaluate import calculate_metrics
from model.explainability.gradcam import GradCAM
from model.explainability.spectral import extract_spectral_features
from model.robustness import (
    TRANSFORMATION_REGISTRY,
    compute_authenticity_stability_score,
    compute_mean_probability_drift,
    compute_prediction_flip_rate,
)

def load_convnext_checkpoint(path: Path, device: torch.device) -> nn.Module:
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
        # Apply degradation transform
        if self.transform_fn:
            pil_img = self.transform_fn(pil_img)
        tensor = self.base_transform(pil_img)
        return tensor, torch.tensor([label], dtype=torch.float32), gen

def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
    return np.array(all_targets, dtype=int), np.array(all_probs, dtype=float)

def benchmark_robustness(
    models: Dict[str, nn.Module],
    val_samples: List[Tuple[Path, int, str]],
    device: torch.device,
    subset_size: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"\n--- Running Robustness Benchmark on {subset_size} samples ---")
    rng = np.random.RandomState(seed)
    real_samples = [s for s in val_samples if s[1] == 0]
    fake_samples = [s for s in val_samples if s[1] == 1]
    n_half = subset_size // 2
    idx_r = rng.permutation(len(real_samples))[:n_half]
    idx_f = rng.permutation(len(fake_samples))[:n_half]
    eval_samples = [real_samples[i] for i in idx_r] + [fake_samples[i] for i in idx_f]

    base_transform = get_default_transforms(image_size=224, is_training=False)
    results = {}

    transform_keys = ["original", "jpeg_95", "jpeg_85", "jpeg_70", "resize", "screenshot", "light_edit"]

    for m_name, model in models.items():
        print(f"Evaluating robustness for {m_name}...")
        model_res = {"transformations": {}}
        orig_probs = None

        for t_key in transform_keys:
            t_fn, meta = TRANSFORMATION_REGISTRY[t_key]
            ds = TransformedDataset(eval_samples, t_fn, base_transform)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
            y_true, probs = predict_probabilities(model, loader, device)

            metrics = calculate_metrics(y_true, probs, operating_threshold=0.50)

            if t_key == "original":
                orig_probs = probs
                flip_rate = 0.0
                mean_drift = 0.0
            else:
                flip_rate = compute_prediction_flip_rate(orig_probs, probs, threshold=0.50)
                mean_drift = compute_mean_probability_drift(orig_probs, probs)

            model_res["transformations"][t_key] = {
                "description": meta.get("description", t_key),
                "roc_auc": round(float(metrics["roc_auc"]), 4),
                "accuracy": round(float(metrics["accuracy"]), 4),
                "macro_f1": round(float(metrics["macro_f1"]), 4),
                "fpr": round(float(metrics["fpr"]), 4),
                "prediction_flip_rate": round(float(flip_rate), 4),
                "mean_drift": round(float(mean_drift), 4),
            }

        # Calculate stability scores
        stab_scores = []
        for i in range(len(eval_samples)):
            p_orig = orig_probs[i]
            p_trans = [model_res["transformations"][k]["probabilities_placeholder"] if False else 0.0 for k in transform_keys if k != "original"]
            # Gather individual perturbed probs for sample i
            pass

        # Summary averages across all non-original transforms
        degraded_keys = [k for k in transform_keys if k != "original"]
        mean_auc = float(np.mean([model_res["transformations"][k]["roc_auc"] for k in degraded_keys]))
        mean_acc = float(np.mean([model_res["transformations"][k]["accuracy"] for k in degraded_keys]))
        mean_flip = float(np.mean([model_res["transformations"][k]["prediction_flip_rate"] for k in degraded_keys]))
        mean_drift = float(np.mean([model_res["transformations"][k]["mean_drift"] for k in degraded_keys]))

        model_res["summary"] = {
            "mean_degraded_auc": round(mean_auc, 4),
            "mean_degraded_accuracy": round(mean_acc, 4),
            "mean_flip_rate": round(mean_flip, 4),
            "mean_probability_drift": round(mean_drift, 4),
        }
        results[m_name] = model_res

    return results

def benchmark_calibration(
    models: Dict[str, nn.Module],
    train_samples: List[Tuple[Path, int, str]],
    val_old_samples: List[Tuple[Path, int, str]],
    val_new_samples: List[Tuple[Path, int, str]],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n--- Running Calibration Benchmark (Temperature Scaling) ---")
    # Take 2,000 samples from train pool for temperature fitting
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_samples))[:2000]
    cal_samples = [train_samples[i] for i in idx]

    transform = get_default_transforms(image_size=224, is_training=False)
    cal_loader = DataLoader(SignalScopeDataset(cal_samples, transform=transform), batch_size=64, shuffle=False, num_workers=0)
    val_old_loader = DataLoader(SignalScopeDataset(val_old_samples[:2000], transform=transform), batch_size=64, shuffle=False, num_workers=0)
    val_new_loader = DataLoader(SignalScopeDataset(val_new_samples, transform=transform), batch_size=64, shuffle=False, num_workers=0)

    results = {}

    for m_name, model in models.items():
        print(f"Fitting temperature for {m_name}...")
        model.eval()
        
        # Get calibration logits
        cal_logits, cal_targets = [], []
        with torch.no_grad():
            for imgs, tgts, _ in cal_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                cal_logits.append(logits.squeeze(-1).cpu())
                cal_targets.append(tgts.squeeze(-1).cpu())
        c_logits = torch.cat(cal_logits)
        c_tgts = torch.cat(cal_targets)

        scaler = TemperatureScaler()
        T = scaler.fit(c_logits.to(device), c_tgts.to(device))

        # Evaluate on Val-Old
        val_old_logits, val_old_tgts = [], []
        with torch.no_grad():
            for imgs, tgts, _ in val_old_loader:
                imgs = imgs.to(device)
                val_old_logits.append(model(imgs).squeeze(-1).cpu())
                val_old_tgts.append(tgts.squeeze(-1).cpu())
        vo_logits = torch.cat(val_old_logits)
        vo_tgts = torch.cat(val_old_tgts).numpy().astype(int)

        raw_probs_vo = torch.sigmoid(vo_logits).numpy()
        cal_probs_vo = scaler.calibrate(vo_logits.to(device)).detach().cpu().numpy()

        raw_ece_vo, _, _ = compute_ece(vo_tgts, raw_probs_vo)
        cal_ece_vo, _, _ = compute_ece(vo_tgts, cal_probs_vo)
        raw_brier_vo = compute_brier_score(vo_tgts, raw_probs_vo)
        cal_brier_vo = compute_brier_score(vo_tgts, cal_probs_vo)

        # Evaluate on Val-New
        val_new_logits, val_new_tgts = [], []
        with torch.no_grad():
            for imgs, tgts, _ in val_new_loader:
                imgs = imgs.to(device)
                val_new_logits.append(model(imgs).squeeze(-1).cpu())
                val_new_tgts.append(tgts.squeeze(-1).cpu())
        vn_logits = torch.cat(val_new_logits)
        vn_tgts = torch.cat(val_new_tgts).numpy().astype(int)

        raw_probs_vn = torch.sigmoid(vn_logits).numpy()
        cal_probs_vn = scaler.calibrate(vn_logits.to(device)).detach().cpu().numpy()

        raw_ece_vn, _, _ = compute_ece(vn_tgts, raw_probs_vn)
        cal_ece_vn, _, _ = compute_ece(vn_tgts, cal_probs_vn)
        raw_brier_vn = compute_brier_score(vn_tgts, raw_probs_vn)
        cal_brier_vn = compute_brier_score(vn_tgts, cal_probs_vn)

        results[m_name] = {
            "fitted_temperature": round(float(T), 4),
            "val_old": {
                "raw_ece": round(float(raw_ece_vo), 4),
                "calibrated_ece": round(float(cal_ece_vo), 4),
                "raw_brier": round(float(raw_brier_vo), 4),
                "calibrated_brier": round(float(cal_brier_vo), 4),
            },
            "val_new": {
                "raw_ece": round(float(raw_ece_vn), 4),
                "calibrated_ece": round(float(cal_ece_vn), 4),
                "raw_brier": round(float(raw_brier_vn), 4),
                "calibrated_brier": round(float(cal_brier_vn), 4),
            }
        }

    return results

def verify_explainability(
    models: Dict[str, nn.Module],
    val_new_samples: List[Tuple[Path, int, str]],
    device: torch.device,
) -> Dict[str, Any]:
    print("\n--- Verifying Grad-CAM and Spectral Explainability ---")
    test_sample = val_new_samples[0]
    img_path, label, gen = test_sample

    with Image.open(img_path) as img:
        pil_img = img.convert("RGB")
    transform = get_default_transforms(image_size=224, is_training=False)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    results = {}
    for m_name, model in models.items():
        gradcam = GradCAM(model)
        logit = float(model(input_tensor).item())
        heatmap = gradcam.generate_heatmap(input_tensor)
        mean_act = float(heatmap.mean())
        max_act = float(heatmap.max())
        min_act = float(heatmap.min())
        shape = list(heatmap.shape)
        results[m_name] = {
            "target_layer": str(gradcam.target_layer),
            "output_logit": round(float(logit), 4),
            "heatmap_shape": shape,
            "heatmap_min": round(min_act, 4),
            "heatmap_max": round(max_act, 4),
            "heatmap_mean": round(mean_act, 4),
            "gradcam_valid": bool(max_act > 0.0 and shape == [224, 224]),
        }
    
    # Spectral extraction check
    fft_2d_np, radial_np, hf_ratio, spec_pil = extract_spectral_features(pil_img)
    results["spectral_verification"] = {
        "fft_map_shape": list(fft_2d_np.shape),
        "radial_profile_bins": len(radial_np),
        "high_freq_energy_ratio": round(float(hf_ratio), 4),
        "spectral_features_valid": bool(fft_2d_np.shape == (32, 32)),
    }
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = {
        "baseline": PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt",
        "exp2": PROJECT_ROOT / "checkpoints" / "additional_training" / "exp2" / "best_model.pt",
        "exp3": PROJECT_ROOT / "checkpoints" / "additional_training" / "exp3" / "best_model.pt",
    }

    models = {}
    for name, path in ckpt_paths.items():
        print(f"Loading {name} from {path}...")
        models[name] = load_convnext_checkpoint(path, device)

    train_off, val_old = get_official_splits(seed=42)
    train_v2, val_new = get_v2_splits()

    robustness_res = benchmark_robustness(models, val_old, device, subset_size=1000)
    calibration_res = benchmark_calibration(models, train_off, val_old, val_new, device)
    explainability_res = verify_explainability(models, val_new, device)

    final_report = {
        "robustness": robustness_res,
        "calibration": calibration_res,
        "explainability": explainability_res,
    }

    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "robustness_calibration_xai_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nAll comparisons completed successfully. Saved to {out_file}")
