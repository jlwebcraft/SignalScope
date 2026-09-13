"""SignalScope Second-Stage Extended Training Runner (Up to 5 Epochs with Early Stopping).

Supports:
- exp2_5ep: Balanced mixed-domain training
- exp3_5ep: Balanced mixed-domain training + forensic augmentations (multi-scale, JPEG, mild blur, color jitter)

Evaluates every epoch on:
1. LOCAL VALIDATION (15,000 samples, 32x32)
2. NEW-DATA VALIDATION (1,118 samples, multi-resolution across 19 generators)
Computes generator-stratified AUCs, confusion matrices, and calibration metrics.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import logger
from model.architectures.convnext import build_convnext_tiny
from model.calibration import compute_brier_score, compute_ece
from model.dataset import SignalScopeDataset, get_default_transforms
from model.dataset_splits import (
    get_official_splits,
    get_v2_real_samples,
    get_v2_splits,
    verify_split_integrity,
)
from model.evaluate import calculate_metrics, find_optimal_threshold_youden
from model.train_experiments import JPEGCompressionSim, get_experiment3_transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_domain_extended(
    model: nn.Module,
    samples: List[Tuple[Path, int, str]],
    device: torch.device,
    domain_name: str,
    batch_size: int = 128,
) -> Dict[str, Any]:
    transform = get_default_transforms(image_size=224, is_training=False)
    dataset = SignalScopeDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    all_targets = []
    all_probs = []
    all_logits = []
    all_gens = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets, gens in loader:
            images = images.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets_dev)
                probs = torch.sigmoid(logits)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())
            all_logits.extend(logits.squeeze(-1).cpu().numpy().astype(float).tolist())
            all_gens.extend(gens)

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)
    avg_loss = float(total_loss / max(1, total_samples))

    metrics = calculate_metrics(y_true, y_scores, operating_threshold=0.50)
    opt_th, youden_j = find_optimal_threshold_youden(y_true, y_scores)
    opt_metrics = calculate_metrics(y_true, y_scores, operating_threshold=opt_th)
    brier = compute_brier_score(y_true, y_scores)
    ece_val, mce_val, _ = compute_ece(y_true, y_scores, n_bins=15)

    res = {
        "domain": domain_name,
        "sample_count": len(samples),
        "loss": round(avg_loss, 4),
        "roc_auc": round(float(metrics["roc_auc"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "accuracy": round(float(metrics["accuracy"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "fpr": round(float(metrics["fpr"]), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece_val), 4),
        "mce": round(float(mce_val), 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "optimal_threshold": {
            "threshold": round(float(opt_th), 4),
            "youden_j": round(float(youden_j), 4),
            "accuracy": round(float(opt_metrics["accuracy"]), 4),
            "macro_f1": round(float(opt_metrics["macro_f1"]), 4),
            "fpr": round(float(opt_metrics["fpr"]), 4),
            "confusion_matrix": opt_metrics["confusion_matrix"],
        },
    }

    # Generator-stratified evaluation for domains with multiple generative families
    distinct_gens = sorted(list(set(g for g in all_gens if g not in {"real", "unknown"})))
    if distinct_gens:
        gen_aucs = {}
        real_indices = [i for i, lbl in enumerate(all_targets) if lbl == 0]
        for g_name in distinct_gens:
            g_indices = [i for i, g in enumerate(all_gens) if g == g_name]
            if g_indices:
                sub_indices = real_indices + g_indices
                sub_y_true = y_true[sub_indices]
                sub_y_scores = y_scores[sub_indices]
                if len(np.unique(sub_y_true)) > 1:
                    g_auc = float(roc_auc_score(sub_y_true, sub_y_scores))
                    gen_aucs[g_name] = {
                        "samples": len(g_indices),
                        "roc_auc": round(g_auc, 4),
                        "recall_at_050": round(float(np.mean(sub_y_scores[len(real_indices):] >= 0.50)), 4),
                        "mean_confidence": round(float(np.mean(sub_y_scores[len(real_indices):])), 4),
                    }

        if gen_aucs:
            auc_vals = [v["roc_auc"] for v in gen_aucs.values()]
            strongest_gen = max(gen_aucs.items(), key=lambda x: x[1]["roc_auc"])
            weakest_gen = min(gen_aucs.items(), key=lambda x: x[1]["roc_auc"])
            spread = strongest_gen[1]["roc_auc"] - weakest_gen[1]["roc_auc"]
            res["generator_stratified_evaluation"] = {
                "num_generators": len(gen_aucs),
                "macro_avg_auc": round(float(np.mean(auc_vals)), 4),
                "strongest_generator": {"name": strongest_gen[0], "auc": strongest_gen[1]["roc_auc"]},
                "weakest_generator": {"name": weakest_gen[0], "auc": weakest_gen[1]["roc_auc"]},
                "auc_spread": round(float(spread), 4),
                "generators": gen_aucs,
            }

    return res


def run_extended_experiment(
    experiment_id: str,
    max_epochs: int = 5,
    patience: int = 2,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info(f" EXTENDED TRAINING STUDY: {experiment_id.upper()} (Max Epochs: {max_epochs}, Patience: {patience})")
    logger.info(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.info("=" * 70)

    # 1. Dataset splits
    train_official, val_old_samples = get_official_splits(seed=seed)
    train_v2, val_new_samples = get_v2_splits()

    rng = np.random.RandomState(seed)
    real_off = [s for s in train_official if s[1] == 0]
    fake_off = [s for s in train_official if s[1] == 1]

    # Balanced pool: 10k official fake, 5k official real, 9.7k AIGen (4.8k R / 4.8k F), 5k photo-real
    idx_r = rng.permutation(len(real_off))[:5000]
    idx_f = rng.permutation(len(fake_off))[:10000]
    sub_official = [real_off[i] for i in idx_r] + [fake_off[i] for i in idx_f]
    v2_real = get_v2_real_samples(limit_per_category=556, seed=seed)
    train_samples = sub_official + train_v2 + v2_real

    rng.shuffle(train_samples)
    verify_split_integrity(train_samples, val_old_samples, val_new_samples)

    num_real = sum(1 for s in train_samples if s[1] == 0)
    num_fake = sum(1 for s in train_samples if s[1] == 1)
    logger.info(f"Balanced Dataset: {len(train_samples):,} samples ({num_real:,} Real, {num_fake:,} Synthetic, {round(100*num_real/len(train_samples), 1)}% Real)")
    logger.info(f"Validation Pools: Val-Old={len(val_old_samples):,}, Val-New={len(val_new_samples):,}")

    if experiment_id.startswith("exp2"):
        train_transform = get_default_transforms(image_size=224, is_training=True)
        config_desc = "Exp2 Extended: Balanced Mixed-Domain Pool (Standard Augmentations)"
    elif experiment_id.startswith("exp3"):
        train_transform = get_experiment3_transforms(image_size=224)
        config_desc = "Exp3 Extended: Balanced Mixed-Domain Pool + Forensic Multi-Scale/JPEG/Blur Augmentations"
    else:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")

    train_dataset = SignalScopeDataset(train_samples, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    # 2. Warm-start model from baseline checkpoint
    baseline_ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    model = build_convnext_tiny(pretrained=True).to(device)
    if baseline_ckpt_path.exists():
        logger.info(f"Warm-starting from validated baseline: {baseline_ckpt_path}")
        b_ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
        b_state = b_ckpt.get("state_dict", b_ckpt)
        cleaned = {k.replace("module.", ""): v for k, v in b_state.items()}
        model.load_state_dict(cleaned)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir = PROJECT_ROOT / "checkpoints" / "additional_training" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    history = []
    best_composite_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_seen = 0

        logger.info(f"\n--- [{experiment_id.upper()}] Starting Epoch {epoch}/{max_epochs} (LR: {scheduler.get_last_lr()[0]:.2e}) ---")
        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            total_seen += bs

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                logger.info(f"  [{experiment_id.upper()}] Epoch [{epoch}/{max_epochs}] Step [{step + 1}/{len(train_loader)}] - Batch Loss: {loss.item():.4f}")
                sys.stdout.flush()

        scheduler.step()
        train_loss = float(running_loss / max(1, total_seen))
        epoch_duration = time.time() - epoch_start
        logger.info(f"Epoch {epoch} training completed in {epoch_duration:.1f}s. Evaluating validation domains...")

        # Evaluate both domains
        val_old_res = evaluate_domain_extended(model, val_old_samples, device, domain_name="LOCAL VALIDATION")
        val_new_res = evaluate_domain_extended(model, val_new_samples, device, domain_name="NEW-DATA VALIDATION")

        # Composite validation score: 50% Old AUC + 50% New AUC
        composite_score = 0.5 * val_old_res["roc_auc"] + 0.5 * val_new_res["roc_auc"]

        logger.info(
            f"Epoch {epoch} Results:\n"
            f"  Train Loss: {train_loss:.4f}\n"
            f"  Val-Old Loss: {val_old_res['loss']:.4f} | AUC: {val_old_res['roc_auc']:.4f} | F1: {val_old_res['macro_f1']:.4f} | FPR: {val_old_res['fpr']:.4f}\n"
            f"  Val-New Loss: {val_new_res['loss']:.4f} | AUC: {val_new_res['roc_auc']:.4f} | F1: {val_new_res['macro_f1']:.4f} | FPR: {val_new_res['fpr']:.4f}\n"
            f"  Composite Score: {composite_score:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "epoch_duration_seconds": round(epoch_duration, 1),
            "composite_score": round(composite_score, 4),
            "val_old": val_old_res,
            "val_new": val_new_res,
        }
        history.append(epoch_record)

        # Check for improvement
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_epoch = epoch
            epochs_without_improvement = 0
            logger.info(f"[*] New best composite score ({best_composite_score:.4f})! Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_architecture": "convnext_tiny",
                "state_dict": model.state_dict(),
                "composite_score": best_composite_score,
                "train_loss": train_loss,
                "val_old_metrics": val_old_res,
                "val_new_metrics": val_new_res,
                "config": config_desc,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement}/{patience} epoch(s). Best was Epoch {best_epoch} ({best_composite_score:.4f}).")
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered at Epoch {epoch} due to convergence.")
                break

    total_duration = time.time() - t_start
    best_sha256 = compute_file_sha256(best_checkpoint_path) if best_checkpoint_path.exists() else "NONE"

    final_results = {
        "experiment_id": experiment_id,
        "description": config_desc,
        "max_epochs": max_epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_composite_score": round(best_composite_score, 4),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_sha256": best_sha256,
        "total_duration_seconds": round(total_duration, 1),
        "history": history,
    }

    results_json = output_dir / "extended_experiment_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Finished {experiment_id} in {total_duration:.1f}s. Results saved to {results_json}")
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Additional Training Experiment")
    parser.add_argument("--experiment", type=str, required=True, choices=["exp2_5ep", "exp3_5ep"])
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    run_extended_experiment(
        args.experiment,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
