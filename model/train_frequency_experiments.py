"""Training and evaluation harness for SignalScope Phase 3 Frequency Experiments.

Supports:
1. Frequency-Only Model (2D FFT Log-Magnitude on native 32x32)
2. DCT-Only Ablation (2D DCT Type-II on native 32x32)
3. RGB + FFT Dual-Branch Fusion Model (224x224 ConvNeXt-Tiny + 32x32 Frequency CNN)

Strict Fairness Rules:
- Same 85,000 train / 15,000 validation split constructed from train/ with seed 42.
- Held-out test set (test/) is STRICTLY EXCLUDED.
- Saves distinct structured JSON records in report/experiments/.
- Saves checkpoints under checkpoints/<experiment_name>/.
- Generates distinct training curve, ROC curve, and confusion matrix figures in report/figures/.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.utils.logger import logger
from model.architectures.frequency import (
    FrequencyCNNBranch,
    compute_dct_2d,
    compute_fft_2d,
)
from model.architectures.fusion import DualBranchFusionDetector
from model.dataset import (
    SignalScopeDataset,
    SignalScopeDualDataset,
    create_honest_splits,
    get_default_transforms,
    get_native_transforms,
    scan_dataset_directory,
)
from model.evaluate import (
    calculate_metrics,
    find_optimal_threshold_youden,
    find_threshold_for_target_fpr,
    plot_confusion_matrix_figure,
    plot_roc_curve,
    plot_training_history,
    print_metrics_report,
)


def get_current_git_commit() -> str:
    """Returns current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def set_seed(seed: int = 42) -> None:
    """Fixes deterministic seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch_frequency(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mode: str = "fft",  # "fft" or "dct"
    use_amp: bool = True,
) -> float:
    """Trains frequency-only or DCT-only model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Compute frequency representation on GPU
        if mode == "dct":
            freq_map = compute_dct_2d(images, normalize=True)
        else:
            freq_map = compute_fft_2d(images, shift=True, normalize=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(freq_map)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

    return float(running_loss / max(1, total_samples))


def evaluate_loader_frequency(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str = "fft",
    operating_threshold: float = 0.50,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluates frequency-only model on validation loader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mode == "dct":
                freq_map = compute_dct_2d(images, normalize=True)
            else:
                freq_map = compute_fft_2d(images, shift=True, normalize=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(freq_map)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits)

            bs = images.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            all_targets.extend(targets.squeeze(-1).cpu().numpy().tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().tolist())

    avg_loss = float(running_loss / max(1, total_samples))
    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)
    metrics = calculate_metrics(y_true, y_scores, operating_threshold=operating_threshold)
    return avg_loss, y_true, y_scores, metrics


def train_epoch_fusion(
    model: DualBranchFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Trains RGB + FFT Fusion model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for step, (spatial_imgs, native_imgs, targets, _) in enumerate(loader):
        spatial_imgs = spatial_imgs.to(device, non_blocking=True)
        native_imgs = native_imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(spatial_imgs, x_native=native_imgs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = spatial_imgs.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % 200 == 0 or (step + 1) == len(loader):
            logger.info(f"  Step [{step + 1}/{len(loader)}] - Batch Loss: {loss.item():.4f}")

    return float(running_loss / max(1, total_samples))


def evaluate_loader_fusion(
    model: DualBranchFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    operating_threshold: float = 0.50,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluates RGB + FFT Fusion model on validation loader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for spatial_imgs, native_imgs, targets, _ in loader:
            spatial_imgs = spatial_imgs.to(device, non_blocking=True)
            native_imgs = native_imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(spatial_imgs, x_native=native_imgs)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits)

            bs = spatial_imgs.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            all_targets.extend(targets.squeeze(-1).cpu().numpy().tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().tolist())

    avg_loss = float(running_loss / max(1, total_samples))
    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)
    metrics = calculate_metrics(y_true, y_scores, operating_threshold=operating_threshold)
    return avg_loss, y_true, y_scores, metrics


def run_experiment(
    experiment_type: str,  # "frequency_only", "dct_only", or "fusion"
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs a complete, reproducible Phase 3 experiment."""
    logger.info("=" * 60)
    logger.info(f" SignalScope Phase 3 Experiment: {experiment_type.upper()}")
    logger.info("=" * 60)

    set_seed(seed)
    git_commit = get_current_git_commit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Dataset directory
    train_dir = Path(r"C:\Programming\SignalScope-data\train")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training partition not found at {train_dir}")

    # Scan train samples
    all_train_samples = scan_dataset_directory(train_dir)
    train_samples, val_samples, _ = create_honest_splits(
        all_train_samples,
        val_ratio=0.15,
        test_ratio=0.0,
        seed=seed,
    )

    if limit_train:
        train_samples = train_samples[:limit_train]
    if limit_val:
        val_samples = val_samples[:limit_val]

    logger.info(f"Samples: Train={len(train_samples):,} | Val={len(val_samples):,}")

    # Output directories
    report_exp_dir = PROJECT_ROOT / "report" / "experiments"
    report_exp_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    # Setup model and datasets according to experiment type
    criterion = nn.BCEWithLogitsLoss()
    use_amp = True

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_fpr": [],
        "lr": [],
        "epoch_duration": [],
    }

    start_time = time.time()
    best_val_auc = -1.0
    best_epoch = -1

    if experiment_type in ("frequency_only", "dct_only"):
        mode = "dct" if experiment_type == "dct_only" else "fft"
        native_train_tf = get_native_transforms(image_size=32, is_training=True)
        native_val_tf = get_native_transforms(image_size=32, is_training=False)

        train_ds = SignalScopeDataset(train_samples, transform=native_train_tf)
        val_ds = SignalScopeDataset(val_samples, transform=native_val_tf)

        # BS can be 128 for 32x32 frequency CNN
        bs = batch_size if batch_size != 64 else 128
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        model = FrequencyCNNBranch(in_channels=1, feature_dim=128, dropout_rate=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

        arch_name = f"FrequencyCNNBranch ({mode.upper()} 32x32)"

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            curr_lr = float(optimizer.param_groups[0]["lr"])

            train_loss = train_epoch_frequency(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                mode=mode,
                use_amp=use_amp,
            )

            val_loss, y_true_val, y_scores_val, val_metrics = evaluate_loader_frequency(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                mode=mode,
                operating_threshold=0.50,
            )

            scheduler.step()
            ep_dur = time.time() - ep_start

            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_auc"].append(val_metrics["roc_auc"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_fpr"].append(val_metrics["fpr"])
            history["lr"].append(curr_lr)
            history["epoch_duration"].append(round(ep_dur, 2))

            logger.info(
                f"Epoch [{epoch}/{epochs}] ({ep_dur:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val FPR: {val_metrics['fpr']*100:.2f}%"
            )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": arch_name,
                        "state_dict": model.state_dict(),
                        "val_auc": best_val_auc,
                        "val_metrics": val_metrics,
                        "seed": seed,
                    },
                    best_checkpoint_path,
                )

        # Final evaluation with best checkpoint
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"])
        final_val_loss, y_true_final, y_scores_final, final_metrics_def = evaluate_loader_frequency(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            mode=mode,
            operating_threshold=0.50,
        )

    elif experiment_type == "fusion":
        spatial_train_tf = get_default_transforms(image_size=224, is_training=True)
        spatial_val_tf = get_default_transforms(image_size=224, is_training=False)
        native_train_tf = get_native_transforms(image_size=32, is_training=True)
        native_val_tf = get_native_transforms(image_size=32, is_training=False)

        train_ds = SignalScopeDualDataset(
            train_samples,
            spatial_transform=spatial_train_tf,
            native_transform=native_train_tf,
        )
        val_ds = SignalScopeDualDataset(
            val_samples,
            spatial_transform=spatial_val_tf,
            native_transform=native_val_tf,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize spatial branch with the verified baseline checkpoint
        spatial_ckpt = str(PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt")
        model = DualBranchFusionDetector(
            pretrained=True,
            dropout_rate=0.2,
            frequency_feature_dim=128,
            spatial_checkpoint=spatial_ckpt if Path(spatial_ckpt).exists() else None,
        ).to(device)

        # Train fusion head and frequency branch, fine-tuning spatial backbone with smaller LR
        params = [
            {"params": model.fusion_head.parameters(), "lr": lr},
            {"params": model.frequency_branch.parameters(), "lr": lr},
            {"params": model.spatial_branch.parameters(), "lr": lr * 0.2},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

        arch_name = "DualBranchFusionDetector (ConvNeXt-Tiny 224x224 + FFT 32x32)"

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            curr_lr = float(optimizer.param_groups[0]["lr"])
            logger.info(f"\n--- Fusion Epoch [{epoch}/{epochs}] (lr: {curr_lr:.2e}) ---")

            train_loss = train_epoch_fusion(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
            )

            val_loss, y_true_val, y_scores_val, val_metrics = evaluate_loader_fusion(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                operating_threshold=0.50,
            )

            scheduler.step()
            ep_dur = time.time() - ep_start

            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_auc"].append(val_metrics["roc_auc"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_fpr"].append(val_metrics["fpr"])
            history["lr"].append(curr_lr)
            history["epoch_duration"].append(round(ep_dur, 2))

            logger.info(
                f"Epoch [{epoch}/{epochs}] ({ep_dur:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val FPR: {val_metrics['fpr']*100:.2f}%"
            )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": arch_name,
                        "state_dict": model.state_dict(),
                        "val_auc": best_val_auc,
                        "val_metrics": val_metrics,
                        "seed": seed,
                    },
                    best_checkpoint_path,
                )

        # Final evaluation with best checkpoint
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"])
        final_val_loss, y_true_final, y_scores_final, final_metrics_def = evaluate_loader_fusion(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            operating_threshold=0.50,
        )

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    total_training_time = time.time() - start_time
    logger.info(f"\nExperiment {experiment_type} completed in {total_training_time:.1f}s.")
    logger.info(f"Best Val ROC-AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Optimal threshold via Youden's J
    opt_thresh_youden, max_j = find_optimal_threshold_youden(y_true_final, y_scores_final)
    opt_metrics = calculate_metrics(y_true_final, y_scores_final, operating_threshold=opt_thresh_youden)

    print_metrics_report(final_metrics_def, title=f"{experiment_type} Metrics (Threshold = 0.50)")

    # Generate Figures
    prefix = "frequency_only" if experiment_type == "frequency_only" else ("dct_only" if experiment_type == "dct_only" else "rgb_frequency_fusion")
    curves_path = figures_dir / f"{prefix}_training_curves.png"
    roc_path = figures_dir / f"{prefix}_roc_curve.png"
    cm_path = figures_dir / f"{prefix}_confusion_matrix.png"

    plot_training_history(history, curves_path, title=f"{arch_name} Training History")
    plot_roc_curve(y_true_final, y_scores_final, roc_path, title=f"{arch_name} ROC Curve (AUC = {final_metrics_def['roc_auc']:.4f})")
    plot_confusion_matrix_figure(opt_metrics["confusion_matrix"]["matrix"], cm_path, title=f"{arch_name} Confusion Matrix (Thr = {opt_thresh_youden:.2f})")

    # Save Predictions array for complementarity analysis
    pred_save_path = report_exp_dir / f"{prefix}_val_predictions.npz"
    np.savez_compressed(
        pred_save_path,
        y_true=y_true_final,
        y_scores=y_scores_final,
    )

    # Save lightweight structured JSON record
    record = {
        "experiment_name": experiment_type,
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "model_architecture": arch_name,
        "dataset_split": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_partition": "STRICTLY_HELD_OUT",
        },
        "random_seed": seed,
        "input_size": "32x32" if experiment_type != "fusion" else "224x224_and_32x32",
        "transform": experiment_type,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "batch_size": batch_size if experiment_type == "fusion" else bs,
        "epoch_count": epochs,
        "best_epoch": best_epoch,
        "loss": round(final_val_loss, 5),
        "metrics_default_thr": final_metrics_def,
        "metrics_optimal_thr": opt_metrics,
        "threshold_analysis": {
            "default_threshold": 0.50,
            "optimal_threshold_youden": round(opt_thresh_youden, 4),
            "max_youden_j": round(max_j, 4),
        },
        "training_time_seconds": round(total_training_time, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device.type == "cuda" else 0.0,
        "checkpoint_path": str(best_checkpoint_path),
        "figures": {
            "training_curves": str(curves_path),
            "roc_curve": str(roc_path),
            "confusion_matrix": str(cm_path),
        },
        "notes": "Experimental evaluation on local validation split. Raw model probabilities are uncalibrated.",
    }

    json_record_path = report_exp_dir / f"{prefix}.json"
    with open(json_record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Saved experiment record to {json_record_path}")

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Frequency Experiment Runner")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["frequency_only", "dct_only", "fusion"],
        help="Experiment type to execute",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit train samples")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit val samples")
    args = parser.parse_args()

    run_experiment(
        experiment_type=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
