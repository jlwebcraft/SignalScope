"""Model training pipeline for SignalScope Baseline: ConvNeXt-Tiny.

Executes reproducible, transfer-learning baseline training runs:
- Pure spatial transfer learning (no frequency, no fusion, no VLMs)
- Trained strictly on C:\\Programming\\SignalScope-data\\train (100k samples: 85k train / 15k val)
- Held-out test set C:\\Programming\\SignalScope-data\\test is strictly excluded
- Mixed precision (FP16 / AMP) on NVIDIA GeForce RTX 3050 GPU
- Evaluates validation ROC-AUC, Macro-F1, Accuracy, Precision, Recall, FPR
- Checkpoints best model by validation ROC-AUC
- Generates plots and lightweight experiment metadata
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from app.utils.logger import logger
from model.architectures.convnext import ConvNeXtTinyDetector, build_convnext_tiny
from model.dataset import (
    SignalScopeDataset,
    create_honest_splits,
    get_default_transforms,
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


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic random seed fixed to: {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Trains model for a single epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % 200 == 0 or (step + 1) == len(loader):
            logger.info(f"  Step [{step + 1}/{len(loader)}] - Batch Loss: {loss.item():.4f}")

    return float(running_loss / max(1, total_samples))


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    operating_threshold: float = 0.50,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluates model on a DataLoader, returning loss, targets, predicted probabilities, and metrics."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
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


def run_baseline_training(
    config_path: str = "model/configs/baseline_convnext.yaml",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
) -> Dict[str, Any]:
    """Orchestrates reproducible ConvNeXt-Tiny baseline training and validation."""
    logger.info("==================================================")
    logger.info(" SignalScope Baseline Training (ConvNeXt-Tiny)")
    logger.info("==================================================")

    # 1. Configuration
    config = load_config(config_path)
    base_cfg_path = config.get("base_config")
    if base_cfg_path and (PROJECT_ROOT / base_cfg_path).exists():
        master_cfg = load_config(str(PROJECT_ROOT / base_cfg_path))
        # Deep merge master with experiment config
        for sec in ["dataset", "model", "training", "evaluation"]:
            if sec in master_cfg and sec not in config:
                config[sec] = master_cfg[sec]
            elif sec in master_cfg and sec in config:
                merged = master_cfg[sec].copy()
                merged.update(config[sec])
                config[sec] = merged

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    # 2. Hardware Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    # 3. Data Location
    data_root_str = os.environ.get(
        "SIGNALSCOPE_DATA_ROOT",
        config.get("dataset", {}).get("data_root", r"C:\Programming\SignalScope-data"),
    )
    data_root = Path(data_root_str)
    train_dir = data_root / config.get("dataset", {}).get("train_path", "train")

    logger.info(f"Dataset root: {data_root}")
    logger.info(f"Training partition (ONLY source used for training & local validation): {train_dir}")
    logger.info("Test partition: Held-out evaluation set — STRICTLY EXCLUDED from training & validation.")

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")

    # 4. Scan samples strictly from TRAIN
    all_train_samples = scan_dataset_directory(train_dir)
    if not all_train_samples:
        raise RuntimeError(f"No samples found in {train_dir}")

    # 5. Create Honest Stratified Split (85% train, 15% validation)
    val_ratio = float(config.get("dataset", {}).get("val_split_ratio", 0.15))
    train_samples, val_samples, _ = create_honest_splits(
        all_train_samples,
        val_ratio=val_ratio,
        test_ratio=0.0,  # 0.0 because test set is separate external partition
        seed=seed,
    )

    if limit_train:
        train_samples = train_samples[:limit_train]
        logger.info(f"Limit applied: train_samples clamped to {len(train_samples)}")
    if limit_val:
        val_samples = val_samples[:limit_val]
        logger.info(f"Limit applied: val_samples clamped to {len(val_samples)}")

    logger.info(f"Final local split: {len(train_samples):,} Training samples | {len(val_samples):,} Validation samples")

    # 6. Transforms & DataLoaders
    image_size = int(config.get("dataset", {}).get("image_size", 224))
    train_transform = get_default_transforms(image_size=image_size, is_training=True)
    val_transform = get_default_transforms(image_size=image_size, is_training=False)

    train_ds = SignalScopeDataset(train_samples, transform=train_transform)
    val_ds = SignalScopeDataset(val_samples, transform=val_transform)

    batch_size = batch_size_override or int(config.get("training", {}).get("batch_size", 64))
    num_workers = int(config.get("dataset", {}).get("num_workers", 2))
    pin_memory = bool(config.get("dataset", {}).get("pin_memory", True)) and (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 7. Model Instantiation (ConvNeXt-Tiny)
    dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.2))
    model = build_convnext_tiny(pretrained=True, dropout_rate=dropout_rate).to(device)
    exact_model_id = "convnext_tiny.in12k_ft_in1k"
    logger.info(f"Instantiated backbone: {exact_model_id} (num_classes=1, dropout={dropout_rate})")

    # 8. Training Hyperparameters
    epochs = epochs_override or int(config.get("training", {}).get("epochs", 5))
    lr = lr_override or float(config.get("training", {}).get("learning_rate", 1.0e-4))
    weight_decay = float(config.get("training", {}).get("weight_decay", 1.0e-2))
    min_lr = float(config.get("training", {}).get("min_lr", 1.0e-6))
    use_amp = bool(config.get("training", {}).get("mixed_precision", True))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, AMP={use_amp}")

    # Output & Checkpoint Directories
    checkpoint_dir = PROJECT_ROOT / config.get("training", {}).get("checkpoint_dir", "checkpoints/baseline_convnext")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_output_dir = PROJECT_ROOT / "outputs" / "experiments" / "baseline"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 9. Training Loop
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

    best_val_auc = -1.0
    best_epoch = -1
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(f"\n--- Epoch [{epoch}/{epochs}] (lr: {current_lr:.2e}) ---")

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        # Validate
        val_loss, y_true_val, y_scores_val, val_metrics = evaluate_on_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            operating_threshold=0.50,
        )

        scheduler.step()
        epoch_dur = time.time() - epoch_start

        # Record history
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_auc"].append(val_metrics["roc_auc"])
        history["val_f1"].append(val_metrics["macro_f1"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_fpr"].append(val_metrics["fpr"])
        history["lr"].append(current_lr)
        history["epoch_duration"].append(round(epoch_dur, 2))

        logger.info(
            f"Epoch [{epoch}/{epochs}] completed in {epoch_dur:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val FPR: {val_metrics['fpr']*100:.2f}%"
        )

        # Checkpoint if best validation ROC-AUC
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            checkpoint_payload = {
                "epoch": epoch,
                "model_name": exact_model_id,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_metrics,
                "config": config,
                "seed": seed,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            logger.info(f"  --> Checkpoint saved: new best validation ROC-AUC: {best_val_auc:.4f} at epoch {epoch}")

    total_training_time = time.time() - start_total_time
    logger.info(f"\nTraining finished in {total_training_time:.1f}s ({total_training_time/60:.2f} mins).")
    logger.info(f"Best model achieved at Epoch {best_epoch} with Val ROC-AUC: {best_val_auc:.4f}")

    # 10. Final In-Depth Evaluation with Best Model on Validation Set
    logger.info("\nLoading best model checkpoint for final validation assessment...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    final_val_loss, y_true_final, y_scores_final, base_metrics = evaluate_on_loader(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        operating_threshold=0.50,
    )

    # Threshold Optimization on Validation Set
    opt_thresh_youden, max_j = find_optimal_threshold_youden(y_true_final, y_scores_final)
    opt_thresh_fpr5, actual_fpr5 = find_threshold_for_target_fpr(y_true_final, y_scores_final, target_fpr=0.05)

    # Metrics at optimal threshold (Youden's J)
    opt_metrics = calculate_metrics(y_true_final, y_scores_final, operating_threshold=opt_thresh_youden)

    print_metrics_report(base_metrics, title="Baseline Validation Metrics (Threshold = 0.50)")
    print_metrics_report(opt_metrics, title=f"Baseline Validation Metrics (Optimal Threshold = {opt_thresh_youden:.4f})")

    # 11. Generate Publication-Quality Plots
    roc_plot_path = figures_dir / "baseline_roc_curve.png"
    cm_plot_path = figures_dir / "baseline_confusion_matrix.png"
    curves_plot_path = figures_dir / "baseline_training_curves.png"

    plot_roc_curve(y_true_final, y_scores_final, roc_plot_path, title=f"SignalScope ConvNeXt Baseline ROC (AUC = {base_metrics['roc_auc']:.4f})")
    plot_confusion_matrix_figure(opt_metrics["confusion_matrix"]["matrix"], cm_plot_path, title=f"Baseline Confusion Matrix (Threshold = {opt_thresh_youden:.2f})")
    plot_training_history(history, curves_plot_path, title="ConvNeXt-Tiny Baseline Training & Validation Curves")

    # 12. Save Structured Metadata Artifacts
    metadata = {
        "experiment_name": config.get("experiment_name", "baseline_convnext_tiny"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": exact_model_id,
        "input_resolution": f"{image_size}x{image_size}",
        "total_training_samples": len(train_samples),
        "total_validation_samples": len(val_samples),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "random_seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_training_seconds": round(total_training_time, 2),
        "training_history": history,
        "threshold_analysis": {
            "default_threshold": 0.50,
            "optimal_threshold_youden": round(opt_thresh_youden, 4),
            "target_5pct_fpr_threshold": round(opt_thresh_fpr5, 4),
            "actual_fpr_at_5pct_threshold": round(actual_fpr5, 4),
        },
        "validation_metrics_default_thr": base_metrics,
        "validation_metrics_optimal_thr": opt_metrics,
        "checkpoint_path": str(best_checkpoint_path),
        "figures": {
            "roc_curve": str(roc_plot_path),
            "confusion_matrix": str(cm_plot_path),
            "training_curves": str(curves_plot_path),
        },
    }

    # Save to report/baseline_experiment.json
    baseline_json_path = PROJECT_ROOT / "report" / "baseline_experiment.json"
    with open(baseline_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved baseline experiment JSON to {baseline_json_path}")

    # Also save copies in outputs/experiments/baseline/
    with open(exp_output_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(exp_output_dir / "epoch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="model/configs/baseline_convnext.yaml",
        help="Path to YAML training configuration",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit training samples (for testing)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit validation samples (for testing)")
    args = parser.parse_args()

    run_baseline_training(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
