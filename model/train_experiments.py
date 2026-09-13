"""Scientific Training Pipeline for SignalScope Additional Training Experiments.

Runs:
- Experiment 1: Full Augmented Baseline (Official 85k + AIGen 9.7k)
- Experiment 2: Balanced Mixed-Domain Run (Controlled Low-Res + High-Res AIGen + Photographic Real)
- Experiment 3: Forensic Multi-Scale & Compression Augmentation

Strict Safety Guarantee:
C:\\Programming\\SignalScope-data\\test is NEVER ACCESSED, LOADED, OR INSPECTED.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


class JPEGCompressionSim:
    """Simulates JPEG recompression on PIL image without destroying spatial layout."""
    def __init__(self, p: float = 0.3, quality_range: Tuple[int, int] = (65, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            from io import BytesIO
            q = random.randint(self.quality_range[0], self.quality_range[1])
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


def get_experiment3_transforms(image_size: int = 224) -> transforms.Compose:
    """Robustness-oriented multi-scale augmentations for Experiment 3."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        JPEGCompressionSim(p=0.3, quality_range=(65, 95)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])


def evaluate_domain(
    model: nn.Module,
    samples: List[Tuple[Path, int, str]],
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    domain_name: str = "VAL",
) -> Dict[str, Any]:
    transform = get_default_transforms(image_size=224, is_training=False)
    dataset = SignalScopeDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

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

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)

    metrics = calculate_metrics(y_true, y_scores, operating_threshold=0.50)
    opt_th, _ = find_optimal_threshold_youden(y_true, y_scores)
    brier = compute_brier_score(y_true, y_scores)
    ece_val, _, _ = compute_ece(y_true, y_scores, n_bins=15)

    return {
        "domain": domain_name,
        "sample_count": len(samples),
        "roc_auc": metrics["roc_auc"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fpr": metrics["fpr"],
        "optimal_threshold": round(float(opt_th), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece_val), 4),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def run_experiment(
    experiment_id: str,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 65)
    logger.info(f" Launching Additional Training Experiment: {experiment_id.upper()}")
    logger.info(f" Hardware Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.info("=" * 65)

    # 1. Load splits
    train_official, val_old_samples = get_official_splits(seed=seed)
    train_v2, val_new_samples = get_v2_splits()

    # Configure dataset composition based on experiment
    custom_transforms = None
    if experiment_id == "exp1":
        # Full Augmented Pool: All Official (85k) + AIGen Train (9.7k)
        train_samples = train_official + train_v2
        description = "Experiment 1: Full Augmented Pool (Official 85k + AIGen 9.7k)"
    elif experiment_id == "exp2":
        # Balanced Mixed-Domain Pool: 20k Official + 9.7k AIGen + 10k Photographic Real
        rng = np.random.RandomState(seed)
        real_off = [s for s in train_official if s[1] == 0]
        fake_off = [s for s in train_official if s[1] == 1]
        
        # 10k official real + 10k official fake
        idx_r = rng.permutation(len(real_off))[:10000]
        idx_f = rng.permutation(len(fake_off))[:10000]
        sub_official = [real_off[i] for i in idx_r] + [fake_off[i] for i in idx_f]

        # 9.7k AIGen train (4,879 real + 4,879 fake)
        # Plus photographic real samples from REAL
        v2_real = get_v2_real_samples(limit_per_category=1100, seed=seed) # ~10k photographic real
        train_samples = sub_official + train_v2 + v2_real
        description = "Experiment 2: Balanced Mixed-Domain Pool (20k official + 9.7k AIGen + 10k photo-real)"
    elif experiment_id == "exp3":
        # Balanced Mixed-Domain Pool + Multi-Scale & JPEG Compression Augmentation
        rng = np.random.RandomState(seed)
        real_off = [s for s in train_official if s[1] == 0]
        fake_off = [s for s in train_official if s[1] == 1]
        idx_r = rng.permutation(len(real_off))[:10000]
        idx_f = rng.permutation(len(fake_off))[:10000]
        sub_official = [real_off[i] for i in idx_r] + [fake_off[i] for i in idx_f]
        v2_real = get_v2_real_samples(limit_per_category=1100, seed=seed)
        train_samples = sub_official + train_v2 + v2_real
        custom_transforms = get_experiment3_transforms(image_size=224)
        description = "Experiment 3: Balanced Mixed-Domain + Multi-Scale/Compression Augmentation"
    else:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")

    # Shuffle training samples deterministically
    rng = np.random.RandomState(seed)
    rng.shuffle(train_samples)

    # Verify zero leakage
    verify_split_integrity(train_samples, val_old_samples, val_new_samples)

    num_real = sum(1 for s in train_samples if s[1] == 0)
    num_fake = sum(1 for s in train_samples if s[1] == 1)
    logger.info(f"Dataset: {len(train_samples):,} training samples ({num_real:,} Real, {num_fake:,} Synthetic, {round(100*num_real/len(train_samples), 1)}% Real)")
    logger.info(f"Evaluation targets: Val-Old={len(val_old_samples):,}, Val-New={len(val_new_samples):,}")

    # 2. Dataloaders
    train_transform = custom_transforms or get_default_transforms(image_size=224, is_training=True)
    train_dataset = SignalScopeDataset(train_samples, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))

    # 3. Model warm-started from baseline ConvNeXt-Tiny weights
    baseline_ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    model = build_convnext_tiny(pretrained=True).to(device)
    if baseline_ckpt_path.exists():
        logger.info(f"Warm-starting from validated baseline checkpoint: {baseline_ckpt_path}")
        b_ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
        b_state = b_ckpt.get("state_dict", b_ckpt)
        cleaned = {k.replace("module.", ""): v for k, v in b_state.items()}
        model.load_state_dict(cleaned)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir = PROJECT_ROOT / "checkpoints" / "additional_training" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    history = []
    best_composite_score = -1.0
    best_epoch = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_seen = 0
        ep_t0 = time.time()

        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            total_seen += bs

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, total_seen)
        ep_time = time.time() - ep_t0

        # Evaluate on both validation domains
        logger.info(f"Epoch {epoch} completed in {ep_time:.1f}s. Evaluating validation domains...")
        val_old_res = evaluate_domain(model, val_old_samples, batch_size=128, device=device, domain_name="LOCAL VALIDATION")
        val_new_res = evaluate_domain(model, val_new_samples, batch_size=128, device=device, domain_name="NEW-DATA VALIDATION")

        # Composite generalization score: Harmonic mean of AUCs
        auc_old = val_old_res["roc_auc"]
        auc_new = val_new_res["roc_auc"]
        composite_score = 2 * (auc_old * auc_new) / max(1e-6, (auc_old + auc_new))

        logger.info(
            f"Epoch {epoch} Summary: Train Loss={train_loss:.4f} | "
            f"Val-Old AUC={auc_old:.4f} (F1={val_old_res['macro_f1']:.4f}, FPR={val_old_res['fpr']:.4f}) | "
            f"Val-New AUC={auc_new:.4f} (F1={val_new_res['macro_f1']:.4f}, FPR={val_new_res['fpr']:.4f}) | "
            f"Composite Score={composite_score:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_old": val_old_res,
            "val_new": val_new_res,
            "composite_score": round(composite_score, 4),
            "epoch_duration_seconds": round(ep_time, 1),
        }
        history.append(epoch_record)

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_old_metrics": val_old_res,
                "val_new_metrics": val_new_res,
                "composite_score": composite_score,
                "experiment_id": experiment_id,
            }, best_checkpoint_path)
            logger.info(f"[*] Best model checkpoint saved to {best_checkpoint_path} (Composite={composite_score:.4f})")

    total_duration = time.time() - t_start
    best_sha256 = compute_file_sha256(best_checkpoint_path)

    # Save complete experiment results
    experiment_results = {
        "experiment_id": experiment_id,
        "description": description,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "total_training_samples": len(train_samples),
        "class_balance": {"real": num_real, "synthetic": num_fake},
        "best_epoch": best_epoch,
        "best_composite_score": round(best_composite_score, 4),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_sha256": best_sha256,
        "total_duration_seconds": round(total_duration, 1),
        "gpu_memory_used_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0,
        "history": history,
    }

    results_json = output_dir / "experiment_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(f"Experiment {experiment_id} finished in {total_duration:.1f}s. Results written to {results_json}")
    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SignalScope additional training experiments")
    parser.add_argument("--experiment", type=str, required=True, choices=["exp1", "exp2", "exp3"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    run_experiment(args.experiment, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
