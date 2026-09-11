"""SignalScope system health, environment verification, and ML stack diagnostic utility.

Verifies:
- Python 3.13 isolated environment
- PyTorch + CUDA + GPU acceleration
- Repository directory structure
- YAML experiment configurations
- Official dataset presence check
- ConvNeXt-Tiny model instantiation and forward-pass smoke test
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import logger
from model.train import load_config


def check_python_environment() -> bool:
    """Verifies Python version and key packages."""
    logger.info("Checking Python environment...")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Python Version: {sys.version.split()[0]}")

    core_packages = ["PIL", "numpy", "yaml", "sklearn", "scipy", "timm", "fastapi"]
    all_ok = True
    for pkg in core_packages:
        try:
            __import__(pkg)
            logger.info(f"  [OK] {pkg}")
        except ImportError:
            logger.error(f"  [MISSING] {pkg}")
            all_ok = False

    # Check PyTorch & GPU
    try:
        import torch
        import torchvision
        import timm

        logger.info(f"  [OK] torch v{torch.__version__}")
        logger.info(f"  [OK] torchvision v{torchvision.__version__}")
        logger.info(f"  [OK] timm v{timm.__version__}")

        cuda_avail = torch.cuda.is_available()
        logger.info(f"  CUDA Available: {cuda_avail}")
        if cuda_avail:
            logger.info(f"  CUDA Runtime: {torch.version.cuda}")
            logger.info(f"  Device Count: {torch.cuda.device_count()}")
            logger.info(f"  Device Name: {torch.cuda.get_device_name(0)}")
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            logger.info(f"  GPU VRAM: {vram_gb} GB")
        else:
            logger.warning("  [NOTE] CUDA is not available. PyTorch running on CPU.")
    except ImportError as e:
        logger.error(f"  [FAIL] PyTorch stack import error: {e}")
        all_ok = False

    return all_ok


def check_directory_structure() -> bool:
    """Ensures required repository structure exists."""
    logger.info("Checking directory structure...")
    required_dirs = [
        "app",
        "app/api",
        "app/inference",
        "app/services",
        "app/utils",
        "model",
        "model/architectures",
        "model/losses",
        "model/configs",
        "report",
        "notebooks",
        "scripts",
        "tests",
        "frontend",
    ]
    all_ok = True
    for d in required_dirs:
        p = PROJECT_ROOT / d
        if p.exists() and p.is_dir():
            logger.info(f"  [EXISTS] {d}")
        else:
            logger.error(f"  [MISSING] {d}")
            all_ok = False
    return all_ok


def check_configurations() -> bool:
    """Validates configuration files."""
    logger.info("Checking YAML configuration files...")
    config_paths = [
        "model/configs/default.yaml",
        "model/configs/baseline_convnext.yaml",
    ]
    all_ok = True
    for cp in config_paths:
        p = PROJECT_ROOT / cp
        if not p.exists():
            logger.error(f"  [MISSING] {cp}")
            all_ok = False
            continue
        try:
            cfg = load_config(str(p))
            assert isinstance(cfg, dict)
            logger.info(f"  [VALID] {cp}")
        except Exception as exc:
            logger.error(f"  [INVALID] {cp}: {exc}")
            all_ok = False
    return all_ok


def check_official_dataset() -> bool:
    """Checks whether the official SIH dataset is locally present."""
    logger.info("Checking official dataset presence...")
    data_dir = PROJECT_ROOT / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and any(train_dir.iterdir()):
        logger.info(f"  [FOUND] Official dataset located at {data_dir}")
        return True
    else:
        logger.warning(
            "  [NOT FOUND] Official SIH dataset not found locally — "
            "baseline training cannot begin until the official training data is provided."
        )
        return False


def check_model_forward_pass() -> bool:
    """Smoke test: instantiates ConvNeXt-Tiny and runs a forward pass on a tiny batch."""
    logger.info("Running baseline model forward-pass smoke test...")
    try:
        import torch
        from model.architectures.convnext import build_convnext_tiny
        from model.dataset import get_default_transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Target device: {device}")

        # Instantiate model (try pretrained weights with fallback to uninitialized for offline)
        try:
            model = build_convnext_tiny(pretrained=True)
            logger.info("  [OK] ConvNeXt-Tiny instantiated with pretrained ImageNet weights.")
        except Exception as exc:
            logger.warning(f"  Pretrained weights download failed ({exc}); instantiating without pretraining.")
            model = build_convnext_tiny(pretrained=False)
            logger.info("  [OK] ConvNeXt-Tiny instantiated with random initialization.")

        model = model.to(device)
        model.eval()

        # Generate tiny batch of 2 dummy images through dataset preprocessing
        transform = get_default_transforms(image_size=224, is_training=False)
        dummy_img1 = Image.new("RGB", (256, 256), color=(100, 150, 200))
        dummy_img2 = Image.new("RGB", (300, 300), color=(50, 80, 120))

        tensor1 = transform(dummy_img1).unsqueeze(0)
        tensor2 = transform(dummy_img2).unsqueeze(0)
        batch = torch.cat([tensor1, tensor2], dim=0).to(device)

        assert batch.shape == (2, 3, 224, 224), f"Unexpected batch shape: {batch.shape}"
        logger.info(f"  [OK] Preprocessing produced expected tensor shape: {batch.shape}")

        # Forward pass without gradient computation (zero training)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)

        assert logits.shape == (2, 1), f"Unexpected output shape: {logits.shape}"
        assert probs.shape == (2, 1), f"Unexpected probabilities shape: {probs.shape}"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1] range"

        logger.info(f"  [OK] Forward pass completed on {device}. Logits shape: {logits.shape}")
        logger.info(f"  Sample synthetic probabilities: {probs.cpu().numpy().flatten().tolist()}")
        return True

    except Exception as exc:
        logger.error(f"  [FAIL] Baseline model forward pass failed: {exc}")
        return False


def main() -> int:
    logger.info("==================================================")
    logger.info(" SignalScope Pre-Flight & ML Stack Diagnostics")
    logger.info("==================================================")

    env_ok = check_python_environment()
    dirs_ok = check_directory_structure()
    cfgs_ok = check_configurations()
    check_official_dataset()
    forward_ok = check_model_forward_pass()

    logger.info("==================================================")
    if env_ok and dirs_ok and cfgs_ok and forward_ok:
        logger.info(" System and baseline model checks passed successfully.")
        return 0
    else:
        logger.error(" Diagnostics identified issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
