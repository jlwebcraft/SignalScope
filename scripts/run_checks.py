"""SignalScope system health and verification utility.

Runs pre-flight diagnostics on environment, directory structure,
configuration files, and core dependencies.
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
    logger.info(f"Python Version: {sys.version}")

    core_packages = ["PIL", "numpy", "yaml", "sklearn", "scipy"]
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
        logger.info(f"  [OK] torch v{torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        logger.info(f"  CUDA Available: {cuda_avail}")
        if cuda_avail:
            logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("  [NOTE] torch not yet installed in active environment.")

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


def main() -> int:
    logger.info("==================================================")
    logger.info(" SignalScope Pre-Flight Diagnostics")
    logger.info("==================================================")

    env_ok = check_python_environment()
    dirs_ok = check_directory_structure()
    cfgs_ok = check_configurations()

    logger.info("==================================================")
    if env_ok and dirs_ok and cfgs_ok:
        logger.info(" System check passed successfully.")
        return 0
    else:
        logger.error(" System check identified issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
