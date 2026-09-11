"""Model training pipeline for SignalScope.

Executes reproducible training runs with:
- Configurable random seed
- Checkpointing and experiment tracking
- Validation monitoring (ROC-AUC, Macro-F1, FPR)
- Mixed precision support
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from app.utils.logger import logger


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed fixed to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_experiment_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    commit_hash: str = "unknown",
) -> None:
    """Saves experiment run parameters and final evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": commit_hash,
        "config": config,
        "final_metrics": metrics,
    }
    meta_path = output_dir / "experiment_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Experiment metadata recorded at {meta_path}")


def train(config_path: str) -> None:
    """Main training execution function."""
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    logger.info(f"Initiating SignalScope training on device: {device}")
    logger.info(f"Configuration: {config_path}")

    # Checkpoint output directory
    checkpoint_dir = Path(config.get("training", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training pipeline initialized. Awaiting dataset ingest in Phase 2. "
        f"Target architecture: {config.get('model', {}).get('name', 'convnext_tiny')}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="model/configs/default.yaml",
        help="Path to YAML training configuration",
    )
    args = parser.parse_args()
    train(args.config)
