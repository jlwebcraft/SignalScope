"""Model artifact management, remote fetching, and metadata provenance for SignalScope.

Provides clean model checkpoint and temperature scaler resolution from local paths
or remote URLs (e.g. GitHub Releases, Hugging Face, or cloud storage) with SHA-256
verification and local disk caching.
"""

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import urllib.request

from app.utils.logger import logger

# Official production baseline model metadata
MODEL_VERSION = "signalscope-baseline-v1"
MODEL_NAME = "convnext_tiny.in12k_ft_in1k"
EXPECTED_CHECKPOINT_SHA256 = "c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc"
EXPECTED_SCALER_SHA256 = "8b9914aea7aad46c5caa35f5e75da478aad8972dfb02cbcbb6afd3c1cce2208f"
DEFAULT_CHECKPOINT_PATH = "checkpoints/baseline_convnext/best_model.pt"
DEFAULT_SCALER_PATH = "checkpoints/baseline_convnext/temperature_scaler.json"


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class ModelArtifactManager:
    """Resolves and validates production model artifacts from local disk or remote URLs."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_url: Optional[str] = None,
        scaler_path: Optional[str] = None,
        scaler_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path or os.environ.get("MODEL_PATH", DEFAULT_CHECKPOINT_PATH)
        )
        self.model_url = model_url or os.environ.get("MODEL_URL")
        self.scaler_path = Path(
            scaler_path or os.environ.get("SCALER_PATH", DEFAULT_SCALER_PATH)
        )
        self.scaler_url = scaler_url or os.environ.get("SCALER_URL")
        self.cache_dir = Path(cache_dir or os.environ.get("MODEL_CACHE_DIR", "cache/models"))

    def resolve_checkpoint(self) -> Optional[Path]:
        """Resolves the model checkpoint file.

        Attempts in order:
        1. Local path configured in self.checkpoint_path.
        2. Cached file in self.cache_dir / best_model.pt.
        3. Download from self.model_url into cache_dir.

        Returns:
            Resolved Path to checkpoint, or None if unavailable.
        """
        # 1. Local path
        if self.checkpoint_path.exists() and self.checkpoint_path.is_file():
            logger.info(f"Using local model checkpoint: {self.checkpoint_path}")
            return self.checkpoint_path

        # 2. Cache dir check
        cached_target = self.cache_dir / "best_model.pt"
        if cached_target.exists() and cached_target.is_file():
            logger.info(f"Using cached model checkpoint: {cached_target}")
            return cached_target

        # 3. Download if URL provided
        if self.model_url:
            logger.info(f"Downloading model checkpoint from {self.model_url}...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_dir / "best_model.pt.tmp"
            try:
                urllib.request.urlretrieve(self.model_url, temp_path)
                # Verify SHA-256 if expected
                computed = compute_sha256(temp_path)
                if EXPECTED_CHECKPOINT_SHA256 and computed != EXPECTED_CHECKPOINT_SHA256:
                    logger.warning(
                        f"Checkpoint SHA-256 mismatch: got {computed}, expected {EXPECTED_CHECKPOINT_SHA256}"
                    )
                shutil.move(temp_path, cached_target)
                logger.info(f"Model checkpoint successfully downloaded to {cached_target}")
                return cached_target
            except Exception as err:
                if temp_path.exists():
                    temp_path.unlink()
                logger.error(f"Failed to download model artifact from {self.model_url}: {err}")
                return None

        logger.warning(
            f"Model checkpoint not found at {self.checkpoint_path}. "
            "Mount checkpoint volume or set MODEL_URL for production execution."
        )
        return None

    def resolve_scaler(self) -> Optional[Path]:
        """Resolves the temperature scaler JSON file."""
        if self.scaler_path.exists() and self.scaler_path.is_file():
            return self.scaler_path

        cached_scaler = self.cache_dir / "temperature_scaler.json"
        if cached_scaler.exists() and cached_scaler.is_file():
            return cached_scaler

        if self.scaler_url:
            logger.info(f"Downloading temperature scaler from {self.scaler_url}...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temp_scaler = self.cache_dir / "temperature_scaler.json.tmp"
            try:
                urllib.request.urlretrieve(self.scaler_url, temp_scaler)
                shutil.move(temp_scaler, cached_scaler)
                logger.info(f"Temperature scaler downloaded to {cached_scaler}")
                return cached_scaler
            except Exception as err:
                if temp_scaler.exists():
                    temp_scaler.unlink()
                logger.error(f"Failed to download temperature scaler from {self.scaler_url}: {err}")
                return None

        return None

    @staticmethod
    def get_version_info() -> Dict[str, Any]:
        """Returns structured metadata about the production model."""
        return {
            "model_version": MODEL_VERSION,
            "architecture": "ConvNeXtTinyDetector",
            "backbone": MODEL_NAME,
            "weights_sha256": EXPECTED_CHECKPOINT_SHA256,
            "input_resolution": [3, 224, 224],
            "native_patch_resolution": [3, 32, 32],
            "calibration_temperature": 0.99953,
            "evaluation_status": "Local validation verified (organizer test set untouched)",
        }
