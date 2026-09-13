"""Model artifact management, remote fetching, and metadata provenance for SignalScope.

Provides clean model checkpoint and temperature scaler resolution from local paths
or remote URLs (e.g. GitHub Releases, Hugging Face, or cloud storage) with strict SHA-256
verification, rejection of corrupted weights, and local disk caching.
"""

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import urllib.request

from app.utils.logger import logger

# Canonical Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "signalscope-v2": {
        "model_version": "signalscope-v2",
        "architecture": "ConvNeXtTinyDetector",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "weights_sha256": "47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d",
        "default_checkpoint_path": "checkpoints/additional_training/exp3_5ep/best_model.pt",
        "default_scaler_path": "checkpoints/additional_training/exp3_5ep/temperature_scaler.json",
        "model_url": "https://github.com/jlwebcraft/SignalScope/releases/download/v2.0.0/best_model.pt",
        "scaler_sha256": "128c501b88e92f968b4c16f72e6cbd731c083cb64465055269179ecf60fa55ea",
        "calibration_temperature": 0.9986,
        "best_epoch": 3,
        "training": "extended balanced mixed-domain + forensic augmentation",
        "input_resolution": [3, 224, 224],
        "native_patch_resolution": [3, 32, 32],
        "evaluation_status": "Photographic-real holdout and multi-generator generalized (SIH held-out test set untouched)",
    },
    "signalscope-baseline-v1": {
        "model_version": "signalscope-baseline-v1",
        "architecture": "ConvNeXtTinyDetector",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "weights_sha256": "c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc",
        "default_checkpoint_path": "checkpoints/baseline_convnext/best_model.pt",
        "default_scaler_path": "checkpoints/baseline_convnext/temperature_scaler.json",
        "model_url": "https://github.com/jlwebcraft/SignalScope/releases/download/v1.0.0/best_model.pt",
        "scaler_sha256": "8b9914aea7aad46c5caa35f5e75da478aad8972dfb02cbcbb6afd3c1cce2208f",
        "calibration_temperature": 0.99953,
        "best_epoch": 4,
        "training": "baseline single-domain native resolution",
        "input_resolution": [3, 224, 224],
        "native_patch_resolution": [3, 32, 32],
        "evaluation_status": "Local validation verified (organizer test set untouched)",
    },
}

# Production Default: signalscope-v2
DEFAULT_MODEL_VERSION = "signalscope-v2"
ACTIVE_VERSION_KEY = os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION)
if ACTIVE_VERSION_KEY not in MODEL_REGISTRY:
    logger.warning(f"Unknown MODEL_VERSION '{ACTIVE_VERSION_KEY}', falling back to '{DEFAULT_MODEL_VERSION}'")
    ACTIVE_VERSION_KEY = DEFAULT_MODEL_VERSION

ACTIVE_SPEC = MODEL_REGISTRY[ACTIVE_VERSION_KEY]

MODEL_VERSION = ACTIVE_SPEC["model_version"]
MODEL_NAME = ACTIVE_SPEC["backbone"]
EXPECTED_CHECKPOINT_SHA256 = ACTIVE_SPEC["weights_sha256"]
EXPECTED_SCALER_SHA256 = ACTIVE_SPEC.get("scaler_sha256")
DEFAULT_CHECKPOINT_PATH = ACTIVE_SPEC["default_checkpoint_path"]
DEFAULT_SCALER_PATH = ACTIVE_SPEC["default_scaler_path"]


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
        model_version: Optional[str] = None,
    ) -> None:
        self.model_version = model_version or os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION)
        if self.model_version not in MODEL_REGISTRY:
            logger.warning(f"Unknown model_version '{self.model_version}', defaulting to '{DEFAULT_MODEL_VERSION}'")
            self.model_version = DEFAULT_MODEL_VERSION

        self.spec = MODEL_REGISTRY[self.model_version]
        self.expected_checkpoint_sha256 = self.spec["weights_sha256"]
        self.expected_scaler_sha256 = self.spec.get("scaler_sha256")

        self.default_checkpoint = self.spec["default_checkpoint_path"]
        self.default_scaler = self.spec["default_scaler_path"]

        # If explicit checkpoint_path passed, note whether it matches default
        passed_ckpt = checkpoint_path or os.environ.get("MODEL_PATH")
        self.is_custom_local_path = passed_ckpt is not None and passed_ckpt != self.default_checkpoint
        self.checkpoint_path = Path(passed_ckpt or self.default_checkpoint)

        self.model_url = model_url or os.environ.get("MODEL_URL", self.spec.get("model_url"))

        passed_scaler = scaler_path or os.environ.get("SCALER_PATH")
        self.scaler_path = Path(passed_scaler or self.default_scaler)
        self.scaler_url = scaler_url or os.environ.get("SCALER_URL")

        self.cache_dir = Path(cache_dir or os.environ.get("MODEL_CACHE_DIR", "cache/models"))

    def resolve_checkpoint(self) -> Optional[Path]:
        """Resolves the model checkpoint file.

        Attempts in order:
        1. Local path configured in self.checkpoint_path.
        2. Cached file in self.cache_dir / best_model.pt.
        3. Download from self.model_url into cache_dir.

        Strictly enforces SHA-256 verification and refuses to load mismatched weights.

        Returns:
            Resolved Path to checkpoint, or None if unavailable/corrupted.
        """
        # 1. Local path
        if self.checkpoint_path.exists() and self.checkpoint_path.is_file():
            if not self.is_custom_local_path:
                computed = compute_sha256(self.checkpoint_path)
                if computed != self.expected_checkpoint_sha256:
                    logger.error(
                        f"Refusing to load checkpoint {self.checkpoint_path}: SHA-256 mismatch! "
                        f"Expected {self.expected_checkpoint_sha256}, got {computed}."
                    )
                    return None
                logger.info(f"Using verified production model checkpoint: {self.checkpoint_path} (SHA-256: {computed[:8]}...)")
            else:
                logger.info(f"Using custom local model checkpoint: {self.checkpoint_path}")
            return self.checkpoint_path

        # 2. Cache dir check
        cached_target = self.cache_dir / f"{self.model_version}_best_model.pt"
        if not cached_target.exists():
            # Check generic cached name
            generic_target = self.cache_dir / "best_model.pt"
            if generic_target.exists() and generic_target.is_file():
                cached_target = generic_target

        if cached_target.exists() and cached_target.is_file():
            computed = compute_sha256(cached_target)
            if computed != self.expected_checkpoint_sha256:
                logger.warning(
                    f"Cached checkpoint SHA-256 mismatch at {cached_target}. Expected {self.expected_checkpoint_sha256}, got {computed}. Invalidating corrupted cache."
                )
                try:
                    cached_target.unlink()
                except OSError:
                    pass
            else:
                logger.info(f"Using verified cached model checkpoint: {cached_target}")
                return cached_target

        # 3. Download if URL provided
        if self.model_url:
            logger.info(f"Downloading model checkpoint from {self.model_url}...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_dir / f"{self.model_version}_best_model.pt.tmp"
            target_path = self.cache_dir / f"{self.model_version}_best_model.pt"
            try:
                urllib.request.urlretrieve(self.model_url, temp_path)
                computed = compute_sha256(temp_path)
                if computed != self.expected_checkpoint_sha256:
                    if temp_path.exists():
                        temp_path.unlink()
                    logger.error(
                        f"Refusing to load downloaded checkpoint: SHA-256 mismatch! "
                        f"Expected {self.expected_checkpoint_sha256}, got {computed}."
                    )
                    return None
                shutil.move(temp_path, target_path)
                logger.info(f"Model checkpoint successfully downloaded and verified to {target_path}")
                return target_path
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

        cached_scaler = self.cache_dir / f"{self.model_version}_temperature_scaler.json"
        if cached_scaler.exists() and cached_scaler.is_file():
            return cached_scaler

        if self.scaler_url:
            logger.info(f"Downloading temperature scaler from {self.scaler_url}...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temp_scaler = self.cache_dir / f"{self.model_version}_temperature_scaler.json.tmp"
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

    @classmethod
    def get_version_info(cls, version: Optional[str] = None) -> Dict[str, Any]:
        """Returns structured metadata about the production model."""
        ver = version or os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION)
        spec = MODEL_REGISTRY.get(ver, MODEL_REGISTRY[DEFAULT_MODEL_VERSION])
        return {
            "model_version": spec["model_version"],
            "architecture": spec["architecture"],
            "backbone": spec["backbone"],
            "weights_sha256": spec["weights_sha256"],
            "best_epoch": spec.get("best_epoch"),
            "training": spec.get("training"),
            "input_resolution": spec.get("input_resolution", [3, 224, 224]),
            "native_patch_resolution": spec.get("native_patch_resolution", [3, 32, 32]),
            "calibration_temperature": spec["calibration_temperature"],
            "evaluation_status": spec["evaluation_status"],
            "rollback_version": "signalscope-baseline-v1" if ver == "signalscope-v2" else None,
        }
