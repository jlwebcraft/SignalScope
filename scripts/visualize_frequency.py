"""Spectral analysis visualization utility for SignalScope.

Demonstrates 2D FFT log-magnitude and 2D DCT representations
computed directly on native 32x32 pixel matrices for authentic vs synthetic media.
Saves side-by-side diagnostic figures to report/figures/spectral_comparison_fft_dct.png.
"""

import os
from pathlib import Path
import sys

# Prevent OpenMP multiple runtime initialization error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from app.utils.logger import logger
from model.architectures.frequency import (
    compute_azimuthal_average,
    compute_dct_2d,
    compute_fft_2d,
)
from model.dataset import get_native_transforms, scan_dataset_directory


def generate_spectral_comparison(
    real_img_path: Path,
    fake_img_path: Path,
    output_path: Path,
) -> None:
    """Generates comparative 2D FFT, 2D DCT, and radial decay curves for Real vs Fake."""
    logger.info("Generating spectral comparison...")
    logger.info(f"  Real sample: {real_img_path}")
    logger.info(f"  Fake sample: {fake_img_path}")

    # Load images as native PIL and convert to tensor
    transform = get_native_transforms(image_size=32, is_training=False)
    pil_real = Image.open(real_img_path).convert("RGB")
    pil_fake = Image.open(fake_img_path).convert("RGB")

    tensor_real = transform(pil_real).unsqueeze(0)  # (1, 3, 32, 32)
    tensor_fake = transform(pil_fake).unsqueeze(0)  # (1, 3, 32, 32)

    # Compute FFT spectra
    fft_real = compute_fft_2d(tensor_real, shift=True, normalize=True).squeeze().numpy()
    fft_fake = compute_fft_2d(tensor_fake, shift=True, normalize=True).squeeze().numpy()

    # Compute DCT spectra
    dct_real = compute_dct_2d(tensor_real, normalize=True).squeeze().numpy()
    dct_fake = compute_dct_2d(tensor_fake, normalize=True).squeeze().numpy()

    # Compute Radial Azimuthal Profiles
    radial_real = compute_azimuthal_average(torch.from_numpy(fft_real)).numpy()
    radial_fake = compute_azimuthal_average(torch.from_numpy(fft_fake)).numpy()
    radius_bins = np.arange(len(radial_real))

    # Plotting 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Authentic Real
    axes[0, 0].imshow(pil_real.resize((128, 128), Image.Resampling.NEAREST))
    axes[0, 0].set_title("Authentic Real (32x32)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    im_fft_real = axes[0, 1].imshow(fft_real, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("2D FFT Log-Magnitude\n(DC Centered)", fontsize=11, fontweight="bold")
    plt.colorbar(im_fft_real, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_dct_real = axes[0, 2].imshow(dct_real, cmap="plasma", vmin=0, vmax=1)
    axes[0, 2].set_title("2D DCT-II Representation\n(Top-Left DC)", fontsize=11, fontweight="bold")
    plt.colorbar(im_dct_real, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].plot(radius_bins, radial_real, "o-", color="#16a34a", lw=2, label="Real Radial Decay")
    axes[0, 3].set_title("Radial Energy Profile", fontsize=11, fontweight="bold")
    axes[0, 3].set_xlabel("Frequency Radius (r)", fontsize=10)
    axes[0, 3].set_ylabel("Normalized Energy", fontsize=10)
    axes[0, 3].set_ylim(0, 1)
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].legend()

    # Row 2: Synthetic / AI
    axes[1, 0].imshow(pil_fake.resize((128, 128), Image.Resampling.NEAREST))
    axes[1, 0].set_title("Synthetic AI (32x32)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    im_fft_fake = axes[1, 1].imshow(fft_fake, cmap="inferno", vmin=0, vmax=1)
    axes[1, 1].set_title("2D FFT Log-Magnitude\n(Periodic Artifacts)", fontsize=11, fontweight="bold")
    plt.colorbar(im_fft_fake, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_dct_fake = axes[1, 2].imshow(dct_fake, cmap="plasma", vmin=0, vmax=1)
    axes[1, 2].set_title("2D DCT-II Representation\n(High-Freq Grid Peaks)", fontsize=11, fontweight="bold")
    plt.colorbar(im_dct_fake, ax=axes[1, 2], fraction=0.046, pad=0.04)

    axes[1, 3].plot(radius_bins, radial_fake, "s-", color="#dc2626", lw=2, label="Synthetic Radial Decay")
    axes[1, 3].set_title("Radial Energy Profile", fontsize=11, fontweight="bold")
    axes[1, 3].set_xlabel("Frequency Radius (r)", fontsize=10)
    axes[1, 3].set_ylabel("Normalized Energy", fontsize=10)
    axes[1, 3].set_ylim(0, 1)
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].legend()

    plt.suptitle(
        "SignalScope: Native 32x32 Spectral Analysis (2D FFT vs 2D DCT)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Spectral analysis plot saved to {output_path}")


def main() -> int:
    train_dir = Path(r"C:\Programming\SignalScope-data\train")
    real_sample = train_dir / "REAL" / "0000 (2).jpg"
    fake_sample = train_dir / "FAKE" / "1000 (2).jpg"

    if not real_sample.exists() or not fake_sample.exists():
        logger.warning("Default train sample paths not found. Scanning train directory...")
        samples = scan_dataset_directory(train_dir)
        reals = [s[0] for s in samples if s[1] == 0]
        fakes = [s[0] for s in samples if s[1] == 1]
        if not reals or not fakes:
            logger.error("Could not find real and fake samples.")
            return 1
        real_sample = Path(reals[0])
        fake_sample = Path(fakes[0])

    output_plot = PROJECT_ROOT / "report" / "figures" / "spectral_comparison_fft_dct.png"
    generate_spectral_comparison(real_sample, fake_sample, output_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
