"""Spectral Evidence Visualization module for SignalScope.

Extracts 2D Fast Fourier Transform (FFT) log-magnitude spectrum maps and 1D
radial azimuthal energy profiles directly from native 32x32 image inputs.

Note:
- Bright regions in the frequency spectrum indicate localized periodic patterns,
  sinc/checkerboard grid structures, or compression boundaries.
- Described as: "Observed spectral structure used by the frequency analysis component."
- Does not claim that every high-frequency marker is causal proof of generation.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from model.architectures.frequency import compute_azimuthal_average, compute_fft_2d
from model.dataset import get_native_transforms


def extract_spectral_features(
    image_pil: Image.Image,
) -> Tuple[np.ndarray, np.ndarray, float, Image.Image]:
    """Computes 2D FFT log-magnitude spectrum, radial energy profile, and visual map.

    Args:
        image_pil: Native 32x32 RGB PIL image.

    Returns:
        (fft_2d_np, radial_profile_np, hf_energy_ratio, spectrum_pil_image)
    """
    native_tf = get_native_transforms(image_size=32, is_training=False)
    x_native = native_tf(image_pil).unsqueeze(0)  # (1, 3, 32, 32)

    with torch.no_grad():
        # Centered log-magnitude spectrum in [0, 1]
        fft_tensor = compute_fft_2d(x_native, shift=True, normalize=True)  # (1, 1, 32, 32)
        fft_2d = fft_tensor.squeeze(0).squeeze(0)  # (32, 32)
        radial_profile = compute_azimuthal_average(fft_2d, num_bins=16)

    fft_2d_np = fft_2d.cpu().numpy()
    radial_np = radial_profile.cpu().numpy()

    # High-frequency energy ratio: energy in outer half of radial spectrum
    total_energy = float(np.sum(radial_np))
    outer_energy = float(np.sum(radial_np[len(radial_np) // 2:]))
    hf_energy_ratio = float(outer_energy / max(1e-8, total_energy))
    hf_energy_ratio = float(max(0.0, min(1.0, hf_energy_ratio)))

    # Colorize 2D FFT map with 'inferno' colormap for intuitive inspection
    cmap = matplotlib.colormaps["inferno"]
    colored_spec = cmap(fft_2d_np)[:, :, :3]
    spec_pil = Image.fromarray((colored_spec * 255).astype(np.uint8), mode="RGB")

    return fft_2d_np, radial_np, round(hf_energy_ratio, 4), spec_pil


def render_multimodal_evidence_panel(
    original_pil: Image.Image,
    gradcam_overlay_pil: Image.Image,
    spectrum_pil: Image.Image,
    radial_profile_np: np.ndarray,
    verdict: str,
    calibrated_prob: float,
    stability_score: float,
    save_path: Path,
) -> None:
    """Renders a comprehensive side-by-side 4-panel visual evidence dashboard."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    # 1. Original
    axes[0].imshow(original_pil.resize((224, 224), Image.Resampling.NEAREST))
    axes[0].set_title("Input Image (Native 32x32)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM Overlay
    axes[1].imshow(gradcam_overlay_pil.resize((224, 224), Image.Resampling.BILINEAR))
    axes[1].set_title("Spatial Attribution (Grad-CAM)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. 2D FFT Spectrum
    axes[2].imshow(spectrum_pil.resize((224, 224), Image.Resampling.NEAREST))
    axes[2].set_title("2D FFT Spectrum (Centered)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Radial Energy Decay
    r_bins = np.arange(len(radial_profile_np))
    axes[3].plot(r_bins, radial_profile_np, "o-", color="#ff7f0e", linewidth=2, markersize=5)
    axes[3].set_xlabel("Radial Frequency Radius (DC -> Nyquist)", fontsize=9)
    axes[3].set_ylabel("Spectral Energy", fontsize=9)
    axes[3].set_title("Azimuthal Radial Decay", fontsize=11, fontweight="bold")
    axes[3].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(
        f"SignalScope Multimodal Evidence Panel | Verdict: {verdict.upper()} | "
        f"Calibrated Prob: {calibrated_prob:.3f} | Stability: {stability_score:.3f}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
