"""Frequency-domain branch architecture and spectral preprocessing for SignalScope.

Extracts spectral representations using 2D Fast Fourier Transform (FFT)
and Discrete Cosine Transform (DCT) to expose periodic generative artifacts.
Directly consumes native 32x32 resolution images to prevent upsampling interpolation bias.
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_grayscale_luminance(x: torch.Tensor) -> torch.Tensor:
    """Converts RGB tensor to standard Rec.601 grayscale luminance.

    Args:
        x: Image tensor of shape (B, C, H, W) or (C, H, W) where C in {1, 3}.

    Returns:
        Grayscale tensor of shape (B, 1, H, W) or (1, H, W).
    """
    if x.ndim == 3:
        if x.shape[0] == 3:
            return 0.2989 * x[0:1, :, :] + 0.5870 * x[1:2, :, :] + 0.1140 * x[2:3, :, :]
        return x
    elif x.ndim == 4:
        if x.shape[1] == 3:
            return 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        return x
    else:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {x.shape}")


def compute_fft_2d(
    x: torch.Tensor,
    shift: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Computes centered 2D Fast Fourier Transform log-magnitude spectrum.

    Args:
        x: Input image tensor (B, C, H, W) or (C, H, W).
        shift: Whether to apply fftshift so DC component (0, 0) is centered.
        normalize: Whether to apply per-image min-max normalization to [0, 1].

    Returns:
        Log-magnitude spectrum tensor with shape (B, 1, H, W) or (1, H, W).
    """
    was_3d = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        was_3d = True

    # Convert RGB to luminance
    gray = rgb_to_grayscale_luminance(x)

    # 2D Fast Fourier Transform
    fft2 = torch.fft.fft2(gray)
    if shift:
        fft2 = torch.fft.fftshift(fft2, dim=(-2, -1))

    # Magnitude and numerically stable logarithmic scaling: log(1 + |X|)
    magnitude = torch.abs(fft2)
    log_mag = torch.log1p(magnitude)

    if normalize:
        min_v = log_mag.amin(dim=(-2, -1), keepdim=True)
        max_v = log_mag.amax(dim=(-2, -1), keepdim=True)
        norm_mag = (log_mag - min_v) / (max_v - min_v + 1e-8)
    else:
        norm_mag = log_mag

    if was_3d:
        norm_mag = norm_mag.squeeze(0)

    return norm_mag


def get_dct2_basis_matrix(
    n: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes the orthonormal N x N 2D Discrete Cosine Transform (DCT Type-II) matrix.

    Basis:
        D_{k, n} = sqrt(1/N) if k=0 else sqrt(2/N) * cos(pi * (2n + 1) * k / (2N))
    """
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)

    weights = torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
    d = weights * torch.cos((math.pi * (2 * idx + 1) * k) / (2.0 * n))
    d[0, :] = torch.sqrt(torch.tensor(1.0 / n, device=device, dtype=dtype))
    return d


def compute_dct_2d(
    x: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Computes orthonormal 2D Discrete Cosine Transform log-magnitude spectrum.

    Args:
        x: Input image tensor (B, C, H, W) or (C, H, W).
        normalize: Whether to apply per-sample min-max normalization to [0, 1].

    Returns:
        Log-magnitude DCT spectrum tensor with shape (B, 1, H, W) or (1, H, W).
    """
    was_3d = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        was_3d = True

    gray = rgb_to_grayscale_luminance(x)
    _, _, h, w = gray.shape

    # Construct DCT projection matrices for dimensions H and W
    d_h = get_dct2_basis_matrix(h, device=gray.device, dtype=gray.dtype)
    d_w = get_dct2_basis_matrix(w, device=gray.device, dtype=gray.dtype)

    # 2D DCT: D_h @ Y @ D_w^T
    # gray is (B, 1, H, W)
    # Using torch.matmul
    dct_h = torch.matmul(d_h, gray)
    dct_2d = torch.matmul(dct_h, d_w.t())

    magnitude = torch.abs(dct_2d)
    log_dct = torch.log1p(magnitude)

    if normalize:
        min_v = log_dct.amin(dim=(-2, -1), keepdim=True)
        max_v = log_dct.amax(dim=(-2, -1), keepdim=True)
        norm_dct = (log_dct - min_v) / (max_v - min_v + 1e-8)
    else:
        norm_dct = log_dct

    if was_3d:
        norm_dct = norm_dct.squeeze(0)

    return norm_dct


def compute_azimuthal_average(
    spectrum_2d: torch.Tensor,
    num_bins: Optional[int] = None,
) -> torch.Tensor:
    """Computes 1D radial azimuthal energy decay from DC center to Nyquist edge.

    Args:
        spectrum_2d: Centered 2D spectrum map of shape (H, W).
        num_bins: Number of radial bins (defaults to H // 2).

    Returns:
        1D tensor of shape (num_bins,) representing radial energy decay.
    """
    if spectrum_2d.ndim != 2:
        raise ValueError(f"Expected 2D matrix (H, W), got {spectrum_2d.shape}")

    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    if num_bins is None:
        num_bins = min(cy, cx)

    y, x = torch.meshgrid(
        torch.arange(h, device=spectrum_2d.device, dtype=torch.float32),
        torch.arange(w, device=spectrum_2d.device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    radial_profile = torch.zeros(num_bins, device=spectrum_2d.device, dtype=spectrum_2d.dtype)
    r_int = torch.round(r).long()

    for bin_idx in range(num_bins):
        mask = r_int == bin_idx
        if mask.any():
            radial_profile[bin_idx] = spectrum_2d[mask].mean()

    return radial_profile


class FrequencyCNNBranch(nn.Module):
    """Lightweight Convolutional Neural Network dedicated to frequency-domain features.

    Operates on native-resolution frequency maps (e.g. 1x32x32) without
    introducing upsampling interpolation artifacts. Produces a compact 128-d
    frequency embedding and auxiliary classification logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # 3-Stage Convolutional Encoder tailored for 32x32 resolution
        self.encoder = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Projection head to compact frequency embedding
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        # Auxiliary standalone classification head
        self.classifier = nn.Linear(feature_dim, 1)

    def extract_features(self, freq_map: torch.Tensor) -> torch.Tensor:
        """Extracts 128-dimensional frequency embedding from a frequency map."""
        conv_out = self.encoder(freq_map)
        embeddings = self.fc(conv_out)
        return embeddings

    def forward(self, freq_map: torch.Tensor) -> torch.Tensor:
        """Forward pass computing auxiliary classification logits from frequency map."""
        features = self.extract_features(freq_map)
        logits = self.classifier(features)
        return logits

    @staticmethod
    def compute_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible helper calling compute_fft_2d."""
        return compute_fft_2d(x, shift=True, normalize=True)


# Backward-compatible alias
FrequencyFeatureExtractor = FrequencyCNNBranch
