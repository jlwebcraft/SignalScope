"""Frequency-domain branch architecture for SignalScope.

Extracts spectral representations using 2D Fast Fourier Transform (FFT)
and Discrete Cosine Transform (DCT) to expose periodic generative artifacts.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class FrequencyFeatureExtractor(nn.Module):
    """Computes 2D Fourier log-magnitude spectrum and azimuthal energy distributions."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Convolutional encoder for Fourier spectrum
        self.spec_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(feature_dim, 1)

    @staticmethod
    def compute_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Computes centered 2D Fast Fourier Transform log-magnitude spectrum.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Log-magnitude spectrum tensor of shape (B, 1, H, W).
        """
        # Convert RGB to grayscale luminance if needed
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        else:
            gray = x

        # 2D Fast Fourier Transform
        fft2 = torch.fft.fft2(gray)
        fft2_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
        magnitude = torch.abs(fft2_shifted)
        # Log scaling: log(1 + |FFT|)
        log_mag = torch.log1p(magnitude)

        # Normalize per image to [0, 1] range for network stability
        min_v = log_mag.amin(dim=(-2, -1), keepdim=True)
        max_v = log_mag.amax(dim=(-2, -1), keepdim=True)
        norm_mag = (log_mag - min_v) / (max_v - min_v + 1e-8)
        return norm_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes FFT and extracts frequency-domain features and logits."""
        fft_mag = self.compute_fft_magnitude(x)
        features = self.spec_conv(fft_mag)
        logits = self.classifier(features)
        return logits
