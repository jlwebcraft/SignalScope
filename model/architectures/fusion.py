"""Multimodal Evidence Fusion Architecture for SignalScope.

Combines RGB Spatial branch (ConvNeXt-Tiny) with Fourier Frequency branch
into a unified calibrated authenticity classifier.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from model.architectures.convnext import ConvNeXtTinyDetector
from model.architectures.frequency import FrequencyFeatureExtractor


class DualBranchFusionDetector(nn.Module):
    """Dual-branch network fusing spatial RGB anomalies with frequency spectrum artifacts."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        frequency_feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_branch = ConvNeXtTinyDetector(
            pretrained=pretrained,
            num_classes=1,
            dropout_rate=dropout_rate,
        )
        self.frequency_branch = FrequencyFeatureExtractor(
            in_channels=1,
            feature_dim=frequency_feature_dim,
        )

        # Spatial feature dimension from ConvNeXt backbone
        spatial_dim = getattr(self.spatial_branch.backbone, "num_features", 768)

        # Fusion projection head
        fusion_in_dim = spatial_dim + frequency_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both branches.

        Returns:
            Tuple of (fused_logits, spatial_logits, frequency_logits).
        """
        # Spatial branch
        spatial_feat = self.spatial_branch.extract_features(x)
        spatial_logits = self.spatial_branch.classifier(spatial_feat)

        # Frequency branch
        fft_mag = self.frequency_branch.compute_fft_magnitude(x)
        freq_feat = self.frequency_branch.spec_conv(fft_mag)
        freq_logits = self.frequency_branch.classifier(freq_feat)

        # Concatenate and fuse
        combined = torch.cat([spatial_feat, freq_feat], dim=-1)
        fused_logits = self.fusion_head(combined)

        return fused_logits, spatial_logits, freq_logits
