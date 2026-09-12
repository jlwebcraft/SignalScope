"""Multimodal Evidence Fusion Architecture for SignalScope.

Combines RGB Spatial branch (ConvNeXt-Tiny on 224x224 upscaled input)
with Fourier Frequency branch (FrequencyCNNBranch on native 32x32 spectrum)
via simple learned feature concatenation and MLP classification head.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from model.architectures.convnext import ConvNeXtTinyDetector
from model.architectures.frequency import FrequencyCNNBranch, compute_fft_2d


class DualBranchFusionDetector(nn.Module):
    """Dual-branch network fusing spatial RGB anomalies with frequency spectrum artifacts.

    Architecture:
                     Original 32x32
                           |
                +----------+----------+
                |                     |
                v                     v
         resize 224x224             FFT
                |                     |
                v                     v
           ConvNeXt             Frequency CNN
                |                     |
                v                     v
          RGB embedding        Frequency embedding
                |                     |
                +----------+----------+
                           |
                     concatenate
                           |
                        MLP head
                           |
                     binary output
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        frequency_feature_dim: int = 128,
        spatial_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Spatial branch (ConvNeXt-Tiny)
        self.spatial_branch = ConvNeXtTinyDetector(
            pretrained=pretrained,
            num_classes=1,
            dropout_rate=dropout_rate,
        )

        # Optionally load verified spatial baseline checkpoint
        if spatial_checkpoint:
            ckpt = torch.load(spatial_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            self.spatial_branch.load_state_dict(state_dict)

        # Frequency branch (Lightweight 3-stage CNN)
        self.frequency_branch = FrequencyCNNBranch(
            in_channels=1,
            feature_dim=frequency_feature_dim,
            dropout_rate=dropout_rate,
        )

        # Dimensions
        self.spatial_dim = getattr(self.spatial_branch.backbone, "num_features", 768)
        self.frequency_feature_dim = frequency_feature_dim
        fusion_in_dim = self.spatial_dim + frequency_feature_dim

        # Simple learned fusion MLP head (no complex attention/gating)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_native: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through spatial and frequency branches.

        Args:
            x_spatial: Spatial RGB image tensor of shape (B, 3, 224, 224).
            x_native: Native RGB image tensor of shape (B, 3, 32, 32). If provided,
                      FFT log-magnitude is computed automatically.
            x_freq: Precomputed frequency map tensor of shape (B, 1, 32, 32).
            return_aux: Whether to return auxiliary logits from individual branches.

        Returns:
            fused_logits of shape (B, 1) or (fused_logits, spatial_logits, freq_logits).
        """
        # 1. Spatial branch feature extraction
        spatial_feat = self.spatial_branch.extract_features(x_spatial)

        # 2. Frequency branch feature extraction
        if x_freq is not None:
            freq_map = x_freq
        elif x_native is not None:
            freq_map = compute_fft_2d(x_native, shift=True, normalize=True)
        else:
            raise ValueError("Either x_native or x_freq must be provided to DualBranchFusionDetector.")

        freq_feat = self.frequency_branch.extract_features(freq_map)

        # 3. Concatenate representations
        fused_features = torch.cat([spatial_feat, freq_feat], dim=-1)

        # 4. Fusion MLP classification
        fused_logits = self.fusion_head(fused_features)

        if return_aux:
            spatial_logits = self.spatial_branch.classifier(spatial_feat)
            freq_logits = self.frequency_branch.classifier(freq_feat)
            return fused_logits, spatial_logits, freq_logits

        return fused_logits

    def predict_probability(
        self,
        x_spatial: torch.Tensor,
        x_native: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes sigmoid output probability."""
        logits = self.forward(x_spatial, x_native=x_native, x_freq=x_freq, return_aux=False)
        return torch.sigmoid(logits)
