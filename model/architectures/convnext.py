"""ConvNeXt-Tiny RGB Spatial Architecture for SignalScope."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ConvNeXtTinyDetector(nn.Module):
    """ConvNeXt-Tiny spatial branch for synthetic media detection.

    Extracts deep hierarchical spatial features from RGB inputs,
    identifying subtle pixel anomalies, generative blending borders,
    and texture synthesis artifacts.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if HAS_TIMM:
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                num_classes=0,  # Remove default classifier head to extract features
            )
            feature_dim = self.backbone.num_features
        else:
            # Fallback sequential architecture if timm is not yet installed in local environment
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                nn.LayerNorm([96, 56, 56]),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            feature_dim = 96

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )

        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate feature representation."""
        if HAS_TIMM:
            return self.backbone(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Returns synthetic probability via sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def load_weights(self, checkpoint_path: str) -> None:
        """Loads model weights safely."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.load_state_dict(state_dict, strict=False)


def build_convnext_tiny(
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    checkpoint_path: Optional[str] = None,
) -> ConvNeXtTinyDetector:
    """Factory helper to build ConvNeXtTinyDetector."""
    return ConvNeXtTinyDetector(
        pretrained=pretrained,
        num_classes=1,
        dropout_rate=dropout_rate,
        checkpoint_path=checkpoint_path,
    )
