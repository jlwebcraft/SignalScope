"""Grad-CAM (Gradient-weighted Class Activation Mapping) for ConvNeXt-Tiny.

Identifies spatial image regions that most influenced the classifier's output logit.

Methodology:
- Hooks into the final convolutional stage of ConvNeXt-Tiny: `stages[-1].blocks[-1]`.
- Computes gradients of the target class logit with respect to the 7x7x768 feature maps.
- Computes channel weights via global average pooling of gradients:
    alpha_k = (1/Z) * sum_{i,j} (d y_c / d A^k_{i,j})
- Computes ReLU-activated linear combination:
    L_GradCAM = ReLU(sum_k alpha_k * A^k)
- Normalizes heatmap to [0, 1] and resizes for visualization overlay.

Note:
- The native image is 32x32. The heatmap is computed on the 7x7 feature space of the
  224x224 upscaled input and visually interpolated for display.
- Grad-CAM highlights regions most influential to the model's decision;
  it does NOT prove the presence of generative artifacts.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils.logger import logger
from model.architectures.convnext import ConvNeXtTinyDetector
from model.dataset import get_default_transforms


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ConvNeXtTinyDetector."""

    def __init__(
        self,
        model: ConvNeXtTinyDetector,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.model.eval()

        # Target last convolutional block in stage 3
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
            self.target_layer = model.backbone.stages[-1].blocks[-1]
        else:
            raise ValueError("Could not automatically locate ConvNeXt target layer.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attaches forward and backward hooks to target convolutional layer."""

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where first element is gradient w.r.t layer output
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        x_tensor: torch.Tensor,
        target_class_index: int = 0,
    ) -> np.ndarray:
        """Generates 2D Grad-CAM heatmap in [0, 1] for input tensor (1, 3, 224, 224).

        Args:
            x_tensor: Normalized image tensor of shape (1, 3, 224, 224).
            target_class_index: Target logit index (0 for single-output binary logit).

        Returns:
            Normalized 2D numpy array of shape (H, W) in range [0.0, 1.0].
        """
        self.model.zero_grad()
        logits = self.model(x_tensor)

        # Target score
        if logits.shape[-1] == 1:
            score = logits[0, 0]
        else:
            score = logits[0, target_class_index]

        # Backward pass to compute gradients at target layer
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # activations: (1, C, H_feat, W_feat) e.g. (1, 768, 7, 7)
        # gradients:   (1, C, H_feat, W_feat)
        acts = self.activations[0]
        grads = self.gradients[0]

        # Global average pooling of gradients -> channel weights alpha_k
        alpha_k = torch.mean(grads, dim=(-2, -1), keepdim=True)  # (C, 1, 1)

        # Weighted combination across channels
        cam = torch.sum(alpha_k * acts, dim=0)  # (H_feat, W_feat)

        # Apply ReLU to retain positive influences
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()

        # Min-max normalization
        min_v = np.min(cam_np)
        max_v = np.max(cam_np)
        if max_v - min_v > 1e-8:
            norm_cam = (cam_np - min_v) / (max_v - min_v)
        else:
            norm_cam = np.zeros_like(cam_np)

        return norm_cam.astype(np.float32)

    def generate_overlay(
        self,
        original_pil: Image.Image,
        heatmap_2d: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "jet",
    ) -> Image.Image:
        """Blends 2D Grad-CAM heatmap over original PIL Image.

        Args:
            original_pil: RGB PIL Image (native 32x32 or any size).
            heatmap_2d: 2D numpy array in [0, 1].
            alpha: Transparency weight for heatmap overlay (0 = image only, 1 = heatmap only).
            colormap_name: Matplotlib colormap name.

        Returns:
            RGB PIL Image with overlaid visual heatmap.
        """
        w, h = original_pil.size
        # Resize heatmap to match image dimensions
        heatmap_img = Image.fromarray((heatmap_2d * 255).astype(np.uint8))
        heatmap_resized = np.array(heatmap_img.resize((w, h), Image.Resampling.BILINEAR)) / 255.0

        # Apply colormap
        cmap = matplotlib.colormaps[colormap_name]
        colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # drop alpha channel -> (H, W, 3) in [0, 1]

        orig_np = np.array(original_pil.convert("RGB")).astype(np.float32) / 255.0

        # Blend
        blended = (1.0 - alpha) * orig_np + alpha * colored_heatmap
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")


def compute_spatial_attribution(
    model: ConvNeXtTinyDetector,
    image_pil: Image.Image,
    device: torch.device,
) -> Tuple[np.ndarray, Image.Image, float]:
    """Convenience helper computing Grad-CAM heatmap, visual overlay, and concentration.

    Returns:
        (heatmap_2d, overlay_pil, concentration_score)
    """
    gradcam = GradCAM(model)
    tf = get_default_transforms(image_size=224, is_training=False)
    x_tensor = tf(image_pil).unsqueeze(0).to(device)

    heatmap_2d = gradcam.generate_heatmap(x_tensor)
    overlay_pil = gradcam.generate_overlay(image_pil, heatmap_2d, alpha=0.50)

    # Spatial concentration metric: fraction of heatmap energy in top 20% of area
    flat = np.sort(heatmap_2d.flatten())[::-1]
    top_20_count = max(1, int(0.20 * len(flat)))
    concentration = float(np.sum(flat[:top_20_count]) / max(1e-8, np.sum(flat)))
    concentration = float(max(0.0, min(1.0, concentration)))

    return heatmap_2d, overlay_pil, round(concentration, 4)
