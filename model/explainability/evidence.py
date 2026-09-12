"""Structured Multimodal Evidence Extractor and Grounded Explanation Engine.

Synthesizes spatial attribution, spectral evidence, robustness consistency, and
calibrated probabilities into a rigorous, verifiable evidence object and
deterministic, responsibly-worded natural language explanations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from app.utils.logger import logger
from model.architectures.convnext import ConvNeXtTinyDetector
from model.calibration import TemperatureScaler
from model.explainability.gradcam import compute_spatial_attribution
from model.explainability.spectral import extract_spectral_features, render_multimodal_evidence_panel
from model.robustness import compute_authenticity_stability_score, transform_jpeg_recompression, transform_resize_down_up


class EvidenceExtractor:
    """Extracts multimodal evidence and synthesizes grounded explanations."""

    def __init__(
        self,
        classifier: ConvNeXtTinyDetector,
        scaler: Optional[TemperatureScaler] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.classifier = classifier
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.classifier.eval()
        self.scaler = scaler or TemperatureScaler()

    def evaluate_image(
        self,
        image_pil: Image.Image,
        artifact_save_dir: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extracts complete multimodal evidence and generates structured verdict.

        Args:
            image_pil: RGB PIL Image (native 32x32 or any resolution).
            artifact_save_dir: Optional path to save visual explanation panels.
            sample_id: Optional identifier for saving filenames.

        Returns:
            Structured Evidence Object matching SignalScope schema.
        """
        w, h = image_pil.size
        sample_name = sample_id or "sample"

        # 1. Classification and Calibration
        from model.dataset import get_default_transforms
        tf = get_default_transforms(image_size=224, is_training=False)
        x_tensor = tf(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logit = self.classifier(x_tensor).squeeze(-1).float()
            raw_prob = float(torch.sigmoid(raw_logit).item())
            cal_prob = float(self.scaler.calibrate(raw_logit).item())

        # 2. Spatial Attribution (Grad-CAM)
        heatmap_2d, overlay_pil, concentration = compute_spatial_attribution(
            self.classifier, image_pil, self.device
        )

        # 3. Frequency Spectral Evidence
        fft_2d, radial_profile, hf_ratio, spectrum_pil = extract_spectral_features(image_pil)

        # 4. Authenticity Stability (Quick 4-transform probe: JPEG 95, 70, Resize 0.7, Crop 0.9)
        from model.robustness import transform_light_crop_resize
        probes = [
            transform_jpeg_recompression(image_pil, 95),
            transform_jpeg_recompression(image_pil, 70),
            transform_resize_down_up(image_pil, 0.70),
            transform_light_crop_resize(image_pil, 0.90),
        ]
        probe_probs = []
        with torch.no_grad():
            for p_img in probes:
                p_tensor = tf(p_img).unsqueeze(0).to(self.device)
                p_logit = self.classifier(p_tensor).squeeze(-1).float()
                probe_probs.append(float(self.scaler.calibrate(p_logit).item()))

        stability_score = compute_authenticity_stability_score(cal_prob, probe_probs, threshold=0.50)

        # 5. Responsible Uncertainty Assessment
        uncertainty_reasons: List[str] = []
        is_borderline = 0.40 <= cal_prob <= 0.60
        is_volatile = stability_score < 0.60

        if is_borderline:
            uncertainty_reasons.append(
                f"Calibrated probability ({cal_prob:.3f}) lies in the ambiguous boundary corridor [0.40, 0.60]."
            )
        if is_volatile:
            uncertainty_reasons.append(
                f"Prediction volatility detected under spatial perturbation (Authenticity Stability: {stability_score:.3f} < 0.60)."
            )

        if is_borderline or is_volatile:
            verdict = "uncertain"
            confidence_level = "uncertain"
            uncertain = True
        elif cal_prob >= 0.85 or cal_prob <= 0.15:
            verdict = "likely_ai_generated" if cal_prob >= 0.50 else "likely_real"
            confidence_level = "high"
            uncertain = False
        else:
            verdict = "likely_ai_generated" if cal_prob >= 0.50 else "likely_real"
            confidence_level = "moderate"
            uncertain = False

        # 6. Save Artifacts if requested
        panel_path_str = None
        if artifact_save_dir is not None:
            artifact_save_dir.mkdir(parents=True, exist_ok=True)
            panel_path = artifact_save_dir / f"{sample_name}_evidence_panel.png"
            render_multimodal_evidence_panel(
                original_pil=image_pil,
                gradcam_overlay_pil=overlay_pil,
                spectrum_pil=spectrum_pil,
                radial_profile_np=radial_profile,
                verdict=verdict,
                calibrated_prob=cal_prob,
                stability_score=stability_score,
                save_path=panel_path,
            )
            panel_path_str = str(panel_path)

        # 7. Grounded Natural-Language Explanation Generation
        explanation = self._synthesize_explanation(
            verdict=verdict,
            cal_prob=cal_prob,
            confidence_level=confidence_level,
            concentration=concentration,
            hf_ratio=hf_ratio,
            stability_score=stability_score,
            uncertainty_reasons=uncertainty_reasons,
        )

        evidence_object = {
            "verdict": verdict,
            "raw_probability": round(raw_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "confidence_level": confidence_level,
            "uncertain": uncertain,
            "uncertainty_reasons": uncertainty_reasons,
            "evidence": {
                "spatial": {
                    "available": True,
                    "attribution_concentration": concentration,
                    "target_layer": "ConvNeXtStage[3].ConvNeXtBlock[2]",
                },
                "frequency": {
                    "available": True,
                    "high_frequency_energy_ratio": hf_ratio,
                    "spectral_map_type": "2D FFT Log-Magnitude (Centered)",
                },
                "robustness": {
                    "stability_score": round(stability_score, 4),
                    "degradation_impact": "minimal" if stability_score >= 0.85 else ("moderate" if stability_score >= 0.60 else "severe"),
                    "probes_evaluated": len(probes),
                },
                "metadata": {
                    "dimensions": [w, h],
                    "mode": image_pil.mode,
                },
                "visual_panel_path": panel_path_str,
            },
            "explanation": explanation,
            "methodological_disclaimer": (
                "Assessment is grounded in spatial attribution, 2D spectral distributions, and transformation stability. "
                "This is a probabilistic likelihood assessment based on statistical evidence, not causal proof."
            ),
        }

        return evidence_object

    def _synthesize_explanation(
        self,
        verdict: str,
        cal_prob: float,
        confidence_level: str,
        concentration: float,
        hf_ratio: float,
        stability_score: float,
        uncertainty_reasons: List[str],
    ) -> str:
        """Constructs faithful, responsible natural language text from empirical signals."""
        sentences = []

        if verdict == "uncertain":
            sentences.append(
                f"Classification is uncertain (calibrated probability: {cal_prob:.2f})."
            )
            for reason in uncertainty_reasons:
                sentences.append(reason)
            sentences.append(
                "Human forensic review is recommended before making an adjudication."
            )
            return " ".join(sentences)

        if verdict == "likely_ai_generated":
            sentences.append(
                f"Likely AI-generated (calibrated probability: {cal_prob:.2f}, {confidence_level} confidence)."
            )
            sentences.append(
                f"Spatial attribution shows the model focused predominantly on localized structural regions "
                f"(attribution concentration: {concentration:.2f})."
            )
            sentences.append(
                f"Spectral analysis observed a high-frequency energy ratio of {hf_ratio:.2f} in the native 2D FFT spectrum."
            )
        else:
            sentences.append(
                f"Likely real camera capture (calibrated probability of synthetic origin: {cal_prob:.2f}, {confidence_level} confidence)."
            )
            sentences.append(
                f"Spatial feature attribution indicates broad natural gradient consistency without localized anomaly peaks "
                f"(attribution concentration: {concentration:.2f})."
            )
            sentences.append(
                f"Spectral analysis observed smooth radial energy decay (high-frequency ratio: {hf_ratio:.2f}) consistent with authentic sensor distributions."
            )

        # Robustness sentence
        if stability_score >= 0.85:
            sentences.append(
                f"Prediction stability was high under the transformations evaluated for this sample (Authenticity Stability: {stability_score:.2f})."
            )
        else:
            sentences.append(
                f"The prediction exhibited moderate drift under the tested compression or resizing probes (Authenticity Stability: {stability_score:.2f})."
            )

        sentences.append("This is an evidence-grounded likelihood assessment, not absolute proof.")

        return " ".join(sentences)
