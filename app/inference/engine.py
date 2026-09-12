"""Inference engine orchestration for SignalScope.

Integrates RGB spatial branch, frequency branch, metadata extraction,
and authenticity stability testing into a unified evidence fusion verdict.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from app.api.schemas import (
    ConfidenceLevelEnum,
    EvidenceItem,
    PredictionResponse,
    VerdictEnum,
)
from app.services.explainability import generate_grounded_explanation
from app.services.metadata import extract_metadata
from app.services.stability import evaluate_authenticity_stability
from app.utils.logger import logger


def pil_to_base64_data_uri(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL Image as Base64 data URI string."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{img_str}"


class SignalScopeInferenceEngine:
    """Orchestrates image authenticity inference and multimodal evidence fusion."""


    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        operating_threshold: float = 0.50,
        uncertainty_band: float = 0.10,
    ) -> None:
        """Initializes the inference engine.

        Args:
            config_path: Path to model configuration YAML.
            checkpoint_path: Path to trained PyTorch weights.
            operating_threshold: Calibrated decision boundary for synthetic classification.
            uncertainty_band: Half-width around threshold where verdict is deemed 'uncertain'.
        """
        self.config_path = config_path
        if checkpoint_path is None:
            default_ckpt = Path("checkpoints/baseline_convnext/best_model.pt")
            if default_ckpt.exists():
                checkpoint_path = str(default_ckpt)
        self.checkpoint_path = checkpoint_path
        self.operating_threshold = operating_threshold
        self.uncertainty_band = uncertainty_band
        self.model = None
        self.device = "cpu"
        self._is_ready = False
        self.has_trained_weights = False
        self.scaler = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initializes model components or falls back to baseline/test mode."""
        from model.calibration import TemperatureScaler
        self.scaler = TemperatureScaler()
        self.model_name = "unloaded"

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                import torch
                from model.architectures.convnext import build_convnext_tiny
                from model.dataset import get_default_transforms

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device = str(device)
                ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

                model = build_convnext_tiny(pretrained=False).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                self.transform = get_default_transforms(image_size=224, is_training=False)
                self.has_trained_weights = True
                self.model_name = ckpt.get("model_name", "convnext_tiny.in12k_ft_in1k") if isinstance(ckpt, dict) else "convnext_tiny"

                # Check for companion temperature scaler
                scaler_path = Path(self.checkpoint_path).parent / "temperature_scaler.json"
                if scaler_path.exists():
                    try:
                        self.scaler.load(scaler_path)
                        logger.info(f"Loaded temperature scaler (T={self.scaler.temperature:.4f}) from {scaler_path}")
                    except Exception as s_err:
                        logger.warning(f"Could not load temperature scaler from {scaler_path}: {s_err}")


                self._is_ready = True
                logger.info(f"Loaded trained checkpoint from {self.checkpoint_path} ({self.model_name}) onto {self.device}")
            except Exception as exc:
                logger.error(f"Failed to load checkpoint {self.checkpoint_path}: {exc}")
                self.has_trained_weights = False
                self._is_ready = False
        else:
            logger.warning(
                "DEVELOPMENT PLACEHOLDER MODE: No trained checkpoint loaded. "
                "Predictions are heuristic scaffolding, not valid model outputs."
            )
            self.has_trained_weights = False
            self._is_ready = True

    @property
    def is_ready(self) -> bool:
        """Whether the inference engine is loaded and operational."""
        return self._is_ready

    def predict_raw_probability(self, image: Image.Image) -> float:
        """Computes synthetic probability for a given PIL Image.

        When model is loaded: routes through PyTorch ConvNeXt-Tiny forward pass.
        In test/scaffolding mode without weights: computes deterministic visual signal.
        """
        if self.model is not None and self.has_trained_weights:
            import torch
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(tensor)
                prob = float(torch.sigmoid(logit).item())
            return float(np.clip(prob, 0.0, 1.0))

        # Deterministic lightweight baseline heuristic for scaffolding and testing
        # Analyzes color distribution and edge variance
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        std_per_channel = np.std(img_np, axis=(0, 1))
        # Scaled to [0.2, 0.8] range to demonstrate stability and scoring
        norm_score = float(np.mean(std_per_channel) / 128.0)
        return float(np.clip(norm_score, 0.05, 0.95))

    def predict_calibrated_probability(self, image: Image.Image) -> float:
        """Computes temperature-calibrated synthetic probability."""
        if self.model is not None and self.has_trained_weights:
            import torch
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(tensor)
                cal_prob = float(self.scaler.calibrate(logit).item())
            return float(np.clip(cal_prob, 0.0, 1.0))
        return self.predict_raw_probability(image)


    def analyze(self, image: Image.Image) -> PredictionResponse:
        """Runs the full SignalScope pipeline on an input image.

        Pipeline:
        1. Base RGB/spatial synthetic probability (raw and temperature-calibrated)
        2. Saliency and spatial attribution (Grad-CAM)
        3. Frequency-domain spectral cues (2D FFT)
        4. Metadata & provenance inspection
        5. Authenticity stability test under degradation
        6. Multimodal evidence fusion
        7. Threshold calibration & responsible uncertainty determination
        8. Evidence-grounded explanation generation
        """
        # Step 1: Base predictions (raw and calibrated)
        raw_prob = self.predict_raw_probability(image)
        cal_prob = self.predict_calibrated_probability(image)

        # Step 2: Spatial Attribution (Grad-CAM)
        heatmap_available = False
        heatmap_b64 = None
        concentration = 0.50
        if self.model is not None and self.has_trained_weights:
            from model.explainability.gradcam import compute_spatial_attribution
            try:
                _, overlay_pil, concentration = compute_spatial_attribution(self.model, image, self.device)
                heatmap_available = True
                heatmap_b64 = pil_to_base64_data_uri(overlay_pil)
            except Exception as exc:
                logger.warning(f"Grad-CAM attribution computation skipped: {exc}")
                heatmap_available = False

        # Step 3: Frequency Spectral Evidence (2D FFT)
        hf_ratio = 0.50
        spectrum_b64 = None
        try:
            from model.explainability.spectral import extract_spectral_features
            _, _, hf_ratio, spectrum_pil = extract_spectral_features(image)
            spectrum_b64 = pil_to_base64_data_uri(spectrum_pil)
        except Exception as exc:
            logger.warning(f"Frequency spectral extraction skipped: {exc}")

        # Step 4: Metadata & provenance inspection
        meta_info = extract_metadata(image)

        # Step 5: Authenticity stability test under controlled degradation
        stability_info = evaluate_authenticity_stability(
            image=image,
            predict_fn=self.predict_calibrated_probability,
            original_prob=cal_prob,
        )

        # Compute flip rate and mean drift
        drift_values = [res.delta_from_original for res in stability_info.transform_results]
        mean_drift = float(np.mean(drift_values)) if drift_values else 0.0
        flips = [
            1 for res in stability_info.transform_results
            if (res.predicted_probability >= self.operating_threshold) != (cal_prob >= self.operating_threshold)
        ]
        flip_rate = float(len(flips) / len(stability_info.transform_results)) if stability_info.transform_results else 0.0

        # Step 6: Assemble backwards-compatible multimodal evidence items
        evidence_items: List[EvidenceItem] = []

        # RGB spatial branch evidence
        rgb_supports_ai = cal_prob >= self.operating_threshold
        evidence_items.append(
            EvidenceItem(
                source="rgb_spatial",
                metric="convnext_feature_anomaly",
                score=round(cal_prob, 4),
                weight=1.0,
                description=(
                    f"Spatial ConvNeXt branch yields calibrated probability of {cal_prob:.2f} "
                    f"({'high generative indicators' if rgb_supports_ai else 'natural texture distribution'})."
                ),
                supports_synthetic=rgb_supports_ai,
            )
        )

        # Frequency evidence (FFT 2D spectral energy)
        freq_supports_ai = hf_ratio > 0.40
        evidence_items.append(
            EvidenceItem(
                source="frequency_domain",
                metric="fft_spectral_distribution",
                score=round(hf_ratio, 4),
                weight=0.75,
                description=(
                    f"Fourier spectral analysis observed high-frequency energy ratio of {hf_ratio:.2f} "
                    f"({'elevated periodic high-frequency residual' if freq_supports_ai else 'expected natural spectral decay'})."
                ),
                supports_synthetic=freq_supports_ai,
            )
        )

        # Metadata evidence
        if meta_info.anomalies and any("AI" in a for a in meta_info.anomalies):
            evidence_items.append(
                EvidenceItem(
                    source="metadata",
                    metric="synthetic_software_signature",
                    score=0.98,
                    weight=0.5,
                    description="AI generator tag or prompt parameter block detected in file chunks.",
                    supports_synthetic=True,
                )
            )
        elif meta_info.has_exif and meta_info.camera_make:
            evidence_items.append(
                EvidenceItem(
                    source="metadata",
                    metric="camera_provenance",
                    score=0.15,
                    weight=0.5,
                    description=f"Authentic camera provenance detected ({meta_info.camera_make}).",
                    supports_synthetic=False,
                )
            )

        # Check for disagreement between modalities
        sources_ai = [item.supports_synthetic for item in evidence_items]
        evidence_disagreement = len(set(sources_ai)) > 1

        # Step 7: Responsible Uncertainty & Decision Boundary
        lower_uncertain = self.operating_threshold - self.uncertainty_band  # e.g., 0.40
        upper_uncertain = self.operating_threshold + self.uncertainty_band  # e.g., 0.60
        is_borderline = lower_uncertain <= cal_prob <= upper_uncertain
        is_volatile = stability_info.stability_score < 0.60

        if is_borderline or is_volatile:
            verdict = VerdictEnum.UNCERTAIN
            confidence_level = ConfidenceLevelEnum.LOW
            uncertain = True
        elif cal_prob >= upper_uncertain:
            verdict = VerdictEnum.LIKELY_AI_GENERATED
            uncertain = False
            confidence_level = (
                ConfidenceLevelEnum.HIGH
                if (cal_prob >= 0.85 and stability_info.is_stable and not evidence_disagreement)
                else ConfidenceLevelEnum.MEDIUM
            )
        else:
            verdict = VerdictEnum.LIKELY_REAL
            uncertain = False
            confidence_level = (
                ConfidenceLevelEnum.HIGH
                if (cal_prob <= 0.15 and stability_info.is_stable and not evidence_disagreement)
                else ConfidenceLevelEnum.MEDIUM
            )

        # Structured multimodal evidence object matching Phase 6 Section 7 contract
        evidence = {
            "spatial": {
                "available": heatmap_available,
                "heatmap": heatmap_b64,
                "attribution_concentration": round(concentration, 4) if concentration is not None else None,
                "target_layer": "ConvNeXtStage[3].ConvNeXtBlock[2]" if self.has_trained_weights else "baseline_scaffolding",
            },
            "spectral": {
                "available": True,
                "spectrum": spectrum_b64,
                "high_frequency_energy_ratio": round(hf_ratio, 4) if hf_ratio is not None else None,
                "representation": "2D Log-Magnitude Fast Fourier Transform",
            },
            "robustness": {
                "available": True,
                "stability_score": round(stability_info.stability_score, 4),
                "prediction_flip_rate": round(flip_rate, 4),
                "mean_probability_drift": round(mean_drift, 4),
                "is_stable": stability_info.is_stable,
                "degradation_impact": stability_info.degradation_impact,
                "transform_results": [
                    t.model_dump() if hasattr(t, "model_dump") else t.dict()
                    for t in stability_info.transform_results
                ],
            },
            "metadata": {
                "available": True,
                "has_exif": meta_info.has_exif,
                "c2pa_present": meta_info.c2pa_detected,
                "camera_make": meta_info.camera_make,
                "camera_model": meta_info.camera_model,
                "software": meta_info.software,
                "anomalies": meta_info.anomalies,
            },
        }
        # Frequency alias for backwards compatibility
        evidence["frequency"] = evidence["spectral"]

        # Step 8: Faithful Explanation
        explanation = generate_grounded_explanation(
            verdict=verdict,
            probability=cal_prob,
            confidence_level=confidence_level,
            stability_score=stability_info.stability_score,
            evidence_items=evidence_items,
            metadata_info=meta_info,
            evidence_disagreement=evidence_disagreement,
        )

        return PredictionResponse(
            schema_version="1.0",
            verdict=verdict,
            probability=round(cal_prob, 4),
            raw_probability=round(raw_prob, 4),
            calibrated_probability=round(cal_prob, 4),
            confidence_level=confidence_level,
            stability_score=stability_info.stability_score,
            evidence_disagreement=evidence_disagreement,
            uncertain=uncertain,
            is_development_placeholder=not self.has_trained_weights,
            evidence=evidence,
            structured_evidence=evidence,
            evidence_items=evidence_items,
            stability=stability_info,
            metadata=meta_info,
            explanation=explanation,
            heatmap_available=heatmap_available,
        )



