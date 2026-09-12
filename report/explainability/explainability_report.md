# SignalScope Phase 5: Faithful Explainability & Multimodal Evidence Report
**Timestamp**: 2026-09-12T19:12:07.512833+00:00
**Model Evaluated**: ConvNeXt-Tiny Spatial Baseline (`checkpoints/baseline_convnext/best_model.pt`)
> [!IMPORTANT]
> **Anti-Leakage & Grounding Protocol**:
> - All representative examples derive exclusively from the local validation split of `train/`.
> - The organizer-held-out test set `C:\Programming\SignalScope-data\test` was **NOT** used.
> - Explanations are strictly grounded in empirical model attribution, 2D FFT spectrum, and degradation stability.

## 1. Multimodal Evidence Architecture
SignalScope rejects superficial natural-language generation in favor of a deterministic evidence synthesis engine. Before any verdict or explanation is rendered, three independent modalities extract evidence from the image:

1. **Spatial Attribution (Grad-CAM)**: Backpropagates logit gradients through the final convolutional block (`stages[-1].blocks[-1]`) to identify pixel regions influencing model activation.
2. **Spectral Signatures (2D FFT)**: Extracts log-magnitude frequency maps and 1D azimuthal radial decay curves on native 32x32 inputs to quantify high-frequency energy concentration.
3. **Authenticity Stability Probing**: Tests prediction invariance under controlled degradations (JPEG Q=95, Q=70, Resizing 0.7x, Cropping 0.90) to measure volatility.

## 2. Representative Validation Case Studies

### Case 1: Authentic Camera Capture (High Confidence Real)
- **File**: `0955 (6).jpg`
- **Ground Truth**: Authentic Real
- **Verdict**: `likely_real`
- **Calibrated Synthetic Probability**: `0.0000`
- **Authenticity Stability Score**: `1.0000`
- **Explanation**: *"Likely real camera capture (calibrated probability of synthetic origin: 0.00, high confidence). Spatial feature attribution indicates broad natural gradient consistency without localized anomaly peaks (attribution concentration: 0.62). Spectral analysis observed smooth radial energy decay (high-frequency ratio: 0.33) consistent with authentic sensor distributions. The assessment remained highly consistent under controlled degradation tests (Authenticity Stability: 1.00). This is an evidence-grounded likelihood assessment, not absolute proof."*
- **Visual Evidence Panel**: `report/explainability/sample_authentic_real_evidence_panel.png`

### Case 2: Synthetic AI-Generated Media (High Confidence AI)
- **File**: `3244 (9).jpg`
- **Ground Truth**: Synthetic AI
- **Verdict**: `likely_ai_generated`
- **Calibrated Synthetic Probability**: `1.0000`
- **Authenticity Stability Score**: `1.0000`
- **Explanation**: *"Likely AI-generated (calibrated probability: 1.00, high confidence). Spatial attribution shows the model focused predominantly on localized structural regions (attribution concentration: 0.74). Spectral analysis observed a high-frequency energy ratio of 0.31 in the native 2D FFT spectrum. The assessment remained highly consistent under controlled degradation tests (Authenticity Stability: 1.00). This is an evidence-grounded likelihood assessment, not absolute proof."*
- **Visual Evidence Panel**: `report/explainability/sample_synthetic_fake_evidence_panel.png`

### Case 3: Borderline / Uncertain Sample (Responsible Uncertainty Triggered)
- **File**: `5457 (8).jpg`
- **Ground Truth**: Synthetic AI
- **Verdict**: `uncertain`
- **Calibrated Probability**: `1.0000`
- **Stability Score**: `0.2833`
- **Uncertainty Reasons**: `['Prediction volatility detected under spatial perturbation (Authenticity Stability: 0.283 < 0.60).']`
- **Explanation**: *"Classification is uncertain (calibrated probability: 1.00). Prediction volatility detected under spatial perturbation (Authenticity Stability: 0.283 < 0.60). Human forensic review is recommended before making an adjudication."*
- **Visual Evidence Panel**: `report/explainability/sample_borderline_uncertain_evidence_panel.png`

## 3. Responsible AI & Faithfulness Guardrails
- **No Over-Claiming**: The system never uses absolute terms such as '100% fake' or 'artifacts definitely prove generation'.
- **Explicit Uncertainty**: When probability hovers in the ambiguous corridor ($[0.40, 0.60]$) or stability degrades under transformation ($S < 0.60$), the engine marks the verdict as `uncertain` and prompts for human forensic review.
- **Display Disclaimer**: Because native images are 32x32, heatmaps are visually interpolated for presentation only; documentation clearly states that upsampling does not increase forensic resolution.