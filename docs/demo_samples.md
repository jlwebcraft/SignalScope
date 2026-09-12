# SignalScope Deterministic Judge Demo Sample Catalog

This catalog documents the curated, reproducible demo image samples packaged in `frontend/public/samples/` and used during hackathon evaluations.

> [!IMPORTANT]
> **Provenance & Evaluation Scope Notice**: All samples in this catalog originate strictly from the training-derived local validation partition (`C:\Programming\SignalScope-data\train`). The organizer's held-out evaluation partition (`C:\Programming\SignalScope-data\test`) was **NOT accessed or used**. Observed outcomes reflect local validation behavior and are not competition test metrics.

---

## Sample 1: Authentic Natural Photograph

- **File Identifier**: `frontend/public/samples/authentic_real.jpg`
- **Origin**: Derived from local validation split of `train/REAL` (`0955 (6).jpg`).
- **Dimensions**: 32×32 pixels (upsampled to 224×224 for ConvNeXt feature extraction).
- **Target Role in Demo**: Demonstrates accurate detection of authentic photographic capture with low false positive tendency.
- **Empirical Prediction Profile**:
  - **Calibrated Probability**: **0.0000** (0.0% synthetic likelihood)
  - **Raw Probability**: 0.0000
  - **Final Verdict**: `likely_real`
  - **Confidence Level**: `high`
  - **Uncertain Flag**: `false`
  - **Authenticity Stability Score ($S$)**: **1.000** (0 flips across 4 perturbations, 0.0 mean drift)
- **Explainability Elements to Highlight**:
  1. **Grad-CAM Spatial Heatmap**: Uniform, non-concentrated gradient attribution without localized anomaly spikes.
  2. **2D FFT Spectral Distribution**: Natural continuous azimuthal power decay obeying standard optical physics ($\sim 1/f^\alpha$ power law, low high-frequency ratio: 22.4%).
  3. **Robustness Matrix**: 100% invariant across JPEG 95, JPEG 85, 0.75× Rescaling, and Screenshot simulation.

---

## Sample 2: Synthetic AI-Generated Image

- **File Identifier**: `frontend/public/samples/synthetic_ai.jpg`
- **Origin**: Derived from local validation split of `train/FAKE` (`3244 (9).jpg`).
- **Dimensions**: 32×32 pixels.
- **Target Role in Demo**: Demonstrates detection of AI generator structural artifacts and elevated spectral harmonics.
- **Empirical Prediction Profile**:
  - **Calibrated Probability**: **1.0000** (100.0% synthetic likelihood)
  - **Raw Probability**: 1.0000
  - **Final Verdict**: `likely_ai_generated`
  - **Confidence Level**: `medium`
  - **Uncertain Flag**: `false`
  - **Authenticity Stability Score ($S$)**: **1.000** (0 flips, 0.0 drift)
- **Explainability Elements to Highlight**:
  1. **Grad-CAM Spatial Heatmap**: Distinct focal clustering on synthesized object boundaries (attribution concentration: 32.4%).
  2. **2D FFT Spectral Distribution**: High-frequency energy accumulation caused by deconvolution upsampling and discrete latent decoding (high-frequency ratio: 28.1%).
  3. **Robustness Matrix**: All 4 degradation conditions remain classified as synthetic.

---

## Sample 3: Borderline / Volatile Sample (Uncertainty Trigger)

- **File Identifier**: `frontend/public/samples/borderline_uncertain.jpg`
- **Origin**: Derived from local validation split of `train/FAKE` (`5457 (8).jpg`).
- **Dimensions**: 32×32 pixels.
- **Target Role in Demo**: Demonstrates the Responsible Uncertainty Framework intervening to prevent overconfident erroneous predictions on fragile visual cues.
- **Empirical Prediction Profile**:
  - **Calibrated Probability (Pristine)**: 1.0000
  - **Authenticity Stability Score ($S$)**: **0.136** (Severe Degradation Impact)
  - **Final Verdict**: `uncertain`
  - **Confidence Level**: `low`
  - **Uncertain Flag**: `true`
  - **Adjudication Advisory**: *"Human review recommended before making an adjudication."*
- **Explainability Elements to Highlight**:
  1. **Degradation Volatility**: Although the pristine image leans synthetic, 0.75× downsampling and screenshot simulation cause severe probability drift ($S = 0.136 < 0.60$).
  2. **Responsible AI Guardrail**: SignalScope refuses to produce an overconfident verdict when stability collapses, transparently signaling ambiguity to the user.

---

## Summary Comparison Matrix

| Sample Name | Ground Truth | Calibrated Prob | Stability Score ($S$) | Final Verdict | Confidence | Key Evidence |
|---|---|---|---|---|---|---|
| `authentic_real.jpg` | REAL | 0.00% | 1.000 | `likely_real` | High | Natural spectral decay, zero drift |
| `synthetic_ai.jpg` | FAKE | 100.00% | 1.000 | `likely_ai_generated` | Medium | Localized spatial saliency, elevated HF |
| `borderline_uncertain.jpg` | Volatile | 100.00% | 0.136 | `uncertain` | Low | Extreme drift under resizing ($S < 0.60$) |
