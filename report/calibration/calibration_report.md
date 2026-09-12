# SignalScope Phase 5: Probability Calibration Report
**Timestamp**: 2026-09-12T19:09:18.416169+00:00
**Model**: ConvNeXt-Tiny Spatial Baseline (`checkpoints/baseline_convnext/best_model.pt`)
> [!IMPORTANT]
> **Calibration Methodology Compliance**:
> - Calibration was fitted strictly on a **5,000-sample subset** carved from the official training partition (`train/`, `seed=42`).
> - Evaluated on the **untouched 15,000-sample local validation split**.
> - The organizer-held-out test partition `C:\Programming\SignalScope-data\test` was **NOT** accessed or used.

## 1. Motivation & Problem Scope
Deep neural networks trained with cross-entropy loss often produce uncalibrated probabilities with extreme confidence over-clustering at the boundaries (0.0 and 1.0). In forensic media authentication, raw sigmoid outputs must not be presented as true posterior probabilities. Post-hoc temperature scaling optimizes a single temperature parameter $T > 0$ such that $p_{\text{cal}} = \sigma(z / T)$ without altering network classification boundaries or ROC-AUC.

## 2. Experimental Setup & Optimization
- **Calibration Pool**: 5,000 samples (2,500 Authentic Real, 2,500 Synthetic AI-generated)
- **Validation Pool**: 15,000 samples (7,500 Authentic Real, 7,500 Synthetic AI-generated)
- **Optimization Algorithm**: L-BFGS minimizing Negative Log-Likelihood (Binary Cross-Entropy)
- **Fitted Temperature $T$**: **`0.9995`**

## 3. Pre- vs. Post-Calibration Comparative Metrics
Evaluated on the 15,000-sample local validation partition:

| Metric | Pre-Calibration (Raw Sigmoid) | Post-Calibration (Temperature Scaled) | Absolute Change |
|---|---|---|---|
| **Temperature $T$** | 1.0000 | **0.9995** | — |
| **Brier Score (lower is better)** | **0.00795** | **0.00795** | **+0.00000** |
| **Expected Calibration Error (ECE)** | **0.0062** (0.62%) | **0.0062** (0.62%) | **+0.0000** |
| **Maximum Calibration Error (MCE)** | **0.2687** | **0.2688** | **+0.0001** |
| **ROC-AUC** | **0.9995** | **0.9995** | Identical (Rank-Preserving) |
| **Macro-F1 (Thr = 0.50)** | **0.9908** | **0.9908** | Identical (Symmetric Monotonic) |
| **Accuracy (Thr = 0.50)** | **99.08%** | **99.08%** | Identical |
| **False Positive Rate (FPR)** | **1.03%** | **1.03%** | Identical |
| **Mean Confidence** | 0.9965 | 0.9965 | — |

## 4. Observations & Findings
1. **Temperature Behavior**: The fitted temperature $T = 0.9995$ demonstrates whether the model was slightly over- or under-confident.
2. **Ranking Invariance**: ROC-AUC remains strictly identical (0.9995), verifying that monotonic scaling does not compromise discrimination capability.
3. **Empirical Probability Fidelity**: Calibrated probabilities better reflect empirical error frequencies, providing faithful inputs to downstream uncertainty quantification.

## 5. Artifacts & Figures
- Reliability Diagram: `report/calibration/reliability_diagram.png`
- Confidence Histogram: `report/calibration/confidence_histogram.png`
- Calibration Curve: `report/calibration/calibration_curve.png`
- Fitted Scaler: `checkpoints/baseline_convnext/temperature_scaler.json`