# SignalScope Phase 4: Robustness & Authenticity Stability Report
**Date**: 2026-09-12T18:58:45.019587+00:00
**Evaluation Partition**: Local Validation Split from `train/` (3,000 samples evaluated out of 15,000 validation pool)
> [!IMPORTANT]
> **Data Compliance**: These are local validation robustness results. The official organizer held-out partition `C:\Programming\SignalScope-data\test` was completely untouched.

## 1. Executive Summary
Detectors in practical media authentication environments confront diverse transmission channel perturbations, including social media recompression, resolution rescaling, screenshot captures, and cropping. This benchmark systematically measures how classification accuracy, decision stability, and output probabilities behave under controlled, deterministic transformations without retraining.

## 2. Transformation Benchmark Matrix
| Transform | Parameter Details | Description |
|---|---|---|
| **original** | None (Pristine) | Pristine native 32x32 |
| **jpeg_95** | quality=95 | High-quality JPEG (Q=95) |
| **jpeg_85** | quality=85 | Standard web JPEG (Q=85) |
| **jpeg_70** | quality=70 | Aggressive social compression (Q=70) |
| **resize** | scale_factor=0.7 | Bilinear downsample (0.7x) + Bicubic restore |
| **screenshot** | blur_radius=0.5, quality=80 | Subpixel box blur (r=0.5) + JPEG Q=80 |
| **light_edit** | crop_fraction=0.9 | 90% center crop + Bilinear restore |

## 3. Comparative Robustness Results

### ConvNeXt Baseline vs. RGB + FFT Fusion
| Model | Transform | ROC-AUC | Macro-F1 | Accuracy | FPR | Flip Rate | Mean Drift |
|---|---|---|---|---|---|---|---|
| **Baseline** | original | **0.9995** | 0.9910 | 99.10% | 1.00% | 0.00% | 0.0000 |
| **Fusion** | original | **0.9991** | 0.9857 | 98.57% | 1.00% | 0.00% | 0.0000 |
| **Baseline** | jpeg_95 | **0.9995** | 0.9903 | 99.03% | 0.93% | 0.13% | 0.0013 |
| **Fusion** | jpeg_95 | **0.9993** | 0.9853 | 98.53% | 0.93% | 0.10% | 0.0020 |
| **Baseline** | jpeg_85 | **0.9994** | 0.9883 | 98.83% | 1.33% | 0.27% | 0.0031 |
| **Fusion** | jpeg_85 | **0.9990** | 0.9843 | 98.43% | 1.13% | 0.27% | 0.0033 |
| **Baseline** | jpeg_70 | **0.9995** | 0.9887 | 98.87% | 1.53% | 0.30% | 0.0039 |
| **Fusion** | jpeg_70 | **0.9990** | 0.9847 | 98.47% | 1.27% | 0.23% | 0.0033 |
| **Baseline** | resize | **0.9494** | 0.6197 | 66.27% | 0.13% | 33.77% | 0.3348 |
| **Fusion** | resize | **0.9521** | 0.6352 | 67.40% | 0.00% | 32.17% | 0.3196 |
| **Baseline** | screenshot | **0.9213** | 0.5312 | 60.40% | 0.20% | 39.63% | 0.3932 |
| **Fusion** | screenshot | **0.9159** | 0.5325 | 60.53% | 0.00% | 39.03% | 0.3872 |
| **Baseline** | light_edit | **0.9925** | 0.8594 | 86.20% | 0.13% | 13.77% | 0.1375 |
| **Fusion** | light_edit | **0.9870** | 0.8491 | 85.23% | 0.07% | 14.27% | 0.1441 |

## 4. Authenticity Stability Score Analysis
The Authenticity Stability Score $S \in [0, 1]$ measures the deterministic invariance of an image's classification across all $K=6$ transformations:
$$S = C \times \left(1 - \frac{\bar{D} + D_{\max}}{2}\right)$$
- $C$: Prediction Consistency (fraction of transformations retaining original hard decision)
- $\bar{D}$: Mean Absolute Probability Drift
- $D_{\max}$: Maximum Absolute Probability Drift

| Metric | ConvNeXt Baseline | RGB + FFT Fusion |
|---|---|---|
| **Mean Stability Score** | **0.6820** | **0.6885** |
| **Median Stability Score** | 0.9991 | 0.9996 |
| **Std Deviation** | 0.3807 | 0.3819 |
| **Min Stability Score** | 0.0542 | 0.0000 |
| **Max Stability Score** | 1.0000 | 1.0000 |
| **Stable Fraction ($S \ge 0.75$)** | **59.23%** | **60.20%** |

## 5. Stability Case Studies & Responsible Uncertainty

### Case 1: Baseline stable while Fusion degrades
- **Image**: `3364 (9).jpg`
- **Ground Truth**: Synthetic (1)
- **Baseline**: Original prob = `1.0000`, Stability = `0.9137`
- **Fusion**: Original prob = `0.9995`, Stability = `0.5295`
- **Observation**: Baseline spatial representation remained consistent under compression, while frequency spectrum suffered degradation that lowered fusion confidence.

### Case 2: Fusion stable while Baseline degrades
- **Image**: `4203 (10).jpg`
- **Ground Truth**: Synthetic (1)
- **Baseline**: Original prob = `1.0000`, Stability = `0.3887`
- **Fusion**: Original prob = `0.9995`, Stability = `0.9486`
- **Observation**: Spatial perturbations drifted ConvNeXt prediction, but multimodal FFT representation grounded the decision, maintaining high fusion stability.

### Case 3: Both models highly stable
- **Image**: `4130 (2).jpg`
- **Ground Truth**: Real (0)
- **Baseline**: Original prob = `0.0000`, Stability = `1.0000`
- **Fusion**: Original prob = `0.0003`, Stability = `1.0000`
- **Observation**: Both architectures decisively recognized salient features that persisted across all degradation transforms.

### Case 4: Both models unstable (Severe degradation impact)
- **Image**: `5159 (8).jpg`
- **Ground Truth**: Synthetic (1)
- **Baseline**: Original prob = `1.0000`, Stability = `0.1254`
- **Fusion**: Original prob = `0.9995`, Stability = `0.1400`
- **Observation**: Subtle boundary features were distorted by recompression and downsampling, inducing high probability drift and prediction flips across both models.

## 6. Recommendations & Technical Conclusion
**Recommendation State**: `C. Baseline and Fusion are similar in overall robustness`

**Rationale**: Both architectures exhibit close average degraded AUC (0.9769 vs. 0.9754) and similar average flip rates (14.64% vs. 14.34%).
