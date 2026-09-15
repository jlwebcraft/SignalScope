# SignalScope: Telling Real From Synthetic in the Age of Generative Media

[![Event: SIH 2026](https://img.shields.io/badge/Event-SIH%202026%20Internal%20Hackathon-4f46e5.svg)]()
[![Model: signalscope--v2](https://img.shields.io/badge/Model-signalscope--v2-059669.svg)](https://github.com/jlwebcraft/SignalScope/releases/tag/v2.0.0)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-3776ab.svg)](https://www.python.org/)
[![PyTorch: 2.6](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI: Production](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js: 16.3](https://img.shields.io/badge/Next.js-16.3%20Turbopack-000000.svg)](https://nextjs.org/)
[![Backend Tests: 78/78 Passed](https://img.shields.io/badge/Tests-78%2F78%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> [!IMPORTANT]
> **Official Evaluation Scope & Anti-Leakage Disclosure**:  
> In strict compliance with Smart India Hackathon (SIH 2026) integrity rules, the organizer's held-out test partition (`SignalScope-data/test`) was **intentionally never accessed, inspected, or leaked** at any stage of development (zero accesses). All development metrics reported below represent local in-domain validation, external multi-domain validation, or independent photographic holdouts. **The official held-out test score is determined solely by the SIH organizers.**

---

## At a Glance

| Resource / Property | Target / Value |
| :--- | :--- |
| **Live Workstation (Frontend)** | [https://sih.deskcraft.online](https://sih.deskcraft.online) (Vercel edge deployment; preview fallback: [https://signal-scope-weld.vercel.app](https://signal-scope-weld.vercel.app)) |
| **Production API (Backend)** | [https://signalscope-backend-780176122274.asia-south1.run.app](https://signalscope-backend-780176122274.asia-south1.run.app) (Google Cloud Run, `asia-south1`) |
| **Production Model** | `signalscope-v2` (`ConvNeXtTinyDetector`, backbone: `convnext_tiny.in12k_ft_in1k`) |
| **Weights SHA-256** | `47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d` (111,359,988 bytes) |
| **Rollback Model** | `signalscope-baseline-v1` (`c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc`, 318.61 MB) |
| **Calibration Temperature** | $T = 0.9986$ (fitted post-hoc via L-BFGS on disjoint calibration split) |
| **Core Detection Task** | Binary media authenticity adjudication (`likely_ai_generated` vs. `likely_real`) with responsible uncertainty (`uncertain` / human review recommended) |
| **Demo Video** | **TODO — add final 3–5 minute demo video URL before submission.** |

### Official SIH 2026 Bonus Module Status

| SIH Bonus Module | Status | Technical Implementation |
| :--- | :---: | :--- |
| **A. Faithful Explanation** | **IMPLEMENTED** | Grad-CAM spatial layer attribution (Stage 3 Block 2), normalized attribution concentration score, and 2D FFT spectral energy analysis. |
| **B. Generator Attribution** | **NOT IMPLEMENTED** | *Not implemented as a separate multi-class classifier. Do not claim.* |
| **C. Robustness to Degradation** | **IMPLEMENTED** | Bounded Authenticity Stability Score $S \in [0, 1]$ evaluated across 4 controlled web degradations (JPEG recompression, down/up-sampling, screenshot simulation). |
| **D. Provenance & Metadata** | **IMPLEMENTED** | Structured EXIF metadata extraction, hardware make/model tags, software signature analysis, and C2PA Content Credentials verification. |
| **E. Image + Text Consistency** | **NOT IMPLEMENTED** | *Not implemented. Out of scope for pure visual forensic workstation.* |
| **F. Real-Time / Deployable** | **IMPLEMENTED** | Containerized FastAPI backend on Google Cloud Run with readiness probes, non-root user, streaming uploads, sub-150ms GPU / sub-2.6s CPU latency, and Next.js 16 frontend. |
| **G. Active Defence Analysis** | **PARTIALLY ANALYZED** | Empirical perturbation stability analysis conducted; *full adversarial-defense module NOT claimed*. |

### Quick Documentation Links
- **Final Submission Report**: [`report/final_submission_report.md`](report/final_submission_report.md)
- **Additional Training & Model Selection Report**: [`report/additional_training_experiment.md`](report/additional_training_experiment.md)
- **SIH Judging Scorecard**: [`docs/final_scorecard.md`](docs/final_scorecard.md)
- **3–5 Minute Evaluation Demo Script**: [`docs/demo_script.md`](docs/demo_script.md)
- **Deterministic Evaluation Demo Samples**: [`docs/demo_samples.md`](docs/demo_samples.md)
- **Cloud Run Deployment Architecture**: [`docs/deployment_architecture.md`](docs/deployment_architecture.md)

---

## Try the Live Demo

**Live application:** https://sih.deskcraft.online

No local installation is required to try the deployed system.

### Basic workflow

1. Open **https://sih.deskcraft.online**
2. Upload a JPEG, PNG, WEBP, or BMP image using drag-and-drop or the file picker.
3. Wait for SignalScope to complete the analysis.
4. Read the primary assessment:
   - **Likely Real (Authentic)**
   - **Likely AI-Generated**
   - **Uncertain / Indeterminate**
5. Review the calibrated likelihood and confidence.
6. Check the **Authenticity Stability** result.
7. Open the forensic evidence:
   - Grad-CAM spatial attribution
   - 2D Fourier / FFT spectral evidence
   - degradation / perturbation matrix
   - EXIF / C2PA provenance information
8. Read the grounded explanation and human-review guidance.
9. Use **Examine New Exhibit** to analyze another image.
10. Use **Print Report** when a printable analysis record is needed.

> [!NOTE]
> **Probabilistic Notice**: SignalScope provides a probabilistic authenticity assessment based on statistical signal and spectral features, not definitive or legal proof.

---

## Judge Quick Demo

For the fastest demonstration, use the built-in deterministic benchmark samples already available in the interface.

| Sample | Purpose | Expected result |
|---|---|---|
| `#0955` | Authentic real image | `Likely Real` |
| `#3244` | Synthetic AI image | `Likely AI-Generated` |
| `#5457` | Stability/uncertainty case | `Uncertain / Indeterminate` |

### #0955 — authentic example
Demonstrates:
- image ingestion
- real-image assessment
- high stability
- Grad-CAM
- spectral evidence
- provenance inspection

### #3244 — synthetic example
Demonstrates:
- synthetic-image assessment
- calibrated likelihood
- confidence
- spatial attribution
- spectral evidence
- robustness results

### #5457 — uncertainty example
Demonstrates SignalScope's most important trust feature:
- strong baseline classifier signal
- low perturbation stability
- automatic uncertainty/adjudication override
- human forensic review recommendation

---

## 1. Problem Framing & Challenge Context

With the rapid proliferation of modern generative imaging models—such as Latent Diffusion Models (Stable Diffusion, SDXL, FLUX), Generative Adversarial Networks (GANs), and Flow Matching architectures—visual media can be synthesized with exceptional perceptual fidelity. This creates acute societal challenges including visual disinformation, synthetic evidence injection, and automated media manipulation.

Conventional deepfake classifiers typically suffer from two systemic failures:
1. **Generator Overfitting & Domain Brittleness**: Classifiers memorize high-frequency artifacts unique to a specific training generator family (e.g., latent deconvolution upsampling grids) and experience catastrophic failure when exposed to unseen generator engines or natural high-resolution camera photography.
2. **Perturbation Fragility**: Subtle synthetic markers often disintegrate under routine digital distribution channels (e.g., social media recompression, resolution downsampling, screenshotting), causing classifiers to produce erratic, overconfident false alarms.

**SignalScope** solves these challenges not by treating authenticity as a single opaque probability, but as a **calibrated, multi-modal forensic inspection workflow** combining deep spatial representations, Fourier spectral evidence, degradation stability probing, and provenance metadata under an overarching **Responsible Uncertainty Framework**.

---

## 2. What SignalScope Delivers

SignalScope provides a complete forensic analysis workstation designed for technical evaluators, investigative teams, and media forensic examiners:

- **Calibrated Authenticity Assessment**: Evaluates images through post-hoc temperature scaling ($T = 0.9986$), ensuring probabilistic scores accurately reflect empirical risk rather than overconfident network logits.
- **Responsible Uncertainty Guardrails**: Enforces a strict decision corridor $[0.40, 0.60]$ and volatility threshold ($S < 0.60$). If visual evidence is ambiguous or volatile under perturbation, SignalScope explicitly outputs `uncertain` with *"Human review recommended"* rather than forcing a hazardous binary decision.
- **Optical Inspection Comparator (Grad-CAM)**: Exposes internal spatial features via Grad-CAM on ConvNeXt-Tiny's final block with split-slider, overlay, and attribution concentration metrics.
- **Signal-Analysis Frequency Deck (2D FFT)**: Computes two-dimensional discrete Fourier log-magnitude spectra and radial high-frequency energy distribution benchmarks, exposing synthetic upsampling grid harmonics and unnatural spectral roll-off.
- **Laboratory Degradation Matrix (Stability Probing)**: Evaluates the image against controlled transformations (JPEG 95, JPEG 85, down/up-sampling, screenshot simulation) to measure prediction drift ($\Delta p$) and decision flips.
- **Structured Provenance Property Sheet**: Inspects embedded EXIF tags, camera hardware signatures, editing software flags, and cryptographically signed C2PA Content Credentials.
- **Authoritative Docket & Lightweight Print Report**: Features forensic exhibit tracking, natural resolution detection, deterministic sample quick-loaders, and a dedicated `@media print` layout.

---

## 3. Multimodal System Architecture

```
                                [ Single Input Image (≤ 25 MB) ]
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │      Ingestion, Security & Bounds Validation │
                         │  - MIME Whitelist (JPEG, PNG, WEBP, BMP)     │
                         │  - Decompression Bomb Check (MAX 64M pixels) │
                         │  - Dimension Bounds (16 ≤ W, H ≤ 4096 px)    │
                         └──────────────────────┬───────────────────────┘
                                                │
                         ┌──────────────────────┴───────────────────────┐
                         ▼                                              ▼
          ┌─────────────────────────────┐                ┌─────────────────────────────┐
          │   RGB Spatial Backbone      │                │   Frequency 2D FFT Branch   │
          │  - ConvNeXt-Tiny            │                │  - Centered Log-Magnitude   │
          │  - 28.6M parameters         │                │  - Radial Energy Ratio      │
          │  - Spatial Logits z(x)      │                │  - Harmonic Spikes Analysis │
          └──────────────┬──────────────┘                └──────────────┬──────────────┘
                         │                                              │
                         ▼                                              │
          ┌─────────────────────────────┐                               │
          │   Probability Calibration   │                               │
          │  - Temperature T = 0.9986   │                               │
          │  - Calibrated p_cal(x)      │                               │
          └──────────────┬──────────────┘                               │
                         │                                              │
                         ├──────────────────────┐                       │
                         ▼                      ▼                       │
          ┌─────────────────────────────┐ ┌───────────────────────────┐ │
          │    Spatial Attribution      │ │    Stability Matrix       │ │
          │  - Grad-CAM Stage 3 Block 2 │ │  - 4 Degradation Probes   │ │
          │  - Attribution Concentration│ │  - Probability Drift Δp   │ │
          │  - Local Saliency Mask      │ │  - Bounded Score S ∈ [0,1]│ │
          └──────────────┬──────────────┘ └─────────────┬─────────────┘ │
                         │                              │               │
                         └──────────────┬───────────────┴───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────────────────────┐
                         │       Responsible Evidence Fusion            │
                         │  - Probability Corridor: p ∈ [0.40, 0.60]    │
                         │  - Stability Gate: S < 0.60                  │
                         │  - EXIF Provenance & C2PA Verification       │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Structured Assessment & Telemetry Output   │
                         │  - Adjudication: likely_real / AI / uncertain│
                         │  - Subordinated Baseline Probe Context       │
                         │  - Grounded Narrative Text (JSON Schema v1.0)│
                         └──────────────────────────────────────────────┘
```

---

## 4. Responsible Uncertainty & Ethical AI Scope

### Ethical Boundaries
SignalScope adheres strictly to responsible AI principles:
- **No Biometric or Facial Recognition**: SignalScope is strictly an image-signal authenticity detector. It does not perform facial recognition, personal identification, emotion detection, or biometric profiling.
- **No Accusations or Legal Proof**: Detection outputs are statistical likelihood estimates. They must never be weaponized as definitive legal accusations.
- **Non-Accusatory Terminology**: Adjudications use probabilistic classifications: `likely_ai_generated`, `likely_real`, or `uncertain`.

### Subordinated Uncertain-Result Hierarchy
On volatile images, conventional detectors present an apparent contradiction: reporting "Uncertain" alongside an unperturbed baseline probe of "100.0%". 

SignalScope eliminates this ambiguity through a clear information hierarchy:
1. **Primary Adjudication**: Prominently displays `Adjudication: Inconclusive / Uncertain` with an `[Override Active]` badge.
2. **Subordinated Baseline Probe**: Contextualizes the raw classifier output (`Baseline Probe: 100.0%`), explaining that this score represents only the unperturbed baseline.
3. **Primary Volatility Metric**: Surfaces `Authenticity Stability: 25.5% (Volatile)` in warning amber/red.
4. **Dynamic Technical Explanation**: Dynamically explains *why* the assessment is inconclusive: *"The baseline classifier produced a strong synthetic signal, but the prediction changed materially under image perturbations. SignalScope therefore withholds a binary authenticity assessment and recommends human review."*

---

## 5. Dataset Architecture & Anti-Leakage Protocol

SignalScope was engineered under strict data partitioning to guarantee zero data leakage.

```
SignalScope-data/
├── train/                                    # Official SIH 2026 Training Pool (100,000 images)
│   ├── REAL/                                 # 50,000 natural camera images (32×32 JPEG)
│   └── FAKE/                                 # 50,000 synthetic AI images (32×32 JPEG)
│       ├── Carved Local Train Pool           # 85,000 images (deterministic seed=42)
│       └── Untouched Local Validation Pool   # 15,000 images (7,500 real / 7,500 fake)
│
├── version-2/                                # External Domain Diversity Dataset (143,404 images)
│   ├── AIGenImages2026/
│   │   ├── train/                            # 9,758 paired samples (4,879 real / 4,879 fake)
│   │   └── val/                              # 1,118 paired samples (559 real / 559 fake across 19 generators)
│   └── REAL/                                 # 132,528 multi-resolution real photographs
│       ├── Used in v2 Training               # 5,004 photographic real images (balanced pool)
│       ├── Excluded from Training            # 24,096 images (illustrations, memes — non-photographic)
│       └── Independent Validation Holdout    # 9,000 unused real photos across 9 categories
│
└── test/                                     # Official SIH Organizer Held-Out Test Pool
    └── [RESTRICTED]                          # STRICTLY UNTOUCHED (0 accesses throughout project)
```

### Dataset Audit & Integrity Controls
- **Zero Official Test Access**: `SignalScope-data/test` was **never opened, read, indexed, or evaluated** by any script in this repository.
- **AIGenImages2026 Label Audit**: An automated audit discovered that `AIGenImages2026` contains strictly paired `0_real` and `1_fake` subdirectories. A naive assumption that the entire folder was synthetic would have poisoned training with 5,438 real images labeled as synthetic.
- **Exclusion of Non-Photographic Real Artifacts**: The categories `REAL/illustrations` (digital vector art) and `REAL/meme` (web text collages) were filtered out of training to prevent distorting the model's representation of genuine camera photography.
- **Anti-Leakage Assertion**: Partition assertions confirm mathematical disjointness:
  $$\text{Train} \cap \text{Val}_{\text{InDomain}} = \emptyset, \quad \text{Train} \cap \text{Val}_{\text{External}} = \emptyset, \quad \text{Train} \cap \text{Holdout}_{\text{Real}} = \emptyset$$

---

## 6. Model Development & Empirical Evaluation

SignalScope progressed through two rigorous model generations: `signalscope-baseline-v1` (trained exclusively on the SIH 32×32 training pool) and **`signalscope-v2`** (`exp3_5ep`, trained on a balanced multi-domain pool with forensic multi-scale augmentations).

### Comprehensive Model Comparison

| Evaluation Metric | Baseline Model (`baseline-v1`) | Candidate Exp 2 (`exp2_5ep`) | Production Model (`signalscope-v2`) | Best Performer |
| :--- | :---: | :---: | :---: | :--- |
| **Model Weights SHA-256** | `c2e7881e...` | `bb0e5c7e...` | `47f6b2a19d61...` | `signalscope-v2` |
| **Model Checkpoint Size** | 318.61 MB | 106.20 MB | **106.20 MB** (111.3 MB disk) | `signalscope-v2` |
| **Local Val ROC-AUC** (15k in-domain) | **0.9992** | 0.9988 | **0.9991** | Baseline / v2 (<0.0001 diff) |
| **Local Val Macro-F1** (15k in-domain)| **0.9908** (99.08%) | 0.9877 | **0.9851** (98.51%) | Baseline |
| **Local Val Accuracy** (15k in-domain)| **99.08%** | 98.77% | **98.51%** | Baseline |
| **Local Val FPR (@ 0.50 threshold)**  | **1.03%** | 1.00% | **1.85%** | Exp 2 / Baseline |
| **External Val ROC-AUC** (1,118 multi-domain)| 0.5950 | **0.9267** | **0.9220** | Exp 2 / v2 |
| **External Val Macro-F1** (1,118 multi-domain)| 0.4313 | 0.8505 | **0.8578** (85.78%) | **`signalscope-v2`** |
| **External Val Accuracy** (default 0.50)| 53.22% | 85.15% | **85.78%** | **`signalscope-v2`** |
| **External Val FPR** (default 0.50)   | 88.91% | 17.53% | **13.95%** | **`signalscope-v2`** |
| **Optimized Threshold**              | 0.9900 | 0.9751 | **0.6338** (near ideal) | **`signalscope-v2`** |
| **Optimized Accuracy (at opt thresh)**| 57.78% | 85.96% | **86.40%** | **`signalscope-v2`** |
| **Optimized FPR (at opt thresh)**     | 68.69% | 6.44% | **9.84%** | Exp 2 |
| **External Val ECE** (Calibration)    | 0.4462 | 0.1095 | **0.0664** (lowest error) | **`signalscope-v2`** |
| **External Val Brier Score**          | 0.4480 | 0.1265 | **0.1098** | **`signalscope-v2`** |
| **Mean Degraded AUC** (Robustness)    | 0.9759 | 0.9770 | **0.9872** | **`signalscope-v2`** |
| **Degraded Flip Rate**                | 15.03% | 16.27% | **12.95%** | **`signalscope-v2`** |
| **Photographic Holdout FPR** (9,000 real)| **74.69%** | — | **1.37%** (**-98.17% relative drop**) | **`signalscope-v2`** |
| **Photographic Specificity** (9,000 real)| 25.31% | — | **98.63%** | **`signalscope-v2`** |

### Independent Photographic-Real Holdout Gate ($N = 9,000$)
To test whether `signalscope-v2` eliminated resolution-induced false positives on natural camera photography, a strict validation gate was evaluated on **9,000 completely unused real photographs** across 9 categories (1,000 images per category, zero training overlap):

```
Category         Baseline FPR (%)     signalscope-v2 FPR (%)     Absolute Reduction
───────────────────────────────────────────────────────────────────────────────────
apparel               71.20%                   0.90%                  -70.30%
cars                  79.40%                   0.30%                  -79.10%
dishes                87.30%                   0.70%                  -86.60%
furniture             56.70%                   0.70%                  -56.00%
landmark              90.50%                   3.10%                  -87.40%
packaged              77.10%                   0.10%                  -77.00%
storefronts           85.40%                   2.20%                  -83.20%
toys                  50.60%                   0.80%                  -49.80%
artwork               74.00%                   3.50%                  -70.50%
───────────────────────────────────────────────────────────────────────────────────
OVERALL (9,000)       74.69%                   1.37%                  -73.32%
```
- **Median Synthetic Probability on Real Photography**: Dropped from **0.9956** (v1) down to **0.0019** (v2).
- **90th Percentile Synthetic Probability**: Dropped from **1.0000** down to **0.0498**.

### External Cross-Generator Stratified Evaluation ($N = 1,118$)
Evaluated across all 19 modern generative families in `AIGenImages2026/val/1_fake/` against 559 real validation images:
- **Macro-Average Generator AUC**: **0.9237** (Exp3-5ep) vs 0.5961 (Baseline v1).
- **Highest Performing Family**: `ideogram-v3` (AUC: **0.9809**, 100.0% recall @ 0.50).
- **Strong Performers**: `gemini-25-flash-image` (AUC: 0.9790), `firefly_image5` (AUC: 0.9540), `gpt-image-1.5` (AUC: 0.9634), `flux-2-max` (AUC: 0.9552), `flux-2-pro` (AUC: 0.9387).
- **Most Challenging Family**: `midjourney_v7` (AUC: **0.7582**, 67.7% recall @ 0.50), reflecting modern ultra-smooth texturing.
- *(Note: Labeled explicitly as external generator-stratified validation, not organizer held-out testing).*

---

## 7. Probability Calibration ($T = 0.9986$)

Deep convolutional neural networks trained with cross-entropy frequently output uncalibrated, overconfident probabilities near 0.0 or 1.0. SignalScope applies post-hoc **Temperature Scaling** to the network logits $z(x)$:

$$\hat{p}_{\text{cal}}(x) = \sigma\left(\frac{z(x)}{T}\right)$$

- **Optimal Temperature Parameter**: $T = 0.9986$ (fitted via L-BFGS on disjoint calibration split).
- **Expected Calibration Error (ECE)**: Reduced from **0.4462** (baseline) down to **0.0664** on out-of-domain evaluation.
- **Brier Score**: Reduced from **0.4480** to **0.1098**, proving that predicted probabilities reliably mirror statistical likelihood.

---

## 8. Faithful Explainability & Spatial Attribution (Grad-CAM)

*(Fulfills SIH Bonus Module A: Faithful Explanation)*

Rather than presenting an uninterpretable decision, SignalScope exposes spatial evidence via **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

- **Target Layer**: Layer-normalized feature representations in ConvNeXt-Tiny's final block (`model.backbone.stages[-1].blocks[-1]`).
- **Attribution Map Generation**: Computes gradients of the synthetic score with respect to feature activation maps $A^k$, generating $7 \times 7 \times 768$ saliency tensors interpolated to input dimensions:
  $$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right), \quad \alpha_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y_{\text{synthetic}}}{\partial A_{i,j}^k}$$
- **Normalized Attribution Concentration Score**: Measures spatial focal clustering of gradients ($0.0$ = diffuse global activation, $1.0$ = hyper-concentrated anomalous focal point).
- **Scientific Grounding & Limitations**:
  - Grad-CAM indicates *which image regions most strongly influenced the model's feature representations*.
  - **It does NOT constitute definitive causal proof of synthetic generation or pixel-level tampering.**

---

## 9. Frequency-Domain Evidence (2D Fast Fourier Transform)

Generative decoders (transposed convolutions, latent upsamplers) frequently imprint subtle periodic artifacts and unnatural energy roll-offs in the frequency domain.

SignalScope extracts discrete 2D Fast Fourier Transform (FFT) features:
1. **Centered Log-Magnitude Spectrum**: Computes 2D spatial FFT, applies logarithmic dynamic-range compression, and shifts DC zero-frequency components to the center:
   $$F(u, v) = \log\left(1 + \left|\mathcal{F}\{I(x, y)\}\right|\right)$$
2. **High-Frequency Energy Ratio**: Calculates the ratio of spectral energy beyond the normalized radial cutoff $r \ge r_{\text{cutoff}}$ relative to total spectral power:
   $$\text{HFER} = \frac{\sum_{r \ge r_{\text{cutoff}}} |F(u, v)|^2}{\sum_{u, v} |F(u, v)|^2}$$
3. **Azimuthal Radial Distribution Benchmark**: Evaluates energy decay against physical photographic optics (which exhibit smooth $1/f^\alpha$ natural decay), flagging anomalous high-frequency energy spikes.
4. **Scientific Grounding**: Spectral anomalies provide supporting signal evidence; high-frequency energy can also stem from uncompressed textures or sharpening filters.

---

## 10. Robustness & Authenticity Stability Score ($S$)

*(Fulfills SIH Bonus Module C: Robustness to Degradation)*

To prevent false confidence on brittle visual artifacts, SignalScope evaluates every submitted image through a standardized **Controlled Degradation Matrix**:
1. **Pristine Native Image**
2. **JPEG Recompression (Quality 95)**: Mild compression artifact test.
3. **JPEG Recompression (Quality 85)**: Standard web compression artifact test.
4. **Resolution Rescaling (0.75× Downsample & Bicubic Upsample)**: Pixel decimation and interpolation test.
5. **Screenshot Simulation**: Bilinear downsampling with subtle spatial blurring and re-encoding.

### Bounded Authenticity Stability Formulation
The Authenticity Stability Score $S \in [0, 1]$ combines prediction consistency $C$ (invariance of binary decision) with penalties for mean probability drift $\bar{D}$ and maximum drift $D_{\max}$:

$$S = C \times \left(1 - 0.5(\bar{D} + D_{\max})\right)$$

- **`S >= 0.80`**: Stable (`minimal` degradation impact).
- **`0.60 <= S < 0.80`**: Moderately stable (`moderate` degradation impact).
- **`S < 0.60`**: Volatile (`severe` degradation impact; triggers automatic `uncertain` verdict override).

---

## 11. Provenance & Metadata Inspection (EXIF & C2PA)

*(Fulfills SIH Bonus Module D: Provenance & Metadata)*

SignalScope includes a dedicated forensic property inspection service:
- **EXIF Metadata Parsing**: Inspects TIFF/JPEG header tags for camera manufacturer (`Make`), hardware model (`Model`), exposure parameters, and lens specifications.
- **Software Signature Analysis**: Flags generation tags (e.g., `"Midjourney"`, `"DALL-E"`, `"Stable Diffusion"`, `"Adobe Firefly"`).
- **C2PA Content Credentials**: Detects cryptographic provenance manifests adhering to the Coalition for Content Provenance and Authenticity (C2PA) standard.
- **Forensic Policy**: The *absence* of metadata is **not** treated as evidence of synthetic creation, because social media platforms routinely strip metadata for privacy.

---

## 12. Forensic Analysis Workstation (Web Application)

The frontend is implemented in Next.js 16 (App Router, Turbopack, TypeScript, Tailwind CSS) as a specialized forensic instrument:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  SignalScope  [Backend Live] [signalscope-v2 · CPU]              [?] Method & Limits   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  EXAMINATION DOCKET · EXHIBIT #5457 · JPEG · 32×32 px · signalscope-v2   [Print Report]│
├────────────────────────────────────────────────────────────────────────────────────────┤
│  ADJUDICATION STATUS: INCONCLUSIVE / UNCERTAIN                 [Override Active]       │
│  Baseline Probe: 100.0% · Stability Override: Inconclusive                             │
│  Authenticity Stability: 25.5% (Volatile)                                              │
│  "Baseline classifier produced a synthetic signal, but prediction changed materially..."│
├────────────────────────────────────────────────────────────────────────────────────────┤
│  OPTICAL INSPECTOR (Grad-CAM)               │  SIGNAL-ANALYSIS SPECTRUM (2D FFT)       │
│  [Darkroom Comparator · Stage 3 Block 2]   │  [Log-Magnitude Spectrum · High-Freq]    │
│  Attribution Concentration: 32.4%           │  High-Frequency Ratio: 28.1%             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  ROBUSTNESS DATA SHEET (Stability Matrix)   │  PROVENANCE PROPERTY SHEET               │
│  JPEG 95: 100.0% (Δ 0.00) · INVARIANT       │  Hardware Make/Model: None               │
│  Resize:   35.2% (Δ 0.65) · DECISION FLIP   │  Software Signature: None                │
│  Screenshot: 28.1% (Δ 0.72) · DECISION FLIP │  C2PA Credentials: Not Detected          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  SYNTHESIZED FINDINGS REPORT                                                           │
│  Evidence Corroboration: Inconclusive (Spatial synthetic vs Stability Volatile)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Interactive Comparator**: Side-by-side, split-slider, and opacity overlay controls.
- **Diagnostic Telemetry Console**: Displays real-time engine specifications (backbone, calibration temperature, supported formats, limits).
- **Zero Template Cards**: Structurally differentiated layouts tailored to each evidence type.
- **Lightweight Print Report**: Dedicated CSS print stylesheet triggered directly from docket header.

---

## 13. Production API Contract

The FastAPI backend exposes versioned, standards-compliant endpoints (`schema_version: "1.0"`):

- `GET /health`: Liveness probe (HTTP 200).
- `GET /ready`: Readiness probe distinguishing HTTP boot from model weight loading (HTTP 200 when ready, HTTP 503 if weights missing).
- `GET /api/v1/info`: Model metadata, architecture, active version (`signalscope-v2`), SHA-256 hash, and calibration parameters.
- `POST /api/v1/predict`: Primary inference endpoint accepting multipart image upload (`file`).

### Sample Prediction Response (`POST /api/v1/predict`)
```json
{
  "schema_version": "1.0",
  "verdict": "uncertain",
  "probability": 1.0,
  "raw_probability": 1.0,
  "calibrated_probability": 1.0,
  "confidence_level": "low",
  "stability_score": 0.255,
  "evidence_disagreement": true,
  "uncertain": true,
  "is_development_placeholder": false,
  "evidence": {
    "spatial": {
      "available": true,
      "heatmap": "data:image/png;base64,...",
      "attribution_concentration": 0.324,
      "target_layer": "model.backbone.stages.3.blocks.2"
    },
    "spectral": {
      "available": true,
      "spectrum": "data:image/png;base64,...",
      "high_frequency_energy_ratio": 0.281,
      "representation": "2D Log-Magnitude Fast Fourier Transform"
    },
    "robustness": {
      "available": true,
      "stability_score": 0.255,
      "prediction_flip_rate": 0.50,
      "mean_probability_drift": 0.342,
      "is_stable": false,
      "degradation_impact": "severe",
      "transform_results": [
        {"transform_name": "original", "predicted_probability": 1.0, "delta_from_original": 0.0},
        {"transform_name": "jpeg_recompression", "predicted_probability": 1.0, "delta_from_original": 0.0},
        {"transform_name": "resize_down_up", "predicted_probability": 0.352, "delta_from_original": 0.648},
        {"transform_name": "screenshot_simulation", "predicted_probability": 0.281, "delta_from_original": 0.719}
      ]
    },
    "metadata": {
      "available": true,
      "has_exif": false,
      "c2pa_present": false,
      "camera_make": null,
      "camera_model": null,
      "software": null
    }
  },
  "explanation": "The baseline classifier produced a strong synthetic signal, but the prediction changed materially under image perturbations. SignalScope therefore withholds a binary authenticity assessment and recommends human review.",
  "disclaimer": "SignalScope outputs are statistical estimates of authenticity signals. They do not constitute legal proof or personal accusations. Outputs should be evaluated in context with supporting human review."
}
```

---

## 14. Security & Hardening

- **MIME & Magic Byte Whitelist**: Accepts only verified `image/jpeg`, `image/png`, `image/webp`, and `image/bmp`. Rejects executable payloads.
- **Upload Size Limit**: Strict 25 MB payload ceiling (`HTTP 413 Payload Too Large`).
- **Decompression Bomb Prevention**: PIL `Image.MAX_IMAGE_PIXELS = 64_000_000` enforced.
- **Dimension Sanity Checks**: Images must satisfy $16 \le \text{width}, \text{height} \le 4096$.
- **Privacy-Conscious In-Memory Streaming**: Images are streamed into byte buffers and discarded immediately following analysis; zero user images are stored on server disk.
- **Model Checksum Gate**: Artifact manager asserts SHA-256 hash before loading (`47f6b2a19d61...`), refusing corrupted or tampered checkpoints.
- **Fail-Safe Container Execution**: Runs as non-root user `appuser` (UID 1000).

---

## 15. Local Run & Reproducibility (10-Minute Guide)

In compliance with the SIH submission contract, evaluators can reproduce predictions and run verification tests locally in under 10 minutes.

### 1. Clone

```bash
git clone https://github.com/jlwebcraft/SignalScope.git
cd SignalScope
```

### 2. Python Environment

Create and activate an isolated Python environment (Python 3.11 – 3.13):

**Option A: Virtual Environment (venv)**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

**Option B: Conda Environment**
```bash
conda env create -f environment.yml
conda activate signalscope
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Production Model Checkpoint (`signalscope-v2`)

The production model artifact (`best_model.pt`, SHA-256: `47f6b2a19d61...`) is hosted on GitHub Releases v2.0.0:

**On Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path checkpoints/additional_training/exp3_5ep
Invoke-WebRequest -Uri "https://github.com/jlwebcraft/SignalScope/releases/download/v2.0.0/best_model.pt" -OutFile "checkpoints/additional_training/exp3_5ep/best_model.pt"
```

**On Linux / macOS (curl):**
```bash
mkdir -p checkpoints/additional_training/exp3_5ep
curl -L -o checkpoints/additional_training/exp3_5ep/best_model.pt "https://github.com/jlwebcraft/SignalScope/releases/download/v2.0.0/best_model.pt"
```

### 5. Start Backend API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- API interactive docs: `http://localhost:8000/docs`
- Health probe: `http://localhost:8000/health`
- Readiness probe: `http://localhost:8000/ready`

### 6. Start Frontend Web Application

Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the forensic analysis workstation.

### 7. Run Standalone CLI Prediction

Run prediction directly from the command line without launching the web server:
```bash
python model/predict.py --image frontend/public/samples/authentic_real.jpg --checkpoint checkpoints/additional_training/exp3_5ep/best_model.pt --json
```

### 8. Run Full Verification Test Suite (78 Tests)

```bash
python -m pytest tests/ -q
# Expected: 78 passed in ~85s
```

---

## 16. Production Deployment & Telemetry

*(Fulfills SIH Bonus Module F: Real-Time / Deployable)*

SignalScope is deployed live in production across Google Cloud Run and Vercel:

| Infrastructure Layer | Production Configuration |
| :--- | :--- |
| **Backend Platform** | **Google Cloud Run** (Fully managed serverless container runtime) |
| **GCP Project & Region** | `signalscope-508420` · Region: `asia-south1` (Mumbai) |
| **Cloud Run Service** | `signalscope-backend` |
| **Container Image** | Python 3.13-slim runtime, non-root user `appuser` (UID 1000) |
| **Allocated Resources** | 2 vCPU, 4 GiB RAM, automatic request concurrency |
| **Environment Variables** | `MODEL_VERSION=signalscope-v2`, `APP_ENV=production` |
| **Model Resolution** | Remote artifact download with SHA-256 verification and local cache |
| **Frontend Platform** | **Vercel Edge Network** (`sih.deskcraft.online`) |

---

## 17. Computational Performance & Latency

Evaluated across production CPU and local GPU runtimes:

| Processing Stage | Local GPU (NVIDIA RTX 3050 Laptop) | Cloud Run CPU (2 vCPU x86_64) |
| :--- | :---: | :---: |
| **Image Ingestion & Bounds Check** | 1.8 ms | 4.2 ms |
| **Primary ConvNeXt Forward Pass** | **16.1 ms** | **61.5 ms** |
| **Temperature Calibration ($T=0.9986$)** | < 0.05 ms | < 0.1 ms |
| **Grad-CAM Spatial Layer Attribution** | 37.7 ms | 310.0 ms |
| **2D FFT Frequency Feature Extraction** | 2.9 ms | 18.5 ms |
| **Authenticity Stability Probes (4 transforms)** | 65.1 ms | 1,980.0 ms |
| **Metadata & EXIF Extraction** | 0.8 ms | 1.2 ms |
| **Total Pipeline Execution Latency** | **~124.4 ms** | **~2,375.5 ms (2.38 s)** |
| **Full HTTP Request-Response Latency** | **~142.0 ms** | **~2,650.0 ms (2.65 s)** |

- **Real-Time Admissibility**: Full multi-modal evidence synthesis completes in **2.65 seconds on serverless CPU** (or **142 ms on GPU**), satisfying real-time deployability requirements without requiring expensive GPU servers.

---

## 18. Known Limitations

In the spirit of honest scientific disclosure, SignalScope documents the following operational boundaries:
1. **Extremely Low Resolutions (< 32×32 px)**: Tiny image patches lack sufficient spatial context for reliable Grad-CAM localization or high-frequency spectral decomposition.
2. **Ultra-Modern Diffusion Texturing (`midjourney_v7`)**: On generator-stratified evaluation, `midjourney_v7` yielded an AUC of 0.7582, demonstrating that state-of-the-art texturing models can partially evade spatial edge detectors.
3. **Absence of Provenance Is Non-Informative**: Because messaging apps strip metadata, absence of EXIF/C2PA is not treated as evidence of AI generation.
4. **Digital Vector Art & Memes**: Illustrations and typography collages contain sharp synthetic-like vector lines; these should be screened upstream.

---

## 19. SIH Bonus Module Mapping

| SIH Module | Problem Statement Specification | SignalScope Implementation |
| :--- | :--- | :--- |
| **Module A: Faithful Explanation** | Explain why image was flagged; highlight spatial regions; non-hallucinatory. | **Implemented**: Grad-CAM on ConvNeXt Stage 3 Block 2, attribution concentration metric, 2D FFT spectral high-frequency ratio, and strictly grounded template summaries. |
| **Module B: Generator Attribution** | Attribute specific generator family (e.g. Midjourney, DALL-E, SD). | **Not Implemented**: *Omitted to avoid overconfident misattribution.* |
| **Module C: Robustness to Degradation** | Resilience to JPEG recompression, resizing, blur, and screenshots. | **Implemented**: 4-probe degradation matrix, probability drift $\Delta p$, and bounded Authenticity Stability Score $S \in [0, 1]$. |
| **Module D: Provenance & Metadata** | Analyze EXIF tags, software signatures, and C2PA credentials. | **Implemented**: Comprehensive EXIF parser, software marker scanner, and C2PA Content Credentials detector. |
| **Module E: Image + Text Consistency** | Cross-modal image-text consistency verification. | **Not Implemented**: *Pure visual/spectral forensic workstation.* |
| **Module F: Real-Time / Deployable** | Lightweight inference, low latency, deployable API. | **Implemented**: Sub-2.65s CPU / sub-150ms GPU execution, containerized Cloud Run backend, and responsive Next.js workstation. |
| **Module G: Active Defence Analysis** | Evaluation of adversarial perturbations or evasion attacks. | **Partially Analyzed**: Transformation perturbation stability evaluated; *full adversarial-defense module NOT claimed*. |

---

## 20. Dataset Provenance & Licensing Notes

- **SIH 2026 Training Pool**: 100,000 images (50,000 REAL, 50,000 FAKE) provided for the SIH 2026 Internal Hackathon.
- **AIGenImages2026 Benchmark**: Paired cross-generator dataset utilized exclusively for external multi-domain validation.
- **High-Resolution Photography**: Category partitions drawn from publicly available natural image domains for holdout validation.
- **License**: SignalScope code is released under the [MIT License](LICENSE). Third-party dependencies retain their respective open-source licenses (`timm`, `PyTorch`, `FastAPI`, `Next.js`).

---

## 21. Originality & Phased Development History

SignalScope was developed organically through a clean, auditable Git history across 10 structured engineering phases:
1. **Foundation & Scaffolding**: Pydantic schemas, directory structure, logging, Docker orchestration.
2. **Reproducible ML Environment**: PyTorch 2.6 CUDA stack, dataset audit, ConvNeXt-Tiny baseline training (Val AUC: 0.9992).
3. **Frequency Experiments**: 2D FFT and 2D DCT spectral branches and complementarity analysis.
4. **Degradation Robustness**: Implementation of Authenticity Stability scoring across 6 transformation probes.
5. **Calibration & Explainability**: Temperature scaling ($T=0.9986$), Grad-CAM spatial hook integration, and uncertainty corridors.
6. **Production Web Application**: Next.js 16 frontend with interactive inspection tools.
7. **Cloud Deployment**: Containerized Google Cloud Run service with readiness checks.
8. **Final Hardening**: Test suite expansion (78 passed tests), edge-case API error handling.
9. **Domain Generalization & Model v2**: Discovery of baseline high-res FPR, multi-domain retraining, 9,000-image photographic holdout gate, and promotion to `signalscope-v2`.
10. **Forensic Workstation Refinement**: Elimination of container shells, uncertain adjudication hierarchy fix, forensic docket header, and live Cloud Run promotion.

---

## 22. Submission Checklist & Artifact Catalog

- [x] **Problem Statement**: AI-Generated Synthetic Media Detection (SIH 2026 Internal Hackathon).
- [x] **Production Model**: `signalscope-v2` (`47f6b2a19d61...`).
- [x] **Rollback Model**: `signalscope-baseline-v1` (`c2e7881e...`).
- [x] **Live Workstation**: [https://sih.deskcraft.online](https://sih.deskcraft.online)
- [x] **Live API**: [https://signalscope-backend-780176122274.asia-south1.run.app](https://signalscope-backend-780176122274.asia-south1.run.app)
- [x] **Reproducibility Guide**: Documented above (10-minute setup with verified pytest suite).
- [x] **Organizer Held-Out Test Partition**: Verified completely untouched (0 accesses).
- [x] **Scorecard**: Documented in [`docs/final_scorecard.md`](docs/final_scorecard.md).
- [x] **Evaluation Script**: Documented in [`docs/demo_script.md`](docs/demo_script.md).
- [x] **Deterministic Demo Samples**: Documented in [`docs/demo_samples.md`](docs/demo_samples.md).
- [ ] **Demo Video**: **TODO — add final 3–5 minute demo video URL before submission.**

---

## 23. What SignalScope Explicitly Does NOT Claim

In accordance with scientific integrity and the SIH submission contract, SignalScope explicitly disclaims the following:
1. **Does NOT claim 100% detection accuracy**: No statistical detector can detect every possible generative architecture.
2. **Does NOT claim official competition test scores**: All reported figures are local or external validation metrics. The official held-out score is determined exclusively by SIH organizers.
3. **Does NOT claim to identify individuals or facial deepfakes**: SignalScope evaluates global and local visual signal artifacts, not biometric identities.
4. **Does NOT claim causal tampering proof**: Heatmaps highlight regions influencing neural network feature layers, not legal proof of manipulation.
5. **Does NOT claim generator attribution as an implemented classifier**: Generator family attribution is not implemented and is not claimed.
6. **Does NOT claim metadata absence indicates AI**: Absence of EXIF/C2PA is treated neutrally.

---

## 24. Final Position & Conclusion

SignalScope delivers an enterprise-grade, ethically bounded, and scientifically rigorous image authenticity detection workstation. By marrying deep spatial transfer learning with frequency-domain spectral analysis, empirical perturbation stability testing, and calibrated uncertainty, SignalScope provides evaluators with a dependable, transparent, and deployable instrument for detecting synthetic media.
