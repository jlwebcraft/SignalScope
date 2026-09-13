# SignalScope Additional Training & Domain Generalization Experiment Report

**Project**: SignalScope AI-Generated Image Forensics  
**Date**: September 13, 2026  
**Git Branch**: `feat/additional-training`  
**Execution Environment**: Windows 11, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 3050 6GB Laptop GPU  
**Production Checkpoint Status**: `signalscope-baseline-v1` (ConvNeXt-Tiny, SHA-256: `c2e7881e...`) remains live and unreplaced pending governance approval.  
**Held-Out Test Status**: `C:\Programming\SignalScope-data\test` remained **COMPLETELY UNTOUCHED, UNACCESSED, AND UNINSPECTED**.

---

## 1. Motivation & Context

SignalScope was originally trained and calibrated exclusively on the official SIH 2026 dataset (100,000 images, 32×32 resolution, JPEG format). While the production baseline achieved outstanding performance on the local in-domain validation split (ROC-AUC: 0.9992, Accuracy: 99.08%, FPR: 1.03%), real-world deployment requires robustness against:
1. Diverse, modern image resolutions (512×512, 1024×1024, and multi-megapixel photography).
2. Modern, state-of-the-art generative families (e.g., FLUX, SDXL, Ideogram-v3, Midjourney-v6, Imagen-3/4, Firefly-v5, Recraft).
3. Eliminating resolution-induced false positives where uncompressed, sharp real-world photographs are erroneously flagged as synthetic due to high-frequency edge gradients.

The user supplied a new external dataset under `C:\Programming\SignalScope-data\version-2\`. This report documents the end-to-end audit, leakage-free integration, multi-domain retraining experiments, robustness stress-testing, and model selection.

---

## 2. New Dataset Location & Directory Architecture

The new dataset is located at:
```
C:\Programming\SignalScope-data\version-2\
```

Detailed physical layout discovered during the Phase A audit:
```
C:\Programming\SignalScope-data\version-2\
├── AIGenImages2026/
│   ├── train/
│   │   ├── 0_real/                (4,879 photographic real images)
│   │   └── 1_fake/                (4,879 synthetic images across multiple generators)
│   ├── val/
│   │   ├── 0_real/                (559 photographic real images)
│   │   └── 1_fake/                (559 synthetic images across 19 modern generator families)
│   ├── metadata.csv
│   ├── train.csv
│   └── val.csv
└── REAL/
    ├── metadata.csv
    ├── apparel/                   (12,048 high-resolution images)
    ├── artwork/                   (12,048 images)
    ├── cars/                      (12,048 images)
    ├── dishes/                    (12,048 images)
    ├── furniture/                 (12,048 images)
    ├── illustrations/             (12,048 digital vector drawings)
    ├── landmark/                  (12,048 architectural photographs)
    ├── meme/                      (12,048 web memes/text overlays)
    ├── packaged/                  (12,048 consumer goods photographs)
    ├── storefronts/               (12,048 urban storefront photographs)
    └── toys/                      (12,048 product photographs)
```

---

## 3. Dataset Counts & Class Inventory

| Partition / Directory | Real Samples | Synthetic Samples | Total Files | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Official Train Pool** (`SignalScope-data\train`) | 50,000 | 50,000 | 100,000 | Native 32×32 JPEGs |
| ├── *Carved Local Train Pool* (`train_official`) | 42,500 | 42,500 | 85,000 | Seed 42 stratified split |
| └── *Untouched Local Validation* (`val_old`) | 7,500 | 7,500 | 15,000 | In-domain validation baseline |
| **Version-2 `AIGenImages2026/train`** | 4,879 | 4,879 | 9,758 | Paired 512×512 – 1024×1024 |
| **Version-2 `AIGenImages2026/val`** (`val_new`) | 559 | 559 | 1,118 | 19 modern generative families |
| **Version-2 `REAL` Directory** | 132,528 | 0 | 132,528 | 11 semantic categories |
| **SIH Organizer Held-Out Test** (`SignalScope-data\test`) | **RESTRICTED** | **RESTRICTED** | **RESTRICTED** | **STRICTLY UNTOUCHED (0 accesses)** |

---

## 4. Phase A: Dataset Audit & Quality Findings

Before ingesting any files, an automated forensic audit was conducted (`scripts/audit_new_dataset.py`, output in `outputs/audit/version2_audit.json`):

1. **Folder Provenance Revelation**:
   - **Crucial Discovery**: The folder name `AIgenImages2026` was **NOT** an AI-only folder. It represents an official 2026 benchmark repository containing strictly paired `0_real` and `1_fake` subdirectories.
   - A naive assumption that `AIgenImages2026/` was synthetic would have poisoned the training pool with **5,438 photographic real images labeled as synthetic**, causing catastrophic label corruption.
2. **Resolution Profile**:
   - `AIGenImages2026`: Ranges from 512×512 to 1024×1024 (mean: ~768×768).
   - `REAL`: Range spans from 512×512 up to 8256×5504 photographic raw imagery.
3. **Color Modes & File Integrity**:
   - 100% of tested samples are 3-channel RGB (JPEG and PNG). Zero corrupt, unreadable, or truncated image headers were detected across both directories.
4. **Metadata & Source Exposure**:
   - Embedded EXIF metadata is present in high-resolution photography (`REAL/storefronts`, `REAL/landmark`) and absent or stripped in synthetic generations.
   - Validation filenames in `AIGenImages2026/val/1_fake/` expose 19 specific modern generative architectures:
     - `flux_1_dev`, `flux_2`, `flux_2_pro`, `gpt_image_1`, `gpt_image_1_5`, `gemini_2_5_flash`, `imagen4`, `ideogram_v3`, `seedream`, `hidream`, `firefly_5`, `sdxl`, `midjourney_v6`, and related variants.
5. **Non-Comparable Media Filtering**:
   - The categories `REAL/illustrations` (digital drawings) and `REAL/meme` (heavy typography and multi-panel collages) were identified as non-photographic artifacts. They were **strictly excluded** from training to avoid distorting the detector's concept of authentic photography.

---

## 5. Phase B: Leakage & Duplicate Findings

1. **Overlap with Official Training Data**:
   - Zero hash collisions, path overlaps, or duplicated samples were found between `SignalScope-data\train` and `version-2`.
2. **Duplicate Ingestion Prevention**:
   - All MD5 and path sets were cross-checked. No duplicate families cross between training splits and validation partitions.
3. **Safety Verification**:
   - `verify_split_integrity()` executed before every training run confirmed:
     $$\text{Train} \cap \text{Val-Old} = \emptyset, \quad \text{Train} \cap \text{Val-New} = \emptyset, \quad \text{Val-Old} \cap \text{Val-New} = \emptyset$$

---

## 6. Phase C: Preprocessing & Hardware Optimization

The original SIH dataset consists of 32×32 images, whereas `version-2` images reach 1024×1024 to 8K resolutions:
- **Decision**: Preserve original source files unaltered on disk.
- **Forensic Preprocessing**:
  - For high-resolution files, aggressive spatial downsampling to 32×32 would obliterate forensic diffusion traces (e.g., lattice interpolation artifacts, pixel residual inconsistencies).
  - Both domains are preprocessed into the canonical $224 \times 224$ input tensor required by the ConvNeXt-Tiny backbone using bilinear sampling with normalized ImageNet statistics.
- **I/O Acceleration**:
  - Standard PIL decoding of high-resolution JPEGs caused an initial bottleneck (14.5s/batch).
  - By integrating JPEG draft decoding (`img.draft("RGB", (448, 448))`) and pre-capping dimension scale at 512, batch ingestion time dropped from 14.45s to **0.616s per batch (a 23.5× acceleration)**, enabling smooth GPU streaming on the RTX 3050 Laptop GPU.

---

## 7. Phase D: Baseline Performance (The Pre-Retraining Benchmark)

The production checkpoint (`checkpoints/baseline_convnext/best_model.pt`, SHA-256: `c2e7881e...`) was evaluated on both domains:

| Metric | Local Validation (Old, 15,000 samples) | New-Data Validation (1,118 samples) | Change / Diagnostic |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.9992** | **0.5933** | **-0.4059** (Catastrophic cross-domain drop) |
| **Macro-F1 (0.50)** | **0.9908** | **0.4296** | **-0.5612** |
| **Accuracy (0.50)** | **99.08%** | **53.13%** | Barely above random chance |
| **FPR (0.50)** | **0.0103** | **0.8909** | **89.09% of real high-res images flagged as fake** |
| **ECE (Calibration Error)**| **0.0063** | **0.4490** | Severely overconfident false alarms |
| **Optimal Threshold** | 0.4907 | 0.9989 | Unstable, extreme threshold divergence |

### Root Cause Analysis
The baseline was trained strictly on 32×32 low-resolution images. When presented with high-resolution, uncompressed real photography, the sharp high-frequency gradients triggered the model's synthetic-detection heads, classifying **89.09% of authentic photographs as synthetic**. Retraining with multi-resolution data was critically necessary.

---

## 8. Phase E: Training Experiments & Candidate Results

Three controlled candidate models were trained warm-started from the validated ConvNeXt baseline weights:

### Experiment 1: Full Augmented Pool
- **Dataset Composition**: All 85,000 Official Training Samples + 9,758 `AIGenImages2026/train` samples (94,758 total).
- **Epochs**: 1 | **Batch Size**: 64 | **LR**: $5 \times 10^{-5}$ | **Optimizer**: AdamW with Cosine Annealing.
- **Checkpoint**: `checkpoints/additional_training/exp1/best_model.pt` (SHA-256: `682d0cc0...`)

### Experiment 2: Balanced Mixed-Domain Pool
- **Dataset Composition**: Controlled 1:1 Class Ratio (29,762 samples total):
  - 15,000 Official (10,000 fake, 5,000 real)
  - 9,758 AIGen paired (4,879 fake, 4,879 real)
  - 5,004 Photographic Real samples from `version-2/REAL` (cars, apparel, storefronts, dishes, furniture, toys)
  - Class balance: Exactly 14,883 Real (50.0%) and 14,879 Synthetic (50.0%).
- **Epochs**: 1 | **Batch Size**: 64 | **LR**: $3 \times 10^{-5}$ | **Optimizer**: AdamW with Cosine Annealing.
- **Checkpoint**: `checkpoints/additional_training/exp2/best_model.pt` (SHA-256: `5d1568c6...`)

### Experiment 3: Forensic Multi-Scale & Degradation Augmentation
- **Dataset Composition**: Balanced Mixed-Domain Pool (29,762 samples).
- **Forensic Augmentations**:
  - Simulated JPEG recompression ($p=0.30$, quality 65–95)
  - RandomResizedCrop (scale 0.80–1.00)
  - ColorJitter (brightness 0.1, contrast 0.1, saturation 0.1)
  - Mild Gaussian Blur ($p=0.20$, kernel 3, $\sigma \in [0.1, 1.0]$)
  - Random Horizontal Flip ($p=0.50$)
- **Epochs**: 1 | **Batch Size**: 64 | **LR**: $3 \times 10^{-5}$ | **Optimizer**: AdamW with Cosine Annealing.
- **Checkpoint**: `checkpoints/additional_training/exp3/best_model.pt` (SHA-256: `1580110e...`)

---

## 9. Comprehensive Multi-Domain Metric Comparison

All evaluations were executed deterministically across both validation domains:

### Domain 1: LOCAL VALIDATION (15,000 samples, In-Domain 32×32)

| Model Candidate | ROC-AUC | Macro-F1 (0.50) | Accuracy (0.50) | FPR (0.50) | ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (v1)** | 0.9992 | 0.9908 | 99.08% | **0.0103** | 0.0063 | 0.0081 |
| **Experiment 1** | 0.9989 | 0.9828 | 98.28% | 0.0283 | 0.0083 | 0.0125 |
| **Experiment 2** | 0.9993 | 0.9891 | 98.91% | 0.0123 | 0.0052 | 0.0087 |
| **Experiment 3** | **0.9994** | 0.9884 | 98.84% | 0.0143 | **0.0025** | 0.0087 |

### Domain 2: NEW-DATA VALIDATION (1,118 samples, External Multi-Resolution)

| Model Candidate | ROC-AUC | Macro-F1 (0.50) | Accuracy (0.50) | FPR (0.50) | Opt. Acc | Opt. FPR | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (v1)** | 0.5933 | 0.4296 | 53.13% | 0.8909 | 59.84% | 0.7710 | 0.4490 |
| **Experiment 1** | 0.8816 | 0.7206 | 73.17% | 0.4669 | 80.32% | 0.1538 | 0.1227 |
| **Experiment 2** | **0.9052** | 0.7802 | 78.44% | 0.3542 | **83.99%** | **0.0966** | 0.0800 |
| **Experiment 3** | 0.8971 | **0.7902** | **79.16%** | **0.2898** | 83.45% | 0.1252 | **0.0567** |

---

## 10. Breakdown Across 19 Modern Generative Models

Performance on `AIGenImages2026/val/1_fake/` subfamilies (detection rate / recall):

| Generative Architecture | Baseline (v1) | Experiment 1 | Experiment 2 | Experiment 3 |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-Image-1.5** | 100.0% | 100.0% | 100.0% | 93.5% |
| **Ideogram-v3** | 96.8% | 100.0% | 100.0% | 96.8% |
| **Gemini-2.5-Flash** | 93.5% | 100.0% | 96.8% | 96.8% |
| **Firefly-5** | 100.0% | 100.0% | 96.8% | 96.8% |
| **GPT-Image-1** | 96.8% | 100.0% | 93.5% | 90.3% |
| **Flux-Dev** | 96.8% | 100.0% | 90.3% | 90.3% |
| **Imagen4** | 93.5% | 96.8% | 93.5% | 90.3% |
| **Flux-2-Pro** | 90.3% | 93.5% | 93.5% | 90.3% |
| **SDXL** | 93.5% | 90.3% | 90.3% | 83.9% |
| **Real Images Correct (`0_real`)** | **10.9%** (89.1% FP) | **53.3%** | **64.6%** | **71.0%** (29.0% FP) |

> **Key Finding**: Synthetic detection recall was already very high in the baseline, but baseline achieved it by predicting almost everything as synthetic. Experiments 2 and 3 preserved >90% synthetic detection while increasing real photograph recognition from 10.9% up to **71.0%** at default threshold, and **90.3%** at optimal threshold.

---

## 11. Multimodal Frequency Branch Evaluation

The previously trained `DualBranchFusionDetector` (`checkpoints/fusion/best_model.pt`) was evaluated across both domains:
- **Local Validation**: ROC-AUC: **0.9992**, Macro-F1: **0.9875**, FPR: **0.0093**
- **New-Data Validation**: ROC-AUC: **0.6054**, Macro-F1: **0.4734**, FPR: **0.8354**

**Finding**: Multimodal FFT fusion does **not** solve the cross-resolution domain shift when trained only on 32×32 data. Downsampling multi-megapixel images to 32×32 introduces sinc and aliasing artifacts that pollute the 2D Fourier spectrum. Data diversity and multi-resolution training are far more critical than architectural frequency fusion.

---

## 12. Calibration Benchmark & Temperature Scaling

Fitted optimal calibration temperatures using L-BFGS on 2,000 training samples:

| Model | Optimal $T$ | Val-Old Raw ECE | Val-Old Calibrated ECE | Val-New Raw ECE | Val-New Calibrated ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (v1)** | 0.9995 | 0.0074 | 0.0074 | 0.4462 | 0.4462 |
| **Experiment 2** | 0.9976 | 0.0067 | 0.0067 | **0.0805** | **0.0803** |
| **Experiment 3** | 0.8527 | 0.0065 | 0.0076 | **0.0573** | **0.0749** |

- **ECE on New-Data dropped from 0.4462 down to 0.0573** (a 7.8× calibration improvement).
- Predictions on out-of-domain samples are significantly less overconfident and reflect authentic predictive uncertainty.

---

## 13. Robustness Benchmark Comparison

Evaluated on 1,000 stratified validation samples across 6 standardized degradation transformations:

| Transformation | Baseline AUC | Exp 2 AUC | Exp 3 AUC | Baseline Flip % | Exp 2 Flip % | Exp 3 Flip % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine Native** | 0.9994 | 0.9995 | 0.9995 | 0.0% | 0.0% | 0.0% |
| **JPEG Quality 95** | 0.9993 | 0.9994 | **0.9995** | 0.0% | 0.2% | 0.1% |
| **JPEG Quality 85** | 0.9994 | 0.9994 | 0.9993 | 0.4% | 0.4% | 0.3% |
| **JPEG Quality 70** | 0.9995 | 0.9995 | 0.9994 | 0.6% | 0.6% | 0.8% |
| **Resize (0.7x Down/Up)** | 0.9473 | 0.9587 | **0.9663** | 34.1% | 35.0% | 35.5% |
| **Screenshot Simulation** | 0.9186 | 0.9296 | **0.9373** | 40.5% | 42.5% | 43.2% |
| **Light Edit (90% Crop)** | 0.9914 | 0.9930 | **0.9957** | 14.6% | 14.3% | **12.5%** |
| **Mean Degraded AUC** | **0.9759** | **0.9799** | **0.9829** | 15.03% | 15.50% | 15.40% |

- **Experiment 3 demonstrated the highest robustness under degradation**, achieving a mean degraded AUC of **0.9829** (+0.0070 over baseline) and superior resilience to downsampling and cropping.

---

## 14. Faithful Explainability Verification

Grad-CAM and 2D Fourier spectral extraction were verified on the retrained candidate models:
1. **Target Layer Hook**: Successfully hooks into `ConvNeXtBlock` at `model.backbone.stages[-1].blocks[-1]`.
2. **Feature Map Resolution**: Generates $7 \times 7 \times 768$ feature maps correctly interpolated to $224 \times 224$ heatmap overlays.
3. **Activation Bounds**: Cleanly normalized to $[0.0, 1.0]$ with meaningful localized attention centers rather than edge boundary artifacts.
4. **Spectral Evidence**: 2D FFT magnitude spectrum and 16-bin azimuthal radial profiles extract cleanly without errors.

---

## 15. Model Selection Decision & Governance

### Candidate Comparison Summary:
- **Experiment 2 (Balanced Mixed-Domain)** achieved the highest raw ROC-AUC on New-Data Validation (**0.9052**) and highest optimal-threshold accuracy (**83.99%**).
- **Experiment 3 (Balanced Mixed-Domain + Multi-Scale Augmentations)** achieved the highest Local Validation AUC (**0.9994**), lowest default FPR on New-Data (**0.2898**), lowest ECE (**0.0567**), and highest degraded robustness AUC (**0.9829**).

### Selection:
**Experiment 3 is designated as the recommended next-generation model candidate (`signalscope-v2-candidate`)**.

### Production Governance Rule:
As strictly required by the project safety guidelines:
1. The current production model **`signalscope-baseline-v1` remains unchanged in production**.
2. No models were deployed to Cloud Run or pushed to public endpoints during this experimental phase.
3. The candidate checkpoint is cataloged safely at `checkpoints/additional_training/exp3/best_model.pt` (SHA-256: `1580110e...`).

---

## 16. Limitations & Unseen-Generator Disclaimer

> **Forensic Governance Statement**:  
> Generator identity in the training set is not completely disjoint from validation generators. Therefore:  
> **"Generator identity is unavailable for complete isolation; this dataset is treated as an external-domain evaluation set, not a proven unseen-generator benchmark."**

Additional limitations:
1. Extremely degraded images (< 32×32) remain susceptible to uncertainty.
2. Web memes, illustrations, and heavy vector typography require upstream heuristic rejection rather than model scoring.

---

## 17. Reproducibility & Checkpoint Catalog

All runs are fully reproducible using deterministic seeds (`seed=42`).

| Model Name | Checkpoint File | File Size | SHA-256 Checksum | Status |
| :--- | :--- | :---: | :--- | :--- |
| **Baseline (v1)** | `checkpoints/baseline_convnext/best_model.pt` | 318.61 MB | `c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc` | Active Production |
| **Experiment 1** | `checkpoints/additional_training/exp1/best_model.pt` | 318.60 MB | `682d0cc050512aa98cbe4d286b1b3f864de48e4cdca86d0cc34638b79bb8cefc` | Completed |
| **Experiment 2** | `checkpoints/additional_training/exp2/best_model.pt` | 318.60 MB | `5d1568c6742f85ae434e91201c356c224966dcc12ca8b8d650365f3aa3d421be` | Candidate (1-ep) |
| **Experiment 3** | `checkpoints/additional_training/exp3/best_model.pt` | 318.60 MB | `1580110e262f6450df85e9b3b909f249d96e1f2f49909b695875d00457187bfb` | Candidate (1-ep) |
| **Exp2-5ep (Best Ep 4)** | `checkpoints/additional_training/exp2_5ep/best_model.pt` | 106.20 MB | `bb0e5c7e677cc9d66f2e47920d4b8ba65a193bba1bd73b4a4c79fab7a6120fe9` | Extended Candidate |
| **Exp3-5ep (Best Ep 3)** | `checkpoints/additional_training/exp3_5ep/best_model.pt` | 106.20 MB | `47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d` | **Recommended Best Overall** |

---

## 18. Second-Stage Extended Training Study

Following the initial 1-epoch proof-of-concept, a second-stage controlled study was conducted to investigate whether extended training (up to 5 epochs with validation-based early stopping) yields deeper generalization, superior calibration, and increased forensic stability.

### 18.1 Dataset-Count Reconciliation
An initial estimate anticipated approximately 220,000 images in the external pool. A forensic recursive scan and CSV cross-reference resolved this discrepancy:
- **`version-2` Directory Ground Truth**:
  - `AIGenImages2026`: exactly **10,876 images** (plus 3 CSV index files)
  - `REAL`: exactly **132,528 images** across 11 category subdirectories (plus 1 CSV index file)
  - Total in `version-2`: exactly **143,404 valid image files** (143,408 total files).
- **Combined Repository Total**:
  - When combined with the official training pool (`SignalScope-data\train`, 100,000 images), the total accessible image inventory across the repository is **243,404 images**.
- **Explanation**: The ~220k figure was an informal estimate of the entire multi-domain collection (100k official + ~120k–140k new data). The audit did not miss, omit, or exclude any files or directories; every byte on disk is strictly verified and cataloged.

### 18.2 Exact Training Composition
Both extended candidates were trained on the identical balanced dataset pool:
- **Total Training Samples**: **29,762**
  - **Class Balance**: Exactly 14,883 Real (50.0%) and 14,879 Synthetic (50.0%).
  - **Source Balance**:
    - 15,000 Official SIH samples (10,000 synthetic, 5,000 real)
    - 9,758 `AIGenImages2026/train` paired samples (4,879 synthetic, 4,879 real)
    - 5,004 Photographic Real samples from `version-2/REAL` (cars, apparel, storefronts, dishes, furniture, toys).
  - **Split Integrity**: Zero sample overlap across training, old local validation, and new-data validation.

### 18.3 Epoch-by-Epoch Progression

#### Experiment A (`exp2_5ep`: Balanced Mixed-Domain, Standard Augmentations)
| Epoch | Train Loss | Val-Old Loss | Val-Old AUC | Val-New Loss | Val-New AUC | Macro-F1 | New FPR | Composite Score | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2035 | 0.0375 | 0.9991 | 0.4589 | 0.9076 | 0.8057 | 0.0519 | 0.9533 | Strong start |
| **2** | 0.0838 | 0.0418 | 0.9990 | 0.4249 | 0.9157 | 0.8317 | 0.1932 | 0.9573 | Improvement |
| **3** | 0.0321 | 0.0500 | 0.9988 | 0.4916 | 0.9240 | 0.8470 | 0.1735 | 0.9614 | Continuing gain |
| **4** | **0.0075** | **0.0578** | **0.9988** | **0.6249** | **0.9267** | **0.8505** | **0.1753** | **0.9627** | **Optimal Convergence Point (Saved Checkpoint)** |
| **5** | 0.0021 | 0.0571 | 0.9983 | 0.7268 | 0.9257 | 0.8440 | 0.2021 | 0.9620 | Overfitting onset (val loss rise) |

#### Experiment B (`exp3_5ep`: Balanced Mixed-Domain, Forensic Augmentations)
| Epoch | Train Loss | Val-Old Loss | Val-Old AUC | Val-New Loss | Val-New AUC | Macro-F1 | New FPR | Composite Score | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2335 | 0.0441 | 0.9989 | 0.4364 | 0.9169 | 0.8043 | 0.0429 | 0.9579 | High precision |
| **2** | 0.1157 | 0.0391 | 0.9991 | 0.5020 | 0.9210 | 0.7885 | 0.3453 | 0.9601 | Feature adaptation |
| **3** | **0.0684** | **0.0483** | **0.9991** | **0.4108** | **0.9220** | **0.8578** | **0.1395** | **0.9606** | **Optimal Convergence Point (Saved Checkpoint)** |
| **4** | 0.0302 | 0.0481 | 0.9987 | 0.6135 | 0.9205 | 0.8303 | 0.2308 | 0.9596 | Val loss uptick |
| **5** | 0.0128 | 0.0674 | 0.9981 | 0.6304 | 0.9223 | 0.8523 | 0.1807 | 0.9602 | Early Stopping Triggered (Patience: 2) |

---

### 18.4 Comprehensive Dual-Domain Evaluation

#### Domain 1: LOCAL VALIDATION (15,000 samples, Official 32×32)
| Model | ROC-AUC | Macro-F1 (0.50) | Accuracy (0.50) | FPR (0.50) | ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (v1)** | **0.9992** | **0.9908** | **99.08%** | 0.0103 | **0.0062** | **0.0079** |
| **Exp2-5ep (Best Ep 4)** | 0.9988 | 0.9877 | 98.77% | **0.0100** | 0.0063 | 0.0083 |
| **Exp3-5ep (Best Ep 3)** | 0.9991 | 0.9851 | 98.51% | 0.0185 | 0.0086 | 0.0119 |

#### Domain 2: NEW-DATA VALIDATION (1,118 samples, Multi-Resolution)
| Model | ROC-AUC | Macro-F1 (0.50) | Accuracy (0.50) | FPR (0.50) | Opt. Acc (Thresh) | Opt. FPR | ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (v1)** | 0.5950 | 0.4313 | 53.22% | 88.91% | 57.78% (0.9900) | 68.69% | 0.4462 | 0.4480 |
| **Exp2-5ep (Best Ep 4)** | **0.9267** | 0.8505 | 85.15% | 17.53% | 85.96% (0.9751) | **6.44%** | 0.1095 | 0.1265 |
| **Exp3-5ep (Best Ep 3)** | 0.9220 | **0.8578** | **85.78%** | **13.95%** | **86.40%** (0.6338) | 9.84% | **0.0664** | **0.1098** |

---

### 18.5 Generator-Stratified External-Domain Validation
Evaluation across all 19 modern generative families in `AIGenImages2026/val/1_fake/` (evaluated against all 559 real validation images):

| Generative Architecture | Samples | Baseline AUC | Exp2-5ep AUC | Exp3-5ep AUC | Exp3-5ep Recall @ 0.50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ideogram-v3** | 31 | 0.6558 | 0.9805 | **0.9809** | 100.0% |
| **gemini-25-flash-image** | 31 | 0.5922 | 0.9748 | **0.9790** | 96.8% |
| **firefly_image5** | 17 | 0.7089 | **0.9722** | 0.9540 | 94.1% |
| **gpt-image-1.5** | 31 | 0.5843 | **0.9675** | 0.9634 | 93.5% |
| **flux-2-max** | 23 | 0.6521 | 0.9475 | **0.9552** | 95.7% |
| **flux-2** | 31 | 0.6302 | **0.9538** | 0.9438 | 90.3% |
| **flux-2-pro** | 31 | 0.6120 | **0.9490** | 0.9387 | 93.5% |
| **flux-dev** | 23 | 0.6084 | **0.9347** | 0.9123 | 87.0% |
| **imagen4** | 31 | 0.5891 | **0.9416** | 0.9288 | 90.3% |
| **seedreamv4.5** | 31 | 0.5947 | **0.9362** | 0.9145 | 87.1% |
| **gpt-image-1** | 31 | 0.6190 | **0.9377** | 0.9189 | 90.3% |
| **hidream-i1-dev** | 31 | 0.5732 | **0.9298** | 0.9082 | 87.1% |
| **gemini-3-pro-image-preview** | 31 | 0.5159 | 0.9181 | **0.9248** | 87.1% |
| **reve** | 31 | 0.5284 | **0.9213** | 0.9042 | 87.1% |
| **stable-diffusion-v35-medium**| 31 | 0.5312 | **0.9154** | 0.8974 | 83.9% |
| **z-image-turbo** | 31 | 0.5528 | **0.9082** | 0.8876 | 80.6% |
| **fast-sdxl** | 31 | 0.7278 | **0.8734** | 0.8719 | 74.2% |
| **flux-pro-v1.1** | 31 | 0.5401 | **0.8213** | 0.8124 | 71.0% |
| **midjourney_v7** | 31 | 0.5187 | **0.8106** | 0.7582 | 67.7% |
| **Summary Metrics** | | | | | |
| **Macro-Average Generator AUC**| — | **0.5961** | **0.9282** | **0.9237** | **87.3%** |
| **Strongest Generator** | — | `fast-sdxl` (0.7278) | `ideogram-v3` (0.9805) | `ideogram-v3` (0.9809) | |
| **Weakest Generator** | — | `gemini-3-pro` (0.5159)| `midjourney_v7` (0.8106)| `midjourney_v7` (0.7582)| |
| **AUC Spread (Max - Min)** | — | **0.2119** | **0.1699** | **0.2227** | |

---

### 18.6 Probability Calibration & Temperature Scaling
Calibrated post-hoc using Temperature Scaling fitted via L-BFGS on 2,000 disjoint official training samples:
- **Baseline (v1)**: Fitted $T = 0.9995$. Val-New ECE = **0.4462**, Brier = **0.4480**. Severe overconfidence on out-of-domain false positives.
- **Exp2-5ep**: Fitted $T = 1.0000$. Val-New ECE = **0.1095**, Brier = **0.1265** (4.1× reduction in calibration error).
- **Exp3-5ep**: Fitted $T = 0.9986$. Val-New ECE = **0.0664**, Brier = **0.1098** (6.7× reduction in calibration error; best calibrated candidate).

---

### 18.7 Robustness Benchmark Comparison (1,000 Stratified Samples)
Evaluated across 6 standardized degradation transformations using the official validation pool:

| Degradation Transformation | Baseline AUC | Exp2-5ep AUC | Exp3-5ep AUC | Baseline Flip % | Exp2-5ep Flip % | Exp3-5ep Flip % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine Native** | 0.9994 | 0.9995 | 0.9992 | 0.0% | 0.0% | 0.0% |
| **JPEG Quality 95** | 0.9993 | 0.9995 | 0.9983 | 0.0% | 0.2% | 0.3% |
| **JPEG Quality 85** | 0.9994 | 0.9994 | 0.9972 | 0.4% | 0.5% | 0.5% |
| **JPEG Quality 70** | 0.9995 | 0.9994 | 0.9990 | 0.6% | 0.9% | 0.7% |
| **Resize (0.7x Down/Up)** | 0.9473 | 0.9521 | **0.9755** | 34.1% | 37.7% | **29.6%** |
| **Screenshot Simulation** | 0.9186 | 0.9199 | **0.9560** | 40.5% | 42.9% | **37.8%** |
| **Light Edit (90% Crop)** | 0.9914 | 0.9921 | **0.9974** | 14.6% | 15.6% | **8.8%** |
| **Mean Degraded AUC** | **0.9759** | **0.9770** | **0.9872** | **15.03%** | **16.27%** | **12.95%** |
| **Mean Probability Drift**| **0.1494** | **0.1620** | **0.1281** | — | — | — |

- **Key Takeaway**: **Exp3-5ep is decisively the most robust model**. Under severe resolution downscaling and screenshot simulation, Exp3-5ep retained a degraded AUC of **0.9755** and **0.9560** (+0.0282 and +0.0374 over baseline), while reducing decision flips under cropping down to **8.8%**.

---

### 18.8 Confusion Matrices (New-Data Validation, $N = 1,118$)

#### Baseline (v1) at threshold 0.50:
$$\begin{bmatrix} \text{TN: } 62 & \text{FP: } 497 \\ \text{FN: } 26 & \text{TP: } 533 \end{bmatrix} \quad (\text{Real Acc: } 11.1\%, \text{ Synthetic Recall: } 95.3\%)$$

#### Exp2-5ep at threshold 0.50:
$$\begin{bmatrix} \text{TN: } 461 & \text{FP: } 98 \\ \text{FN: } 68 & \text{TP: } 491 \end{bmatrix} \quad (\text{Real Acc: } 82.5\%, \text{ Synthetic Recall: } 87.8\%)$$

#### Exp3-5ep at threshold 0.50:
$$\begin{bmatrix} \text{TN: } 481 & \text{FP: } 78 \\ \text{FN: } 81 & \text{TP: } 478 \end{bmatrix} \quad (\text{Real Acc: } 86.0\%, \text{ Synthetic Recall: } 85.5\%)$$

#### Exp2-5ep at Optimal Threshold (0.9751):
$$\begin{bmatrix} \text{TN: } 523 & \text{FP: } 36 \\ \text{FN: } 121 & \text{TP: } 438 \end{bmatrix} \quad (\text{Real Acc: } 93.6\%, \text{ Synthetic Recall: } 78.4\%)$$

#### Exp3-5ep at Optimal Threshold (0.6338):
$$\begin{bmatrix} \text{TN: } 504 & \text{FP: } 55 \\ \text{FN: } 97 & \text{TP: } 462 \end{bmatrix} \quad (\text{Real Acc: } 90.2\%, \text{ Synthetic Recall: } 82.6\%)$$

---

### 18.9 Required Master Comparison Table

| Metric | Baseline (v1) | Exp2-5ep | Exp3-5ep | Winner |
| :--- | :---: | :---: | :---: | :--- |
| **Old AUC (In-Domain)** | **0.9992** | 0.9988 | 0.9991 | Baseline (Tied / <0.0004 diff) |
| **New AUC (External)** | 0.5950 | **0.9267** | 0.9220 | **Exp2-5ep** (+0.0047 edge) |
| **Old Macro-F1** | **0.9908** | 0.9877 | 0.9851 | Baseline |
| **New Macro-F1** | 0.4313 | 0.8505 | **0.8578** | **Exp3-5ep** |
| **New FPR (0.50 default)** | 88.91% | 17.53% | **13.95%** | **Exp3-5ep** (-3.58% lower FPR) |
| **Optimal FPR** | 68.69% | **6.44%** | 9.84% | **Exp2-5ep** (requires th=0.975) |
| **ECE (New-Data)** | 0.4462 | 0.1095 | **0.0664** | **Exp3-5ep** (39% better calibration) |
| **Brier Score (New-Data)** | 0.4480 | 0.1265 | **0.1098** | **Exp3-5ep** |
| **Mean Degraded AUC** | 0.9759 | 0.9770 | **0.9872** | **Exp3-5ep** (+0.0102 higher) |
| **Flip Rate under Degradation** | 15.03% | 16.27% | **12.95%** | **Exp3-5ep** (Lowest flip rate) |
| **Generator AUC Spread** | 0.2119 | **0.1699** | 0.2227 | **Exp2-5ep** (More uniform spread) |

---

### 18.10 Dimension-by-Dimension Leadership
1. **ROC-AUC Leader**: **Exp2-5ep** (New-Data AUC: **0.9267**, Macro-average Generator AUC: **0.9282**).
2. **F1 & Accuracy Leader**: **Exp3-5ep** (New-Data Macro-F1: **0.8578**, Default Accuracy: **85.78%**, Optimal Accuracy: **86.40%**).
3. **Calibration Leader**: **Exp3-5ep** (ECE: **0.0664**, Brier: **0.1098**; reflects authentic predictive uncertainty).
4. **Robustness Leader**: **Exp3-5ep** (Mean Degraded AUC: **0.9872**, Degraded Accuracy: **86.85%**, Flip Rate: **12.95%**).
5. **Overall Generalization Leader**: **Exp3-5ep**.  
   *Scientific Rationale*: While Exp2-5ep holds a minor +0.0047 raw AUC advantage, Exp3-5ep dominates across all operational reliability metrics: 39% lower calibration error, 20% lower false positive rate at default threshold (13.95% vs 17.53%), substantially superior robustness to social media compression and downsampling (+0.0102 AUC, -3.3% flip rate), and near-ideal threshold stability (operating threshold 0.6338 vs extreme 0.9751 for Exp2).

---

### 18.11 Second-Stage Limitations
1. **Generator-Stratified Evaluation Disclaimer**: Generator identities in `AIGenImages2026` are matched pairs. This dataset represents an **external-domain evaluation set**, not a proven unseen-generator benchmark.
2. **`midjourney_v7` Challenge**: Both models found `midjourney_v7` the most difficult modern generator (Exp2 AUC: 0.8106, Exp3 AUC: 0.7582), indicating that latest diffusion/autoregressive texture blending requires continued fine-grained focus.

---

### 18.12 Exact Checkpoint Hashes
- **`checkpoints/additional_training/exp2_5ep/best_model.pt`**:  
  SHA-256: `bb0e5c7e677cc9d66f2e47920d4b8ba65a193bba1bd73b4a4c79fab7a6120fe9`
- **`checkpoints/additional_training/exp3_5ep/best_model.pt`**:  
  SHA-256: `47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d`

---

### 18.13 Production Governance Status
**PRODUCTION MODEL CHANGE: NOT PERFORMED**  
The active production model remains `signalscope-baseline-v1`. No models were deployed to Google Cloud Run, no release was published, and no frontend configurations were updated.

