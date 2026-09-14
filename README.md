# SignalScope
### *"Detecting Real vs Synthetic Media with Responsible Uncertainty"*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.3-black.svg)](https://nextjs.org/)
[![SIH 2026](https://img.shields.io/badge/Event-SIH%202026%20Internal%20Hackathon-purple.svg)]()

> [!IMPORTANT]
> **Official Evaluation Scope & Anti-Leakage Notice**: All reported development and validation metrics in this repository are evaluated strictly on a 15,000-image validation split constructed from `train/`. In strict adherence to SIH 2026 competition rules, the organizer's held-out test partition (`C:\Programming\SignalScope-data\test`) was **kept strictly untouched**. No competition test claims are made.

---

## Quick Navigation for Judges & Evaluators
- **Final Submission Report**: [`report/final_submission_report.md`](report/final_submission_report.md)
- **SIH Judging Scorecard**: [`docs/final_scorecard.md`](docs/final_scorecard.md)
- **3–5 Minute Demo Script**: [`docs/demo_script.md`](docs/demo_script.md)
- **Deterministic Demo Samples**: [`docs/demo_samples.md`](docs/demo_samples.md)
- **Deployment Architecture & Host Feasibility**: [`docs/deployment_architecture.md`](docs/deployment_architecture.md)

---


## 1. Project Overview & Official Problem Requirements

In the era of advanced generative modeling (Latent Diffusion Models, GANs, Flow Matching, Autoregressive visual transformers), distinguishing authentic physical photography from synthetic generations is a critical societal and technical imperative.

**SignalScope** is an evidence-fusion authenticity detection system designed specifically for **unseen-generator generalization** and **degradation resilience**.

### Core Requirements Matrix

| Requirement | Implementation Status | Technical Details |
|---|---|---|
| **Single Image Ingestion** | Implemented (Phase 1) | Strict format/boundary validation, decompression bomb protection |
| **Classification (Real vs AI)** | Foundation Ready (Phase 1) | Binary classification with probabilistic thresholding |
| **Confidence Score** | Implemented (Phase 1) | Calibrated probability margin + stability weighting |
| **Honest Train/Val/Test** | Designed (Phase 1) | Generator-aware stratified split preventing data leakage |
| **ROC-AUC, Macro-F1, FPR, CM** | Implemented (Phase 1) | Mandatory metrics evaluation suite (`model/evaluate.py`) |
| **Unseen-Generator Generalization** | Designed (Phases 3-4) | Dual-branch spatial + frequency fusion architecture |
| **Authenticity Stability Score** | Implemented (Phase 1) | Invariance testing across JPEG, resizing, & screenshots |
| **Working Prediction Interface** | Implemented (Phase 1) | FastAPI REST endpoints (`/health`, `/predict`) + CLI runner |
| **Reproducibility** | Implemented (Phase 1) | Dockerfile, docker-compose, configs, deterministic seeds |

---

## 2. Ethical Scope & Responsible AI Principles

SignalScope strictly adheres to ethical safeguards:
- **No Personal Identification**: SignalScope is **not** an identity verification, deepfake facial recognition, or personal profiling tool.
- **No Accusatory Language**: Results are reported probabilistically as `Likely AI-generated`, `Likely real`, or `Uncertain`.
- **Transparent Uncertainty**: Borderline predictions and conflicting evidence signals are surfaced as `Uncertain` with detailed diagnostic reports.
- **No Political Adjudication**: Focused strictly on visual signal anomalies, provenance, and spectral statistics.

---

## 3. High-Level Architecture & Technical Direction

Instead of relying solely on an isolated black-box classifier (`image -> CNN -> fake`), SignalScope implements a **Multimodal Evidence Fusion** pipeline:

```
                           +------------------------+
                           |      Input Image       |
                           +-----------+------------+
                                       |
                   +-------------------+--------------------+
                   |                                        |
                   v                                        v
        +--------------------+                   +--------------------+
        | RGB Spatial Branch |                   |  Frequency Branch  |
        |   ConvNeXt-Tiny    |                   |   2D FFT/DCT Spec  |
        +----------+---------+                   +---------+----------+
                   |                                        |
                   +-------------------+--------------------+
                                       |
                                       v
                             +-------------------+
                             |  Evidence Fusion  |
                             +---------+---------+
                                       |
                                       v
                            +--------------------+
                            |    Calibration     |
                            +----+----------+----+
                                 |          |
                    +------------+          +------------+
                    v                                    v
           +-----------------+                  +-----------------+
           | Authenticity    |                  | Grounded        |
           | Verdict         |                  | Explanation     |
           +-----------------+                  +-----------------+
```

### Key Innovation: Authenticity Stability Score
Synthetic images often exhibit brittle, high-frequency artifacts that disintegrate or swing wildly when subjected to routine web transformations. SignalScope tests every image against controlled transformations:
- **Pristine Original**
- **JPEG Recompression** (Quality = 70)
- **Resolution Rescaling** (Downsample 0.6x & Bicubic Upsample)
- **Screenshot Simulation** (Subpixel blur + recompression)
- **Light Crop & Resize** (92% frame crop)

If predictions swing drastically across these operations, the system lowers the stability score and flags high degradation impact.

---

## 4. Repository Structure

```
SignalScope/
|
├── README.md               # Master documentation and setup guide
├── requirements.txt        # Python dependency manifest
├── environment.yml         # Conda environment definition (Python 3.11)
├── .gitignore              # Strict exclusions for checkpoints, datasets, & caches
├── LICENSE                 # Open-source MIT License
├── Dockerfile              # Containerized production deployment
├── docker-compose.yml      # Multi-container orchestration
|
├── app/                    # FastAPI backend service
│   ├── main.py             # Application entry point and lifespan
│   ├── api/                # API schemas and router
│   │   ├── schemas.py      # Pydantic v2 schemas for evidence & verdicts
│   │   └── v1/             # Endpoints (/health, /info, /predict)
│   ├── inference/          # Inference orchestration engine
│   │   └── engine.py       # Evidence fusion engine
│   ├── services/           # Domain services
│   │   ├── stability.py    # Authenticity Stability Score evaluation
│   │   ├── metadata.py     # EXIF provenance and AI generator signatures
│   │   └── explainability.py # Evidence-grounded explanation synthesis
│   └── utils/              # Logging and safe image ingestion
│       ├── logger.py       # Centralized structured logger
│       └── image.py        # Validation and bounds checking
|
├── model/                  # Deep learning pipelines
│   ├── train.py            # Training harness with seed and experiment tracking
│   ├── evaluate.py         # Mandatory evaluation metrics (AUC, F1, FPR, CM)
│   ├── dataset.py          # Generator-aware PyTorch dataset loaders
│   ├── predict.py          # Standalone CLI prediction interface
│   ├── architectures/      # Model definitions (ConvNeXt, FFT, Fusion)
│   │   ├── convnext.py     # ConvNeXt-Tiny spatial branch
│   │   ├── frequency.py    # 2D FFT spectral branch
│   │   └── fusion.py       # Dual-branch multimodal fusion head
│   ├── losses/             # Loss functions (Label smoothed BCE, Focal)
│   └── configs/            # Reproducible YAML experiment configs
|
├── report/                 # Official hackathon evaluation reports
│   └── model_report.md     # In-depth technical report and metric logs
|
├── notebooks/              # Exploratory and analysis notebooks
├── scripts/                # Diagnostic and helper scripts
│   └── run_checks.py       # Pre-flight diagnostic runner
├── tests/                  # Pytest verification suite
└── frontend/               # Next.js web application (Phase 10)
```

---

## 5. Quick Start & Setup Instructions

### Prerequisites
- Python 3.11+ (or Anaconda / Miniconda)
- Git
- NVIDIA GPU (Optional, recommended for training)

### Option A: Standard Python Virtual Environment
```bash
# 1. Clone repository
git clone https://github.com/jlwebcraft/SignalScope.git
cd SignalScope

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda Environment
```bash
conda env create -f environment.yml
conda activate signalscope
```

### Option C: Full Stack Docker Deployment
```bash
docker compose up --build
```
- **Backend API**: `http://localhost:8000` (interactive OpenAPI docs at `http://localhost:8000/docs`)
- **Frontend Web UI**: `http://localhost:3000`

---

## 6. Running the System & Local Development

### 1. Launch Backend API (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Launch Frontend Web Application (Next.js + TypeScript + Tailwind CSS)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to access the interactive web application.

### 3. Build Frontend for Production
```bash
cd frontend
npm run build
npm run start
```

### 4. Run CLI Prediction Runner
```bash
# Standalone execution using trained baseline checkpoint:
python model/predict.py --image path/to/image.jpg --checkpoint checkpoints/baseline_convnext/best_model.pt

# Output as structured JSON (schema v1.0):
python model/predict.py --image path/to/image.jpg --checkpoint checkpoints/baseline_convnext/best_model.pt --json
```

### 5. Run Verification & Diagnostics
```bash
# Pre-flight environment and hardware check:
python scripts/run_checks.py

# Complete Pytest integration test suite (61 tests):
pytest -v

# Automated End-to-End production verification benchmark:
python scripts/verify_production_e2e.py
```

---

## 7. Production API Endpoints & Response Contract

- `GET /health`: Basic service liveness health check (status, version).
- `GET /ready`: Readiness probe distinguishing HTTP service availability from model checkpoint loading.
- `GET /info`: Model architecture (`ConvNeXtTinyDetector`), production model version (`signalscope-v2`), weights SHA-256 hash (`47f6b2a1...`), calibration temperature ($T=0.9986$), and rollback coordinates.
- `POST /predict`: Production inference endpoint accepting multipart image upload (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`, max 25MB).

### Response Schema Contract (`schema_version: "1.0"`)
```json
{
  "schema_version": "1.0",
  "verdict": "likely_ai_generated",
  "probability": 1.0,
  "confidence_level": "medium",
  "uncertain": false,
  "evidence": {
    "spatial": {
      "available": true,
      "heatmap": "data:image/png;base64,...",
      "attribution_concentration": 0.324
    },
    "spectral": {
      "available": true,
      "spectrum": "data:image/png;base64,...",
      "high_frequency_energy_ratio": 0.281
    },
    "robustness": {
      "available": true,
      "stability_score": 1.0,
      "prediction_flip_rate": 0.0,
      "mean_probability_drift": 0.0
    },
    "metadata": {
      "available": true,
      "has_exif": false,
      "c2pa_present": false
    }
  },
  "explanation": "Predicted as likely AI-generated with 100.0% probability. Grad-CAM spatial attribution observed localized activation patterns. Spectral analysis observed elevated high-frequency energy. Prediction stability was high under the transformations evaluated for this sample.",
  "disclaimer": "This assessment is a probabilistic estimation of visual signal artifacts and does not constitute definitive proof of synthetic origin."
}
```

### Responsible Language & Uncertainty Framework
SignalScope enforces strict linguistic integrity:
- Results are reported probabilistically as `Likely AI-generated`, `Likely real`, or `Uncertain` (with `"Human review recommended."`).
- Raw sigmoid outputs are never termed "calibrated" without the empirical post-hoc temperature scaling layer.
- High-frequency energy is reported as an *observed spectral statistic*, not "universal proof of AI generation".
- Transformation stability is reported as *sample-specific stability under tested transformations*, never generalized to all image alterations.

---

## 8. Latency & Performance Breakdown

### Consumer GPU Latency (NVIDIA GeForce RTX 3050 Laptop GPU, native 32×32 input)
- **Model Loading & Cold Start**: ~467 ms
- **Primary ConvNeXt Forward Pass**: **16.1 ms**
- **Temperature Scaling ($T=0.9995$)**: **< 0.05 ms**
- **Grad-CAM Saliency Extraction**: **37.7 ms**
- **2D FFT Spectral Feature Generation**: **2.9 ms**
- **Authenticity Stability Probing (4 transformations)**: **65.1 ms**
- **Total Single-Image Inference Pipeline**: **~121.8 ms**
- **Full HTTP Request-Response Latency**: **~139.5 ms**

### Cloud CPU Latency (Standard x86_64 CPU Inference, no GPU required)
- **Cold Start & Deserialization**: ~4.84 s
- **Single ConvNeXt Forward Pass**: **61.5 ms**
- **Total Analyze (XAI Grad-CAM + 4-Probe Stability)**: **2.50 s**
- **Full HTTP Request-Response Latency**: **~2.65 s**

---

## 9. Deployment Architecture & Artifact Management

SignalScope is engineered for reproducible deployment across cloud environments without hard-coding local filesystem paths:

### 1. External Model Artifact Strategy
- The production checkpoint (`best_model.pt`, 318.61 MB, SHA-256 `c2e7881e9206...`) and temperature scaler (`temperature_scaler.json`) are resolved dynamically via `app.inference.artifacts.ModelArtifactManager`:
  1. Local disk path `checkpoints/baseline_convnext/best_model.pt`.
  2. Local persistent cache `cache/models/best_model.pt`.
  3. Remote URL download (`MODEL_URL` / `SCALER_URL`) with automatic SHA-256 integrity verification.
- The service fails safely and reports HTTP 503 (`not_ready`) on `/ready` if model artifacts cannot be verified.

### 2. Containerized Deployment (Python 3.13-slim)
- Production Dockerfile uses `python:3.13-slim` matching the verified development runtime.
- Runs as non-root user `appuser` (UID 1000) for security hardening.
- Dynamic port binding via `${PORT:-8000}`.
- Comprehensive platform research and memory budgeting documented in [`docs/deployment_architecture.md`](docs/deployment_architecture.md).
- Hackathon 3–5 minute presentation script documented in [`docs/demo_script.md`](docs/demo_script.md).

---

## 10. Official Dataset Layout & Anti-Leakage Protocol

SignalScope consumes the official dataset through a decoupled, configurable data root:
- **Default Location**: `C:\Programming\SignalScope-data` (or configurable via `SIGNALSCOPE_DATA_ROOT` environment variable / `model/configs/default.yaml`).

### Dataset Hierarchy
```
<DATA_ROOT>/
├── train/
│   ├── REAL/       # 50,000 authentic images (32x32 RGB JPEG)
│   └── FAKE/       # 50,000 synthetic AI-generated images (32x32 RGB JPEG)
└── test/           # 20,000 evaluation images (10,000 REAL, 10,000 FAKE)
```

### Training & Validation Methodology
1. **Zero-Leakage Policy**: In strict adherence to SIH 2026 guidelines, **`test/` is completely excluded from model training, validation, threshold tuning, and feature selection**.
2. **Local Stratified Splits**: A reproducible validation split (85,000 train / 15,000 validation) is constructed exclusively from `train/` using deterministic stratified sampling (`seed=42`).
3. **Disjoint Verification**: Path overlap is asserted to be zero (`assert len(train_paths ∩ val_paths) == 0`).

---

## 11. Empirical Baseline Model Results (Phase 2C)

Trained strictly on the local training partition (85,000 train / 15,000 val) without touching `test/`:
- **Backbone**: `convnext_tiny.in12k_ft_in1k` (pure spatial transfer learning)
- **Local Validation ROC-AUC**: **0.9992**
- **Local Validation Macro-F1**: **0.9908** (99.08%)
- **Local Validation Accuracy**: **99.08%**
- **Local Validation False Positive Rate**: **1.03%** (at default threshold 0.50)
- **Local Validation Specificity**: **98.97%**
- **Confusion Matrix (Val 15,000)**: TN=7,423 | FP=77 | FN=61 | TP=7,439
- **Official Held-Out Test**: Pending official organizer evaluation

---

## 12. Known Limitations

1. **Resolution Scale**: Model is trained on 32×32 patches; upsampled Grad-CAM heatmaps represent visual localization regions rather than high-frequency microscopic forensic artifacts.
2. **Degradation Sensitivity**: Aggressive resizing and downsampling can induce prediction drift on borderline samples, captured by the Authenticity Stability score.
3. **Absence of Provenance**: Lack of C2PA manifest does not imply synthetic origin; metadata is supporting evidence only.

---

## 13. Development Roadmap & Phased Execution

- [x] **Phase 1 — Foundation**: Repository structure, configuration, logging, testing suite, Docker, schemas, baseline CLI.
- [x] **Phase 2A — Reproducible ML Stack**: Python 3.13 isolated Conda environment (`signalscope`), PyTorch 2.6.0+cu124, timm, RTX 3050 GPU verification.
- [x] **Phase 2B — Dataset Pipeline**: Non-destructive audit of 100,000 training images, verified class balance (1.00:1), fast scandir loader, zero-leakage split logic, model smoke test.
- [x] **Phase 2C — Baseline Model Training**: Pretrained ConvNeXt-Tiny classifier training on local split, baseline metrics (ROC-AUC: 0.9992, Macro-F1: 0.9908, FPR: 1.03% @ threshold 0.50).
- [x] **Phase 3 — Frequency Features + Generalization**: 2D FFT & 2D DCT spectral branches, Dual-Branch Fusion Detector (ConvNeXt-Tiny + FFT 32x32: ROC-AUC: 0.9992, Macro-F1: 0.9875, FPR: 0.93%), complementarity analysis.
- [x] **Phase 4 — Robustness & Authenticity Stability**: Controlled degradation benchmark across 7 conditions (JPEG 95, 85, 70, Resize, Screenshot, Light Edit). Bounded Authenticity Stability Score $S \in [0, 1]$ ($S = C \times (1 - 0.5(\bar{D} + D_{\max}))$, Fusion mean $S = 0.6885$, Baseline mean $S = 0.6820$).
- [x] **Phase 5 — Calibration + Faithful Explainability**: Post-hoc probability calibration (Temperature Scaling $T=0.9995$, ECE $0.0062$, Brier $0.00795$ on 15k validation set), Grad-CAM spatial attribution on ConvNeXt-Tiny stage 3 block 2, 2D FFT spectral visualizer with azimuthal decay profiles, structured multimodal evidence representation, responsible uncertainty framework (borderline $[0.40, 0.60]$ corridor, volatility threshold $S < 0.60$), and deterministic evidence-grounded explanation synthesis.
- [x] **Phase 6 — Production Inference + Web Application**: FastAPI production serving, Next.js 16 + TypeScript + Tailwind CSS web interface, interactive Grad-CAM heatmap blending, 2D FFT spectrum viewer, transformation robustness benchmark matrix, EXIF metadata inspector, full docker compose stack, and sub-150ms end-to-end latency.
- [x] **Phase 7 — Deployment + Public Demo**: Reproducible Python 3.13-slim production Docker container, non-root execution, readiness probe (`/ready`), model artifact resolution with SHA-256 verification and local caching, CPU inference verification (61ms forward pass), cloud host feasibility research (Hugging Face Spaces / Render Starter / Vercel), and 3-5 minute demo recording script.
- [x] **Phase 8 — Final Hardening, Demo, Metrics, and Submission**: Comprehensive repository audit, failure-path API test hardening, dynamic frontend error UX, deterministic judge demo catalog (`docs/demo_samples.md`), project scorecard (`docs/final_scorecard.md`), and final submission report (`report/final_submission_report.md`).

> [!NOTE]
> **Validation Notice**: All reported calibration, explainability, stability, and inference metrics are local validation results evaluated on partitions constructed from `train/`. They are not the organizer's unseen-generator test results. The official held-out test partition `test/` remains strictly untouched.


---

## 14. Originality & Third-Party Declarations

- **Codebase Originality**: All architecture wrappers, evidence fusion logic, stability testing services, and API endpoints are original implementations built for the SIH 2026 hackathon.
- **Third-Party Libraries**: `PyTorch`, `torchvision`, `timm` (Ross Wightman), `scikit-learn`, `FastAPI`, `Pillow`, `NumPy`, `SciPy`.
- **Metrics Integrity**: All reported metrics will be derived solely from empirical evaluations conducted on verified datasets without synthetic manipulation or placeholder fabrication.

