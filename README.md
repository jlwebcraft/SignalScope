# SignalScope
### *"Telling Real From Synthetic in the Age of Generative Media"*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![SIH 2026](https://img.shields.io/badge/Event-SIH%202026%20Internal%20Hackathon-purple.svg)]()

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

### Option C: Docker Deployment
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

---

## 6. Verification & Running the System

### Run Pre-Flight Diagnostics
```bash
python scripts/run_checks.py
```

### Run Unit Tests
```bash
pytest -v
```

### Launch the REST API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Run CLI Prediction
```bash
python model/predict.py --image path/to/image.jpg
```

---

## 7. Development Roadmap & Phased Execution

- [x] **Phase 1 — Foundation**: Repository structure, configuration, logging, testing suite, Docker, schemas, baseline CLI.
- [ ] **Phase 2 — Dataset**: Ingest official training data, inspect class balance/resolution, implement honest train/val/test generator-aware split.
- [ ] **Phase 3 — Baseline Model**: Pretrained ConvNeXt-Tiny classifier, train on official dataset, establish baseline metrics.
- [ ] **Phase 4 — Generalization**: Frequency-domain representation (2D FFT), spatial-frequency fusion, evaluate unseen generator AUC.
- [ ] **Phase 5 — Robustness**: Controlled degradation evaluation (JPEG, resize, screenshots) and stability score benchmarking.
- [ ] **Phase 6 — Calibration**: Temperature scaling, probability calibration, threshold optimization for target 5% FPR.
- [ ] **Phase 7 — Explainability**: Grad-CAM saliency heatmaps, spectral anomaly plots, grounded natural language explanations.
- [ ] **Phase 8 — Bonus Modules**: C2PA Content Credentials provenance, generator attribution.
- [ ] **Phase 9 — Backend**: Hardened production API endpoints and streaming.
- [ ] **Phase 10 — Frontend**: Next.js + TypeScript interactive authenticity analysis dashboard.
- [ ] **Phase 11 — Deployment**: Dockerized container deployment.
- [ ] **Phase 12 — Final Verification**: End-to-end judge reproducibility audit (< 10 minutes).

---

## 8. Originality & Third-Party Declarations

- **Codebase Originality**: All architecture wrappers, evidence fusion logic, stability testing services, and API endpoints are original implementations built for the SIH 2026 hackathon.
- **Third-Party Libraries**: `PyTorch`, `torchvision`, `timm` (Ross Wightman), `scikit-learn`, `FastAPI`, `Pillow`, `NumPy`, `SciPy`.
- **Metrics Integrity**: All reported metrics will be derived solely from empirical evaluations conducted on verified datasets without synthetic manipulation or placeholder fabrication.
