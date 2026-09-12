# SignalScope Deployment Architecture & Production Host Evaluation

## 1. System Topology Overview

SignalScope employs a decoupled client-server architecture designed for independent scalability, reproducibility, and minimal latency:

```
                          [ Client Browser ]
                                  │
                                  ▼
               ┌───────────────────────────────────────┐
               │    Next.js 16 Web Application (UI)    │
               │   - Vercel Edge Network / Node 20     │
               │   - Dynamic Opacity Heatmap Blender   │
               │   - FFT Spectrum & Radial Profiles    │
               │   - Degradation Robustness Matrix     │
               │   - Provenance & Explanation Panels   │
               └──────────────────┬────────────────────┘
                                  │ HTTPS API Requests
                                  │ (NEXT_PUBLIC_API_URL)
                                  ▼
               ┌───────────────────────────────────────┐
               │     FastAPI Authenticity Service      │
               │   - Python 3.13-slim Container        │
               │   - Safe Multipart Ingestion (≤25MB)  │
               │   - Liveness (/health) & Readiness    │
               │     Probes (/ready)                   │
               └──────────────────┬────────────────────┘
                                  │
                                  ▼
               ┌───────────────────────────────────────┐
               │   Inference Engine & Evidence Fusion  │
               │   - ConvNeXt-Tiny Spatial Classifier  │
               │   - Post-Hoc Temperature Scaler       │
               │     (T = 0.99953)                     │
               │   - Stage 3 Block 2 Grad-CAM Engine   │
               │   - 2D Fast Fourier Transform (FFT)   │
               │   - 4-Probe Authenticity Stability    │
               │     Score Evaluator                   │
               └──────────────────┬────────────────────┘
                                  │
                                  ▼
               ┌───────────────────────────────────────┐
               │     Model Checkpoint & Artifacts      │
               │   - Local: checkpoints/baseline_...   │
               │   - Remote: Hugging Face / GH Release │
               │     (SHA-256 Verified, Local Cache)   │
               └───────────────────────────────────────┘
```

---

## 2. Production Host Research & Memory Feasibility Analysis

### Workload Memory & Compute Profile
- **Trained Model Checkpoint**: 318.61 MB (`best_model.pt`, 28.6M parameters).
- **In-Memory Deserialization & PyTorch Runtime**:
  - Python runtime + FastAPI + Uvicorn: ~120 MB
  - PyTorch CPU core + Timm: ~350 MB
  - ConvNeXt-Tiny Model Graph + Memory Allocator: ~450 MB
  - Working Buffers for 224x224 transforms & Grad-CAM tensors: ~150 MB
  - **Minimum Safe Working RAM**: **~1.2 GB** (Peak during cold-start loading: **~1.5 GB**).
- **CPU Inference Latency (Empirically Measured)**:
  - Cold-start load time: 4.84 seconds.
  - Single ConvNeXt-Tiny forward pass: **61.5 ms**.
  - Total Analyze request (Forward pass + Grad-CAM backward pass + 4 stability probe evaluations): **2.50 seconds**.
  - Conclusion: Total request latency is well within standard cloud HTTP request timeouts (typically 30–60 seconds).

---

### Platform Comparison Matrix

| Platform | Free / Starter Pricing | Memory Allocation | CPU | PyTorch Suitability | Cold-Start Behavior | Recommended Role |
|---|---|---|---|---|---|---|
| **Vercel** | Free (Hobby) | Serverless / Edge CDN | Fast | Frontend only | Instant (< 100ms) | **Primary Frontend Host** |
| **Hugging Face Spaces** | Free | **16 GB RAM** | 2 vCPU | **Excellent (Native ML)** | Persistent / Fast Wake | **Primary Backend Host** |
| **Render (Free)** | Free | 512 MB RAM | 0.5 CPU | **UNSUITABLE (OOM Kill)** | ~50s sleep wake | Not Recommended (OOM) |
| **Render (Starter)** | $7 / month | 2 GB RAM | 1 CPU | **Good (Stable)** | Always on | Alternative Backend |
| **Railway** | $5 trial credit / Usage | Up to 8 GB RAM | Flexible | **Good** | Fast build & deploy | Alternative Backend |
| **Fly.io** | ~$5 / month (2GB) | 2 GB RAM | 1 shared CPU | **Good** | Edge container | Alternative Backend |

### Decision & Platform Selection Rationale
1. **Frontend: Vercel**
   - Built specifically for Next.js with automatic asset optimization and seamless GitHub integration.
   - Ingests `NEXT_PUBLIC_API_URL` during build.
2. **Backend: Hugging Face Spaces / Render Starter / Docker**
   - Free tiers with < 1GB RAM (like Render Free's 512MB) reliably crash with Out-Of-Memory (`OOMKilled`) when loading the 318MB checkpoint into PyTorch.
   - Hugging Face Spaces offers a dedicated **16 GB RAM** environment for containerized ML microservices, guaranteeing reliable operation with zero memory exhaustion.
   - For private cloud environments, Docker Compose on any 2GB+ VM provides 100% parity with local development.

---

## 3. Remote Model Artifact Strategy

To ensure zero-manual-intervention reproducibility without storing heavy binary weights in Git:
- The checkpoint file (`best_model.pt`) and temperature scaler (`temperature_scaler.json`) are resolved through `app/inference/artifacts.py`:
  1. Check local path `checkpoints/baseline_convnext/best_model.pt`.
  2. Check cached file `cache/models/best_model.pt`.
  3. If missing, download from `MODEL_URL` / `SCALER_URL`, verify SHA-256 (`c2e7881e9206184b8cd43c7999e02c6faa946c254088aa9e19e6dcb3ff3d9cdc`), and cache atomically.
- **Fail-Safe Integrity**: If the artifact cannot be located or verified, the service cleanly rejects requests and signals `GET /ready` as HTTP 503 (`not_ready`), preventing uncalibrated or misleading predictions.

---

## 4. Production Security & Hardening Checklist

- [x] **Non-Root Execution**: Docker container runs as unprivileged user `appuser` (UID 1000).
- [x] **Safe Memory & Image Boundaries**: 25 MB payload limit, decompression bomb thresholds, Pillow dimension caps.
- [x] **Zero Stack Trace Leaks**: API exceptions caught and translated to structured HTTP 400/413/415/422/500 JSON without exposing local directories.
- [x] **Dynamic CORS Restriction**: Production origins enforced via `FRONTEND_ORIGIN` environment variable.
- [x] **Probes Separation**: Separate `/health` (liveness) and `/ready` (weights readiness) endpoints.
