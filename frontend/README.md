# SignalScope Web Frontend
**Next.js 16 + TypeScript + Tailwind CSS (Turbopack)**

Interactive web application for SignalScope multimodal image authenticity analysis.

---

## Architecture
- **Framework**: Next.js (App Router)
- **Styling**: Tailwind CSS + Lucide Icons
- **Backend Communication**: FastAPI REST API (`http://localhost:8000/api/v1/predict`)
- **Visual Evidence**: Base64 data URI overlays for Grad-CAM heatmaps and 2D FFT magnitude spectra.

## Development Setup

```bash
cd frontend
npm install
npm run dev
```

The application will be running at `http://localhost:3000`.

## Production Build

```bash
cd frontend
npm run build
npm start
```

## Features
1. **Drag-and-Drop Ingestion**: Client-side validation for JPEG, PNG, WEBP, and BMP up to 25MB.
2. **Pre-packaged Local Validation Samples**: Quick one-click testing of Authentic Real, Synthetic AI, and Borderline/Uncertain cases.
3. **Calibrated Verdict & Confidence**: Clear badges for `Likely AI-generated`, `Likely Real`, and `Uncertain` with calibrated synthetic probabilities.
4. **Grad-CAM Spatial Attribution**: Interactive blend slider to smoothly inspect feature activation heatmaps over original images.
5. **2D FFT Frequency Spectrum**: High-frequency energy ratio gauge with continuous decay visualization.
6. **Authenticity Stability Benchmark**: Perturbation test breakdown across 4 transformations with flip rate and drift metrics.
7. **Metadata & Provenance**: Camera hardware provenance and C2PA Content Credentials inspection.
8. **Responsible AI Guardrails**: Prominent ethical disclaimers and human review advisories on ambiguous cases.
