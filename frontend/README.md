# SignalScope Web Frontend
**Next.js + TypeScript + Tailwind CSS**

The SignalScope frontend delivers an interactive interface for evaluating image authenticity in real-time.

## Architecture
- **Framework**: Next.js (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Lucide Icons
- **Backend Connection**: FastAPI REST API (`http://localhost:8000/api/v1`)

## Key UI Components (Phase 10)
1. **Dropzone Ingestion**: Drag-and-drop file upload with format and size validation.
2. **Authenticity Verdict Card**: Clear visual badges (`Likely AI-generated`, `Likely Real`, `Uncertain`) with calibrated probability gauges.
3. **Multimodal Evidence Panel**: Side-by-side inspection of RGB spatial anomalies, FFT frequency spectrum, and metadata provenance.
4. **Authenticity Stability Visualizer**: Interactive chart demonstrating prediction invariance across JPEG recompression, downscale-upscale, and screenshot simulation.
5. **Faithful Explanation View**: Plain-language, evidence-grounded summary.
6. **Responsible AI Guardrails**: Prominent uncertainty communication and ethical disclaimers.

## Development Setup (Phase 10)
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:3000`.
