# SignalScope Digital Media Forensic Workbench — Design System

## 1. Design Philosophy

The SignalScope interface is architected as a **Digital Media Forensic Workbench** rather than an AI startup product dashboard. It rejects consumer AI tropes (neon glow effects, purple-to-cyan gradients, floating glassmorphic pills, and marketing buzzwords) in favor of the visual restraint, precise information density, and evidentiary discipline found in security tooling, scientific instrumentation, and developer environments.

---

## 2. Color System & Semantic Discipline

| Role | Color Value | Tailwind Utility | Semantic Application |
|---|---|---|---|
| **Base Background** | `#0b0f17` | `bg-[#0b0f17]` | Deep charcoal/navy canvas. |
| **Surface Panel** | `#0f172a` | `forensic-panel` | Solid dark slate panel surface. |
| **Panel Border** | `#1e293b` | `border-slate-800` | Crisp 1px structural dividing lines. |
| **Primary Text** | `#f8fafc` | `text-slate-100` | Headings, primary metrics, active values. |
| **Muted Metadata** | `#94a3b8` | `text-slate-400` | Property labels, explanatory scopes. |
| **Subtle Telemetry** | `#64748b` | `text-slate-500` | Timestamps, schema identifiers, disclaimer text. |
| **Authentic / Real** | `#10b981` | `text-emerald-400` | Strictly reserved for `LIKELY REAL` and `Stable` states. |
| **Synthetic / Fake** | `#f43f5e` | `text-rose-400` | Strictly reserved for `LIKELY AI-GENERATED` verdicts. |
| **Uncertainty** | `#f59e0b` | `text-amber-400` | Dedicated to `UNCERTAIN` and `Human Review Recommended`. |
| **Instrumentation** | `#0284c7` | `text-sky-400` | Layer tags, active execution status dots. |

---

## 3. Typography & Hierarchy

- **Primary Interface**: Clean system sans-serif (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto`).
- **Data & Telemetry (`font-mono`)**: Monospace typography is strictly mandated for all quantitative, probabilistic, and hardware values:
  - Synthetic likelihood percentages (`82.4%`, `0.0%`, `100.0%`)
  - Calibration parameters (`T=0.99953`)
  - Probability drift values (`±1.8%`, `Δp`)
  - Layer indicators (`ConvNeXtStage[3].ConvNeXtBlock[2]`)
  - Schema version identifiers (`Schema v1.0`)
  - Device compute status (`CUDA`, `CPU`)

---

## 4. Component Topology & Forensic Layout

```
                                  [ Header ]
                   SIGNALSCOPE / Forensic Workbench · API Online [CUDA]
                                      │
                                      ▼
                        [ Workbench Ingestion Target ]
                      Drop an image here or browse (≤25MB)
            [ Authentic Real ] · [ Synthetic AI ] · [ Borderline / Uncertain ]
                                      │
                                      ▼
                             [ Analysis Result ]
           LIKELY AI-GENERATED · 100.0% Likelihood (Calibrated T=0.9995)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   [ 1. Spatial Attribution ]              [ 2. Frequency Evidence ]
   Grad-CAM (Stage 3 Block 2)              Centered 2D FFT Spectrum
   Blend Slider (0% → 100%)                HF Energy Ratio (32.6% Natural)
                 │                                         │
                 ├────────────────────┬────────────────────┤
                 ▼                                         ▼
   [ 3. Authenticity Stability ]           [ 4. Provenance & Metadata ]
   6-Probe Perturbation Table              EXIF & C2PA Property Sheet
   Stability: 100.0% · Flips: 0%           Camera & Software Signatures
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                   [ 5. Forensic Synthesis & Summary ]
             Deterministic observations + Mandatory Disclaimer
```

---

## 5. Ethical Guardrails & Uncertainty Protocol

1. **Explicit Uncertainty**: Samples with calibrated probability inside the boundary corridor (0.40–0.60) or exhibiting decision flips under standard compression/rescaling trigger an unambiguous amber banner: **Human Forensic Review Recommended**.
2. **Faithfulness**: Spatial feature attributions and 2D Fourier spectra illustrate features guiding the neural network's decisions; they do not represent absolute physical fingerprints.
3. **No Biometric Identification**: SignalScope strictly evaluates low-level signal anomalies and pixel synthesis artifacts; it performs no human profiling or face recognition.
