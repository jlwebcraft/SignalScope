# SignalScope Editorial Forensic Workstation — Design System

## 1. Design Philosophy

SignalScope is architected as an **Editorial Forensic Investigation Workstation**. It deliberately departs from stereotypical AI/cybersecurity dashboard templates (dark navy backgrounds, blue-on-blue panels, cyan glows, floating rounded glassmorphism cards, and marketing fluff) in favor of the visual restraint, authoritative typography, and evidentiary clarity of investigative reporting, scientific exhibits, and official judicial dossiers.

---

## 2. Color System & Semantic Discipline

| Role | Color Value | Tailwind Utility | Semantic Application |
|---|---|---|---|
| **Base Canvas** | `#fbfbf9` | `bg-[#fbfbf9]` | Warm archival bone/paper background. |
| **Secondary Surface** | `#f3f1ea` | `bg-[#f3f1ea]` | Warm stone contrast surface for quantitative sheets. |
| **Exhibit Mount** | `#ffffff` | `bg-[#ffffff]` | Crisp white backing for photographic evidence. |
| **Dividers & Rules** | `#e5e2d9` | `border-[#e5e2d9]` | 1px hairline structural dividers and table rules. |
| **Carbon Ink (Primary)** | `#121316` | `text-[#121316]` | Editorial headings, primary metrics, active controls. |
| **Charcoal Body** | `#2a2d34` | `text-[#2a2d34]` | Narrative finding, evidence descriptions, body text. |
| **Muted Graphite** | `#606570` | `text-[#606570]` | Property labels, technical scopes, section indicators. |
| **Subtle Telemetry** | `#8c8a82` | `text-[#8c8a82]` | Timestamps, schema identifiers, disclaimers. |
| **Editorial Accent** | `#9a3412` | `text-[#9a3412]` | Warm terracotta/rust section numbering (`01`, `02`, `03`). |
| **Authentic / Real** | `#166534` | `text-[#166534]` | Strictly reserved for `Likely Real` and `Stable` states. |
| **Synthetic / Fake** | `#991b1b` | `text-[#991b1b]` | Strictly reserved for `Likely AI-Generated` verdicts. |
| **Uncertainty** | `#92400e` | `text-[#92400e]` | Dedicated to `Uncertain` and `Human Review Recommended`. |

---

## 3. Typography & Hierarchy

- **Mastheads & Headings (`font-editorial`)**: `Newsreader` / `Charter` / `Georgia` serif. Provides historical forensic authority, human craftsmanship, and gravitas.
- **Body & Labels (`font-sans`)**: `Inter` / system-ui sans-serif. Highly legible, neutral, and unobtrusive.
- **Technical Metrics (`data-mono`)**: `JetBrains Mono` monospace. Strictly restricted to quantitative figures:
  - Synthetic likelihood percentages (`82.4%`, `0.0%`, `100.0%`)
  - Calibration parameters (`T=0.99953`)
  - Probability drift values (`±1.8%`, `Δp`)
  - Target layer indicators (`ConvNeXtStage[3].ConvNeXtBlock[2]`)
  - Schema version identifiers (`Schema v1.0`)
  - Device compute status (`CUDA`, `CPU`)

---

## 4. Anti-Card Layout & Visual Rhythm

The interface rejects the repetitive AI card paradigm (nested rounded rectangles with borders and decorative icons). Instead, it structures content through:
1. **Hairline Dividers**: 1px warm stone rules (`border-b border-[#e5e2d9]`) establish clean horizontal pacing.
2. **Numbered Sections**: Editorial sequence numbering (`01`, `02`, `03`, `04`, `05`) creates a logical dossier reading order.
3. **Prominent Visual Exhibits**: The source image and Grad-CAM heatmap are displayed side-by-side in generous 320px mounts (Exhibit A & Exhibit B) rather than cramped inside tiny card frames.
4. **Unboxed Forensic Tables**: The perturbation stability matrix uses open horizontal rules without enclosing card borders.

---

## 5. Component Topology & Dossier Structure

```
                                     [ Masthead ]
                   SIGNALSCOPE / Image Forensics · API: Online · CPU · Guardrails
                                         │
                                         ▼
                             [ Case Intake Station ]
                        Verify an Image (≤25MB Ingestion)
               01 Authentic · 02 Synthetic · 03 Borderline / Uncertain
                                         │
                                         ▼
                               [ Finding Masthead ]
           Case Analysis // LIKELY AI-GENERATED · 100.0% Likelihood (T=0.9995)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      [ 01. Visual Attribution ]                     [ 02. Frequency Harmonics ]
      Exhibit A: Original Image                      Exhibit C: 2D FFT Spectrum
      Exhibit B: Grad-CAM Overlay                    HF Radial Energy Gauge (32.6%)
      Blend Slider (0% → 100%)                       Continuous vs. Grid Residuals
                 │                                               │
                 ├───────────────────────┬───────────────────────┤
                 ▼                                               ▼
      [ 03. Perturbation Matrix ]                    [ 04. Provenance Sheet ]
      6-Probe Table with Rules                       EXIF & C2PA Property Sheet
      Stability: 100.0% · Flips: 0%                  Camera & Software Tags
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                        [ 05. Synthesis & Methodology ]
               Factual Finding + Numbered Factors + Mandatory Disclaimer
```

---

## 6. Responsible Forensics & Uncertainty Standard

1. **Uncertainty as an Active Finding**: Indeterminate predictions in the boundary corridor (0.40–0.60 calibrated probability) or exhibiting sensitivity under standard image degradations trigger an explicit amber callout: **HUMAN FORENSIC REVIEW RECOMMENDED**.
2. **Faithfulness**: Feature maps illustrate network logit focus; they do not represent universal physical fingerprints.
3. **Zero Biometrics**: Analysis is restricted strictly to signal and compression anomalies. No human facial recognition or identity profiling is performed.
