# SignalScope 3–5 Minute Hackathon Video Demonstration Script

**Event**: SIH 2026 Internal Hackathon  
**Project**: SignalScope — *"Telling Real From Synthetic in the Age of Generative Media"*  
**Target Duration**: 3 minutes 45 seconds (Strictly within the 3:00–5:00 hackathon constraint)  
**Presenter Flow**: Voiceover with screen recording of live SignalScope Next.js Web UI and API.

---

## Timeline & Scene-by-Scene Script

### Scene 1: Problem Definition & The Generalization Challenge (00:00 – 00:30)
- **Visual**: Title slide / SignalScope Landing Page showing the tagline, architecture badges, and clean interface.
- **Presenter Voiceover**:
  > *"Welcome. Today, generative models—from Latent Diffusion to Flow Matching—can synthesize photo-realistic imagery in milliseconds. However, most contemporary AI detectors suffer from two fatal flaws: they overfit to specific generator artifacts, and their predictions collapse under routine social-media degradations like resizing and recompression.*
  >
  > *SignalScope was engineered from first principles to solve this. Instead of an isolated black-box classifier, SignalScope is a calibrated, multimodal evidence-fusion detection system that pairs spatial transfer learning with spectral physics and an Authenticity Stability score."*

---

### Scene 2: Interactive Ingestion & Real-Time Analysis (00:30 – 01:00)
- **Visual**: Presenter clicks on the drag-and-drop zone, selects `synthetic_ai.jpg`, or clicks the "Synthetic AI" quick sample button. The image preview renders instantly with file dimensions and metadata. Presenter clicks **"Run Authenticity Analysis"**.
- **Presenter Voiceover**:
  > *"Let's test this in our production Next.js interface. We upload a candidate image. In under 150 milliseconds on consumer hardware, our FastAPI backend ingests the image with strict MIME and size protections, executes our validated ConvNeXt-Tiny classifier, applies post-hoc temperature scaling, extracts Grad-CAM heatmaps, computes 2D Fast Fourier Transforms, and probes stability across four degradation attacks."*

---

### Scene 3: Verdict, Calibrated Probability & Confidence (01:00 – 01:30)
- **Visual**: The UI smoothly transitions to the verdict banner. Shows `LIKELY AI-GENERATED`, probability `100.0%`, `Medium Confidence`, and the responsible disclaimer banner.
- **Presenter Voiceover**:
  > *"Here is our verdict: **Likely AI-generated**. Notice our responsible AI framing: SignalScope never uses accusatory language like '100% fake' or 'proof'. All predictions are probabilistic likelihood assessments. The raw model output is calibrated through an empirical post-hoc temperature scaling layer (T=0.9995) to ensure probabilities reflect true empirical risk."*

---

### Scene 4: Spatial Attribution (Grad-CAM) & 2D FFT Spectral Cues (01:30 – 02:00)
- **Visual**: Presenter scrolls down to the **Spatial & Spectral Evidence** section. Drags the Grad-CAM opacity blend slider back and forth (0% to 100%) to demonstrate overlay localization. Then points to the 2D FFT spectrum and high-frequency energy ratio.
- **Presenter Voiceover**:
  > *"To ensure explainability is faithful and not just post-hoc hallucination, we provide multi-angle evidence. On the left, our interactive Grad-CAM heatmap hooks into Stage 3 Block 2 of ConvNeXt-Tiny. Using our blend slider, judges can inspect exactly which spatial regions drove the prediction.*
  >
  > *On the right, our 2D FFT spectral visualizer observes frequency-domain anomalies. Elevated high-frequency energy ratios indicate synthetic generation artifacts in the power spectrum, corroborating the spatial decision."*

---

### Scene 5: Authenticity Stability Score (Bonus Module C) (02:00 – 02:30)
- **Visual**: Focus shifts to the **Degradation Robustness Matrix**. Points to the 4 transformation rows (JPEG 95, JPEG 85, Resize 0.75x, Screenshot simulation) and the overall Stability Score ($S = 1.00$).
- **Presenter Voiceover**:
  > *"This brings us to SIH Bonus Module C: Degradation Robustness. In the real world, bad actors recompress or screenshot synthetic media. SignalScope subjects the candidate image to real-time controlled perturbations. For this sample, the prediction remained 100% stable across all transformations with zero flips, yielding an Authenticity Stability score of 1.00."*

---

### Scene 6: Responsible Uncertainty & Borderline Handling (02:30 – 03:00)
- **Visual**: Presenter clicks "Reset / Test Another", selects `borderline_uncertain.jpg`, and clicks "Run Authenticity Analysis". The UI renders an orange **UNCERTAIN** badge with the subtext: *"Human review recommended"*.
- **Presenter Voiceover**:
  > *"What happens when an image is degraded or borderline? Here, we analyze a volatile sample. Although the pristine image leans synthetic, under aggressive downsampling and screenshot simulation, the prediction swings wildly—crashing stability down to 0.136.*
  >
  > *Our Responsible Uncertainty Framework immediately intervenes: whenever stability falls below 0.60 or confidence is borderline, SignalScope refuses to guess and transparently flags the image as **Uncertain — Human Review Recommended**."*

---

### Scene 7: Production Architecture, Provenance & Reproducibility (03:00 – 03:30)
- **Visual**: Briefly show the Metadata panel (*"No Content Credentials detected"*), and transition to the terminal showing Docker Compose / FastAPI OpenAPI docs (`/docs`).
- **Presenter Voiceover**:
  > *"Under the hood, our stack is production-ready. The Next.js frontend connects to our FastAPI backend with separate /health and /ready probes. Model checkpoints are fetched with SHA-256 verification and local caching. Everything runs reproducibly via Docker with Python 3.13."*

---

### Scene 8: Empirical Validation & Honest Limitations (03:30 – 03:45)
- **Visual**: Display summary slide showing verified local validation metrics: ROC-AUC 0.9992, Macro-F1 99.08%, FPR 1.03%, sub-150ms GPU latency / 61ms CPU forward pass.
- **Presenter Voiceover**:
  > *"On our strictly separated 15,000-image local validation partition, SignalScope achieves 0.9992 ROC-AUC and 1.03% False Positive Rate, while leaving the official organizer test set completely untouched.*
  >
  > *SignalScope bridges the gap between deep learning research and responsible, battle-tested forensic deployment. Thank you."*
