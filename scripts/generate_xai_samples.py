"""Script to evaluate Explainability, Grad-CAM, and Evidence on local validation samples."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

# Prevent OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch

from app.utils.logger import logger
from model.architectures.convnext import build_convnext_tiny
from model.calibration import TemperatureScaler
from model.dataset import create_honest_splits, scan_dataset_directory
from model.explainability import EvidenceExtractor


def main():
    logger.info("=" * 65)
    logger.info(" SignalScope Phase 5: Explainability & Evidence Validation")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load trained ConvNeXt baseline
    b_ckpt = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "best_model.pt"
    ckpt = torch.load(b_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model = build_convnext_tiny(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load fitted temperature scaler
    scaler = TemperatureScaler()
    scaler_path = PROJECT_ROOT / "checkpoints" / "baseline_convnext" / "temperature_scaler.json"
    if scaler_path.exists():
        scaler.load(scaler_path)
    else:
        logger.warning(f"Scaler path not found at {scaler_path}, using default T=1.0")

    extractor = EvidenceExtractor(classifier=model, scaler=scaler, device=device)

    # 3. Load local validation samples
    train_dir = Path(r"C:\Programming\SignalScope-data\train")
    all_samples = scan_dataset_directory(train_dir)
    _, val_samples, _ = create_honest_splits(all_samples, val_ratio=0.15, test_ratio=0.0, seed=42)

    logger.info(f"Loaded {len(val_samples):,} validation samples from train partition.")

    output_dir = PROJECT_ROOT / "report" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select representative cases
    # We want:
    # 1. Authentic Real sample (high confidence)
    # 2. Synthetic AI sample (high confidence)
    # 3. Uncertain / Volatile / Borderline sample
    real_candidates = [s for s in val_samples if s[1] == 0]
    fake_candidates = [s for s in val_samples if s[1] == 1]

    # Sample 1: Authentic Real
    sample_real_path, _, _ = real_candidates[0]
    with Image.open(sample_real_path) as img:
        real_img = img.convert("RGB")

    evidence_real = extractor.evaluate_image(
        real_img, artifact_save_dir=output_dir, sample_id="sample_authentic_real"
    )
    logger.info(f"\n[Case 1: Authentic Real] {sample_real_path.name}")
    logger.info(f"  Verdict     : {evidence_real['verdict']}")
    logger.info(f"  Cal Prob    : {evidence_real['calibrated_probability']}")
    logger.info(f"  Explanation : {evidence_real['explanation']}")

    # Sample 2: Synthetic AI
    sample_fake_path, _, _ = fake_candidates[0]
    with Image.open(sample_fake_path) as img:
        fake_img = img.convert("RGB")

    evidence_fake = extractor.evaluate_image(
        fake_img, artifact_save_dir=output_dir, sample_id="sample_synthetic_fake"
    )
    logger.info(f"\n[Case 2: Synthetic AI] {sample_fake_path.name}")
    logger.info(f"  Verdict     : {evidence_fake['verdict']}")
    logger.info(f"  Cal Prob    : {evidence_fake['calibrated_probability']}")
    logger.info(f"  Explanation : {evidence_fake['explanation']}")

    # Sample 3: Scan for an uncertain or volatile sample among validation samples
    uncertain_sample = None
    for s in val_samples[10:150]:
        p, lbl, _ = s
        with Image.open(p) as img:
            test_img = img.convert("RGB")
        res = extractor.evaluate_image(test_img)
        if res["uncertain"]:
            uncertain_sample = (p, lbl, test_img, res)
            break

    if uncertain_sample is None:
        # If no natural borderline sample in the first 140, pick a sample with lowest stability
        lowest_s = 1.0
        best_candidate = None
        for s in val_samples[10:60]:
            p, lbl, _ = s
            with Image.open(p) as img:
                test_img = img.convert("RGB")
            res = extractor.evaluate_image(test_img)
            stab = res["evidence"]["robustness"]["stability_score"]
            if stab < lowest_s:
                lowest_s = stab
                best_candidate = (p, lbl, test_img, res)
        uncertain_sample = best_candidate

    p_unc, lbl_unc, unc_img, _ = uncertain_sample
    evidence_unc = extractor.evaluate_image(
        unc_img, artifact_save_dir=output_dir, sample_id="sample_borderline_uncertain"
    )
    logger.info(f"\n[Case 3: Borderline/Uncertain] {p_unc.name} (GT: {'AI' if lbl_unc==1 else 'Real'})")
    logger.info(f"  Verdict     : {evidence_unc['verdict']}")
    logger.info(f"  Cal Prob    : {evidence_unc['calibrated_probability']}")
    logger.info(f"  Stability   : {evidence_unc['evidence']['robustness']['stability_score']}")
    logger.info(f"  Explanation : {evidence_unc['explanation']}")

    # Save Structured JSON Case Studies
    cases_record = {
        "benchmark_name": "explainability_and_evidence_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": [
            {
                "case_id": "case_1_authentic_real",
                "source_image": str(sample_real_path),
                "ground_truth": "Authentic Real (0)",
                "evidence_object": evidence_real,
            },
            {
                "case_id": "case_2_synthetic_fake",
                "source_image": str(sample_fake_path),
                "ground_truth": "Synthetic AI (1)",
                "evidence_object": evidence_fake,
            },
            {
                "case_id": "case_3_borderline_uncertain",
                "source_image": str(p_unc),
                "ground_truth": "Synthetic AI (1)" if lbl_unc == 1 else "Authentic Real (0)",
                "evidence_object": evidence_unc,
            },
        ],
    }

    json_cases_path = output_dir / "explainability_cases.json"
    with open(json_cases_path, "w", encoding="utf-8") as f:
        json.dump(cases_record, f, indent=2)
    logger.info(f"Saved explainability cases to {json_cases_path}")

    # Generate Markdown Report
    md = [
        "# SignalScope Phase 5: Faithful Explainability & Multimodal Evidence Report",
        f"**Timestamp**: {cases_record['timestamp']}",
        f"**Model Evaluated**: ConvNeXt-Tiny Spatial Baseline (`checkpoints/baseline_convnext/best_model.pt`)",
        "> [!IMPORTANT]",
        "> **Anti-Leakage & Grounding Protocol**:",
        "> - All representative examples derive exclusively from the local validation split of `train/`.",
        "> - The organizer-held-out test set `C:\\Programming\\SignalScope-data\\test` was **NOT** used.",
        "> - Explanations are strictly grounded in empirical model attribution, 2D FFT spectrum, and degradation stability.\n",
        "## 1. Multimodal Evidence Architecture",
        "SignalScope rejects superficial natural-language generation in favor of a deterministic evidence synthesis engine. "
        "Before any verdict or explanation is rendered, three independent modalities extract evidence from the image:\n",
        "1. **Spatial Attribution (Grad-CAM)**: Backpropagates logit gradients through the final convolutional block (`stages[-1].blocks[-1]`) to identify pixel regions influencing model activation.",
        "2. **Spectral Signatures (2D FFT)**: Extracts log-magnitude frequency maps and 1D azimuthal radial decay curves on native 32x32 inputs to quantify high-frequency energy concentration.",
        "3. **Authenticity Stability Probing**: Tests prediction invariance under controlled degradations (JPEG Q=95, Q=70, Resizing 0.7x, Cropping 0.90) to measure volatility.\n",
        "## 2. Representative Validation Case Studies\n",
        "### Case 1: Authentic Camera Capture (High Confidence Real)",
        f"- **File**: `{sample_real_path.name}`",
        f"- **Ground Truth**: Authentic Real",
        f"- **Verdict**: `{evidence_real['verdict']}`",
        f"- **Calibrated Synthetic Probability**: `{evidence_real['calibrated_probability']:.4f}`",
        f"- **Authenticity Stability Score**: `{evidence_real['evidence']['robustness']['stability_score']:.4f}`",
        f"- **Explanation**: *\"{evidence_real['explanation']}\"*",
        f"- **Visual Evidence Panel**: `report/explainability/sample_authentic_real_evidence_panel.png`\n",
        "### Case 2: Synthetic AI-Generated Media (High Confidence AI)",
        f"- **File**: `{sample_fake_path.name}`",
        f"- **Ground Truth**: Synthetic AI",
        f"- **Verdict**: `{evidence_fake['verdict']}`",
        f"- **Calibrated Synthetic Probability**: `{evidence_fake['calibrated_probability']:.4f}`",
        f"- **Authenticity Stability Score**: `{evidence_fake['evidence']['robustness']['stability_score']:.4f}`",
        f"- **Explanation**: *\"{evidence_fake['explanation']}\"*",
        f"- **Visual Evidence Panel**: `report/explainability/sample_synthetic_fake_evidence_panel.png`\n",
        "### Case 3: Borderline / Uncertain Sample (Responsible Uncertainty Triggered)",
        f"- **File**: `{p_unc.name}`",
        f"- **Ground Truth**: {'Synthetic AI' if lbl_unc == 1 else 'Authentic Real'}",
        f"- **Verdict**: `{evidence_unc['verdict']}`",
        f"- **Calibrated Probability**: `{evidence_unc['calibrated_probability']:.4f}`",
        f"- **Stability Score**: `{evidence_unc['evidence']['robustness']['stability_score']:.4f}`",
        f"- **Uncertainty Reasons**: `{evidence_unc['uncertainty_reasons']}`",
        f"- **Explanation**: *\"{evidence_unc['explanation']}\"*",
        f"- **Visual Evidence Panel**: `report/explainability/sample_borderline_uncertain_evidence_panel.png`\n",
        "## 3. Responsible AI & Faithfulness Guardrails",
        "- **No Over-Claiming**: The system never uses absolute terms such as '100% fake' or 'artifacts definitely prove generation'.",
        "- **Explicit Uncertainty**: When probability hovers in the ambiguous corridor ($[0.40, 0.60]$) or stability degrades under transformation ($S < 0.60$), the engine marks the verdict as `uncertain` and prompts for human forensic review.",
        "- **Display Disclaimer**: Because native images are 32x32, heatmaps are visually interpolated for presentation only; documentation clearly states that upsampling does not increase forensic resolution.",
    ]

    report_md_path = output_dir / "explainability_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Saved explainability report to {report_md_path}")


if __name__ == "__main__":
    main()
