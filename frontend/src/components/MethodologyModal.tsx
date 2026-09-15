"use client";

import React, { useEffect } from "react";
import { X, ShieldCheck, Cpu, Activity } from "lucide-react";

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelVersion?: string;
  device?: string;
}

export const MethodologyModal: React.FC<MethodologyModalProps> = ({
  isOpen,
  onClose,
  modelVersion = "signalscope-v2",
  device = "CPU",
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="methodology-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-900/60 backdrop-blur-xs"
    >
      <div className="bg-white border border-slate-300 w-full max-w-3xl max-h-[90vh] flex flex-col shadow-xl text-slate-800">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-600">
                Technical Specification & Ethical Scope
              </span>
              <span className="text-slate-300">/</span>
              <span className="text-[11px] font-mono text-slate-500">
                Model: {modelVersion}
              </span>
            </div>
            <h2 id="methodology-title" className="text-base font-semibold text-slate-900 mt-0.5">
              SignalScope Forensic Architecture & Verification Principles
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close specifications modal"
            className="p-1.5 text-slate-500 hover:text-slate-900 hover:bg-slate-200/60 rounded transition-colors focus-forensic"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="overflow-y-auto px-6 py-6 space-y-6 text-xs text-slate-700 leading-relaxed">
          {/* Section 1: Multimodal Verification Engine */}
          <section className="space-y-2">
            <div className="flex items-center space-x-2 text-slate-900 font-semibold text-sm">
              <Cpu className="w-4 h-4 text-sky-700" />
              <h3>1. Multimodal Evidence Fusion</h3>
            </div>
            <p>
              SignalScope does not rely on a single unverified classifier output. It synthesizes evidence across four distinct forensic modalities:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div className="p-3 bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-semibold text-slate-900">Spatial Feature Attribution</span>
                <p className="text-[11px] text-slate-600">
                  ConvNeXt-Tiny deep feature extractor inspects patch-level anomalies, blended synthetic edges, and unnatural texture gradients via Grad-CAM attribution.
                </p>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-semibold text-slate-900">2D Spectral Fourier Harmonics</span>
                <p className="text-[11px] text-slate-600">
                  Centered 2D Fast Fourier Transform examines frequency power decay. Generative upsamplers frequently introduce high-frequency periodic grid lattice residuals.
                </p>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-semibold text-slate-900">Perturbation Invariance</span>
                <p className="text-[11px] text-slate-600">
                  Every exhibit is stressed across JPEG compression, Gaussian blur, and spatial re-scaling to measure prediction drift (Δp) and detect fragile overconfident artifacts.
                </p>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-semibold text-slate-900">Provenance & Metadata</span>
                <p className="text-[11px] text-slate-600">
                  Parses EXIF camera parameters and inspects C2PA Content Credentials to determine whether cryptographic manifests or camera capture hardware are recorded.
                </p>
              </div>
            </div>
          </section>

          {/* Section 2: Uncertainty & Calibration Policy */}
          <section className="space-y-2 border-t border-slate-200 pt-4">
            <div className="flex items-center space-x-2 text-slate-900 font-semibold text-sm">
              <Activity className="w-4 h-4 text-amber-700" />
              <h3>2. Calibrated Likelihood & Uncertainty Policy</h3>
            </div>
            <p>
              Deep networks frequently produce overconfident probabilities on uncalibrated logits. SignalScope applies post-hoc <strong>Temperature Scaling (T = 0.9986)</strong> fitted on holdout verification data:
            </p>
            <div className="p-3 bg-amber-50/70 border border-amber-200/80 text-amber-950 space-y-1 text-[11px]">
              <div className="font-semibold uppercase tracking-wider font-mono text-[10px]">
                Adjudication & Uncertainty Protocol
              </div>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  <strong>Likely Real (Authentic)</strong>: Calibrated probability &lt; 0.40 with confirmed perturbation stability (stability score ≥ 0.70).
                </li>
                <li>
                  <strong>Likely AI-Generated</strong>: Calibrated probability ≥ 0.60 with confirmed perturbation stability (stability score ≥ 0.70).
                </li>
                <li>
                  <strong>Uncertain / Indeterminate (Human Review)</strong>: Triggered whenever probability falls in the decision corridor (0.40–0.60) <em>or</em> when perturbation stress tests reveal categorical flips or high volatility.
                </li>
              </ul>
            </div>
          </section>

          {/* Section 3: Ethical Boundaries */}
          <section className="space-y-2 border-t border-slate-200 pt-4">
            <div className="flex items-center space-x-2 text-slate-900 font-semibold text-sm">
              <ShieldCheck className="w-4 h-4 text-emerald-700" />
              <h3>3. Ethical Guardrails & Legal Scope</h3>
            </div>
            <ul className="space-y-1.5 text-[11px] text-slate-700">
              <li className="flex items-start gap-2">
                <span className="font-mono text-slate-400 font-semibold">01.</span>
                <span>
                  <strong>No Biometric or Facial Identification</strong>: SignalScope analyzes structural pixel patterns and frequency statistics. It does not perform facial recognition, identity matching, or demographic profiling.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="font-mono text-slate-400 font-semibold">02.</span>
                <span>
                  <strong>Statistical Likelihood, Not Causal Proof</strong>: Results represent empirical likelihood estimates given current model weights. They should serve as supporting evidence for human analysts, fact-checkers, and verification desks, not automated legal proof.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="font-mono text-slate-400 font-semibold">03.</span>
                <span>
                  <strong>Independent Evaluation Integrity</strong>: The organizer&apos;s official held-out test partition remains strictly untouched. Benchmark validations are conducted against held-out photographic captures and cross-generator datasets.
                </span>
              </li>
            </ul>
          </section>

          {/* Runtime Context */}
          <div className="p-3 bg-slate-100 border border-slate-200 text-slate-600 font-mono text-[11px] flex flex-wrap items-center justify-between gap-2">
            <span>Runtime Target: {device}</span>
            <span>Registered Artifact: {modelVersion}</span>
            <span>Calibration: T = 0.9986</span>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <span className="text-[11px] font-mono text-slate-500">
            SignalScope Media Forensics · Open Documentation
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium rounded transition-colors focus-forensic"
          >
            Acknowledge & Close
          </button>
        </div>
      </div>
    </div>
  );
};
