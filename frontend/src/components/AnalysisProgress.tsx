"use client";

import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

interface Step {
  id: string;
  label: string;
}

const STEPS: Step[] = [
  { id: "validate", label: "Validating input image dimensions and security integrity" },
  { id: "spatial", label: "Extracting ConvNeXt-Tiny deep spatial representations" },
  { id: "calib", label: "Applying post-hoc Temperature Scaling calibration (T=0.9995)" },
  { id: "spectral", label: "Computing 2D FFT log-magnitude spectral distribution" },
  { id: "gradcam", label: "Calculating Grad-CAM spatial activation attribution" },
  { id: "robustness", label: "Evaluating Authenticity Stability under 4 perturbation probes" },
];

export const AnalysisProgress: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev < STEPS.length - 1 ? prev + 1 : prev));
    }, 450);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="forensic-panel p-5 max-w-lg mx-auto space-y-4 animate-fadeIn">
      <div className="flex items-center space-x-2.5 border-b border-slate-800 pb-3">
        <Loader2 className="w-4 h-4 text-slate-300 animate-spin" />
        <div>
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Executing Forensic Analysis Pipeline
          </h4>
          <p className="text-[11px] font-mono text-slate-500">
            Running ConvNeXt spatial, 2D Fourier spectral, and perturbation probes
          </p>
        </div>
      </div>

      <div className="space-y-2 font-mono text-xs">
        {STEPS.map((step, idx) => {
          const isDone = idx < activeStep;
          const isCurrent = idx === activeStep;

          return (
            <div
              key={step.id}
              className={`flex items-center space-x-2.5 transition-colors duration-150 ${
                isDone
                  ? "text-slate-400"
                  : isCurrent
                  ? "text-slate-100 font-medium"
                  : "text-slate-600"
              }`}
            >
              <div className="w-4 flex justify-center text-[10px]">
                {isDone ? (
                  <span className="text-emerald-400 font-bold">✓</span>
                ) : isCurrent ? (
                  <span className="text-sky-400 animate-pulse">●</span>
                ) : (
                  <span className="text-slate-700">○</span>
                )}
              </div>
              <span className="text-[11px] truncate">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
