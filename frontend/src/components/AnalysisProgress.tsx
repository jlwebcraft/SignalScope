"use client";

import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

interface Step {
  id: string;
  label: string;
}

const STEPS: Step[] = [
  { id: "validate", label: "Validating input image dimensions & integrity" },
  { id: "spatial", label: "Extracting ConvNeXt-Tiny spatial representations" },
  { id: "calib", label: "Applying Temperature Scaling calibration (T=0.9995)" },
  { id: "spectral", label: "Computing 2D Fourier log-magnitude spectrum" },
  { id: "gradcam", label: "Generating Grad-CAM spatial activation map" },
  { id: "robustness", label: "Evaluating stability across 4 perturbation probes" },
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
    <div className="bg-[#ffffff] border border-[#e5e2d9] p-6 max-w-md mx-auto space-y-4 shadow-sm animate-fadeIn">
      <div className="flex items-center space-x-2.5 border-b border-[#e5e2d9] pb-3">
        <Loader2 className="w-4 h-4 text-[#9a3412] animate-spin" />
        <div>
          <h4 className="text-xs font-mono font-bold text-[#121316] uppercase tracking-wider">
            Executing Forensic Pipeline
          </h4>
          <p className="text-[11px] font-mono text-[#8c8a82]">
            Spatial, spectral, and perturbation evaluations
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
              className={`flex items-center space-x-3 transition-colors duration-150 ${
                isDone
                  ? "text-[#606570]"
                  : isCurrent
                  ? "text-[#121316] font-semibold"
                  : "text-[#8c8a82]/60"
              }`}
            >
              <div className="w-5 text-[11px] font-mono shrink-0">
                {isDone ? (
                  <span className="text-[#166534] font-bold">0{idx + 1}✓</span>
                ) : isCurrent ? (
                  <span className="text-[#9a3412] font-bold">0{idx + 1}►</span>
                ) : (
                  <span className="text-[#8c8a82]/60">0{idx + 1}</span>
                )}
              </div>
              <span className="text-xs truncate">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
