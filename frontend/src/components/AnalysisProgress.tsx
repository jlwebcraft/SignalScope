"use client";

import React, { useState, useEffect } from "react";
import { Loader2, CheckCircle2, Shield, Eye, Activity, Layers, Sliders } from "lucide-react";

interface Step {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const STEPS: Step[] = [
  { id: "validate", label: "Validating image dimensions & security integrity", icon: Shield },
  { id: "spatial", label: "Extracting ConvNeXt-Tiny deep spatial representations", icon: Layers },
  { id: "calib", label: "Applying post-hoc Temperature Scaling calibration (T=0.9995)", icon: Sliders },
  { id: "spectral", label: "Computing 2D FFT log-magnitude spectral distribution", icon: Activity },
  { id: "gradcam", label: "Calculating Grad-CAM spatial activation attribution", icon: Eye },
  { id: "robustness", label: "Evaluating Authenticity Stability under 4 degradation probes", icon: Shield },
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
    <div className="glass-panel rounded-2xl p-6 border border-indigo-500/20 shadow-2xl bg-slate-900/60 max-w-xl mx-auto space-y-5 animate-fadeIn">
      <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
        <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
        <div>
          <h4 className="text-sm font-semibold text-white">Running Multimodal Evidence Pipeline</h4>
          <p className="text-xs text-slate-400">Executing calibrated spatial, spectral, and stability analyses</p>
        </div>
      </div>

      <div className="space-y-3">
        {STEPS.map((step, idx) => {
          const isDone = idx < activeStep;
          const isCurrent = idx === activeStep;
          const Icon = step.icon;

          return (
            <div
              key={step.id}
              className={`flex items-center space-x-3 text-xs transition-all duration-200 ${
                isDone
                  ? "text-slate-300"
                  : isCurrent
                  ? "text-indigo-400 font-medium"
                  : "text-slate-600 opacity-60"
              }`}
            >
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-xs transition-colors ${
                  isDone
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    : isCurrent
                    ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 animate-pulse"
                    : "bg-slate-800/60 text-slate-600"
                }`}
              >
                {isDone ? <CheckCircle2 className="w-3.5 h-3.5" /> : idx + 1}
              </div>
              <span className="truncate">{step.label}</span>
              {isCurrent && <Loader2 className="w-3.5 h-3.5 animate-spin ml-auto shrink-0" />}
            </div>
          );
        })}
      </div>
    </div>
  );
};
