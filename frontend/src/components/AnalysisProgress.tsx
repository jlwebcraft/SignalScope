"use client";

import React from "react";
import { Loader2 } from "lucide-react";

export const AnalysisProgress: React.FC = () => {
  return (
    <div className="bg-white border border-slate-300 rounded p-6 max-w-lg mx-auto space-y-4 shadow-sm">
      <div className="flex items-center space-x-3 border-b border-slate-200 pb-3">
        <Loader2 className="w-5 h-5 text-slate-800 animate-spin shrink-0" />
        <div>
          <h4 className="text-xs font-mono font-bold text-slate-900 uppercase tracking-wider">
            Executing Forensic Pipeline
          </h4>
          <p className="text-[11px] font-mono text-slate-500">
            Evaluating spatial, spectral, perturbation, and provenance layers
          </p>
        </div>
      </div>

      <div className="space-y-2 text-xs font-mono">
        <div className="flex items-center justify-between text-slate-700 py-1 border-b border-slate-100">
          <span className="text-slate-500">01. Spatial Analysis</span>
          <span className="text-slate-900 font-medium">ConvNeXt-Tiny Feature Attribution & Grad-CAM</span>
        </div>
        <div className="flex items-center justify-between text-slate-700 py-1 border-b border-slate-100">
          <span className="text-slate-500">02. Frequency Domain</span>
          <span className="text-slate-900 font-medium">2D Centered Fast Fourier Transform</span>
        </div>
        <div className="flex items-center justify-between text-slate-700 py-1 border-b border-slate-100">
          <span className="text-slate-500">03. Invariance Stress</span>
          <span className="text-slate-900 font-medium">4× Perturbation Robustness Probes</span>
        </div>
        <div className="flex items-center justify-between text-slate-700 py-1 border-b border-slate-100">
          <span className="text-slate-500">04. Calibration</span>
          <span className="text-slate-900 font-medium">Temperature Scaling (T = 0.9986)</span>
        </div>
        <div className="flex items-center justify-between text-slate-700 py-1">
          <span className="text-slate-500">05. Provenance</span>
          <span className="text-slate-900 font-medium">EXIF & C2PA Content Credentials</span>
        </div>
      </div>

      <div className="pt-2 text-[11px] text-slate-500 font-sans leading-relaxed border-t border-slate-100">
        Evaluation is executed locally or via containerized inference on Google Cloud Run. Large images are processed at native resolution.
      </div>
    </div>
  );
};
