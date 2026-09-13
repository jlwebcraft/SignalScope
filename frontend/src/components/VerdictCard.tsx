"use client";

import React from "react";
import { AlertCircle, Sliders } from "lucide-react";
import { PredictionResponse } from "@/types/prediction";

interface VerdictCardProps {
  prediction: PredictionResponse;
}

export const VerdictCard: React.FC<VerdictCardProps> = ({ prediction }) => {
  const {
    verdict,
    probability,
    raw_probability,
    calibrated_probability,
    confidence_level,
    uncertain,
    is_development_placeholder,
  } = prediction;

  const pct = (probability * 100).toFixed(1);
  const rawPct = raw_probability != null ? (raw_probability * 100).toFixed(1) : pct;
  const calPct = calibrated_probability != null ? (calibrated_probability * 100).toFixed(1) : pct;

  const getVerdictDetails = () => {
    switch (verdict) {
      case "likely_ai_generated":
        return {
          title: "LIKELY AI-GENERATED",
          summary: "Feature activations and spectral energy patterns align with generative model characteristics.",
          borderTop: "border-t-rose-500",
          textColor: "text-rose-400",
          statusTag: "bg-rose-950/40 text-rose-300 border-rose-800",
        };
      case "likely_real":
        return {
          title: "LIKELY REAL (AUTHENTIC)",
          summary: "Visual gradients and Fourier power decay conform to physical camera capture.",
          borderTop: "border-t-emerald-500",
          textColor: "text-emerald-400",
          statusTag: "bg-emerald-950/40 text-emerald-300 border-emerald-800",
        };
      case "uncertain":
      default:
        return {
          title: "UNCERTAIN / INDETERMINATE",
          summary: "Evidence lies within the ambiguous boundary corridor or shows instability under perturbation.",
          borderTop: "border-t-amber-500",
          textColor: "text-amber-400",
          statusTag: "bg-amber-950/40 text-amber-300 border-amber-800",
        };
    }
  };

  const details = getVerdictDetails();

  return (
    <div className={`forensic-panel p-5 border-t-2 ${details.borderTop} space-y-4`}>
      {/* Development Scaffolding Guardrail */}
      {is_development_placeholder && (
        <div className="p-2.5 rounded bg-amber-950/30 border border-amber-800/60 text-amber-300 text-xs flex items-center gap-2">
          <Sliders className="w-3.5 h-3.5 shrink-0" />
          <span>Development Scaffolding Mode: Active without trained checkpoint</span>
        </div>
      )}

      {/* Main Verdict Row */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        {/* Left: Classification Details */}
        <div className="space-y-1.5 max-w-xl">
          <div className="flex items-center space-x-2 text-[11px] font-mono text-slate-400 uppercase tracking-wider">
            <span>Analysis Result</span>
            <span>·</span>
            <span>ConvNeXt-Tiny Spatial + 2D FFT</span>
          </div>

          <div className="flex items-baseline space-x-3">
            <h2 className={`text-xl font-bold tracking-tight ${details.textColor}`}>
              {details.title}
            </h2>
            <span className="text-xs font-mono text-slate-300 px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700">
              {confidence_level.toUpperCase()} CONFIDENCE
            </span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed pt-0.5">
            {details.summary}
          </p>
        </div>

        {/* Right: Quantitative Likelihood */}
        <div className="text-left md:text-right border-t md:border-t-0 md:border-l border-slate-800 pt-3 md:pt-0 md:pl-6 shrink-0">
          <div className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">
            Synthetic Likelihood
          </div>
          <div className={`text-3xl font-mono font-bold tracking-tight mt-0.5 ${details.textColor}`}>
            {calPct}%
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-0.5">
            Calibrated (T=0.9995) · Raw: {rawPct}%
          </div>
        </div>
      </div>

      {/* Uncertainty & Human Review Callout */}
      {uncertain && (
        <div className="p-3 rounded border border-amber-800/60 bg-amber-950/25 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <div className="font-semibold text-amber-300">
              Human Forensic Review Recommended
            </div>
            <p className="text-[11px] text-amber-300/80 leading-relaxed">
              The calibrated score is within the ambiguous decision corridor (0.40–0.60) or the prediction
              exhibited instability under perturbation probes. Automated classification should not be treated as conclusive.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
