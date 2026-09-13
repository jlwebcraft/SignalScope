"use client";

import React from "react";
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

  const getVerdictStyle = () => {
    switch (verdict) {
      case "likely_ai_generated":
        return {
          title: "Likely AI-Generated",
          accentColor: "text-[#991b1b]",
          badgeClass: "bg-[#fef2f2] text-[#991b1b] border-[#fecaca]",
          summary: "Spatial feature attribution and frequency power distributions exhibit statistical anomalies characteristic of generative synthesis models.",
        };
      case "likely_real":
        return {
          title: "Likely Real (Authentic)",
          accentColor: "text-[#166534]",
          badgeClass: "bg-[#f0fdf4] text-[#166534] border-[#bbf7d0]",
          summary: "Feature gradients, natural edge transitions, and 2D Fourier power decay conform to physical optical sensor capture.",
        };
      case "uncertain":
      default:
        return {
          title: "Uncertain / Indeterminate",
          accentColor: "text-[#92400e]",
          badgeClass: "bg-[#fffbeb] text-[#92400e] border-[#fde68a]",
          summary: "Model evidence lies within the ambiguous decision corridor (0.40–0.60) or exhibits volatility under standard image perturbations.",
        };
    }
  };

  const style = getVerdictStyle();

  return (
    <div className="space-y-4">
      {/* Development Scaffolding Guardrail */}
      {is_development_placeholder && (
        <div className="p-3 border border-[#fde68a] bg-[#fffbeb] text-[#92400e] text-xs font-mono">
          Development Scaffolding Mode: Active without trained checkpoint
        </div>
      )}

      {/* Case Finding Header Block */}
      <div className="border-b-2 border-[#121316] pb-5">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          {/* Left: Finding Description */}
          <div className="space-y-2 max-w-xl">
            <div className="text-[10px] font-mono tracking-[0.25em] text-[#9a3412] uppercase font-semibold">
              Case Analysis // Finding
            </div>

            <h2 className={`text-3xl sm:text-4xl font-editorial font-semibold tracking-tight ${style.accentColor}`}>
              {style.title}
            </h2>

            <p className="text-xs text-[#606570] leading-relaxed pt-1">
              {style.summary}
            </p>
          </div>

          {/* Right: Quantitative Finding Sheet */}
          <div className="bg-[#f3f1ea] border border-[#e5e2d9] p-4 min-w-[240px] text-right space-y-1 font-mono shrink-0">
            <div className="text-[10px] uppercase tracking-wider text-[#8c8a82]">
              Synthetic Likelihood
            </div>
            <div className={`text-4xl font-bold tracking-tight ${style.accentColor}`}>
              {calPct}%
            </div>
            <div className="text-[10px] text-[#606570] pt-1 border-t border-[#e5e2d9] flex justify-between">
              <span>Confidence:</span>
              <span className="font-semibold text-[#121316] capitalize">{confidence_level}</span>
            </div>
            <div className="text-[10px] text-[#8c8a82] flex justify-between">
              <span>Calibrated (T=0.9995):</span>
              <span>Raw: {rawPct}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Uncertain / Human Review Recommendation Callout */}
      {uncertain && (
        <div className="p-4 border-l-4 border-[#92400e] bg-[#fffbeb] border-y border-r border-[#fde68a] text-xs text-[#92400e] space-y-1">
          <div className="font-mono text-[11px] font-bold uppercase tracking-wider">
            Human Forensic Review Recommended
          </div>
          <p className="text-xs text-[#92400e]/90 leading-relaxed font-sans">
            The automated pipeline declined confident adjudication. The probability score falls near the 0.50
            decision corridor or the classification exhibited sensitivity during perturbation stress tests.
            This case requires human visual and contextual verification.
          </p>
        </div>
      )}
    </div>
  );
};
