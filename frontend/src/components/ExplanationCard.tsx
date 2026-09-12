"use client";

import React from "react";
import { MessageSquareText, ShieldAlert, Sparkles } from "lucide-react";
import { PredictionResponse } from "@/types/prediction";

interface ExplanationCardProps {
  prediction: PredictionResponse;
}

export const ExplanationCard: React.FC<ExplanationCardProps> = ({ prediction }) => {
  const { explanation, disclaimer, verdict, evidence } = prediction;

  const getHighlights = () => {
    const items: string[] = [];

    // Spatial
    if (evidence.spatial.available) {
      if (verdict === "likely_ai_generated") {
        items.push("Spatial feature attribution observed localized structural anomalies in highlighted regions.");
      } else {
        items.push("Spatial feature attribution indicates broad natural gradient consistency without anomalous localized focus.");
      }
    }

    // Spectral
    if (evidence.spectral.available) {
      if (evidence.spectral.high_frequency_energy_ratio && evidence.spectral.high_frequency_energy_ratio > 0.40) {
        items.push("2D Fourier spectral analysis observed elevated high-frequency periodic residuals.");
      } else {
        items.push("2D Fourier spectral analysis observed expected continuous power-law energy decay.");
      }
    }

    // Robustness
    if (evidence.robustness.available) {
      if (evidence.robustness.stability_score >= 0.80) {
        items.push("Prediction stability was high under the transformations evaluated for this sample.");
      } else {
        items.push("Prediction exhibited sensitivity or probability drift under tested compression/rescaling probes.");
      }
    }

    return items;
  };

  const highlights = getHighlights();

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
        <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400">
          <MessageSquareText className="w-4 h-4" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white">Evidence-Grounded Explanation</h3>
          <p className="text-[11px] text-slate-400">Deterministic synthesis of model observations</p>
        </div>
      </div>

      {/* Synthesis Paragraph */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs text-slate-200 leading-relaxed">
        <p className="font-sans">{explanation}</p>
      </div>

      {/* Key Factors Checklist */}
      {highlights.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Primary Contributing Factors:
          </div>
          <ul className="space-y-1.5">
            {highlights.map((item, idx) => (
              <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                <span className="text-indigo-400 font-bold shrink-0">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Mandatory Disclaimer */}
      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
        <ShieldAlert className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Mandatory Disclaimer: </span>
          {disclaimer}
        </div>
      </div>
    </div>
  );
};
