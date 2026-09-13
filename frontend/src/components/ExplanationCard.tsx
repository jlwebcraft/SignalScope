"use client";

import React from "react";
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
        items.push("Spatial feature attribution identified localized structural anomalies in highlighted areas.");
      } else {
        items.push("Spatial feature attribution exhibits natural continuous gradients across the image plane.");
      }
    }

    // Spectral
    if (evidence.spectral.available) {
      if (evidence.spectral.high_frequency_energy_ratio && evidence.spectral.high_frequency_energy_ratio > 0.40) {
        items.push("2D Fourier spectral analysis identified elevated high-frequency periodic residuals typical of generative upsampling.");
      } else {
        items.push("2D Fourier spectral analysis observed standard continuous power-law decay characteristic of physical camera sensors.");
      }
    }

    // Robustness
    if (evidence.robustness.available) {
      if (evidence.robustness.stability_score >= 0.80) {
        items.push("Prediction stability remained high across controlled compression and downscaling transformations.");
      } else {
        items.push("Prediction exhibited sensitivity or decision drift under tested perturbation probes.");
      }
    }

    // Provenance
    if (evidence.metadata.available) {
      if (evidence.metadata.c2pa_present) {
        items.push("C2PA content credentials verified cryptographic provenance manifest.");
      } else if (evidence.metadata.has_exif) {
        items.push("Standard camera EXIF headers are recorded.");
      } else {
        items.push("No EXIF or C2PA provenance headers present (typical for web distribution).");
      }
    }

    return items;
  };

  const highlights = getHighlights();

  return (
    <div className="forensic-panel p-5 space-y-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-2.5">
        <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
          5. Forensic Synthesis & Evidence Summary
        </h3>
        <p className="text-[11px] font-mono text-slate-500 mt-0.5">
          Deterministic summary grounded in quantitative feature observations
        </p>
      </div>

      {/* Synthesis Narrative */}
      <div className="p-3.5 rounded bg-slate-900/60 border border-slate-800 text-xs text-slate-200 leading-relaxed font-sans">
        <div className="font-semibold text-slate-100 mb-1 uppercase text-[10px] font-mono tracking-wider">
          Forensic Finding:
        </div>
        <p>{explanation}</p>
      </div>

      {/* Evidence Factors */}
      {highlights.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-mono font-medium text-slate-400 uppercase tracking-wider">
            Evidence Supporting This Finding:
          </div>
          <ul className="space-y-1.5 font-mono text-xs text-slate-300">
            {highlights.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-slate-500 font-bold shrink-0">—</span>
                <span className="leading-relaxed">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Mandatory Disclaimer */}
      <div className="p-2.5 rounded bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-500 leading-relaxed">
        <span className="font-semibold text-slate-400">Mandatory Forensic Disclaimer: </span>
        {disclaimer}
      </div>
    </div>
  );
};
