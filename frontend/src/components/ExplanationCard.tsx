"use client";

import React from "react";
import { PredictionResponse } from "@/types/prediction";

interface ExplanationCardProps {
  prediction: PredictionResponse;
}

export const ExplanationCard: React.FC<ExplanationCardProps> = ({ prediction }) => {
  const { explanation, disclaimer, verdict, evidence } = prediction;

  const getHighlights = () => {
    const items: { label: string; text: string }[] = [];

    // Spatial
    if (evidence.spatial.available) {
      if (verdict === "likely_ai_generated") {
        items.push({
          label: "Spatial Attribution",
          text: "Spatial activation analysis observed localized structural anomalies in highlighted image regions.",
        });
      } else {
        items.push({
          label: "Spatial Attribution",
          text: "Spatial feature gradients exhibit continuous natural transitions consistent with optical lens capture.",
        });
      }
    }

    // Spectral
    if (evidence.spectral.available) {
      if (
        evidence.spectral.high_frequency_energy_ratio &&
        evidence.spectral.high_frequency_energy_ratio > 0.40
      ) {
        items.push({
          label: "Spectral Harmonics",
          text: "2D Fourier analysis observed elevated high-frequency periodic residuals typical of generative upsampling lattices.",
        });
      } else {
        items.push({
          label: "Spectral Harmonics",
          text: "2D Fourier analysis conformed to smooth power-law energy decay typical of natural physical optical scenes.",
        });
      }
    }

    // Robustness
    if (evidence.robustness.available) {
      if (evidence.robustness.stability_score >= 0.70) {
        items.push({
          label: "Perturbation Invariance",
          text: "Model predictions exhibited high stability and invariance across controlled compression and downscaling transformations.",
        });
      } else {
        items.push({
          label: "Perturbation Invariance",
          text: "Prediction demonstrated significant numerical drift or categorical decision shifts under perturbation stress probes.",
        });
      }
    }

    // Provenance
    if (evidence.metadata.available) {
      if (evidence.metadata.c2pa_present) {
        items.push({
          label: "Provenance Claims",
          text: "Cryptographic C2PA manifest claims are attached and verified.",
        });
      } else if (evidence.metadata.has_exif) {
        items.push({
          label: "Provenance Claims",
          text: "Standard camera EXIF hardware exposure headers are recorded.",
        });
      } else {
        items.push({
          label: "Provenance Claims",
          text: "EXIF headers not recorded; standard for online social platforms and re-encoded media.",
        });
      }
    }

    return items;
  };

  const highlights = getHighlights();

  return (
    <section className="bg-white border border-slate-200 rounded overflow-hidden shadow-xs">
      {/* Section Sub-Header */}
      <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-2 h-2 rounded-[1px] bg-slate-700" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Synthesized Forensic Synthesis & Scope
          </h3>
        </div>
        <span className="text-[11px] font-mono text-slate-500">
          Multimodal Evidence Integration
        </span>
      </div>

      <div className="p-5 space-y-5 text-xs text-slate-700 leading-relaxed">
        {/* Core Synthesized Explanation */}
        <div className="space-y-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 font-semibold block">
            Adjudication Statement
          </span>
          <p className="text-sm font-medium text-slate-900 leading-relaxed">
            {explanation}
          </p>
        </div>

        {/* Supporting Evidence Breakdown */}
        {highlights.length > 0 && (
          <div className="space-y-2 pt-3 border-t border-slate-100">
            <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 font-semibold block">
              Corroborating Multimodal Evidence
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              {highlights.map((item, idx) => (
                <div key={idx} className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="w-4 font-mono text-[10px] text-slate-400 font-bold">
                      0{idx + 1}.
                    </span>
                    <span className="font-mono text-[11px] uppercase font-bold text-slate-800">
                      {item.label}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 pl-6 font-sans">
                    {item.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Methodological Limitations */}
        <div className="space-y-1 pt-3 border-t border-slate-100">
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 font-semibold block">
            Methodological Boundaries
          </span>
          <p className="text-xs text-slate-500">
            Evaluations are statistical likelihood estimates derived from deep ConvNeXt representations and 2D Fourier spectra.
            Novel diffusion architectures, heavy adversarial filtering, or non-standard post-processing may impact classifier confidence.
          </p>
        </div>

        {/* Mandatory Forensic Disclaimer */}
        <div className="p-3 bg-slate-100 border border-slate-200 rounded text-[11px] text-slate-600 space-y-1">
          <strong className="font-mono text-[10px] text-slate-800 uppercase font-semibold block">
            Mandatory Disclaimer:
          </strong>
          <p>{disclaimer}</p>
        </div>
      </div>
    </section>
  );
};
