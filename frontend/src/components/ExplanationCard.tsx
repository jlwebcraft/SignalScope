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
      if (evidence.spectral.high_frequency_energy_ratio && evidence.spectral.high_frequency_energy_ratio > 0.40) {
        items.push({
          label: "Spectral Harmonics",
          text: "2D Fourier analysis observed elevated high-frequency periodic residuals typical of generative upsampling lattices.",
        });
      } else {
        items.push({
          label: "Spectral Harmonics",
          text: "2D Fourier analysis conformed to power-law energy decay typical of natural physical scenes.",
        });
      }
    }

    // Robustness
    if (evidence.robustness.available) {
      if (evidence.robustness.stability_score >= 0.80) {
        items.push({
          label: "Transformation Invariance",
          text: "Model predictions exhibited high stability across controlled compression and downscaling transformations.",
        });
      } else {
        items.push({
          label: "Transformation Invariance",
          text: "Prediction demonstrated numerical sensitivity or categorical shifts under tested perturbation probes.",
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
          text: "Standard camera EXIF headers are recorded.",
        });
      } else {
        items.push({
          label: "Provenance Claims",
          text: "EXIF headers not recorded; typical for web distribution and re-encoding.",
        });
      }
    }

    return items;
  };

  const highlights = getHighlights();

  return (
    <section className="space-y-6 pt-4 border-t border-[#e5e2d9]">
      {/* Section Header */}
      <div className="flex items-baseline space-x-3 border-b border-[#e5e2d9] pb-2">
        <span className="text-[10px] font-mono tracking-[0.2em] text-[#9a3412] uppercase font-semibold">
          05
        </span>
        <h3 className="text-lg font-editorial font-semibold text-[#121316]">
          Forensic Synthesis & Methodology
        </h3>
      </div>

      <div className="space-y-6 text-xs text-[#2a2d34]">
        {/* Finding Paragraph */}
        <div className="space-y-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#606570] font-semibold">
            Finding
          </span>
          <p className="text-sm leading-relaxed text-[#121316] font-sans">
            {explanation}
          </p>
        </div>

        {/* Evidence Factor Breakdown */}
        {highlights.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-[#eeece5]">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#606570] font-semibold">
              Evidence Supporting Finding
            </span>
            <div className="space-y-2 pt-1 font-mono">
              {highlights.map((item, idx) => (
                <div key={idx} className="flex items-start space-x-3">
                  <span className="text-[#9a3412] font-bold text-[11px] shrink-0">
                    0{idx + 1}
                  </span>
                  <div className="text-xs text-[#2a2d34]">
                    <strong className="text-[#121316] uppercase text-[10px] tracking-wider">{item.label}: </strong>
                    <span className="font-sans text-[#2a2d34]">{item.text}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Limitations */}
        <div className="space-y-1 pt-2 border-t border-[#eeece5]">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#606570] font-semibold">
            Methodological Limitations
          </span>
          <p className="text-xs text-[#606570] font-sans leading-relaxed">
            Evaluations are statistical likelihood estimates derived from deep ConvNeXt representations and Fourier spectra.
            Novel generative architectures, unseen post-processing filters, or intentional adversarial perturbations can
            affect classification stability.
          </p>
        </div>

        {/* Mandatory Disclaimer */}
        <div className="p-3 bg-[#f3f1ea] border border-[#e5e2d9] text-[11px] text-[#606570] leading-relaxed">
          <strong className="text-[#121316] font-mono uppercase text-[10px] tracking-wider">Mandatory Forensic Disclaimer: </strong>
          {disclaimer}
        </div>
      </div>
    </section>
  );
};
