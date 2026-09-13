"use client";

import React from "react";
import { SpectralEvidence } from "@/types/prediction";

interface SpectralViewerProps {
  spectral: SpectralEvidence;
}

export const SpectralViewer: React.FC<SpectralViewerProps> = ({ spectral }) => {
  const hfRatio = spectral.high_frequency_energy_ratio;
  const hfPct = hfRatio != null ? (hfRatio * 100).toFixed(1) : null;
  const isElevated = hfRatio != null && hfRatio > 0.40;

  return (
    <section className="space-y-4 pt-4 border-t border-[#e5e2d9]">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between border-b border-[#e5e2d9] pb-2 gap-2">
        <div className="flex items-baseline space-x-3">
          <span className="text-[10px] font-mono tracking-[0.2em] text-[#9a3412] uppercase font-semibold">
            02
          </span>
          <h3 className="text-lg font-editorial font-semibold text-[#121316]">
            Frequency Domain Evidence (2D FFT)
          </h3>
        </div>

        <div className="text-xs font-mono text-[#606570]">
          {spectral.representation || "Centered 2D Fast Fourier Transform"}
        </div>
      </div>

      {/* Analytical Layout: Spectrum Image + Quantitative Metric */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center pt-2">
        {/* Left: 2D FFT Spectrum Display */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase tracking-wider text-[#606570] flex justify-between">
            <span>Exhibit C · Centered 2D Fourier Magnitude</span>
            <span className="text-[#8c8a82]">Log-Scale Energy</span>
          </div>
          <div className="w-full h-64 bg-[#121316] border border-[#e5e2d9] p-2 flex items-center justify-center overflow-hidden">
            {spectral.spectrum ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={spectral.spectrum}
                alt="2D FFT Spectrum"
                className="w-full h-full object-contain pixelated"
              />
            ) : (
              <span className="text-xs font-mono text-[#8c8a82]">Spectral representation unavailable</span>
            )}
          </div>
          <p className="text-[11px] text-[#8c8a82] font-mono">
            DC component centered; radial distance represents spatial frequency.
          </p>
        </div>

        {/* Right: Quantitative Energy Metric & Analytical Interpretation */}
        <div className="space-y-4 bg-[#f3f1ea] border border-[#e5e2d9] p-5 font-mono">
          <div className="space-y-1">
            <div className="text-[10px] uppercase tracking-wider text-[#8c8a82]">
              High-Frequency Radial Energy
            </div>
            <div className="flex items-baseline space-x-2">
              <span className={`text-3xl font-bold tracking-tight ${isElevated ? "text-[#991b1b]" : "text-[#166534]"}`}>
                {hfPct != null ? `${hfPct}%` : "—"}
              </span>
              <span className="text-xs text-[#606570]">
                ({isElevated ? "Elevated Grid Residual" : "Natural Power-Law Decay"})
              </span>
            </div>
          </div>

          {/* Clean Distribution Gauge */}
          {hfRatio != null && (
            <div className="space-y-1.5 pt-1">
              <div className="w-full h-1.5 bg-[#dcd9ce] overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    isElevated ? "bg-[#991b1b]" : "bg-[#166534]"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, hfRatio * 100))}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-[#8c8a82]">
                <span>Natural Continuous Decay (&lt;40%)</span>
                <span>Generative Periodic Grid (&gt;40%)</span>
              </div>
            </div>
          )}

          {/* Scientific Interpretation */}
          <div className="text-xs text-[#606570] font-sans leading-relaxed pt-2 border-t border-[#e5e2d9]">
            <strong className="text-[#121316] font-mono uppercase text-[10px] tracking-wider">Spectral Scope: </strong>
            Natural optical camera sensors exhibit smooth power-law falloff across radial frequencies. Generative upsamplers
            frequently introduce periodic lattice artifacts. This spectral metric serves as supporting context only.
          </div>
        </div>
      </div>
    </section>
  );
};
