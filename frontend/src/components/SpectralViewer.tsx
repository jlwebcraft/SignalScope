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
    <section className="bg-white border border-slate-200 rounded overflow-hidden shadow-xs">
      {/* Section Sub-Header */}
      <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-2 h-2 rounded-[1px] bg-slate-700" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Frequency Domain Analysis (2D FFT Residuals)
          </h3>
        </div>
        <span className="text-[11px] font-mono text-slate-500">
          Centered 2D Log-Magnitude Spectrum
        </span>
      </div>

      <div className="p-5 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          {/* Left: 2D FFT Spectrum Display */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
              <span className="font-semibold text-slate-700">Exhibit C · Fourier Log-Magnitude Spectrogram</span>
              <span>Centered DC</span>
            </div>
            <div className="w-full h-64 bg-slate-950 border border-slate-300 rounded p-1 flex items-center justify-center overflow-hidden">
              {spectral.spectrum ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={spectral.spectrum}
                  alt="2D Centered Fast Fourier Transform Spectrum"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <span className="text-xs font-mono text-slate-500">Spectral representation unavailable</span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              DC center; radial distance reflects spatial frequency. Sharp periodic peaks indicate generative grid lattices.
            </p>
          </div>

          {/* Right: Quantitative Radial Energy Gauge */}
          <div className="space-y-4 bg-slate-50 border border-slate-200 rounded p-5 font-mono text-xs">
            <div className="space-y-1 border-b border-slate-200 pb-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                High-Frequency Energy Ratio
              </div>
              <div className="flex items-baseline space-x-2">
                <span
                  className={`text-3xl font-bold tracking-tight ${
                    isElevated ? "text-rose-700" : "text-emerald-700"
                  }`}
                >
                  {hfPct != null ? `${hfPct}%` : "—"}
                </span>
                <span className="text-xs text-slate-600 font-sans">
                  ({isElevated ? "Elevated Synthetic Grid Residuals" : "Conforms to Natural Optical Decay"})
                </span>
              </div>
            </div>

            {/* Distribution Bar */}
            {hfRatio != null && (
              <div className="space-y-1.5">
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      isElevated ? "bg-rose-600" : "bg-emerald-600"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, hfRatio * 100))}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Natural Smooth Falloff (&lt; 40%)</span>
                  <span>Generative Grid Anomaly (&gt; 40%)</span>
                </div>
              </div>
            )}

            {/* Diagnostic Interpretation */}
            <div className="text-xs text-slate-600 font-sans leading-relaxed pt-1">
              <strong className="font-mono text-[10px] text-slate-900 uppercase font-semibold">Spectral Scope: </strong>
              Physical camera lenses exhibit continuous power-law energy decay across radial frequencies. Generative upsamplers and latent decoders
              frequently introduce periodic lattice artifacts. Spectral evidence serves as supporting corroboration.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
