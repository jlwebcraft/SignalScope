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
    <div className="forensic-panel flex flex-col justify-between p-4 space-y-3.5">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div>
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            2. Frequency Evidence (2D FFT)
          </h3>
          <p className="text-[11px] font-mono text-slate-500 mt-0.5">
            {spectral.representation || "Centered 2D FFT Log-Magnitude"}
          </p>
        </div>

        {hfPct && (
          <div className="text-right">
            <span className="text-[10px] font-mono text-slate-500 uppercase">
              HF Energy Ratio
            </span>
            <div className={`text-xs font-mono font-semibold ${isElevated ? "text-rose-400" : "text-emerald-400"}`}>
              {hfPct}% ({isElevated ? "Elevated" : "Natural Decay"})
            </div>
          </div>
        )}
      </div>

      {/* Spectrum Viewport */}
      <div className="flex flex-col items-center">
        <div className="relative w-56 h-56 rounded border border-slate-700/80 bg-slate-950 flex items-center justify-center overflow-hidden">
          {spectral.spectrum ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={spectral.spectrum}
              alt="2D FFT Spectrum"
              className="w-full h-full object-contain pixelated"
            />
          ) : (
            <div className="p-4 text-center text-xs text-slate-500 font-mono">
              Spectral representation computed during inference
            </div>
          )}
        </div>

        {/* High Frequency Distribution Metric */}
        {hfRatio != null && (
          <div className="w-full max-w-xs mt-3 space-y-1">
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>High-Frequency Radial Energy</span>
              <span className="text-slate-200">{hfPct}%</span>
            </div>
            <div className="w-full h-1.5 rounded bg-slate-800 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  isElevated ? "bg-rose-500" : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, Math.max(0, hfRatio * 100))}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-slate-500">
              <span>Expected 1/f Decay (&lt;40%)</span>
              <span>Periodic Grid Residual (&gt;40%)</span>
            </div>
          </div>
        )}
      </div>

      {/* Scientific Context */}
      <div className="p-2.5 rounded bg-slate-900/50 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
        <span className="font-semibold text-slate-300">Spectral Scope: </span>
        Natural camera sensors exhibit continuous radial decay. Convolutional upsamplers and latent decoders often leave high-frequency grid harmonics. This metric provides supporting context only.
      </div>
    </div>
  );
};
