"use client";

import React from "react";
import { Activity, Info, BarChart3 } from "lucide-react";
import { SpectralEvidence } from "@/types/prediction";

interface SpectralViewerProps {
  spectral: SpectralEvidence;
}

export const SpectralViewer: React.FC<SpectralViewerProps> = ({ spectral }) => {
  const hfRatio = spectral.high_frequency_energy_ratio;
  const hfPct = hfRatio != null ? (hfRatio * 100).toFixed(1) : null;
  const isElevated = hfRatio != null && hfRatio > 0.40;

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Frequency Domain Evidence (2D FFT)</h3>
            <p className="text-[11px] text-slate-400">{spectral.representation || "Centered 2D FFT Spectrum"}</p>
          </div>
        </div>

        {hfPct && (
          <div className="text-right">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">HF Energy Ratio</span>
            <div className={`text-xs font-bold ${isElevated ? "text-rose-400" : "text-emerald-400"}`}>
              {hfPct}% ({isElevated ? "Elevated" : "Natural Decay"})
            </div>
          </div>
        )}
      </div>

      {/* Visual Spectrum Image */}
      <div className="flex flex-col items-center">
        <div className="relative w-64 h-64 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center shadow-lg">
          {spectral.spectrum ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={spectral.spectrum}
              alt="2D FFT Spectrum"
              className="w-full h-full object-contain pixelated"
            />
          ) : (
            <div className="p-4 text-center text-xs text-slate-500">
              Spectral representation computed during forward pass
            </div>
          )}
        </div>

        {/* Energy Meter */}
        {hfRatio != null && (
          <div className="w-full max-w-xs mt-4 space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                <BarChart3 className="w-3 h-3" /> High-Frequency Distribution
              </span>
              <span className="font-mono text-slate-300">{hfPct}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isElevated ? "bg-gradient-to-r from-cyan-500 to-rose-500" : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, Math.max(0, hfRatio * 100))}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Expected 1/f Decay (&lt;40%)</span>
              <span>Periodic Grid Residual (&gt;40%)</span>
            </div>
          </div>
        )}
      </div>

      {/* Scientific Context */}
      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Spectral Note: </span>
          Natural physical photographs exhibit continuous radial energy decay. Generative convolutional upsamplers and latent decoders frequently leave high-frequency grid harmonics.
          This metric is supporting evidence, not independent proof of synthetic origin.
        </div>
      </div>
    </div>
  );
};
