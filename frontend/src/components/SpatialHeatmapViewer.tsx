"use client";

import React, { useState } from "react";
import { SpatialEvidence } from "@/types/prediction";
import { Sliders } from "lucide-react";

interface SpatialHeatmapViewerProps {
  spatial: SpatialEvidence;
  originalPreview: string | null;
}

export const SpatialHeatmapViewer: React.FC<SpatialHeatmapViewerProps> = ({
  spatial,
  originalPreview,
}) => {
  const [heatmapOpacity, setHeatmapOpacity] = useState(70);
  const [viewMode, setViewMode] = useState<"side-by-side" | "overlay">("side-by-side");

  const concentrationPct =
    spatial.attribution_concentration != null
      ? (spatial.attribution_concentration * 100).toFixed(1)
      : null;

  return (
    <section className="bg-white border border-slate-200 rounded overflow-hidden shadow-xs">
      {/* Section Sub-Header */}
      <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2.5">
          <div className="w-2 h-2 rounded-[1px] bg-slate-700" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Spatial Feature Attribution (Grad-CAM)
          </h3>
          <span className="text-slate-300">/</span>
          <span className="text-[11px] font-mono text-slate-500">
            Layer: {spatial.target_layer || "ConvNeXt Stage 3"}
          </span>
        </div>

        <div className="flex items-center space-x-4 text-xs font-mono">
          {concentrationPct && (
            <span className="text-slate-600">
              Concentration: <strong className="text-slate-900">{concentrationPct}%</strong>
            </span>
          )}

          {/* View Mode Toggle */}
          <div className="flex items-center bg-slate-200/70 p-0.5 rounded text-[11px]">
            <button
              type="button"
              onClick={() => setViewMode("side-by-side")}
              className={`px-2 py-0.5 rounded transition-colors ${
                viewMode === "side-by-side"
                  ? "bg-white text-slate-900 font-semibold shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Side-by-Side
            </button>
            <button
              type="button"
              onClick={() => setViewMode("overlay")}
              className={`px-2 py-0.5 rounded transition-colors ${
                viewMode === "overlay"
                  ? "bg-white text-slate-900 font-semibold shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Interactive Overlay
            </button>
          </div>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {viewMode === "side-by-side" ? (
          /* Side-by-Side Comparator */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Exhibit A: Original */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
                <span className="font-semibold text-slate-700">Exhibit A · Source Input</span>
                <span>Original RGB</span>
              </div>
              <div className="w-full h-72 bg-slate-100 border border-slate-200 rounded p-1 flex items-center justify-center overflow-hidden">
                {originalPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={originalPreview}
                    alt="Source forensic exhibit"
                    className="w-full h-full object-contain pixelated"
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-400">Source image unavailable</span>
                )}
              </div>
            </div>

            {/* Exhibit B: Grad-CAM Heatmap */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
                <span className="font-semibold text-slate-700">Exhibit B · Receptive Field Attribution</span>
                <span>Grad-CAM Activation</span>
              </div>
              <div className="relative w-full h-72 bg-slate-100 border border-slate-200 rounded p-1 flex items-center justify-center overflow-hidden">
                {originalPreview && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={originalPreview}
                    alt="Source image for heatmap alignment"
                    className="w-full h-full object-contain pixelated opacity-30"
                  />
                )}
                {spatial.heatmap ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={spatial.heatmap}
                    alt="Grad-CAM spatial heatmap overlay"
                    className="absolute inset-1 w-[calc(100%-8px)] h-[calc(100%-8px)] object-contain mix-blend-multiply"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xs font-mono text-slate-400">Heatmap unavailable</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Single Overlay with Blend Slider */
          <div className="space-y-3">
            <div className="relative w-full h-80 bg-slate-100 border border-slate-200 rounded p-1 flex items-center justify-center overflow-hidden">
              {originalPreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={originalPreview}
                  alt="Base evidence image"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <span className="text-xs font-mono text-slate-400">Source image unavailable</span>
              )}

              {spatial.heatmap && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={spatial.heatmap}
                  alt="Grad-CAM spatial activation map"
                  style={{ opacity: heatmapOpacity / 100 }}
                  className="absolute inset-1 w-[calc(100%-8px)] h-[calc(100%-8px)] object-contain pointer-events-none transition-opacity duration-75"
                />
              )}
            </div>

            {/* Slider control */}
            {spatial.heatmap && (
              <div className="p-3 bg-slate-50 border border-slate-200 rounded flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
                <div className="flex items-center space-x-2 text-slate-700">
                  <Sliders className="w-4 h-4 text-slate-500" />
                  <span>Attribution Heatmap Blend:</span>
                  <span className="font-bold text-slate-900">{heatmapOpacity}%</span>
                </div>
                <div className="flex-1 max-w-xs flex items-center space-x-2">
                  <span className="text-[10px] text-slate-400">0%</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    aria-label="Attribution heatmap opacity"
                    className="w-full h-1.5 bg-slate-300 rounded accent-slate-900 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-400">100%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Scientific Context & Methodological Limitations */}
        <div className="p-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-600 leading-relaxed font-sans">
          <strong className="font-mono text-[10px] text-slate-900 uppercase font-semibold">Attribution Scope: </strong>
          Highlighted warm regions indicate receptive fields that exerted the greatest numerical weight on the classifier logit.
          Native ConvNeXt patch features are 32×32 and bilinearly upsampled for visual inspection. Attribution illustrates model focus;
          it does not constitute standalone causal proof of artificial generation.
        </div>
      </div>
    </section>
  );
};
