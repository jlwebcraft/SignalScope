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
    <section className="bg-slate-900 border border-slate-800 rounded overflow-hidden shadow-sm text-slate-200">
      {/* Precision Inspection Toolbar */}
      <div className="px-5 py-3 border-b border-slate-800 bg-slate-950 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2.5">
          <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" aria-hidden="true" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-100">
            Spatial Receptive Field Attribution
          </h3>
          <span className="text-slate-600">/</span>
          <span className="text-[11px] font-mono text-slate-400">
            {spatial.target_layer || "ConvNeXt Stage 3 (Patch 32×32)"}
          </span>
        </div>

        <div className="flex items-center space-x-4 text-xs font-mono">
          {concentrationPct && (
            <span className="text-slate-300">
              Concentration: <strong className="text-sky-400">{concentrationPct}%</strong>
            </span>
          )}

          {/* View Mode Toggle */}
          <div className="flex items-center bg-slate-800 p-0.5 rounded text-[11px]">
            <button
              type="button"
              onClick={() => setViewMode("side-by-side")}
              className={`px-2.5 py-1 rounded transition-colors ${
                viewMode === "side-by-side"
                  ? "bg-slate-700 text-white font-semibold shadow-xs"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Side-by-Side
            </button>
            <button
              type="button"
              onClick={() => setViewMode("overlay")}
              className={`px-2.5 py-1 rounded transition-colors ${
                viewMode === "overlay"
                  ? "bg-slate-700 text-white font-semibold shadow-xs"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Overlay Blend
            </button>
          </div>
        </div>
      </div>

      {/* Optical Comparator Canvas */}
      <div className="p-5 space-y-3">
        {viewMode === "side-by-side" ? (
          /* Side-by-Side Comparator */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Exhibit A: Original */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span className="font-semibold text-slate-200">Exhibit A · Source Input</span>
                <span>Original RGB</span>
              </div>
              <div className="w-full h-72 bg-slate-950 border border-slate-800 rounded flex items-center justify-center overflow-hidden p-2">
                {originalPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={originalPreview}
                    alt="Source forensic exhibit"
                    className="w-full h-full object-contain pixelated"
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-600">Source image unavailable</span>
                )}
              </div>
            </div>

            {/* Exhibit B: Grad-CAM Heatmap */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span className="font-semibold text-slate-200">Exhibit B · Attribution Field</span>
                <span>Grad-CAM Activation Map</span>
              </div>
              <div className="relative w-full h-72 bg-slate-950 border border-slate-800 rounded flex items-center justify-center overflow-hidden p-2">
                {originalPreview && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={originalPreview}
                    alt="Source image for heatmap alignment"
                    className="w-full h-full object-contain pixelated opacity-25"
                  />
                )}
                {spatial.heatmap ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={spatial.heatmap}
                    alt="Grad-CAM spatial heatmap overlay"
                    className="absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)] object-contain mix-blend-screen"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xs font-mono text-slate-600">Heatmap unavailable</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Single Overlay with Blend Slider */
          <div className="space-y-3">
            <div className="relative w-full h-80 bg-slate-950 border border-slate-800 rounded flex items-center justify-center overflow-hidden p-2">
              {originalPreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={originalPreview}
                  alt="Base evidence image"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <span className="text-xs font-mono text-slate-600">Source image unavailable</span>
              )}

              {spatial.heatmap && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={spatial.heatmap}
                  alt="Grad-CAM spatial activation map"
                  style={{ opacity: heatmapOpacity / 100 }}
                  className="absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)] object-contain pointer-events-none transition-opacity duration-75"
                />
              )}
            </div>

            {/* Slider control */}
            {spatial.heatmap && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
                <div className="flex items-center space-x-2 text-slate-300">
                  <Sliders className="w-4 h-4 text-sky-400" />
                  <span>Overlay Heatmap Blend:</span>
                  <span className="font-bold text-sky-400">{heatmapOpacity}%</span>
                </div>
                <div className="flex-1 max-w-xs flex items-center space-x-2">
                  <span className="text-[10px] text-slate-500">0%</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    aria-label="Attribution heatmap opacity"
                    className="w-full h-1.5 bg-slate-700 rounded accent-sky-400 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-500">100%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Integrated technical caption (replaces repeated gray scope box) */}
        <p className="text-[11px] font-mono text-slate-400 pt-1">
          Patch features (32×32) bilinearly upsampled. Warm highlights identify receptive fields exerting top mathematical weight on model logit.
        </p>
      </div>
    </section>
  );
};
