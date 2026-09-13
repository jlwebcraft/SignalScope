"use client";

import React, { useState } from "react";
import { SpatialEvidence } from "@/types/prediction";

interface SpatialHeatmapViewerProps {
  spatial: SpatialEvidence;
  originalPreview: string | null;
}

export const SpatialHeatmapViewer: React.FC<SpatialHeatmapViewerProps> = ({
  spatial,
  originalPreview,
}) => {
  const [heatmapOpacity, setHeatmapOpacity] = useState(70);

  const concentrationPct =
    spatial.attribution_concentration != null
      ? (spatial.attribution_concentration * 100).toFixed(1)
      : null;

  return (
    <div className="forensic-panel flex flex-col justify-between p-4 space-y-3.5">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div>
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            1. Spatial Attribution (Grad-CAM)
          </h3>
          <p className="text-[11px] font-mono text-slate-500 mt-0.5">
            Target Layer: {spatial.target_layer || "ConvNeXt-Tiny Stage 3"}
          </p>
        </div>

        {concentrationPct && (
          <div className="text-right">
            <span className="text-[10px] font-mono text-slate-500 uppercase">
              Attribution Concentration
            </span>
            <div className="text-xs font-mono font-semibold text-slate-300">
              {concentrationPct}%
            </div>
          </div>
        )}
      </div>

      {/* Primary Image Viewport */}
      <div className="flex flex-col items-center">
        <div className="relative w-56 h-56 rounded border border-slate-700/80 bg-slate-950 flex items-center justify-center overflow-hidden">
          {/* Base Original Image */}
          {originalPreview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={originalPreview}
              alt="Original preview"
              className="absolute inset-0 w-full h-full object-contain pixelated"
            />
          ) : (
            <div className="text-xs text-slate-600 font-mono">Image unavailable</div>
          )}

          {/* Grad-CAM Heatmap Layer */}
          {spatial.heatmap && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={spatial.heatmap}
              alt="Grad-CAM Overlay"
              style={{
                opacity: heatmapOpacity / 100,
              }}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-75"
            />
          )}

          {!spatial.heatmap && (
            <div className="absolute inset-0 bg-slate-900/90 flex items-center justify-center p-4 text-center">
              <span className="text-xs text-slate-500 font-mono">Heatmap unavailable</span>
            </div>
          )}
        </div>

        {/* Heatmap Blend Slider */}
        {spatial.heatmap && (
          <div className="w-full max-w-xs mt-3 space-y-1">
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Heatmap Overlay Blend</span>
              <span className="text-slate-200">{heatmapOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-full h-1 bg-slate-800 rounded accent-slate-300 cursor-pointer appearance-none"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-600">
              <span>0% (Original)</span>
              <span>50%</span>
              <span>100% (Heatmap)</span>
            </div>
          </div>
        )}
      </div>

      {/* Scientific Faithfulness Note */}
      <div className="p-2.5 rounded bg-slate-900/50 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
        <span className="font-semibold text-slate-300">Localization Scope: </span>
        Highlighted regions indicate spatial receptive fields that most heavily influenced model logit output.
        Native features are 32×32 and bilinearly upsampled for visual inspection.
      </div>
    </div>
  );
};
