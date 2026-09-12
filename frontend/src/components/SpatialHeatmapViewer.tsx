"use client";

import React, { useState } from "react";
import { Eye, Layers, Info, Sliders } from "lucide-react";
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
  const [showOnlyHeatmap, setShowOnlyHeatmap] = useState(false);

  const concentrationPct =
    spatial.attribution_concentration != null
      ? (spatial.attribution_concentration * 100).toFixed(1)
      : null;

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Spatial Attribution (Grad-CAM)</h3>
            <p className="text-[11px] text-slate-400">
              Layer: {spatial.target_layer || "ConvNeXt-Tiny Stage 3"}
            </p>
          </div>
        </div>

        {concentrationPct && (
          <div className="text-right">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">Attribution Concentration</span>
            <div className="text-xs font-bold text-indigo-400">{concentrationPct}%</div>
          </div>
        )}
      </div>

      {/* Visual Image Overlay Container */}
      <div className="flex flex-col items-center">
        <div className="relative w-64 h-64 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center shadow-lg">
          {/* Base Original Image */}
          {originalPreview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={originalPreview}
              alt="Original preview"
              className="absolute inset-0 w-full h-full object-contain pixelated"
            />
          ) : (
            <div className="text-xs text-slate-500">Image unavailable</div>
          )}

          {/* Grad-CAM Heatmap Overlay */}
          {spatial.heatmap && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={spatial.heatmap}
              alt="Grad-CAM Overlay"
              style={{
                opacity: showOnlyHeatmap ? 1 : heatmapOpacity / 100,
              }}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
            />
          )}

          {!spatial.heatmap && (
            <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center p-4 text-center">
              <span className="text-xs text-slate-400">Grad-CAM overlay available with trained weights</span>
            </div>
          )}
        </div>

        {/* Heatmap Controls */}
        {spatial.heatmap && (
          <div className="w-full max-w-xs mt-4 space-y-2">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                <Sliders className="w-3 h-3" /> Heatmap Blend
              </span>
              <span className="font-mono text-slate-300">{heatmapOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Original (0%)</span>
              <span>Overlay (50%)</span>
              <span>Heatmap (100%)</span>
            </div>
          </div>
        )}
      </div>

      {/* Saliency Interpretation Note */}
      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Faithfulness Note: </span>
          Native 32×32 input features are upsampled strictly for visualization.
          Highlighted regions show areas that most strongly guided the classifier; they do not constitute standalone proof of synthetic origin.
        </div>
      </div>
    </div>
  );
};
