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
    <section className="space-y-4 pt-2">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between border-b border-[#e5e2d9] pb-2 gap-2">
        <div className="flex items-baseline space-x-3">
          <span className="text-[10px] font-mono tracking-[0.2em] text-[#9a3412] uppercase font-semibold">
            01
          </span>
          <h3 className="text-lg font-editorial font-semibold text-[#121316]">
            Visual Attribution Evidence
          </h3>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono text-[#606570]">
          <span>Layer: {spatial.target_layer || "ConvNeXt-Tiny Stage 3"}</span>
          {concentrationPct && (
            <>
              <span className="text-[#dcd9ce]">/</span>
              <span>Attribution Concentration: <strong className="text-[#121316]">{concentrationPct}%</strong></span>
            </>
          )}
        </div>
      </div>

      {/* Side-by-Side Large Visual Exhibits */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
        {/* Exhibit A: Original */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase tracking-wider text-[#606570] flex justify-between">
            <span>Exhibit A · Original Image</span>
            <span className="text-[#8c8a82]">Source Input</span>
          </div>
          <div className="w-full h-72 sm:h-80 bg-[#ffffff] border border-[#e5e2d9] p-2 flex items-center justify-center overflow-hidden">
            {originalPreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalPreview}
                alt="Original evidence"
                className="w-full h-full object-contain pixelated"
              />
            ) : (
              <span className="text-xs font-mono text-[#8c8a82]">Original image unavailable</span>
            )}
          </div>
          <p className="text-[11px] text-[#8c8a82] font-mono">
            Full spatial input evaluated at native resolution.
          </p>
        </div>

        {/* Exhibit B: Grad-CAM Overlay */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase tracking-wider text-[#606570] flex justify-between">
            <span>Exhibit B · Grad-CAM Attribution</span>
            <span className="text-[#8c8a82]">Activation Map</span>
          </div>
          <div className="relative w-full h-72 sm:h-80 bg-[#ffffff] border border-[#e5e2d9] p-2 flex items-center justify-center overflow-hidden">
            {originalPreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalPreview}
                alt="Base image for attribution"
                className="w-full h-full object-contain pixelated"
              />
            ) : (
              <span className="text-xs font-mono text-[#8c8a82]">Base image unavailable</span>
            )}

            {spatial.heatmap ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={spatial.heatmap}
                alt="Grad-CAM Attribution Overlay"
                style={{ opacity: heatmapOpacity / 100 }}
                className="absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)] object-contain pointer-events-none transition-opacity duration-75"
              />
            ) : (
              <div className="absolute inset-0 bg-[#f3f1ea]/90 flex items-center justify-center p-4">
                <span className="text-xs font-mono text-[#8c8a82]">Heatmap computation unavailable</span>
              </div>
            )}
          </div>

          {/* Interactive Blend Control */}
          {spatial.heatmap && (
            <div className="space-y-1 pt-1">
              <div className="flex items-center justify-between text-[11px] font-mono text-[#606570]">
                <span>Heatmap Overlay Blend</span>
                <span className="font-semibold text-[#121316]">{heatmapOpacity}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                className="w-full h-1 bg-[#dcd9ce] rounded accent-[#121316] cursor-pointer appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8c8a82]">
                <span>0% (Original)</span>
                <span>50%</span>
                <span>100% (Heatmap Only)</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Localization Scope & Faithfulness Note */}
      <div className="text-xs text-[#606570] leading-relaxed border-t border-[#eeece5] pt-3">
        <strong className="text-[#121316] font-mono uppercase text-[10px] tracking-wider">Faithfulness Note: </strong>
        Highlighted warm regions indicate receptive fields that exerted the greatest numerical influence on the classifier logit.
        Native ConvNeXt patch features are 32×32 and bilinearly upsampled for visual inspection. Attribution identifies model focus,
        not standalone causal proof of artificial generation.
      </div>
    </section>
  );
};
