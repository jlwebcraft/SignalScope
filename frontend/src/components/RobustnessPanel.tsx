"use client";

import React from "react";
import { RobustnessEvidence } from "@/types/prediction";

interface RobustnessPanelProps {
  robustness: RobustnessEvidence;
  operatingThreshold: number;
}

export const RobustnessPanel: React.FC<RobustnessPanelProps> = ({
  robustness,
  operatingThreshold,
}) => {
  const stabilityPct = (robustness.stability_score * 100).toFixed(1);
  const flipPct =
    robustness.prediction_flip_rate != null
      ? (robustness.prediction_flip_rate * 100).toFixed(1)
      : "0.0";
  const meanDrift =
    robustness.mean_probability_drift != null
      ? (robustness.mean_probability_drift * 100).toFixed(1)
      : "0.0";

  const getImpactDetails = () => {
    switch (robustness.degradation_impact.toLowerCase()) {
      case "minimal":
        return {
          text: "Minimal Impact",
          className: "text-[#166534] bg-[#f0fdf4] border-[#bbf7d0]",
        };
      case "moderate":
        return {
          text: "Moderate Impact",
          className: "text-[#92400e] bg-[#fffbeb] border-[#fde68a]",
        };
      case "severe":
      default:
        return {
          text: "Severe Sensitivity",
          className: "text-[#991b1b] bg-[#fef2f2] border-[#fecaca]",
        };
    }
  };

  const impact = getImpactDetails();

  return (
    <section className="space-y-4 pt-4 border-t border-[#e5e2d9]">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between border-b border-[#e5e2d9] pb-2 gap-2">
        <div className="flex items-baseline space-x-3">
          <span className="text-[10px] font-mono tracking-[0.2em] text-[#9a3412] uppercase font-semibold">
            03
          </span>
          <h3 className="text-lg font-editorial font-semibold text-[#121316]">
            Perturbation Stability Benchmark
          </h3>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono text-[#606570]">
          <span>Operating Threshold: 0.50</span>
          <span className="text-[#dcd9ce]">/</span>
          <span className={`px-2 py-0.5 text-[11px] font-medium border ${impact.className}`}>
            {impact.text}
          </span>
        </div>
      </div>

      {/* Summary Metrics Bar */}
      <div className="grid grid-cols-3 gap-4 bg-[#f3f1ea] border border-[#e5e2d9] p-3 text-center font-mono">
        <div>
          <div className="text-[10px] text-[#8c8a82] uppercase">Stability Score</div>
          <div className="text-base font-bold text-[#121316] mt-0.5">{stabilityPct}%</div>
        </div>
        <div className="border-x border-[#e5e2d9]">
          <div className="text-[10px] text-[#8c8a82] uppercase">Decision Flips</div>
          <div className="text-base font-bold text-[#121316] mt-0.5">{flipPct}%</div>
        </div>
        <div>
          <div className="text-[10px] text-[#8c8a82] uppercase">Mean Drift</div>
          <div className="text-base font-bold text-[#121316] mt-0.5">±{meanDrift}%</div>
        </div>
      </div>

      {/* Clean Editorial Table with Horizontal Rules */}
      {robustness.transform_results && robustness.transform_results.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-b border-[#e5e2d9]">
            <thead className="text-[10px] uppercase text-[#606570] border-b border-[#121316]">
              <tr>
                <th className="py-2.5 px-2 font-semibold">Transformation</th>
                <th className="py-2.5 px-2 font-semibold">Likelihood</th>
                <th className="py-2.5 px-2 font-semibold">Drift (Δp)</th>
                <th className="py-2.5 px-2 text-right font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#eeece5] text-[#2a2d34]">
              {robustness.transform_results.map((res, idx) => {
                const probPct = (res.predicted_probability * 100).toFixed(1);
                const deltaPct = (res.delta_from_original * 100).toFixed(1);
                const isFlipped =
                  idx > 0 &&
                  (res.predicted_probability >= operatingThreshold) !==
                    (robustness.transform_results[0].predicted_probability >= operatingThreshold);
                const hasHighDrift = res.delta_from_original > 0.15;

                return (
                  <tr key={idx} className="hover:bg-[#f3f1ea]/70 transition-colors">
                    <td className="py-2.5 px-2 font-sans font-medium text-[#121316]">
                      {res.transform_name}
                    </td>
                    <td className="py-2.5 px-2">{probPct}%</td>
                    <td className="py-2.5 px-2 text-[#606570]">
                      {idx === 0 ? "—" : `±${deltaPct}%`}
                    </td>
                    <td className="py-2.5 px-2 text-right">
                      {isFlipped || hasHighDrift ? (
                        <span className="text-[11px] font-semibold text-[#92400e]">
                          {isFlipped ? "Flipped" : "Shift"}
                        </span>
                      ) : (
                        <span className="text-[11px] font-semibold text-[#166534]">
                          Stable
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Stability Context Note */}
      <div className="text-xs text-[#606570] font-sans leading-relaxed pt-1">
        <strong className="text-[#121316] font-mono uppercase text-[10px] tracking-wider">Robustness Scope: </strong>
        Synthetic generative cues often degrade under standard compression, rescaling, or blurring. Probes displaying high drift values
        or categorical flips trigger responsible uncertainty rather than overconfident classifications.
      </div>
    </section>
  );
};
