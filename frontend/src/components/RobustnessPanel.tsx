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

  const getImpactBadge = () => {
    switch (robustness.degradation_impact.toLowerCase()) {
      case "minimal":
        return {
          text: "Minimal Impact",
          className: "text-emerald-400 border-emerald-800 bg-emerald-950/40",
        };
      case "moderate":
        return {
          text: "Moderate Impact",
          className: "text-amber-400 border-amber-800 bg-amber-950/40",
        };
      case "severe":
      default:
        return {
          text: "Severe Sensitivity",
          className: "text-rose-400 border-rose-800 bg-rose-950/40",
        };
    }
  };

  const impact = getImpactBadge();

  return (
    <div className="forensic-panel flex flex-col justify-between p-4 space-y-3.5">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div>
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            3. Authenticity Stability (Perturbation Probing)
          </h3>
          <p className="text-[11px] font-mono text-slate-500 mt-0.5">
            4 Controlled Transformations · Decision Threshold: 0.50
          </p>
        </div>

        <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium border ${impact.className}`}>
          {impact.text}
        </span>
      </div>

      {/* Summary Metrics Bar */}
      <div className="grid grid-cols-3 gap-2 p-2 rounded bg-slate-900/60 border border-slate-800 text-center font-mono">
        <div>
          <div className="text-[10px] text-slate-500 uppercase">Stability Score</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">{stabilityPct}%</div>
        </div>
        <div className="border-x border-slate-800">
          <div className="text-[10px] text-slate-500 uppercase">Decision Flips</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">{flipPct}%</div>
        </div>
        <div>
          <div className="text-[10px] text-slate-500 uppercase">Mean Drift</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">±{meanDrift}%</div>
        </div>
      </div>

      {/* Forensic Transformation Table */}
      {robustness.transform_results && robustness.transform_results.length > 0 && (
        <div className="overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 text-[10px] uppercase border-b border-slate-800">
              <tr>
                <th className="py-2 px-3 font-semibold">Transformation</th>
                <th className="py-2 px-3 font-semibold">Likelihood</th>
                <th className="py-2 px-3 font-semibold">Drift (Δp)</th>
                <th className="py-2 px-3 text-right font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 text-[11px]">
              {robustness.transform_results.map((res, idx) => {
                const probPct = (res.predicted_probability * 100).toFixed(1);
                const deltaPct = (res.delta_from_original * 100).toFixed(1);
                const isFlipped =
                  idx > 0 &&
                  (res.predicted_probability >= operatingThreshold) !==
                    (robustness.transform_results[0].predicted_probability >= operatingThreshold);
                const hasHighDrift = res.delta_from_original > 0.15;

                return (
                  <tr key={idx} className="hover:bg-slate-900/40">
                    <td className="py-1.5 px-3 font-sans text-slate-200 font-medium">
                      {res.transform_name}
                    </td>
                    <td className="py-1.5 px-3 text-slate-200">{probPct}%</td>
                    <td className="py-1.5 px-3 text-slate-400">
                      {idx === 0 ? "—" : `±${deltaPct}%`}
                    </td>
                    <td className="py-1.5 px-3 text-right">
                      {isFlipped || hasHighDrift ? (
                        <span className="text-[11px] font-semibold text-amber-400">
                          {isFlipped ? "Flipped" : "Shift"}
                        </span>
                      ) : (
                        <span className="text-[11px] font-semibold text-emerald-400">
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
      <div className="p-2.5 rounded bg-slate-900/50 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed font-sans">
        <span className="font-semibold text-slate-300">Robustness Scope: </span>
        Synthetic generative cues often exhibit sensitivity to downscaling or recompression. High drift values trigger uncertainty rather than overconfident classifications.
      </div>
    </div>
  );
};
