"use client";

import React from "react";
import { RobustnessEvidence } from "@/types/prediction";

interface RobustnessPanelProps {
  robustness: RobustnessEvidence;
  operatingThreshold?: number;
}

export const RobustnessPanel: React.FC<RobustnessPanelProps> = ({
  robustness,
  operatingThreshold = 0.50,
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
          text: "Minimal Sensitivity (Invariance Confirmed)",
          className: "text-emerald-800 bg-emerald-50 border-emerald-200",
        };
      case "moderate":
        return {
          text: "Moderate Drift (Boundary Proximity)",
          className: "text-amber-800 bg-amber-50 border-amber-200",
        };
      case "severe":
      default:
        return {
          text: "Severe Instability (Categorical Flips)",
          className: "text-rose-800 bg-rose-50 border-rose-200",
        };
    }
  };

  const impact = getImpactDetails();

  return (
    <section className="bg-white border border-slate-200 rounded overflow-hidden shadow-xs">
      {/* Section Sub-Header */}
      <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2.5">
          <div className="w-2 h-2 rounded-[1px] bg-slate-700" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Perturbation Stability Matrix (Stress-Testing)
          </h3>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <span className="text-slate-500">Threshold: {operatingThreshold.toFixed(2)}</span>
          <span className="text-slate-300">/</span>
          <span className={`px-2 py-0.5 text-[11px] font-medium border rounded ${impact.className}`}>
            {impact.text}
          </span>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Core Stability Diagnostics Bar */}
        <div className="grid grid-cols-3 gap-3 bg-slate-50 border border-slate-200 rounded p-3 text-center font-mono">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Stability Score</div>
            <div className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">{stabilityPct}%</div>
          </div>
          <div className="border-x border-slate-200">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Decision Flip Rate</div>
            <div className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">{flipPct}%</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Mean Probability Drift</div>
            <div className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">±{meanDrift}%</div>
          </div>
        </div>

        {/* Tabular Stress Probe Results */}
        {robustness.transform_results && robustness.transform_results.length > 0 && (
          <div className="border border-slate-200 rounded overflow-x-auto">
            <table className="w-full text-left text-xs font-mono divide-y divide-slate-200">
              <thead className="bg-slate-50 text-[10px] uppercase text-slate-500 font-semibold">
                <tr>
                  <th className="py-2.5 px-3">Degradation Probe</th>
                  <th className="py-2.5 px-3">Likelihood</th>
                  <th className="py-2.5 px-3">Drift (Δp)</th>
                  <th className="py-2.5 px-3 text-right">Adjudication Invariance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {robustness.transform_results.map((res, idx) => {
                  const probPct = (res.predicted_probability * 100).toFixed(1);
                  const deltaPct = (res.delta_from_original * 100).toFixed(1);
                  const isFlipped =
                    idx > 0 &&
                    (res.predicted_probability >= operatingThreshold) !==
                      (robustness.transform_results[0].predicted_probability >= operatingThreshold);
                  const hasHighDrift = res.delta_from_original > 0.15;

                  return (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="py-2.5 px-3 font-sans font-medium text-slate-900">
                        {res.transform_name}
                      </td>
                      <td className="py-2.5 px-3 text-slate-800">{probPct}%</td>
                      <td className="py-2.5 px-3 text-slate-500">
                        {idx === 0 ? "— (Baseline)" : `±${deltaPct}%`}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        {isFlipped ? (
                          <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase bg-rose-50 text-rose-800 border border-rose-200">
                            Decision Flip
                          </span>
                        ) : hasHighDrift ? (
                          <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase bg-amber-50 text-amber-800 border border-amber-200">
                            High Shift
                          </span>
                        ) : (
                          <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase bg-emerald-50 text-emerald-800 border border-emerald-200">
                            Invariant
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

        {/* Methodological Context */}
        <div className="p-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-600 leading-relaxed font-sans">
          <strong className="font-mono text-[10px] text-slate-900 uppercase font-semibold">Robustness Scope: </strong>
          Synthetic generative cues often degrade sharply under standard compression, rescaling, or blurring. Probes displaying high drift values
          or categorical flips trigger responsible uncertainty rather than overconfident classifications.
        </div>
      </div>
    </section>
  );
};
