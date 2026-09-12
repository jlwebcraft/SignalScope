"use client";

import React from "react";
import { ShieldCheck, AlertTriangle, CheckCircle2, TrendingDown, Info } from "lucide-react";
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
      ? (robustness.mean_probability_drift * 100).toFixed(2)
      : "0.00";

  const getImpactBadge = () => {
    switch (robustness.degradation_impact.toLowerCase()) {
      case "minimal":
        return {
          text: "Minimal Impact",
          className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
        };
      case "moderate":
        return {
          text: "Moderate Impact",
          className: "bg-amber-500/10 text-amber-400 border-amber-500/30",
        };
      case "severe":
      default:
        return {
          text: "Severe Sensitivity",
          className: "bg-rose-500/10 text-rose-400 border-rose-500/30",
        };
    }
  };

  const impact = getImpactBadge();

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Authenticity Stability Benchmark</h3>
            <p className="text-[11px] text-slate-400">Perturbation Invariance under Controlled Degradations</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${impact.className}`}>
            {impact.text}
          </span>
          <div className="text-right">
            <span className="text-sm font-black text-white">{stabilityPct}%</span>
          </div>
        </div>
      </div>

      {/* Stability Metrics Bar */}
      <div className="grid grid-cols-3 gap-3 p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-center">
        <div>
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Stability Score</div>
          <div className="text-sm font-bold text-indigo-400 mt-0.5">{stabilityPct}%</div>
        </div>
        <div className="border-x border-slate-800">
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Decision Flips</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">{flipPct}%</div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Mean Prob Drift</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">{meanDrift}%</div>
        </div>
      </div>

      {/* Per-Transformation Probe Table */}
      {robustness.transform_results && robustness.transform_results.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 text-[10px] uppercase font-semibold border-b border-slate-800">
              <tr>
                <th className="py-2 px-3">Probe Transformation</th>
                <th className="py-2 px-3">Predicted Likelihood</th>
                <th className="py-2 px-3">Drift (Δp)</th>
                <th className="py-2 px-3 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {robustness.transform_results.map((res, idx) => {
                const probPct = (res.predicted_probability * 100).toFixed(1);
                const deltaPct = (res.delta_from_original * 100).toFixed(1);
                const isFlipped =
                  idx > 0 &&
                  (res.predicted_probability >= operatingThreshold) !==
                    (robustness.transform_results[0].predicted_probability >= operatingThreshold);
                const hasHighDrift = res.delta_from_original > 0.15;

                return (
                  <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-2 px-3 font-medium text-white flex items-center gap-1.5">
                      {res.transform_name}
                    </td>
                    <td className="py-2 px-3 font-mono">{probPct}%</td>
                    <td className="py-2 px-3 font-mono text-slate-400">
                      {idx === 0 ? "—" : `±${deltaPct}%`}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {isFlipped || hasHighDrift ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-400">
                          <AlertTriangle className="w-3 h-3" /> Shift
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400">
                          <CheckCircle2 className="w-3 h-3" /> Stable
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

      {/* Stability Note */}
      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Robustness Note: </span>
          Synthetic visual cues frequently degrade under resolution scaling or recompression.
          Evaluations showing high drift or decision flips trigger responsible uncertainty rather than overconfident classifications.
        </div>
      </div>
    </div>
  );
};
