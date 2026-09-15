"use client";

import React from "react";
import { PredictionResponse } from "@/types/prediction";
import { CheckCircle2, AlertTriangle, XCircle, AlertCircle } from "lucide-react";

interface VerdictCardProps {
  prediction: PredictionResponse;
}

export const VerdictCard: React.FC<VerdictCardProps> = ({ prediction }) => {
  const {
    verdict,
    probability,
    raw_probability,
    calibrated_probability,
    confidence_level,
    uncertain,
    stability_score,
    is_development_placeholder,
  } = prediction;

  const pct = (probability * 100).toFixed(1);
  const rawPct = raw_probability != null ? (raw_probability * 100).toFixed(1) : pct;
  const calPct = calibrated_probability != null ? (calibrated_probability * 100).toFixed(1) : pct;
  const stabilityPct = (stability_score * 100).toFixed(1);

  // Dynamic explanation for uncertain adjudication
  const getUncertainExplanation = () => {
    if (stability_score < 0.70) {
      if ((calibrated_probability ?? probability) >= 0.60) {
        return `The baseline classifier produced a strong synthetic signal (${calPct}%), but the prediction changed materially under image perturbations (stability score: ${stabilityPct}%). SignalScope therefore withholds a binary authenticity assessment and recommends human review.`;
      } else if ((calibrated_probability ?? probability) <= 0.40) {
        return `The baseline classifier produced a natural capture signal (${calPct}%), but the prediction demonstrated volatility under image perturbations (stability score: ${stabilityPct}%). SignalScope therefore withholds a binary authenticity assessment and recommends human review.`;
      } else {
        return `The calibrated probability (${calPct}%) falls near the 0.50 decision corridor and exhibited sensitivity under perturbation stress tests (stability score: ${stabilityPct}%). SignalScope withholds automated adjudication and recommends human review.`;
      }
    }
    return `The calibrated likelihood (${calPct}%) falls within the 0.40–0.60 decision boundary corridor where classifier evidence is statistically ambiguous. SignalScope withholds automated adjudication and recommends human review.`;
  };

  const getVerdictPresentation = () => {
    switch (verdict) {
      case "likely_ai_generated":
        return {
          title: "Likely AI-Generated",
          badgeLabel: "Synthetic Likelihood High",
          textColor: "text-rose-900",
          bgColor: "bg-rose-50/70",
          borderColor: "border-rose-200",
          iconColor: "text-rose-700",
          icon: XCircle,
          summary:
            "Spatial patch features, edge gradients, and frequency distributions exhibit statistical anomalies characteristic of generative synthesis models.",
        };
      case "likely_real":
        return {
          title: "Likely Real (Authentic)",
          badgeLabel: "Natural Sensor Likelihood High",
          textColor: "text-emerald-900",
          bgColor: "bg-emerald-50/70",
          borderColor: "border-emerald-200",
          iconColor: "text-emerald-700",
          icon: CheckCircle2,
          summary:
            "Spatial feature gradients, natural edge transitions, and 2D Fourier power decay conform to physical optical lens capture.",
        };
      case "uncertain":
      default:
        return {
          title: "Uncertain / Indeterminate",
          badgeLabel: stability_score < 0.70 ? "Adjudication Overridden (Instability)" : "Boundary Ambiguity (Review Required)",
          textColor: "text-amber-900",
          bgColor: "bg-amber-50/70",
          borderColor: "border-amber-200",
          iconColor: "text-amber-700",
          icon: AlertTriangle,
          summary: getUncertainExplanation(),
        };
    }
  };

  const vStyle = getVerdictPresentation();
  const Icon = vStyle.icon;

  return (
    <div className="space-y-4">
      {/* Development Scaffolding Guardrail */}
      {is_development_placeholder && (
        <div className="p-3 border border-amber-300 bg-amber-50 text-amber-900 text-xs font-mono rounded flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-700" />
          <span>Notice: Development Scaffolding Mode active without production weights.</span>
        </div>
      )}

      {/* Primary Adjudication Sheet */}
      <div className={`p-6 border rounded ${vStyle.borderColor} ${vStyle.bgColor} shadow-xs`}>
        <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6">
          {/* Left Column: Verdict finding */}
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center space-x-2">
              <span className="text-[11px] font-mono uppercase font-bold tracking-wider text-slate-600">
                Final Adjudication
              </span>
              <span className="text-slate-300">/</span>
              <span className={`text-[11px] font-mono font-semibold uppercase px-2 py-0.5 rounded bg-white/80 border ${vStyle.borderColor} ${vStyle.textColor}`}>
                {vStyle.badgeLabel}
              </span>
            </div>

            <div className="flex items-center space-x-3">
              <Icon className={`w-8 h-8 ${vStyle.iconColor} shrink-0`} aria-hidden="true" />
              <h2 className={`text-2xl sm:text-3xl font-bold tracking-tight ${vStyle.textColor}`}>
                {vStyle.title}
              </h2>
            </div>

            <p className="text-xs text-slate-700 leading-relaxed pt-1">
              {vStyle.summary}
            </p>
          </div>

          {/* Right Column: Quantitative Evidence Gauges */}
          <div className="bg-white border border-slate-200 p-4 rounded min-w-[270px] space-y-2 shrink-0 shadow-xs">
            {uncertain ? (
              /* Uncertain Outcome: Headline clearly states Inconclusive */
              <>
                <div className="flex items-baseline justify-between border-b border-slate-100 pb-2">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-amber-700 font-semibold block">
                      Adjudication Status
                    </span>
                    <span className="text-xl font-bold font-mono text-amber-800">
                      Inconclusive
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded uppercase font-semibold">
                    Override Active
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Baseline Probe</span>
                    <span className="font-semibold text-slate-700">
                      {calPct}% (Raw: {rawPct}%)
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Stability Score</span>
                    <span className="font-semibold text-rose-700">
                      {stabilityPct}% (Volatile)
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Confidence Grade</span>
                    <span className="font-semibold text-amber-700 capitalize">
                      Low (Overridden)
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Calibration T</span>
                    <span className="text-slate-600">
                      T = 0.9986
                    </span>
                  </div>
                </div>
              </>
            ) : (
              /* Decisive Real or Synthetic Outcome */
              <>
                <div className="flex items-baseline justify-between border-b border-slate-100 pb-2">
                  <span className="text-[10px] font-mono uppercase text-slate-500 font-semibold">
                    Calibrated Likelihood
                  </span>
                  <span className={`text-2xl font-bold font-mono ${vStyle.textColor}`}>
                    {calPct}%
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Confidence</span>
                    <span className="font-semibold text-slate-800 capitalize">
                      {confidence_level}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Stability</span>
                    <span className="font-semibold text-slate-800">
                      {stabilityPct}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Raw Logit Prob</span>
                    <span className="text-slate-600">
                      {rawPct}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Calibration T</span>
                    <span className="text-slate-600">
                      T = 0.9986
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Prominent Human Review Requirement for Uncertain Outcomes */}
      {uncertain && (
        <div className="p-4 border-l-4 border-amber-600 bg-amber-50 border-y border-r border-amber-200 rounded-r text-xs text-amber-950 space-y-1 shadow-xs">
          <div className="font-mono text-[11px] font-bold uppercase tracking-wider flex items-center gap-1.5 text-amber-900">
            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
            <span>Adjudication Condition: Human Forensic Review Required</span>
          </div>
          <p className="text-xs text-amber-900/90 leading-relaxed font-sans pl-5.5">
            The automated pipeline declined to issue a definitive binary verdict. While the unperturbed classifier probe yielded{" "}
            <strong>{calPct}%</strong>, perturbation stress testing demonstrated categorical instability (stability score:{" "}
            <strong>{stabilityPct}%</strong>). This indicates sensitivity to standard transformations such as rescaling or JPEG re-compression.
            Human verification across the spatial heatmap and spectral harmonics below is required.
          </p>
        </div>
      )}
    </div>
  );
};
