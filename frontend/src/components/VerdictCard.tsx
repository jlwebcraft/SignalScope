"use client";

import React from "react";
import { CheckCircle, AlertTriangle, HelpCircle, ShieldAlert, Sparkles, Sliders } from "lucide-react";
import { PredictionResponse } from "@/types/prediction";

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
    is_development_placeholder,
  } = prediction;

  const pct = (probability * 100).toFixed(1);
  const rawPct = raw_probability != null ? (raw_probability * 100).toFixed(1) : pct;
  const calPct = calibrated_probability != null ? (calibrated_probability * 100).toFixed(1) : pct;

  const getVerdictStyle = () => {
    switch (verdict) {
      case "likely_ai_generated":
        return {
          title: "LIKELY AI-GENERATED",
          subtitle: "Statistical anomalies align with generative model architectures",
          badgeBg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          glow: "from-rose-500/10 via-transparent to-transparent",
          icon: AlertTriangle,
          textColor: "text-rose-400",
          barColor: "bg-rose-500",
        };
      case "likely_real":
        return {
          title: "LIKELY REAL (AUTHENTIC)",
          subtitle: "Visual gradients and spectral decay conform to natural camera capture",
          badgeBg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          glow: "from-emerald-500/10 via-transparent to-transparent",
          icon: CheckCircle,
          textColor: "text-emerald-400",
          barColor: "bg-emerald-500",
        };
      case "uncertain":
      default:
        return {
          title: "UNCERTAIN / AMBIGUOUS",
          subtitle: "Signals lie in the ambiguous boundary corridor or exhibit perturbation volatility",
          badgeBg: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          glow: "from-amber-500/10 via-transparent to-transparent",
          icon: HelpCircle,
          textColor: "text-amber-400",
          barColor: "bg-amber-500",
        };
    }
  };

  const style = getVerdictStyle();
  const Icon = style.icon;

  return (
    <div className={`glass-panel rounded-2xl p-6 border border-slate-800 relative overflow-hidden bg-gradient-to-b ${style.glow}`}>
      {is_development_placeholder && (
        <div className="mb-3 px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs flex items-center gap-1.5 font-medium">
          <Sliders className="w-3.5 h-3.5" />
          <span>Development Scaffolding Mode: Active without trained checkpoint</span>
        </div>
      )}

      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        {/* Left: Verdict & Confidence */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2.5">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${style.badgeBg}`}>
              <Icon className="w-4 h-4" />
              {style.title}
            </span>

            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700 capitalize">
              {confidence_level} Confidence
            </span>
          </div>

          <h2 className="text-xl font-bold text-white tracking-tight">
            Authenticity Assessment
          </h2>
          <p className="text-xs text-slate-400 max-w-lg">
            {style.subtitle}
          </p>
        </div>

        {/* Right: Probability Metric */}
        <div className="flex md:flex-col items-baseline md:items-end justify-between border-t md:border-t-0 md:border-l border-slate-800 pt-4 md:pt-0 md:pl-6">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Synthetic Likelihood
          </div>
          <div className="flex items-baseline gap-1 mt-1">
            <span className={`text-4xl font-black tracking-tight ${style.textColor}`}>
              {calPct}%
            </span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1 flex items-center gap-1">
            <Sliders className="w-3 h-3 text-slate-400" />
            <span>Calibrated (T=0.9995) | Raw: {rawPct}%</span>
          </div>
        </div>
      </div>

      {/* Uncertainty Alert Callout */}
      {uncertain && (
        <div className="mt-5 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />
          <div>
            <span className="font-bold">Human Forensic Review Recommended: </span>
            <span>
              The probability score is near the 0.50 decision corridor or the model exhibited sensitivity under degradation testing.
              Do not treat this statistical assessment as conclusive.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
