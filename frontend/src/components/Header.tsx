"use client";

import React, { useState, useEffect } from "react";
import { Info, X, ShieldAlert } from "lucide-react";
import { SystemHealth } from "@/types/prediction";

interface HeaderProps {
  apiBaseUrl: string;
}

export const Header: React.FC<HeaderProps> = ({ apiBaseUrl }) => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [showEthicalModal, setShowEthicalModal] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/v1/health`);
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        } else {
          setHealth(null);
        }
      } catch {
        setHealth(null);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return (
    <>
      <header className="border-b border-slate-800 bg-[#0d1322] sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-12 flex items-center justify-between">
          {/* Product Brand */}
          <div className="flex items-center">
            <div className="flex items-baseline space-x-2">
              <span className="font-bold text-sm tracking-wider text-slate-100 uppercase">
                SignalScope
              </span>
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest hidden md:inline">
                / Forensic Workbench
              </span>
            </div>
            <span className="text-xs text-slate-400 hidden sm:inline ml-3 pl-3 border-l border-slate-800">
              Image Authenticity Forensics
            </span>
          </div>

          {/* System Telemetry & Utilities */}
          <div className="flex items-center space-x-3 text-xs">
            {/* API Status */}
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  health ? "bg-emerald-400" : "bg-rose-500"
                }`}
              />
              <span className="text-slate-300 text-[11px] font-medium">
                {health ? "API Online" : "API Offline"}
              </span>
              {health?.device && (
                <span className="text-[10px] font-mono text-slate-400 border-l border-slate-800 pl-1.5 ml-1">
                  {health.device.toUpperCase()}
                </span>
              )}
            </div>

            {/* Ethical Guardrails Button */}
            <button
              type="button"
              onClick={() => setShowEthicalModal(true)}
              className="flex items-center space-x-1 text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-900 transition-colors"
              title="View responsible forensics & ethical constraints"
            >
              <Info className="w-3.5 h-3.5 text-slate-400" />
              <span className="hidden sm:inline text-[11px]">Guardrails</span>
            </button>
          </div>
        </div>
      </header>

      {/* Ethical Guardrails Modal */}
      {showEthicalModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 animate-fadeIn">
          <div className="max-w-lg w-full rounded-lg p-5 border border-slate-700 bg-slate-900 text-slate-200 shadow-2xl relative">
            <button
              type="button"
              onClick={() => setShowEthicalModal(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center space-x-2.5 mb-3">
              <ShieldAlert className="w-5 h-5 text-amber-400" />
              <h3 className="text-sm font-bold text-white tracking-wide uppercase">
                Responsible Forensic Principles
              </h3>
            </div>

            <div className="space-y-2.5 text-xs leading-relaxed text-slate-300 border-t border-slate-800 pt-3">
              <p>
                <strong className="text-slate-100">Probabilistic Assessment:</strong> Evaluations represent statistical likelihoods derived from deep ConvNeXt feature activations, 2D Fourier spectra, and controlled perturbation tests. They do not constitute causal or legal proof.
              </p>
              <p>
                <strong className="text-slate-100">No Identity Profiling:</strong> SignalScope analyzes image structure and compression artifacts. It does not perform facial recognition, biometric profiling, or human identification.
              </p>
              <p>
                <strong className="text-slate-100">Responsible Uncertainty:</strong> Predictions in the boundary corridor (0.40–0.60 calibrated likelihood) or showing volatility under recompression/rescaling trigger an explicit <span className="text-amber-400 font-medium">Uncertain</span> classification recommending human verification.
              </p>
              <p>
                <strong className="text-slate-100">Supporting Context:</strong> Spatial heatmaps and frequency profiles illustrate features that guided classifier decisions; they do not represent universal physical signatures.
              </p>
            </div>

            <div className="mt-5 pt-3 border-t border-slate-800 flex justify-end">
              <button
                type="button"
                onClick={() => setShowEthicalModal(false)}
                className="px-3.5 py-1.5 text-xs font-semibold rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
