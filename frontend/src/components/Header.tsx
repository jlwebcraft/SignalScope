"use client";

import React, { useState, useEffect } from "react";
import { Shield, ShieldAlert, Cpu, Activity, Info, X } from "lucide-react";
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
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Shield className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  SignalScope
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  SIH 2026
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
                Multimodal Authenticity & Provenance Intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Backend Status indicator */}
            <div className="flex items-center space-x-2 text-xs px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800">
              <span className={`w-2 h-2 rounded-full ${health ? "bg-emerald-400 animate-pulse" : "bg-rose-500"}`} />
              <span className="text-slate-300 font-medium hidden md:inline">
                {health ? "Inference Ready" : "API Offline"}
              </span>
              {health && (
                <span className="text-[10px] text-slate-500 border-l border-slate-800 pl-2 hidden lg:inline flex items-center gap-1">
                  <Cpu className="w-3 h-3 text-slate-400 inline" /> {health.device.toUpperCase()}
                </span>
              )}
            </div>

            <button
              onClick={() => setShowEthicalModal(true)}
              className="flex items-center space-x-1.5 text-xs text-slate-300 hover:text-white px-3 py-1.5 rounded-lg bg-indigo-600/10 hover:bg-indigo-600/20 border border-indigo-500/20 transition-colors"
            >
              <Info className="w-3.5 h-3.5 text-indigo-400" />
              <span className="hidden sm:inline">Ethical Guardrails</span>
            </button>
          </div>
        </div>
      </header>

      {/* Ethical Guardrails Modal */}
      {showEthicalModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel max-w-lg w-full rounded-2xl p-6 shadow-2xl relative border border-slate-700/60 bg-slate-900/90 text-slate-200">
            <button
              onClick={() => setShowEthicalModal(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center space-x-3 mb-4">
              <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400">
                <ShieldAlert className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">Responsible AI Guardrails</h3>
                <p className="text-xs text-slate-400">SIH 2026 Ethical Design Standard</p>
              </div>
            </div>

            <div className="space-y-3 text-xs leading-relaxed text-slate-300">
              <p>
                <strong className="text-white">Probabilistic Assessments, Not Definite Proof:</strong> All assessments reflect statistical likelihoods derived from trained ConvNeXt feature activations, 2D Fourier spectra, and controlled perturbation tests.
              </p>
              <p>
                <strong className="text-white">No Identity Attribution:</strong> SignalScope does not perform facial recognition, biometric profiling, or human identification.
              </p>
              <p>
                <strong className="text-white">Responsible Uncertainty:</strong> Ambiguous boundary predictions (probability within 0.40–0.60) or samples exhibiting high volatility under compression are explicitly classified as <span className="text-amber-400 font-semibold">Uncertain</span>, recommending human review.
              </p>
              <p>
                <strong className="text-white">Supporting Evidence Only:</strong> Visual heatmaps and spectral profiles show regions and frequencies influencing the classifier; they do not represent universal physical fingerprints.
              </p>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setShowEthicalModal(false)}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
              >
                Understood
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
