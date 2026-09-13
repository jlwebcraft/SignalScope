"use client";

import React, { useState, useEffect } from "react";
import { X } from "lucide-react";
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
      <header className="border-b border-[#e5e2d9] bg-[#fbfbf9] sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          {/* Masthead Identity */}
          <div className="flex items-center">
            <span className="font-semibold text-xs tracking-[0.2em] text-[#121316] uppercase">
              SignalScope
            </span>
            <span className="text-xs text-[#606570] font-editorial italic ml-3 pl-3 border-l border-[#e5e2d9]">
              Image Forensics
            </span>
          </div>

          {/* Minimal Text Telemetry & Utilities */}
          <div className="flex items-center space-x-3 text-xs text-[#606570]">
            {/* API Status */}
            <div className="flex items-center space-x-1.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  health ? "bg-[#166534]" : "bg-[#991b1b]"
                }`}
              />
              <span className="font-medium text-[#2a2d34]">
                {health ? "API Online" : "API Offline"}
              </span>
            </div>

            <span className="text-[#dcd9ce]">/</span>

            {/* Compute Indicator */}
            <span className="font-mono text-[11px] text-[#606570]">
              {health?.device ? health.device.toUpperCase() : "CPU"}
            </span>

            <span className="text-[#dcd9ce]">/</span>

            {/* Ethical Guardrails Button */}
            <button
              type="button"
              onClick={() => setShowEthicalModal(true)}
              className="text-xs uppercase tracking-wider text-[#606570] hover:text-[#121316] underline underline-offset-4 decoration-[#dcd9ce] hover:decoration-[#121316] transition-colors"
            >
              Guardrails
            </button>
          </div>
        </div>
      </header>

      {/* Ethical Guardrails Modal */}
      {showEthicalModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#121316]/50 backdrop-blur-[2px] animate-fadeIn">
          <div className="max-w-lg w-full bg-[#fbfbf9] border border-[#e5e2d9] p-6 shadow-2xl relative text-[#2a2d34]">
            <button
              type="button"
              onClick={() => setShowEthicalModal(false)}
              className="absolute top-4 right-4 text-[#606570] hover:text-[#121316] p-1 transition-colors"
              aria-label="Close modal"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="mb-4">
              <span className="text-[10px] font-mono tracking-widest text-[#9a3412] uppercase font-semibold">
                Ethical Standard · SIH 2026
              </span>
              <h3 className="text-xl font-editorial font-semibold text-[#121316] mt-0.5">
                Responsible Forensic Principles
              </h3>
            </div>

            <div className="space-y-3 text-xs leading-relaxed text-[#2a2d34] border-t border-[#e5e2d9] pt-4">
              <p>
                <strong className="text-[#121316]">01. Probabilistic Assessment:</strong> Evaluations represent statistical likelihoods derived from trained ConvNeXt feature activations, 2D Fourier spectra, and controlled perturbation tests. They do not constitute causal or legal proof.
              </p>
              <p>
                <strong className="text-[#121316]">02. No Biometric Identification:</strong> SignalScope analyzes image structure, frequency residuals, and compression artifacts. It does not perform facial recognition, biometric profiling, or human identification.
              </p>
              <p>
                <strong className="text-[#121316]">03. Responsible Uncertainty:</strong> Predictions in the boundary corridor (0.40–0.60 calibrated likelihood) or displaying volatility under recompression/rescaling trigger an explicit <span className="font-semibold text-[#92400e]">Uncertain</span> classification recommending human verification.
              </p>
              <p>
                <strong className="text-[#121316]">04. Supporting Context:</strong> Spatial heatmaps and frequency profiles illustrate features that guided classifier decisions; they do not represent universal physical signatures.
              </p>
            </div>

            <div className="mt-6 pt-4 border-t border-[#e5e2d9] flex justify-end">
              <button
                type="button"
                onClick={() => setShowEthicalModal(false)}
                className="px-4 py-1.5 text-xs font-medium uppercase tracking-wider bg-[#121316] hover:bg-[#2a2d34] text-[#fbfbf9] transition-colors"
              >
                Acknowledge
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
