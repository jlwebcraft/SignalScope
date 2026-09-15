"use client";

import React, { useState, useEffect } from "react";
import { SystemHealth } from "@/types/prediction";
import { MethodologyModal } from "@/components/MethodologyModal";
import { Shield, ExternalLink } from "lucide-react";

interface HeaderProps {
  apiBaseUrl: string;
}

export const Header: React.FC<HeaderProps> = ({ apiBaseUrl }) => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("signalscope-v2");
  const [showMethodology, setShowMethodology] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchTelemetry = async () => {
      try {
        const [healthRes, readyRes] = await Promise.allSettled([
          fetch(`${apiBaseUrl}/api/v1/health`),
          fetch(`${apiBaseUrl}/ready`),
        ]);

        if (!isMounted) return;

        if (healthRes.status === "fulfilled" && healthRes.value.ok) {
          const healthData = await healthRes.value.json();
          setHealth(healthData);
        } else {
          setHealth(null);
        }

        if (readyRes.status === "fulfilled" && readyRes.value.ok) {
          const readyData = await readyRes.value.json();
          if (readyData.model_version) {
            setModelVersion(readyData.model_version);
          }
        }
      } catch {
        if (isMounted) setHealth(null);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 20000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiBaseUrl]);

  return (
    <>
      <header className="border-b border-slate-200 bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-13 flex items-center justify-between">
          {/* Workstation Identity */}
          <div className="flex items-center space-x-2.5 sm:space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-2.5 h-2.5 bg-slate-900 rounded-[1px]" />
              <span className="font-bold text-sm tracking-tight text-slate-900">
                SignalScope
              </span>
            </div>
            <span className="text-slate-300 hidden sm:inline">/</span>
            <span className="text-xs text-slate-500 font-medium hidden sm:inline">
              Image Forensics Workstation
            </span>
          </div>

          {/* Precision Telemetry & Navigation */}
          <div className="flex items-center space-x-2.5 sm:space-x-4 text-xs">
            {/* API Status Indicator */}
            <div className="flex items-center space-x-1.5 font-mono text-[11px] text-slate-600">
              <span
                className={`w-2 h-2 rounded-full ${
                  health ? "bg-emerald-600" : "bg-rose-600"
                }`}
                aria-hidden="true"
              />
              <span className="font-medium text-slate-700">
                {health ? "Live" : "Offline"}
              </span>
            </div>

            <span className="text-slate-200 hidden xs:inline" aria-hidden="true">|</span>

            {/* Model Version Tag */}
            <div className="hidden md:flex items-center space-x-1 font-mono text-[11px] text-slate-600">
              <span className="text-slate-400">Engine:</span>
              <span className="bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded text-slate-800 font-semibold">
                {modelVersion}
              </span>
            </div>

            {/* Device Compute Tag */}
            <span className="font-mono text-[11px] text-slate-500 hidden lg:inline">
              [{health?.device ? health.device.toUpperCase() : "CPU"}]
            </span>

            <span className="text-slate-200" aria-hidden="true">|</span>

            {/* Methodology & Guardrails Modal Trigger */}
            <button
              type="button"
              onClick={() => setShowMethodology(true)}
              className="flex items-center space-x-1 text-slate-700 hover:text-slate-900 font-medium hover:bg-slate-100 px-2 py-1 rounded transition-colors focus-forensic"
            >
              <Shield className="w-3.5 h-3.5 text-slate-500" />
              <span>Guardrails</span>
            </button>

            {/* Code Repository Link */}
            <a
              href="https://github.com/jlwebcraft/SignalScope"
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1 text-slate-500 hover:text-slate-900 p-1 rounded transition-colors focus-forensic"
              title="GitHub Repository"
            >
              <span className="sr-only">GitHub Repository</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </header>

      <MethodologyModal
        isOpen={showMethodology}
        onClose={() => setShowMethodology(false)}
        modelVersion={modelVersion}
        device={health?.device ? health.device.toUpperCase() : "CPU"}
      />
    </>
  );
};
