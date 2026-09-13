"use client";

import React, { useState } from "react";
import { Header } from "@/components/Header";
import { Dropzone } from "@/components/Dropzone";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { VerdictCard } from "@/components/VerdictCard";
import { SpatialHeatmapViewer } from "@/components/SpatialHeatmapViewer";
import { SpectralViewer } from "@/components/SpectralViewer";
import { RobustnessPanel } from "@/components/RobustnessPanel";
import { MetadataPanel } from "@/components/MetadataPanel";
import { ExplanationCard } from "@/components/ExplanationCard";
import { PredictionResponse } from "@/types/prediction";
import { RotateCcw, AlertCircle } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
    setPrediction(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleClear = () => {
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setErrorMsg(null);
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMsg(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let errDetail = "Inference request failed.";
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) {
            errDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
          }
        } catch {
          errDetail = `HTTP error ${res.status}: ${res.statusText}`;
        }
        throw new Error(errDetail);
      }

      const data: PredictionResponse = await res.json();
      setPrediction(data);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to connect to SignalScope inference server.";
      setErrorMsg(
        message.includes("Failed to fetch")
          ? `Unable to reach the SignalScope backend at ${API_BASE_URL}. Please ensure the API service is running.`
          : message
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f17] text-slate-100">
      <Header apiBaseUrl={API_BASE_URL} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Landing Page: Functional Forensic Workbench Heading */}
        {!prediction && (
          <div className="max-w-2xl mx-auto text-left space-y-1 pt-2 pb-1">
            <h1 className="text-xl font-bold text-white tracking-tight">
              Image Authenticity Analysis
            </h1>
            <p className="text-xs text-slate-400 leading-relaxed">
              Multimodal forensic evaluation combining ConvNeXt-Tiny spatial feature attribution,
              2D Fourier spectral residuals, and perturbation stability.
            </p>
          </div>
        )}

        {/* Ingestion & Upload Section */}
        {!prediction && (
          <div className="max-w-2xl mx-auto space-y-3">
            <Dropzone
              onFileSelected={handleFileSelected}
              isLoading={isLoading}
              selectedPreview={previewUrl}
              onClear={handleClear}
            />

            {selectedFile && !isLoading && (
              <div className="flex items-center justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-1.5 text-xs font-medium rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors"
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={handleRunAnalysis}
                  className="px-4 py-1.5 text-xs font-semibold rounded bg-slate-100 hover:bg-white text-slate-950 transition-colors flex items-center gap-1.5"
                >
                  Analyze image
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading Progress State */}
        {isLoading && (
          <div className="py-6">
            <AnalysisProgress />
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="max-w-2xl mx-auto p-3.5 rounded border border-rose-900/60 bg-rose-950/20 text-rose-300 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-rose-200">Analysis Error: </span>
              <span>{errorMsg}</span>
            </div>
          </div>
        )}

        {/* Results Presentation (Forensic Report Dossier) */}
        {prediction && (
          <div className="space-y-5 animate-fadeIn">
            {/* Top Toolbar */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <span className="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">
                  Forensic Dossier
                </span>
                <span className="text-slate-600">/</span>
                <span className="font-mono text-slate-400">Schema v{prediction.schema_version}</span>
              </div>
              <button
                type="button"
                onClick={handleClear}
                className="px-3 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-slate-200 flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-3 h-3" /> Analyze another image
              </button>
            </div>

            {/* Verdict Header */}
            <VerdictCard prediction={prediction} />

            {/* Evidence Section 1: Spatial Attribution & Frequency Spectrum */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <SpatialHeatmapViewer
                spatial={prediction.evidence.spatial}
                originalPreview={previewUrl}
              />
              <SpectralViewer spectral={prediction.evidence.spectral} />
            </div>

            {/* Evidence Section 2: Perturbation Robustness & Metadata Provenance */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <RobustnessPanel
                robustness={prediction.evidence.robustness}
                operatingThreshold={0.50}
              />
              <MetadataPanel metadata={prediction.evidence.metadata} />
            </div>

            {/* Evidence Section 3: Explanation Finding & Disclaimer */}
            <ExplanationCard prediction={prediction} />
          </div>
        )}
      </main>

      {/* Forensic Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0d1322] py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4">
          SignalScope Forensic Workbench — Evaluations represent statistical likelihood estimates and do not constitute absolute physical proof.
        </div>
      </footer>
    </div>
  );
}
