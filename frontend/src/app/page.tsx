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
import { Play, RotateCcw, AlertCircle, Sparkles, Shield } from "lucide-react";

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
    <div className="min-h-screen flex flex-col bg-[#090d16] text-slate-100 selection:bg-indigo-500/30">
      <Header apiBaseUrl={API_BASE_URL} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Title Hero */}
        {!prediction && (
          <div className="text-center max-w-2xl mx-auto space-y-3 pt-4">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              <span>SIH 2026 Production Authenticity Intelligence</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              Telling Real From Synthetic Media
            </h1>
            <p className="text-sm text-slate-400 leading-relaxed">
              Dual-branch evidence fusion combining spatial ConvNeXt-Tiny feature attribution,
              2D Fourier spectral harmonics, post-hoc probability calibration, and transformation stability.
            </p>
          </div>
        )}

        {/* Ingestion & Upload Section */}
        {!prediction && (
          <div className="max-w-2xl mx-auto space-y-4">
            <Dropzone
              onFileSelected={handleFileSelected}
              isLoading={isLoading}
              selectedPreview={previewUrl}
              onClear={handleClear}
            />

            {selectedFile && !isLoading && (
              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-4 py-2 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={handleRunAnalysis}
                  className="px-5 py-2.5 text-xs font-bold rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-500 hover:opacity-95 text-white shadow-lg shadow-indigo-500/25 flex items-center gap-2 transition-all active:scale-[0.98]"
                >
                  <Play className="w-4 h-4 fill-white" />
                  Evaluate Authenticity
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading Progress State */}
        {isLoading && (
          <div className="py-8">
            <AnalysisProgress />
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="max-w-2xl mx-auto p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Evaluation Notice: </span>
              <span>{errorMsg}</span>
            </div>
          </div>
        )}

        {/* Results Presentation */}
        {prediction && (
          <div className="space-y-6 animate-fadeIn">
            {/* Action Bar */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <Shield className="w-4 h-4 text-indigo-400" />
                <span>SignalScope Analysis Dossier</span>
                <span className="text-slate-600">•</span>
                <span className="font-mono text-slate-300">Schema v{prediction.schema_version}</span>
              </div>
              <button
                type="button"
                onClick={handleClear}
                className="px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-1.5 transition-colors shadow-sm"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Analyze Another Image
              </button>
            </div>

            {/* Top Verdict Card */}
            <VerdictCard prediction={prediction} />

            {/* Evidence Grid: Spatial & Spectral */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SpatialHeatmapViewer
                spatial={prediction.evidence.spatial}
                originalPreview={previewUrl}
              />
              <SpectralViewer spectral={prediction.evidence.spectral} />
            </div>

            {/* Evidence Grid: Robustness & Metadata */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <RobustnessPanel
                robustness={prediction.evidence.robustness}
                operatingThreshold={0.50}
              />
              <MetadataPanel metadata={prediction.evidence.metadata} />
            </div>

            {/* Explanation and Disclaimer */}
            <ExplanationCard prediction={prediction} />
          </div>
        )}
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-slate-900/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4">
          SignalScope — Smart India Hackathon (SIH) 2026 Internal Hackathon.
          Evaluations are probabilistic likelihood assessments based on statistical evidence, not causal proof.
        </div>
      </footer>
    </div>
  );
}
