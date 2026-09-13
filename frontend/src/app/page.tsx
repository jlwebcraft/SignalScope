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
import { AlertCircle } from "lucide-react";

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
          ? `Unable to connect to the backend at ${API_BASE_URL}. Ensure the service is operational.`
          : message
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#fbfbf9] text-[#2a2d34]">
      <Header apiBaseUrl={API_BASE_URL} />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
        {/* Landing State: Editorial Intake Workstation */}
        {!prediction && (
          <div className="max-w-xl mx-auto space-y-8">
            {/* Editorial Title Block */}
            <div className="space-y-2 border-b border-[#e5e2d9] pb-6">
              <span className="text-[10px] font-mono tracking-[0.25em] text-[#9a3412] uppercase font-semibold">
                Case Intake Station
              </span>
              <h1 className="text-3xl sm:text-4xl font-editorial font-normal tracking-tight text-[#121316]">
                Verify an Image
              </h1>
              <p className="text-xs text-[#606570] leading-relaxed pt-1">
                Upload a digital still image for multimodal forensic verification across ConvNeXt spatial feature
                attribution, 2D Fourier spectral residuals, perturbation stability, and metadata provenance.
              </p>
            </div>

            {/* Ingestion Dropzone & Actions */}
            <div className="space-y-4">
              <Dropzone
                onFileSelected={handleFileSelected}
                isLoading={isLoading}
                selectedPreview={previewUrl}
                onClear={handleClear}
              />

              {selectedFile && !isLoading && (
                <div className="flex items-center justify-end space-x-3 pt-2">
                  <button
                    type="button"
                    onClick={handleClear}
                    className="px-4 py-2 text-xs font-mono uppercase tracking-wider text-[#606570] hover:text-[#121316] border border-[#e5e2d9] hover:bg-[#f3f1ea] transition-colors"
                  >
                    Clear
                  </button>
                  <button
                    type="button"
                    onClick={handleRunAnalysis}
                    className="px-5 py-2 text-xs font-mono uppercase tracking-wider font-semibold bg-[#121316] hover:bg-[#2a2d34] text-[#fbfbf9] transition-colors"
                  >
                    Analyze image
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Loading Execution Progress */}
        {isLoading && (
          <div className="py-12">
            <AnalysisProgress />
          </div>
        )}

        {/* Error Notification */}
        {errorMsg && (
          <div className="max-w-xl mx-auto p-4 border border-[#fecaca] bg-[#fef2f2] text-[#991b1b] text-xs flex items-start gap-3">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-semibold uppercase tracking-wider text-[11px] font-mono">
                Intake / Evaluation Notice
              </span>
              <p className="text-xs leading-relaxed">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Result State: Forensic Investigation Dossier */}
        {prediction && (
          <div className="space-y-10 animate-fadeIn">
            {/* Dossier Control Header */}
            <div className="flex items-baseline justify-between border-b border-[#e5e2d9] pb-3">
              <div className="flex items-baseline space-x-2 text-xs font-mono text-[#606570]">
                <span className="text-[#121316] font-semibold tracking-wider uppercase text-[11px]">
                  Case Dossier
                </span>
                <span>/</span>
                <span>SignalScope Baseline v1</span>
                <span>/</span>
                <span className="text-[#8c8a82]">Schema v{prediction.schema_version}</span>
              </div>
              <button
                type="button"
                onClick={handleClear}
                className="text-xs font-mono uppercase tracking-wider text-[#606570] hover:text-[#121316] underline underline-offset-4 decoration-[#dcd9ce] hover:decoration-[#121316] transition-colors"
              >
                Analyze another image
              </button>
            </div>

            {/* Primary Finding Header */}
            <VerdictCard prediction={prediction} />

            {/* Visual Evidence Comparison (Original + Grad-CAM) */}
            <SpatialHeatmapViewer
              spatial={prediction.evidence.spatial}
              originalPreview={previewUrl}
            />

            {/* Frequency Domain & Spectral Evidence */}
            <SpectralViewer spectral={prediction.evidence.spectral} />

            {/* Perturbation Robustness Table */}
            <RobustnessPanel
              robustness={prediction.evidence.robustness}
              operatingThreshold={0.50}
            />

            {/* Provenance & Metadata Inspection */}
            <MetadataPanel metadata={prediction.evidence.metadata} />

            {/* Formal Forensic Finding & Summary */}
            <ExplanationCard prediction={prediction} />
          </div>
        )}
      </main>

      {/* Editorial Colophon / Footer */}
      <footer className="border-t border-[#e5e2d9] bg-[#fbfbf9] py-8 text-center text-xs text-[#8c8a82]">
        <div className="max-w-5xl mx-auto px-4 space-y-1">
          <p className="font-editorial text-sm text-[#606570] italic">
            SignalScope — Media Forensic Investigation System
          </p>
          <p className="text-[11px] font-mono">
            Evaluations represent probabilistic statistical assessments. They do not constitute conclusive causal proof.
          </p>
        </div>
      </footer>
    </div>
  );
}
