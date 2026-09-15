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
import { AlertCircle, ArrowRight, RotateCcw, Cpu, Activity, ShieldCheck, FileSearch } from "lucide-react";

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
    <div className="min-h-screen flex flex-col bg-[#f8f9fa] text-[#334155]">
      <Header apiBaseUrl={API_BASE_URL} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* State A: Initial Ingestion & Examination Workspace */}
        {!prediction && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column (7 cols): Ingestion Console */}
            <div className="lg:col-span-7 space-y-6">
              <div className="space-y-1 border-b border-slate-200 pb-4">
                <div className="flex items-center space-x-2">
                  <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-500">
                    Forensic Intake Console
                  </span>
                </div>
                <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
                  Image Authenticity Examination
                </h1>
                <p className="text-xs text-slate-600 leading-relaxed pt-1">
                  Upload an image to execute multimodal forensic verification across spatial ConvNeXt feature attribution,
                  2D Fourier spectral harmonics, perturbation stability stress-testing, and C2PA provenance.
                </p>
              </div>

              {/* Ingestion Dropzone */}
              <Dropzone
                onFileSelected={handleFileSelected}
                isLoading={isLoading}
                selectedPreview={previewUrl}
                onClear={handleClear}
              />

              {/* Execution Actions */}
              {selectedFile && !isLoading && (
                <div className="flex items-center justify-end space-x-3 pt-2">
                  <button
                    type="button"
                    onClick={handleClear}
                    className="px-4 py-2 text-xs font-mono font-medium text-slate-600 hover:text-slate-900 border border-slate-300 rounded hover:bg-slate-100 transition-colors focus-forensic"
                  >
                    Clear Staged Exhibit
                  </button>
                  <button
                    type="button"
                    onClick={handleRunAnalysis}
                    className="px-5 py-2 text-xs font-mono font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded flex items-center space-x-2 transition-colors focus-forensic shadow-xs"
                  >
                    <span>Execute Analysis</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Right Column (5 cols): Workstation Capabilities Brief */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-xs">
                <div className="border-b border-slate-100 pb-3">
                  <span className="text-[10px] font-mono uppercase font-semibold text-slate-400 block">
                    Verification Pipeline
                  </span>
                  <h2 className="text-sm font-bold text-slate-900 mt-0.5">
                    Multimodal Forensic Architecture
                  </h2>
                </div>

                <div className="space-y-3.5 text-xs">
                  <div className="flex items-start space-x-3">
                    <Cpu className="w-4 h-4 text-sky-700 mt-0.5 shrink-0" />
                    <div>
                      <h3 className="font-semibold text-slate-900">Spatial ConvNeXt Attribution</h3>
                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Evaluates patch-level anomalies, synthetic texture blending, and unnatural edge gradients via Grad-CAM receptive field heatmaps.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <Activity className="w-4 h-4 text-emerald-700 mt-0.5 shrink-0" />
                    <div>
                      <h3 className="font-semibold text-slate-900">2D Fourier Spectral Harmonics</h3>
                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Centered 2D FFT inspects radial energy decay. Generative upsamplers introduce high-frequency periodic grid lattice residuals.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <ShieldCheck className="w-4 h-4 text-amber-700 mt-0.5 shrink-0" />
                    <div>
                      <h3 className="font-semibold text-slate-900">Perturbation Invariance & Uncertainty</h3>
                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Exhibits undergo stress tests across compression, blur, and scaling. Fragile logits or decision corridor scores trigger explicit Human Review.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <FileSearch className="w-4 h-4 text-purple-700 mt-0.5 shrink-0" />
                    <div>
                      <h3 className="font-semibold text-slate-900">Provenance & C2PA Credentials</h3>
                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Inspects camera hardware exposure tags and verifies cryptographic C2PA Content Credentials manifests.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 border border-slate-200 rounded text-[11px] text-slate-600 font-mono">
                  <span className="text-slate-400 block uppercase text-[10px]">Calibration Protocol</span>
                  Temperature Scaling T = 0.9986 applied post-hoc to prevent uncalibrated overconfidence.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State: Pipeline Activity Readout */}
        {isLoading && (
          <div className="py-12">
            <AnalysisProgress />
          </div>
        )}

        {/* Error Notification */}
        {errorMsg && (
          <div className="max-w-2xl mx-auto my-6 p-4 border border-rose-200 bg-rose-50 text-rose-900 text-xs rounded flex items-start gap-3 shadow-xs">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-semibold uppercase tracking-wider text-[11px] font-mono text-rose-800">
                Evaluation Error / Backend Notice
              </span>
              <p className="text-xs leading-relaxed text-rose-700">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* State B: Analyzed Forensic Examination Dossier */}
        {prediction && (
          <div className="space-y-6">
            {/* Dossier Control Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-200 pb-3 gap-3">
              <div className="flex items-center space-x-2 text-xs font-mono text-slate-500">
                <span className="font-bold uppercase tracking-wider text-slate-900 text-[11px]">
                  Examination Dossier
                </span>
                <span className="text-slate-300">/</span>
                <span className="text-slate-700 truncate max-w-xs">
                  {selectedFile?.name || "Uploaded Exhibit"}
                </span>
                <span className="text-slate-300 hidden md:inline">/</span>
                <span className="text-slate-500 hidden md:inline">
                  Schema v{prediction.schema_version}
                </span>
              </div>

              <button
                type="button"
                onClick={handleClear}
                className="flex items-center space-x-1.5 text-xs font-mono font-medium text-slate-700 hover:text-slate-900 bg-white border border-slate-300 hover:bg-slate-50 px-3 py-1.5 rounded transition-colors focus-forensic shadow-xs"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
                <span>Examine Another Image</span>
              </button>
            </div>

            {/* 1. Primary Verdict Banner */}
            <VerdictCard prediction={prediction} />

            {/* 2. Spatial Feature Attribution Studio */}
            <SpatialHeatmapViewer
              spatial={prediction.evidence.spatial}
              originalPreview={previewUrl}
            />

            {/* 3. 2D Fourier Spectral Residuals */}
            <SpectralViewer spectral={prediction.evidence.spectral} />

            {/* 4. Perturbation Stability Matrix */}
            <RobustnessPanel
              robustness={prediction.evidence.robustness}
              operatingThreshold={0.50}
            />

            {/* 5. Metadata & Provenance Inspection */}
            <MetadataPanel metadata={prediction.evidence.metadata} />

            {/* 6. Synthesized Forensic Finding & Methodology Limitations */}
            <ExplanationCard prediction={prediction} />
          </div>
        )}
      </main>

      {/* Forensic Workstation Colophon & Disclaimer Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 space-y-1">
          <div className="flex items-center justify-center space-x-2 text-slate-700 font-medium">
            <span>SignalScope Media Forensics</span>
            <span className="text-slate-300">·</span>
            <span>Smart India Hackathon 2026</span>
          </div>
          <p className="text-[11px] font-mono text-slate-400">
            Evaluations represent probabilistic likelihood assessments derived from deep ConvNeXt representations and 2D Fourier spectra. They do not constitute conclusive causal proof.
          </p>
        </div>
      </footer>
    </div>
  );
}
