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
import { AlertCircle, ArrowRight, RotateCcw, Printer } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
    setPrediction(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Extract natural dimensions
    const img = new Image();
    img.onload = () => {
      setDimensions(`${img.naturalWidth} × ${img.naturalHeight} px`);
    };
    img.src = url;
  };

  const handleClear = () => {
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setDimensions(null);
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

  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#f8f9fa] text-[#334155]">
      <Header apiBaseUrl={API_BASE_URL} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* State A: Initial Ingestion & Workstation Workspace */}
        {!prediction && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column (7 cols): Ingestion Console */}
            <div className="lg:col-span-7 space-y-6">
              <div className="space-y-1 border-b border-slate-200 pb-4">
                <div className="flex items-center space-x-2">
                  <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-sky-800">
                    Forensic Ingestion Console
                  </span>
                </div>
                <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
                  Digital Media Authenticity Verification
                </h1>
                <p className="text-xs text-slate-600 leading-relaxed pt-1">
                  Upload a digital still image to execute multimodal forensic verification across spatial ConvNeXt feature attribution,
                  2D Fourier spectral residuals, perturbation stability stress-testing, and C2PA provenance.
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

            {/* Right Column (5 cols): Engine Specification & Diagnostic Telemetry */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-white border border-slate-200 rounded p-5 space-y-4 shadow-xs">
                <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase font-semibold text-slate-400 block">
                      Workstation Instrumentation
                    </span>
                    <h2 className="text-sm font-bold text-slate-900 mt-0.5">
                      Engine Specification & Architecture
                    </h2>
                  </div>
                  <span className="text-[10px] font-mono bg-sky-50 text-sky-800 border border-sky-200 px-2 py-0.5 rounded font-semibold uppercase">
                    v2.0 Release
                  </span>
                </div>

                {/* Technical Parameter Readout Sheet */}
                <div className="space-y-3 font-mono text-xs divide-y divide-slate-100">
                  <div className="flex items-center justify-between pt-1">
                    <span className="text-slate-500">Detector Backbone</span>
                    <span className="font-semibold text-slate-900 text-right">ConvNeXt-Tiny (12k ft 1k)</span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Registered Artifact</span>
                    <span className="font-semibold text-sky-800 bg-sky-50 px-1.5 py-0.5 rounded border border-sky-200">
                      signalscope-v2
                    </span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Calibration Math</span>
                    <span className="text-slate-800">Platt Scaling (T = 0.9986)</span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Patch Dimension</span>
                    <span className="text-slate-800">3 × 32 × 32 px (Input: 224×224)</span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Decision Policy</span>
                    <span className="text-slate-800">0.50 Threshold / 0.40–0.60 Corridor</span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Perturbation Suite</span>
                    <span className="text-slate-800">JPEG Q=95/85/70, Downscale, Crop</span>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <span className="text-slate-500">Inspection Layers</span>
                    <span className="text-slate-800">Spatial, Fourier, Stress, C2PA</span>
                  </div>
                </div>

                {/* Verification Integrity Notice */}
                <div className="p-3 bg-slate-50 border border-slate-200 rounded text-[11px] text-slate-600 font-sans leading-relaxed">
                  <strong className="font-mono text-[10px] text-slate-900 uppercase font-semibold block">
                    Holdout Integrity Guarantee:
                  </strong>
                  SignalScope v2 is verified across cross-generator datasets and independent photographic holdouts.
                  The official SIH held-out test partition remains strictly untouched.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State: Real-Time Pipeline Activity Readout */}
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
            {/* Forensic Docket Header */}
            <div className="bg-white border border-slate-200 rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-[1px] bg-slate-900" />
                  <span className="font-mono font-bold text-xs uppercase tracking-wider text-slate-900">
                    Forensic Examination Docket
                  </span>
                  <span className="text-slate-300">/</span>
                  <span className="font-mono text-xs text-sky-800 font-medium">
                    Schema v{prediction.schema_version}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs font-mono text-slate-600 pt-0.5">
                  <div>
                    <span className="text-slate-400">Exhibit: </span>
                    <span className="font-semibold text-slate-800">{selectedFile?.name || "Uploaded Exhibit"}</span>
                  </div>
                  {dimensions && (
                    <div>
                      <span className="text-slate-400">Resolution: </span>
                      <span className="text-slate-800">{dimensions}</span>
                    </div>
                  )}
                  <div>
                    <span className="text-slate-400">Engine: </span>
                    <span className="text-slate-800">signalscope-v2</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2 shrink-0">
                <button
                  type="button"
                  onClick={handlePrint}
                  className="flex items-center space-x-1.5 text-xs font-mono font-medium text-slate-700 hover:text-slate-900 bg-slate-50 border border-slate-300 hover:bg-slate-100 px-3 py-1.5 rounded transition-colors focus-forensic"
                  title="Print or export examination report"
                >
                  <Printer className="w-3.5 h-3.5 text-slate-500" />
                  <span>Print Report</span>
                </button>

                <button
                  type="button"
                  onClick={handleClear}
                  className="flex items-center space-x-1.5 text-xs font-mono font-semibold text-white bg-slate-900 hover:bg-slate-800 px-3.5 py-1.5 rounded transition-colors focus-forensic shadow-xs"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Examine New Exhibit</span>
                </button>
              </div>
            </div>

            {/* 1. Primary Verdict & Calibrated Adjudication Banner */}
            <VerdictCard prediction={prediction} />

            {/* 2. Optical Receptive Field Comparator (Dark Canvas) */}
            <SpatialHeatmapViewer
              spatial={prediction.evidence.spatial}
              originalPreview={previewUrl}
            />

            {/* 3. 2D Fourier Spectrogram & High-Frequency Residuals Deck */}
            <SpectralViewer spectral={prediction.evidence.spectral} />

            {/* 4. Perturbation Stability Stress-Testing Matrix */}
            <RobustnessPanel
              robustness={prediction.evidence.robustness}
              operatingThreshold={0.50}
            />

            {/* 5. Provenance & Metadata Property Sheet */}
            <MetadataPanel metadata={prediction.evidence.metadata} />

            {/* 6. Synthesized Forensic Findings & Adjudication Report */}
            <ExplanationCard prediction={prediction} />
          </div>
        )}
      </main>

      {/* Forensic Workstation Colophon Footer */}
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
