"use client";

import React, { useState, useRef } from "react";
import { Upload, AlertCircle, RefreshCw, CheckCircle, AlertTriangle, HelpCircle } from "lucide-react";

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  isLoading: boolean;
  selectedPreview: string | null;
  onClear: () => void;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelected,
  isLoading,
  selectedPreview,
  onClear,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = (file: File) => {
    setErrorMsg(null);
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      setErrorMsg(`Unsupported format (${file.type || "unknown"}). Allowed: JPEG, PNG, WEBP, BMP.`);
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File size exceeds the 25 MB limit.");
      return;
    }
    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const loadSample = async (samplePath: string, sampleName: string) => {
    setErrorMsg(null);
    try {
      const res = await fetch(samplePath);
      if (!res.ok) throw new Error("Could not load sample file.");
      const blob = await res.blob();
      const file = new File([blob], sampleName, { type: "image/jpeg" });
      onFileSelected(file);
    } catch {
      setErrorMsg("Unable to load sample image. Please upload a local file.");
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !selectedPreview && !isLoading && fileInputRef.current?.click()}
        className={`rounded-lg border transition-colors duration-150 p-6 flex flex-col items-center justify-center cursor-pointer ${
          isDragOver
            ? "border-sky-500 bg-sky-950/20"
            : selectedPreview
            ? "border-slate-800 bg-slate-900/60 cursor-default"
            : "border-dashed border-slate-700 hover:border-slate-500 bg-slate-900/30 hover:bg-slate-900/60"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              validateAndSelect(e.target.files[0]);
            }
          }}
        />

        {selectedPreview ? (
          <div className="flex flex-col items-center space-y-3 w-full">
            <div className="relative group w-44 h-44 rounded border border-slate-700 bg-slate-950 flex items-center justify-center overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedPreview}
                alt="Selected image"
                className="w-full h-full object-contain pixelated"
              />
              <div className="absolute inset-0 bg-slate-950/70 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClear();
                  }}
                  disabled={isLoading}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-600 flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3 h-3" /> Replace
                </button>
              </div>
            </div>
            <div className="text-center">
              <span className="text-xs text-slate-300 font-medium">Image staged for evaluation</span>
              <p className="text-[11px] text-slate-500 mt-0.5">Click &apos;Analyze image&apos; to run inference pipeline</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-2 py-4">
            <Upload className="w-6 h-6 text-slate-400 mb-1" />
            <div>
              <p className="text-sm font-medium text-slate-200">
                Drop an image here or <span className="text-slate-100 underline decoration-slate-600 hover:decoration-slate-300">browse</span>
              </p>
              <p className="text-xs text-slate-500 mt-1">
                JPEG · PNG · WEBP · BMP · Maximum 25 MB
              </p>
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="p-3 rounded border border-rose-900/60 bg-rose-950/30 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Representative Test Cases */}
      {!selectedPreview && !isLoading && (
        <div className="pt-2">
          <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider mb-2">
            Representative test cases:
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => loadSample("/samples/authentic_real.jpg", "sample_real_0955.jpg")}
              className="p-2.5 rounded border border-slate-800 hover:border-slate-600 bg-slate-900/40 hover:bg-slate-900 text-left transition-colors flex items-center gap-2.5"
            >
              <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 truncate">
                  Authentic Real
                </div>
                <div className="text-[10px] font-mono text-slate-500 truncate">Local Val #0955</div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/synthetic_ai.jpg", "sample_ai_3244.jpg")}
              className="p-2.5 rounded border border-slate-800 hover:border-slate-600 bg-slate-900/40 hover:bg-slate-900 text-left transition-colors flex items-center gap-2.5"
            >
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 truncate">
                  Synthetic AI
                </div>
                <div className="text-[10px] font-mono text-slate-500 truncate">Local Val #3244</div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/borderline_uncertain.jpg", "sample_uncertain_5457.jpg")}
              className="p-2.5 rounded border border-slate-800 hover:border-slate-600 bg-slate-900/40 hover:bg-slate-900 text-left transition-colors flex items-center gap-2.5"
            >
              <HelpCircle className="w-4 h-4 text-amber-400 shrink-0" />
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 truncate">
                  Borderline / Uncertain
                </div>
                <div className="text-[10px] font-mono text-slate-500 truncate">Perturbation Volatile #5457</div>
              </div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
