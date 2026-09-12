"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, Image as ImageIcon, Sparkles, CheckCircle2, AlertTriangle, HelpCircle, RefreshCw } from "lucide-react";

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
      setErrorMsg("File size exceeds the 25MB limit.");
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
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-8 flex flex-col items-center justify-center cursor-pointer ${
          isDragOver
            ? "border-cyan-400 bg-cyan-950/20 shadow-lg shadow-cyan-500/10"
            : selectedPreview
            ? "border-slate-700 bg-slate-900/40 cursor-default"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/20 hover:bg-slate-900/40"
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
          <div className="flex flex-col items-center space-y-4 w-full">
            <div className="relative group w-48 h-48 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center shadow-xl">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedPreview}
                alt="Selected preview"
                className="w-full h-full object-contain pixelated"
              />
              <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClear();
                  }}
                  disabled={isLoading}
                  className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-lg flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Replace Image
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-400">
              Image loaded and prepared for multimodal inference.
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-3">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 group-hover:scale-105 transition-transform">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">
                Drag and drop your image here, or{" "}
                <span className="text-indigo-400 hover:text-indigo-300 underline">browse</span>
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports JPEG, PNG, WEBP, BMP up to 25MB (Native 32×32 to high-res capture)
              </p>
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Quick Sample Selector */}
      {!selectedPreview && !isLoading && (
        <div className="glass-panel rounded-xl p-3 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Or evaluate representative test cases:
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => loadSample("/samples/authentic_real.jpg", "sample_real_0955.jpg")}
              className="px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-emerald-500/40 text-left transition-all group flex items-center gap-2.5"
            >
              <div className="w-7 h-7 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center shrink-0">
                <CheckCircle2 className="w-4 h-4" />
              </div>
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 group-hover:text-emerald-400 truncate">
                  Authentic Real
                </div>
                <div className="text-[10px] text-slate-500 truncate">Local Val Sample #0955</div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/synthetic_ai.jpg", "sample_ai_3244.jpg")}
              className="px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-rose-500/40 text-left transition-all group flex items-center gap-2.5"
            >
              <div className="w-7 h-7 rounded-lg bg-rose-500/10 text-rose-400 flex items-center justify-center shrink-0">
                <Sparkles className="w-4 h-4" />
              </div>
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 group-hover:text-rose-400 truncate">
                  Synthetic AI
                </div>
                <div className="text-[10px] text-slate-500 truncate">Local Val Sample #3244</div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/borderline_uncertain.jpg", "sample_uncertain_5457.jpg")}
              className="px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-amber-500/40 text-left transition-all group flex items-center gap-2.5"
            >
              <div className="w-7 h-7 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center shrink-0">
                <HelpCircle className="w-4 h-4" />
              </div>
              <div className="overflow-hidden">
                <div className="text-xs font-semibold text-slate-200 group-hover:text-amber-400 truncate">
                  Borderline / Uncertain
                </div>
                <div className="text-[10px] text-slate-500 truncate">Perturbation Volatile #5457</div>
              </div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
