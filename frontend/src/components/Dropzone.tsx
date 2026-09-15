"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, AlertCircle } from "lucide-react";

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  isLoading: boolean;
  selectedPreview: string | null;
  onClear: () => void;
}

interface FileMetadata {
  name: string;
  sizeFormatted: string;
  type: string;
  dimensions?: string;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelected,
  isLoading,
  selectedPreview,
  onClear,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fileMeta, setFileMeta] = useState<FileMetadata | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validateAndSelect = (file: File) => {
    setErrorMsg(null);
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    const fileType = file.type.toLowerCase();

    if (!validTypes.includes(fileType)) {
      setErrorMsg(`Unsupported file format (${file.type || "unknown"}). Allowed: JPEG, PNG, WEBP, BMP.`);
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File size exceeds the 25 MB ingestion limit.");
      return;
    }

    // Extract natural dimensions
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      setFileMeta({
        name: file.name,
        sizeFormatted: formatFileSize(file.size),
        type: file.type || "image/jpeg",
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`,
      });
      URL.revokeObjectURL(objectUrl);
    };
    img.onerror = () => {
      setFileMeta({
        name: file.name,
        sizeFormatted: formatFileSize(file.size),
        type: file.type || "image/jpeg",
      });
      URL.revokeObjectURL(objectUrl);
    };
    img.src = objectUrl;

    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setFileMeta(null);
    setErrorMsg(null);
    onClear();
  };

  const loadSample = async (samplePath: string, sampleName: string) => {
    setErrorMsg(null);
    try {
      const res = await fetch(samplePath);
      if (!res.ok) throw new Error("Could not load benchmark sample file.");
      const blob = await res.blob();
      const file = new File([blob], sampleName, { type: "image/jpeg" });
      validateAndSelect(file);
    } catch {
      setErrorMsg("Unable to load benchmark sample image. Please upload a local image file.");
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Precision Forensic Ingestion Area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !selectedPreview && !isLoading && fileInputRef.current?.click()}
        className={`relative border rounded transition-colors duration-150 p-6 flex flex-col items-center justify-center ${
          isDragOver
            ? "border-sky-600 bg-sky-50/40"
            : selectedPreview
            ? "border-slate-300 bg-white"
            : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/50 cursor-pointer"
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
          <div className="flex flex-col md:flex-row items-center gap-6 w-full">
            {/* Image Preview Box */}
            <div className="relative shrink-0 w-48 h-48 bg-slate-100 border border-slate-200 rounded p-1 flex items-center justify-center overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedPreview}
                alt="Staged forensic exhibit"
                className="w-full h-full object-contain pixelated"
              />
            </div>

            {/* Exhibit Metadata Readout */}
            <div className="flex-1 w-full space-y-3">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-600" />
                  <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Exhibit Staged for Ingestion
                  </span>
                </div>
                <h3 className="text-sm font-semibold text-slate-900 truncate max-w-md">
                  {fileMeta?.name || "Uploaded Image"}
                </h3>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs text-slate-600 bg-slate-50 p-3 rounded border border-slate-200">
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block">Size</span>
                  <span className="font-medium text-slate-800">{fileMeta?.sizeFormatted || "—"}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block">Format</span>
                  <span className="font-medium text-slate-800 uppercase">
                    {fileMeta?.type.replace("image/", "") || "JPEG"}
                  </span>
                </div>
                <div className="col-span-2 sm:col-span-1">
                  <span className="text-[10px] text-slate-400 uppercase block">Resolution</span>
                  <span className="font-medium text-slate-800">{fileMeta?.dimensions || "Calculating..."}</span>
                </div>
              </div>

              <div className="flex items-center space-x-3 pt-1">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  disabled={isLoading}
                  className="text-xs font-medium text-slate-700 hover:text-slate-900 underline underline-offset-4 focus-forensic"
                >
                  Change file
                </button>
                <span className="text-slate-300">|</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClear();
                  }}
                  disabled={isLoading}
                  className="text-xs font-medium text-rose-700 hover:text-rose-900 focus-forensic"
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-3 py-6">
            <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-600 border border-slate-200">
              <UploadCloud className="w-5 h-5 text-slate-700" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold text-slate-900">
                Drag and drop image here, or <span className="text-sky-700 underline underline-offset-4">browse files</span>
              </p>
              <p className="text-xs text-slate-500">
                JPEG, PNG, WebP, BMP · Maximum file size 25 MB
              </p>
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="p-3 border border-rose-200 bg-rose-50 text-rose-800 text-xs rounded flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Benchmark Reference Cases (Curated Partitions) */}
      {!selectedPreview && !isLoading && (
        <div className="border border-slate-200 rounded bg-white overflow-hidden">
          <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700">
              Standard Benchmark Partitions
            </span>
            <span className="text-[11px] font-mono text-slate-500">
              1-Click Reference Cases
            </span>
          </div>

          <div className="divide-y divide-slate-100 text-xs">
            <button
              type="button"
              onClick={() => loadSample("/samples/authentic_real.jpg", "sample_real_0955.jpg")}
              className="w-full px-4 py-2.5 text-left hover:bg-slate-50 flex items-center justify-between group transition-colors focus-forensic"
            >
              <div className="flex items-center space-x-3">
                <span className="w-5 font-mono text-[11px] text-slate-400 font-semibold group-hover:text-slate-900">
                  01
                </span>
                <div>
                  <span className="font-semibold text-slate-900 block group-hover:text-sky-800">
                    Authentic Photographic Capture
                  </span>
                  <span className="text-[11px] text-slate-500">
                    Physical optical lens sensor capture with smooth continuous power-law spectral decay
                  </span>
                </div>
              </div>
              <span className="font-mono text-[11px] text-emerald-800 font-medium bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded shrink-0">
                Case #0955
              </span>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/synthetic_ai.jpg", "sample_ai_3244.jpg")}
              className="w-full px-4 py-2.5 text-left hover:bg-slate-50 flex items-center justify-between group transition-colors focus-forensic"
            >
              <div className="flex items-center space-x-3">
                <span className="w-5 font-mono text-[11px] text-slate-400 font-semibold group-hover:text-slate-900">
                  02
                </span>
                <div>
                  <span className="font-semibold text-slate-900 block group-hover:text-sky-800">
                    Synthetic Generative Model Output
                  </span>
                  <span className="text-[11px] text-slate-500">
                    Deep generative architecture with periodic high-frequency Fourier grid artifacts
                  </span>
                </div>
              </div>
              <span className="font-mono text-[11px] text-rose-800 font-medium bg-rose-50 border border-rose-200 px-2 py-0.5 rounded shrink-0">
                Case #3244
              </span>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/borderline_uncertain.jpg", "sample_uncertain_5457.jpg")}
              className="w-full px-4 py-2.5 text-left hover:bg-slate-50 flex items-center justify-between group transition-colors focus-forensic"
            >
              <div className="flex items-center space-x-3">
                <span className="w-5 font-mono text-[11px] text-slate-400 font-semibold group-hover:text-slate-900">
                  03
                </span>
                <div>
                  <span className="font-semibold text-slate-900 block group-hover:text-sky-800">
                    Perturbation Volatile / Boundary Candidate
                  </span>
                  <span className="text-[11px] text-slate-500">
                    Strong model logit that experiences categorical decision flips under JPEG perturbation
                  </span>
                </div>
              </div>
              <span className="font-mono text-[11px] text-amber-800 font-medium bg-amber-50 border border-amber-200 px-2 py-0.5 rounded shrink-0">
                Case #5457
              </span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
