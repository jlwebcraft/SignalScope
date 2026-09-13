"use client";

import React, { useState, useRef } from "react";
import { AlertCircle } from "lucide-react";

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
      setErrorMsg(`Unsupported file format (${file.type || "unknown"}). Allowed: JPEG, PNG, WEBP, BMP.`);
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File size exceeds 25 MB limit.");
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
    <div className="w-full space-y-6">
      {/* Intake Drop Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !selectedPreview && !isLoading && fileInputRef.current?.click()}
        className={`transition-colors duration-150 p-8 flex flex-col items-center justify-center cursor-pointer border ${
          isDragOver
            ? "border-[#121316] bg-[#f3f1ea]"
            : selectedPreview
            ? "border-[#e5e2d9] bg-[#ffffff] cursor-default"
            : "border-dashed border-[#dcd9ce] hover:border-[#121316] bg-[#ffffff]"
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
            <div className="relative group max-w-sm w-full h-64 bg-[#fbfbf9] border border-[#e5e2d9] flex items-center justify-center overflow-hidden p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedPreview}
                alt="Selected evidence image"
                className="w-full h-full object-contain pixelated"
              />
              <div className="absolute inset-0 bg-[#121316]/70 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClear();
                  }}
                  disabled={isLoading}
                  className="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-[#fbfbf9] text-[#121316] hover:bg-[#ffffff] transition-colors"
                >
                  Replace image
                </button>
              </div>
            </div>
            <div className="text-center">
              <span className="text-xs font-mono uppercase tracking-wider text-[#606570]">
                Exhibit Staged for Ingestion
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-2 py-3">
            <p className="text-sm font-medium tracking-tight text-[#121316]">
              DROP IMAGE HERE <span className="text-[#606570] font-normal">or</span> <span className="underline underline-offset-4 decoration-[#8c8a82]">browse files</span>
            </p>
            <p className="text-xs font-mono text-[#8c8a82]">
              JPEG · PNG · WEBP · BMP · Maximum 25 MB
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="p-3 border border-[#fecaca] bg-[#fef2f2] text-[#991b1b] text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Representative Test Cases (Editorial List) */}
      {!selectedPreview && !isLoading && (
        <div className="space-y-2 pt-2">
          <div className="flex items-baseline justify-between border-b border-[#e5e2d9] pb-1.5">
            <span className="text-[10px] font-mono tracking-[0.2em] text-[#606570] uppercase font-semibold">
              Representative Case Archive
            </span>
            <span className="text-[10px] font-mono text-[#8c8a82]">
              Standard Test Partitions
            </span>
          </div>

          <div className="divide-y divide-[#eeece5]">
            <button
              type="button"
              onClick={() => loadSample("/samples/authentic_real.jpg", "sample_real_0955.jpg")}
              className="w-full py-2.5 px-2 text-left hover:bg-[#f3f1ea] transition-colors flex items-center justify-between group"
            >
              <div className="flex items-baseline space-x-3">
                <span className="text-[11px] font-mono text-[#9a3412]">01</span>
                <span className="text-xs font-medium text-[#121316] group-hover:underline">
                  Authentic photographic capture
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#8c8a82]">
                Local Val #0955
              </span>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/synthetic_ai.jpg", "sample_ai_3244.jpg")}
              className="w-full py-2.5 px-2 text-left hover:bg-[#f3f1ea] transition-colors flex items-center justify-between group"
            >
              <div className="flex items-baseline space-x-3">
                <span className="text-[11px] font-mono text-[#9a3412]">02</span>
                <span className="text-xs font-medium text-[#121316] group-hover:underline">
                  Synthetic generative model output
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#8c8a82]">
                Local Val #3244
              </span>
            </button>

            <button
              type="button"
              onClick={() => loadSample("/samples/borderline_uncertain.jpg", "sample_uncertain_5457.jpg")}
              className="w-full py-2.5 px-2 text-left hover:bg-[#f3f1ea] transition-colors flex items-center justify-between group"
            >
              <div className="flex items-baseline space-x-3">
                <span className="text-[11px] font-mono text-[#9a3412]">03</span>
                <span className="text-xs font-medium text-[#121316] group-hover:underline">
                  Perturbation volatile / uncertain case
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#8c8a82]">
                Volatile #5457
              </span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
