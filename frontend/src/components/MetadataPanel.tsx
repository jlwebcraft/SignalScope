"use client";

import React from "react";
import { MetadataEvidence } from "@/types/prediction";

interface MetadataPanelProps {
  metadata: MetadataEvidence;
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({ metadata }) => {
  const cameraDisplay =
    metadata.camera_make || metadata.camera_model
      ? `${metadata.camera_make || ""} ${metadata.camera_model || ""}`.trim()
      : "Unavailable (No EXIF)";

  return (
    <div className="forensic-panel flex flex-col justify-between p-4 space-y-3.5">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div>
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            4. Provenance & Metadata Inspection
          </h3>
          <p className="text-[11px] font-mono text-slate-500 mt-0.5">
            EXIF Structure · C2PA Manifest Verification
          </p>
        </div>

        <span
          className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium border ${
            metadata.c2pa_present
              ? "text-emerald-400 border-emerald-800 bg-emerald-950/40"
              : "text-slate-400 border-slate-700 bg-slate-900/60"
          }`}
        >
          {metadata.c2pa_present ? "C2PA Verified" : "No C2PA Manifest"}
        </span>
      </div>

      {/* Forensic Property Sheet */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
        <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="text-[10px] text-slate-500 uppercase">EXIF Headers</div>
          <div className="text-slate-200 font-medium">
            {metadata.has_exif ? "Structures Detected" : "Not Present / Stripped"}
          </div>
          <p className="text-[10px] text-slate-500 font-sans">
            {metadata.has_exif ? "Standard camera metadata present." : "Typical for web & social platforms."}
          </p>
        </div>

        <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="text-[10px] text-slate-500 uppercase">Content Credentials</div>
          <div className="text-slate-200 font-medium">
            {metadata.c2pa_present ? "C2PA Claim Attached" : "Not Detected"}
          </div>
          <p className="text-[10px] text-slate-500 font-sans">
            {metadata.c2pa_present ? "Cryptographic provenance verified." : "No signed provenance block."}
          </p>
        </div>

        <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="text-[10px] text-slate-500 uppercase">Camera Hardware</div>
          <div className="text-slate-200 font-medium truncate">
            {cameraDisplay}
          </div>
          <p className="text-[10px] text-slate-500 font-sans">Hardware device recording.</p>
        </div>

        <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="text-[10px] text-slate-500 uppercase">Software Signature</div>
          <div className="text-slate-200 font-medium truncate">
            {metadata.software || "Unavailable"}
          </div>
          <p className="text-[10px] text-slate-500 font-sans">Generator or editing tool tag.</p>
        </div>
      </div>

      {/* Identified Anomalies */}
      {metadata.anomalies && metadata.anomalies.length > 0 && (
        <div className="p-2.5 rounded bg-amber-950/25 border border-amber-800/60 text-amber-200 text-xs space-y-1">
          <div className="font-semibold text-[11px] text-amber-300 uppercase">
            Identified Header Anomalies:
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-300/80 font-mono">
            {metadata.anomalies.map((anom, idx) => (
              <li key={idx}>{anom}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Provenance Scope Guardrail */}
      <div className="p-2.5 rounded bg-slate-900/50 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed font-sans">
        <span className="font-semibold text-slate-300">Provenance Scope: </span>
        Metadata absence is routine across modern web services and does not indicate synthetic origin. It serves as supporting forensic context only.
      </div>
    </div>
  );
};
