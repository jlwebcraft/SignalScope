"use client";

import React from "react";
import { MetadataEvidence } from "@/types/prediction";
import { ShieldAlert } from "lucide-react";

interface MetadataPanelProps {
  metadata: MetadataEvidence;
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({ metadata }) => {
  const cameraDisplay =
    metadata.camera_make || metadata.camera_model
      ? `${metadata.camera_make || ""} ${metadata.camera_model || ""}`.trim()
      : "Not Recorded (EXIF Stripped)";

  return (
    <section className="bg-white border border-slate-200 rounded p-5 shadow-xs space-y-4">
      {/* Section Sub-Header */}
      <div className="flex flex-wrap items-baseline justify-between border-b border-slate-100 pb-3 gap-2">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-[1px] bg-slate-800" aria-hidden="true" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Provenance & Metadata Property Sheet
          </h3>
        </div>

        <span
          className={`px-2 py-0.5 text-[11px] font-mono font-medium rounded border ${
            metadata.c2pa_present
              ? "text-emerald-800 bg-emerald-50 border-emerald-200"
              : "text-slate-600 bg-slate-100 border-slate-200"
          }`}
        >
          {metadata.c2pa_present ? "C2PA Manifest Attached" : "No C2PA Manifest"}
        </span>
      </div>

      {/* Property Sheet Key-Value Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
        <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
          <span className="text-[10px] text-slate-400 uppercase font-semibold block">EXIF Structure</span>
          <div className="font-semibold text-slate-900">
            {metadata.has_exif ? "Structure Present" : "Not Present / Stripped"}
          </div>
          <p className="text-[11px] text-slate-500 font-sans">
            {metadata.has_exif
              ? "Physical exposure and camera hardware tags recorded."
              : "Standard for web re-uploads, social platforms, and synthetic outputs."}
          </p>
        </div>

        <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
          <span className="text-[10px] text-slate-400 uppercase font-semibold block">C2PA Credentials</span>
          <div className="font-semibold text-slate-900">
            {metadata.c2pa_present ? "Cryptographic Manifest Detected" : "No Cryptographic Signature"}
          </div>
          <p className="text-[11px] text-slate-500 font-sans">
            {metadata.c2pa_present
              ? "Cryptographic chain of custody manifest verified."
              : "No C2PA / Content Authenticity Initiative signature block attached."}
          </p>
        </div>

        <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
          <span className="text-[10px] text-slate-400 uppercase font-semibold block">Hardware Sensor Model</span>
          <div className="font-semibold text-slate-900 truncate">
            {cameraDisplay}
          </div>
          <p className="text-[11px] text-slate-500 font-sans">
            Optical capture equipment string.
          </p>
        </div>

        <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
          <span className="text-[10px] text-slate-400 uppercase font-semibold block">Software / Generator Tag</span>
          <div className="font-semibold text-slate-900 truncate">
            {metadata.software || "Not Recorded"}
          </div>
          <p className="text-[11px] text-slate-500 font-sans">
            Processing tool or generative synthesis pipeline signature.
          </p>
        </div>
      </div>

      {/* Identified Anomalies */}
      {metadata.anomalies && metadata.anomalies.length > 0 && (
        <div className="p-3 border border-amber-200 bg-amber-50 rounded text-amber-900 text-xs space-y-1">
          <div className="font-mono text-[11px] font-bold uppercase tracking-wider flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-700 shrink-0" />
            <span>Header Structural Anomalies</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 font-mono text-[11px] text-amber-950/90 pl-5">
            {metadata.anomalies.map((anom, idx) => (
              <li key={idx}>{anom}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Inline caption note */}
      <p className="text-[11px] font-mono text-slate-500 pt-1">
        Metadata omission is standard across web messaging platforms and social networks and does not prove synthetic origin.
      </p>
    </section>
  );
};
