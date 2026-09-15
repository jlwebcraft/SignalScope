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
    <section className="bg-white border border-slate-200 rounded overflow-hidden shadow-xs">
      {/* Section Sub-Header */}
      <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-2 h-2 rounded-[1px] bg-slate-700" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900">
            Provenance & Metadata Inspection
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

      <div className="p-5 space-y-4">
        {/* Technical Property Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">EXIF Structure</span>
            <div className="font-semibold text-slate-900">
              {metadata.has_exif ? "Structure Present" : "Not Present / Stripped"}
            </div>
            <p className="text-[11px] text-slate-500 font-sans">
              {metadata.has_exif
                ? "Physical exposure, lens focal length, or timestamp tags recorded."
                : "Standard for social messaging apps, web re-uploads, and generative outputs."}
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">C2PA Content Credentials</span>
            <div className="font-semibold text-slate-900">
              {metadata.c2pa_present ? "Cryptographic Manifest Detected" : "No Cryptographic Signature"}
            </div>
            <p className="text-[11px] text-slate-500 font-sans">
              {metadata.c2pa_present
                ? "Cryptographic chain of custody manifest attached."
                : "No C2PA / CAI digital signature block detected in file headers."}
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Hardware Identifier</span>
            <div className="font-semibold text-slate-900 truncate">
              {cameraDisplay}
            </div>
            <p className="text-[11px] text-slate-500 font-sans">
              Optical sensor capture equipment manufacturer and model string.
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Software / Processing Tag</span>
            <div className="font-semibold text-slate-900 truncate">
              {metadata.software || "Not Recorded"}
            </div>
            <p className="text-[11px] text-slate-500 font-sans">
              Editing software or generator pipeline signature recorded in headers.
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

        {/* Methodological Guidance */}
        <div className="p-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-600 leading-relaxed font-sans">
          <strong className="font-mono text-[10px] text-slate-900 uppercase font-semibold">Provenance Scope: </strong>
          Metadata omission is ubiquitous across the modern internet due to platform re-encoding (e.g. WhatsApp, X, Instagram) and
          does not prove synthetic origin. Provenance serves solely as corroborating contextual data.
        </div>
      </div>
    </section>
  );
};
