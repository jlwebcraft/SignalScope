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
    <section className="space-y-4 pt-4 border-t border-[#e5e2d9]">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between border-b border-[#e5e2d9] pb-2 gap-2">
        <div className="flex items-baseline space-x-3">
          <span className="text-[10px] font-mono tracking-[0.2em] text-[#9a3412] uppercase font-semibold">
            04
          </span>
          <h3 className="text-lg font-editorial font-semibold text-[#121316]">
            Provenance & Metadata Inspection
          </h3>
        </div>

        <span
          className={`px-2 py-0.5 text-[11px] font-mono font-medium border ${
            metadata.c2pa_present
              ? "text-[#166534] bg-[#f0fdf4] border-[#bbf7d0]"
              : "text-[#606570] bg-[#f3f1ea] border-[#e5e2d9]"
          }`}
        >
          {metadata.c2pa_present ? "C2PA Manifest Attached" : "No C2PA Manifest"}
        </span>
      </div>

      {/* Forensic Evidence Property Sheet */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
        <div className="p-3 bg-[#f3f1ea] border border-[#e5e2d9] space-y-1">
          <div className="text-[10px] text-[#8c8a82] uppercase">EXIF Headers</div>
          <div className="font-semibold text-[#121316]">
            {metadata.has_exif ? "Structure Present" : "Not Present / Stripped"}
          </div>
          <p className="text-[11px] text-[#606570] font-sans">
            {metadata.has_exif ? "Camera exposure tags recorded." : "Typical for web distribution and re-uploads."}
          </p>
        </div>

        <div className="p-3 bg-[#f3f1ea] border border-[#e5e2d9] space-y-1">
          <div className="text-[10px] text-[#8c8a82] uppercase">C2PA Credentials</div>
          <div className="font-semibold text-[#121316]">
            {metadata.c2pa_present ? "Cryptographic Manifest Attached" : "Not Detected"}
          </div>
          <p className="text-[11px] text-[#606570] font-sans">
            {metadata.c2pa_present ? "Cryptographic chain verified." : "No signed provenance block attached."}
          </p>
        </div>

        <div className="p-3 bg-[#f3f1ea] border border-[#e5e2d9] space-y-1">
          <div className="text-[10px] text-[#8c8a82] uppercase">Camera Hardware</div>
          <div className="font-semibold text-[#121316] truncate">
            {cameraDisplay}
          </div>
          <p className="text-[11px] text-[#606570] font-sans">
            Hardware capture identifier.
          </p>
        </div>

        <div className="p-3 bg-[#f3f1ea] border border-[#e5e2d9] space-y-1">
          <div className="text-[10px] text-[#8c8a82] uppercase">Software Tag</div>
          <div className="font-semibold text-[#121316] truncate">
            {metadata.software || "Unavailable"}
          </div>
          <p className="text-[11px] text-[#606570] font-sans">
            Generator or processing tool string.
          </p>
        </div>
      </div>

      {/* Identified Anomalies */}
      {metadata.anomalies && metadata.anomalies.length > 0 && (
        <div className="p-3 border border-[#fde68a] bg-[#fffbeb] text-[#92400e] text-xs space-y-1">
          <div className="font-mono text-[11px] font-bold uppercase tracking-wider">
            Identified Header Anomalies:
          </div>
          <ul className="list-disc list-inside space-y-0.5 font-mono text-[11px] text-[#92400e]/90">
            {metadata.anomalies.map((anom, idx) => (
              <li key={idx}>{anom}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Interpretation & Scope Guardrail */}
      <div className="text-xs text-[#606570] font-sans leading-relaxed pt-1">
        <strong className="text-[#121316] font-mono uppercase text-[10px] tracking-wider">Provenance Interpretation: </strong>
        Metadata absence is standard for modern web platforms and social networks and does not imply synthetic origin.
        Provenance metadata serves as supporting contextual evidence only.
      </div>
    </section>
  );
};
