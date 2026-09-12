"use client";

import React from "react";
import { FileText, Camera, ShieldCheck, AlertCircle, Info } from "lucide-react";
import { MetadataEvidence } from "@/types/prediction";

interface MetadataPanelProps {
  metadata: MetadataEvidence;
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({ metadata }) => {
  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Metadata & Provenance Inspection</h3>
            <p className="text-[11px] text-slate-400">EXIF Headers, C2PA Manifests, and Chunk Tags</p>
          </div>
        </div>

        <span
          className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
            metadata.c2pa_present
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-slate-800 text-slate-400 border-slate-700"
          }`}
        >
          {metadata.c2pa_present ? "C2PA Verified" : "No C2PA Manifest"}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        {/* Camera Provenance */}
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1.5">
          <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center gap-1">
            <Camera className="w-3.5 h-3.5 text-indigo-400" /> Camera Hardware Provenance
          </div>
          <div className="font-semibold text-white">
            {metadata.camera_make || metadata.camera_model
              ? `${metadata.camera_make || ""} ${metadata.camera_model || ""}`.trim()
              : "No Camera EXIF Recorded"}
          </div>
          <p className="text-[11px] text-slate-400">
            {metadata.has_exif ? "EXIF metadata structures detected." : "Stripped or omitted EXIF headers (typical for web/social uploads)."}
          </p>
        </div>

        {/* C2PA & Generator Signatures */}
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1.5">
          <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" /> Content Credentials (C2PA)
          </div>
          <div className="font-semibold text-white">
            {metadata.c2pa_present ? "Cryptographic Manifest Attached" : "No Content Credentials detected."}
          </div>
          <p className="text-[11px] text-slate-400">
            {metadata.software ? `Software tag: ${metadata.software}` : "No generator parameter blocks found."}
          </p>
        </div>
      </div>

      {/* Anomalies if present */}
      {metadata.anomalies && metadata.anomalies.length > 0 && (
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs space-y-1">
          <div className="font-semibold flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> Identified Metadata Anomalies:
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-200/80">
            {metadata.anomalies.map((anom, idx) => (
              <li key={idx}>{anom}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Metadata Note */}
      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
        <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Provenance Guardrail: </span>
          Metadata is supporting contextual evidence only. The absence of C2PA credentials or EXIF headers is common on web platforms and does not imply synthetic origin.
        </div>
      </div>
    </div>
  );
};
